import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer
transformers.logging.set_verbosity_error()

from .layers.embed import PatchEmbedding
from .layers.RevIN import RevIN

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TimeLLM(nn.Module):

    supported_tasks = ["forecasting", "anomaly_detection", "semantic_segmentation"]
    supported_modes = ["univariate"]

    def __init__(self, config, dataset):
        super(TimeLLM, self).__init__()
        self.config = config
        self.model_config = self.config.models.timellm

        self.pred_len = self.config.pred_len
        self.seq_len = self.config.history_len

        self.d_ff = self.model_config.d_ff
        self.d_model = self.model_config.d_model
        self.n_attention_heads = self.model_config.n_heads
        self.num_tokens = self.model_config.num_tokens
        self.dropout = self.config.training.dropout
        self.n_lags = 5

        self.patch_len = self.model_config.patching.patch_len
        self.stride = self.model_config.patching.stride
        self.n_patches = int((self.seq_len - self.patch_len) / self.stride + 2)

        self.task = self.config.task
        self.task_description = self.get_task_description()
        self.dataset_description = dataset.description

        self.setup_llm()

        self.normalize_layers = RevIN(dataset.n_features, affine=False)
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.patch_embedding = PatchEmbedding(self.d_model, self.patch_len, self.stride, self.dropout, pos_embed=False)
        self.reprogramming_layer = ReprogrammingLayer(self.d_model, self.n_attention_heads, self.d_ff, self.d_llm, attention_dropout=self.dropout)
        self.output_projection = FlattenHead(dataset.n_features, self.d_ff * self.n_patches, self.pred_len, head_dropout=0)

        if not self.llm_enabled:
            self.llm_replacement = nn.Sequential(
                nn.Linear(self.d_llm, self.d_llm),
                nn.GELU(),
                nn.Linear(self.d_llm, self.d_ff),
                nn.LayerNorm(self.d_ff),
            )

        if self.task == "semantic_segmentation":
            if dataset.n_classes != 2:
                raise ValueError("TimeLLM only supports binary classification")

    def setup_llm(self):
        self.llm_enabled = self.model_config.llm.enabled
        self.llm_id = self.model_config.llm.llm
        self.llm_layers = self.model_config.llm.llm_layers

        llm_config = AutoConfig.from_pretrained(
            self.llm_id,
            trust_remote_code = True,
        )
        if self.llm_layers > 0:
            llm_config.num_hidden_layers = self.llm_layers
        llm_config.output_hidden_states = True

        llm = AutoModel.from_pretrained(
            self.llm_id,
            config = llm_config,
            load_in_4bit = self.model_config.llm.load_in_4bit,
            trust_remote_code = True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.llm_id,
            trust_remote_code = True,
        )

        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            tokenizer.add_special_tokens({"pad_token": pad_token})
            tokenizer.pad_token = pad_token

        self.word_embeddings = llm.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]

        self.d_llm = llm_config.hidden_size

        if self.llm_enabled:
            self.llm = llm
            self.tokenizer = tokenizer

            for param in self.llm.parameters():
                param.requires_grad = False

    def forward(self, x_enc, x_dec):
        forecast_fn = self.forecast if self.llm_enabled else self.forecast_nollm
        if self.task in ["forecasting", "anomaly_detection"]:
            return forecast_fn(x_enc, x_dec)
        if self.task == "semantic_segmentation":
            pred = forecast_fn(x_enc, x_dec)
            pred = F.sigmoid(pred)
            return pred
        else:
            raise ValueError(f"Task {self.task_name} not implemented")

    def forecast(self, x_enc, x_dec=None):

        x_enc = x_enc.unsqueeze(-1)
        x_enc = self.normalize_layers(x_enc, "norm")

        with torch.no_grad():
            min_values = torch.min(x_enc, dim=1)[0]
            max_values = torch.max(x_enc, dim=1)[0]
            medians = torch.median(x_enc.float(), dim=1).values
            lags = calcute_lags(x_enc.float(), self.n_lags)
            trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.size(0)):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.dataset_description} "
                f"Task description: {self.task_description} "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm.get_input_embeddings()(prompt.to(x_enc.device))

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, _ = self.patch_embedding(x_enc)
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        llm_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm(inputs_embeds=llm_enc_out).last_hidden_state
        dec_out = dec_out.to(x_enc.dtype)

        dec_out = dec_out[:, -self.n_patches:, :self.d_ff]
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.output_projection(dec_out)

        dec_out = self.normalize_layers(dec_out.unsqueeze(-1), "denorm").squeeze(-1)

        return dec_out

    def forecast_nollm(self, x_enc, x_dec=None):
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.unsqueeze(-1)
        x_enc = self.normalize_layers(x_enc, "norm")

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, _ = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        dec_out = self.llm_replacement(enc_out)

        dec_out = dec_out[:, -self.n_patches:, :self.d_ff]
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.output_projection(dec_out)

        dec_out = self.normalize_layers(dec_out.unsqueeze(-1), "denorm").squeeze(-1)

        return dec_out

    def get_task_description(self):
        if self.task == "forecasting":
            self.task_description = f"Forecast the next {self.pred_len} steps given the previous {self.seq_len} steps of data."
        elif self.task == "anomaly_detection":
            self.task_description = f"Reconstruct the past {self.seq_len} steps of data as accurately as possible using the following information."
        elif self.task == "semantic_segmentation":
            self.task_description = f"Classify the past {self.seq_len} steps of data as accurately as possible using the following information."
        else:
            raise ValueError(f"Task {self.task} is not supported.")
        return self.task_description

def calcute_lags(x, n_lags=5):
    x = x.permute(0, 2, 1).contiguous()
    q_fft = torch.fft.rfft(x, dim=-1)
    k_fft = torch.fft.rfft(x, dim=-1)
    res = q_fft * torch.conj(k_fft)
    corr = torch.fft.irfft(res, dim=-1)
    mean_value = torch.mean(corr, dim=1)
    _, lags = torch.topk(mean_value, n_lags, dim=-1)
    return lags


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys, d_llm, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
