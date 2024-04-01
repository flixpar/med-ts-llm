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

    supported_tasks = ["forecasting", "anomaly_detection", "semantic_segmentation", "segmentation"]
    supported_modes = ["univariate", "multivariate"]

    def __init__(self, config, dataset):
        super(TimeLLM, self).__init__()
        self.config = config
        self.model_config = self.config.models.timellm

        self.pred_len = self.config.pred_len
        self.seq_len = self.config.history_len

        self.task = self.config.task
        self.task_description = self.get_task_description()
        self.dataset_description = dataset.description

        self.d_ff = self.model_config.d_ff
        self.d_model = self.model_config.d_model
        self.n_attention_heads = self.model_config.n_heads
        self.num_tokens = self.model_config.num_tokens
        self.dropout = self.config.training.dropout
        self.n_lags = 5

        self.patch_len = self.model_config.patching.patch_len
        self.stride = self.model_config.patching.stride
        self.n_patches = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.d_patch = self.d_model

        self.covariate_mode = self.model_config.covariate_mode
        self.n_features = dataset.n_features

        self.n_classes = dataset.n_classes if self.task in ["classification", "semantic_segmentation"] else 0

        if self.task in ["forecasting", "anomaly_detection"]:
            self.n_outputs_per_step = self.n_features
        elif self.task == "semantic_segmentation":
            self.n_outputs_per_step = self.n_classes if self.n_classes > 2 else 1
        elif self.task == "segmentation":
            self.n_outputs_per_step = 1
        else:
            raise ValueError(f"Task {self.task} is not supported.")
        self.n_outputs = self.n_outputs_per_step * self.pred_len

        match self.covariate_mode:
            case "univariate":
                assert self.n_features == 1
            case "interleave":
                self.n_patches *= self.n_features
            case "concat":
                self.d_model *= self.n_features
            case "independent":
                # self.n_outputs = self.pred_len
                # self.indep_projection = nn.Linear(self.n_features, self.n_outputs_per_step)
                pass
            case "add":
                pass
            case _:
                raise ValueError(f"Unknown covariate mode {self.covariate_mode}")

        self.setup_llm()

        self.normalize_layers = RevIN(self.n_features, affine=False)
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.patch_embedding = PatchEmbedding(self.d_patch, self.patch_len, self.stride, self.dropout, pos_embed=False)
        self.reprogramming_layer = ReprogrammingLayer(self.d_model, self.n_attention_heads, self.d_ff, self.d_llm, attention_dropout=self.dropout)
        self.output_projection = FlattenHead(self.d_ff * self.n_patches, self.n_outputs, head_dropout=0)

        if not self.llm_enabled:
            self.llm_replacement = nn.Sequential(
                nn.Linear(self.d_llm, self.d_llm),
                nn.GELU(),
                nn.Linear(self.d_llm, self.d_ff),
                nn.LayerNorm(self.d_ff),
            )

    def setup_llm(self):
        self.llm_enabled = self.model_config.llm.enabled
        self.llm_id = self.model_config.llm.llm
        self.llm_layers = self.model_config.llm.llm_layers

        cache_dir = self.config.get("paths", {}).get("llm_path")
        if cache_dir == "" or cache_dir == "none":
            cache_dir = None

        llm_config = AutoConfig.from_pretrained(
            self.llm_id,
            cache_dir = cache_dir,
            trust_remote_code = True,
        )
        if self.llm_layers > 0:
            llm_config.num_hidden_layers = self.llm_layers
        llm_config.output_hidden_states = True

        llm = AutoModel.from_pretrained(
            self.llm_id,
            config = llm_config,
            load_in_4bit = self.model_config.llm.load_in_4bit,
            cache_dir = cache_dir,
            trust_remote_code = True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.llm_id,
            cache_dir = cache_dir,
            trust_remote_code = True,
        )

        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            tokenizer.add_special_tokens({"pad_token": pad_token})
            tokenizer.pad_token = pad_token

        self.word_embeddings = llm.get_input_embeddings().weight.clone().detach()
        self.vocab_size = self.word_embeddings.shape[0]

        self.d_llm = llm_config.hidden_size

        if self.llm_enabled:
            self.llm = llm
            self.tokenizer = tokenizer

            for param in self.llm.parameters():
                param.requires_grad = False

    def state_dict(self):
        state_dict = super().state_dict()

        if self.llm_enabled:
            llm_keys = [k for k in state_dict.keys() if k[:4] == "llm."]
            for k in llm_keys:
                del state_dict[k]

        return state_dict

    def forward(self, inputs):
        pred = self.predict(inputs)

        if self.task == "semantic_segmentation":
            if self.n_classes > 2:
                pred = F.softmax(pred, dim=-1)
            else:
                pred = F.sigmoid(pred)
        elif self.task == "segmentation":
            if self.config.tasks.segmentation.mode == "boundary-prediction":
                pred = F.sigmoid(pred)
            elif self.config.tasks.segmentation.mode == "steps-to-boundary":
                raise NotImplementedError("Steps-to-boundary segmentation not yet implemented for TimeLLM")
            else:
                raise ValueError(f"Segmentation mode {self.config.tasks.segmentation.mode} not implemented for TimeLLM")

        return pred

    def predict(self, inputs):
        x_enc = inputs["x_enc"]

        if x_enc.ndim == 2:
            x_enc = x_enc.unsqueeze(-1)

        bs, seq_len, n_features = x_enc.size()
        assert n_features == self.n_features

        x_enc = self.normalize_layers(x_enc, "norm")

        x_enc = x_enc.permute(0, 2, 1).contiguous() # [bs, n_features, seq_len]
        enc_out, _ = self.patch_embedding(x_enc)    # [bs * n_features, n_patches, d_patch]

        if self.covariate_mode == "concat":
            enc_out = enc_out.reshape(bs, n_features, self.n_patches, self.d_patch)
            enc_out = enc_out.permute(0, 2, 1, 3) # [bs, n_patches, n_features, d_patch]
            enc_out = enc_out.reshape(bs, self.n_patches, n_features * self.d_patch)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings) # [bs * n_features, n_patches, d_llm]

        if self.covariate_mode == "add":
            enc_out = enc_out.reshape(bs, n_features, self.n_patches, self.d_llm)
            enc_out = enc_out.mean(dim=1)                           # [bs, n_patches, d_llm]
        elif self.covariate_mode == "interleave":
            enc_out = enc_out.reshape(bs, n_features, -1, self.d_llm) # [bs, n_features, n_patches, d_llm]
            enc_out = enc_out.permute(0, 2, 1, 3)                     # [bs, n_patches, n_features, d_llm]
            enc_out = enc_out.reshape(bs, -1, self.d_llm)             # [bs, n_patches * n_features, d_llm]

        if self.llm_enabled:
            prompt = self.build_prompt(inputs)
            prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
            prompt_embeddings = self.llm.get_input_embeddings()(prompt.to(x_enc.device)) # [bs, n_tok, d_llm]

            if self.covariate_mode == "independent":
                prompt_embeddings = prompt_embeddings.repeat_interleave(n_features, dim=0)

            llm_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
            dec_out = self.llm(inputs_embeds=llm_enc_out).last_hidden_state  # [bs, n_tok + n_patches, d_llm]
            dec_out = dec_out.to(x_enc.dtype)
        else:
            dec_out = self.llm_replacement(enc_out)

        dec_out = dec_out[:, -self.n_patches:, :self.d_ff]
        dec_out = dec_out.permute(0, 2, 1).contiguous() # [bs, d_ff, n_patches]
        dec_out = self.output_projection(dec_out)       # [bs, pred_len * n_features]

        if self.covariate_mode == "independent":
            # dec_out = dec_out.view(bs, self.n_features, self.pred_len)
            # dec_out = dec_out.permute(0, 2, 1).contiguous() # [bs, pred_len, n_features]
            # dec_out = self.indep_projection(dec_out).squeeze(-1) # [bs, pred_len, n_outputs_per_step]
            dec_out = dec_out.view(bs, self.n_features, self.pred_len, self.n_outputs_per_step).squeeze(-1)
            dec_out = dec_out.mean(dim=1)
        else:
            dec_out = dec_out.view(bs, self.pred_len, self.n_outputs_per_step)

        if self.task in ["forecasting", "anomaly_detection"]:
            dec_out = self.normalize_layers(dec_out, "denorm")
        else:
            dec_out = dec_out.squeeze(-1)

        return dec_out

    def build_prompt(self, inputs):
        x_enc = inputs["x_enc"]
        if x_enc.ndim == 2:
            x_enc = x_enc.unsqueeze(-1)

        with torch.no_grad():
            min_values = torch.min(x_enc, dim=1).values
            max_values = torch.max(x_enc, dim=1).values
            medians = torch.median(x_enc.float(), dim=1).values
            lags = calcute_lags(x_enc.float(), self.n_lags)
            trends = x_enc.diff(dim=1).sum(dim=1)
            descriptions = inputs.get("descriptions")

        prompts = []
        for b in range(x_enc.size(0)):
            min_value = min_values[b][0].item()
            max_value = max_values[b][0].item()
            median_value = medians[b][0].item()
            lags_value = lags[b].tolist()
            trend_dir = "upward" if trends[b].mean() > 0 else "downward"
            prompt = (
                f"<|start_prompt|>"
                f"Dataset description: {self.dataset_description} "
                f"{descriptions[b]} " if descriptions is not None else ""
                f"Task description: {self.task_description} "
                f"Input statistics: "
                f"min value = {min_value:.3f}, "
                f"max value = {max_value:.3f}, "
                f"median value = {median_value:.3f}, "
                f"the trend of input is {trend_dir}, "
                f"top 5 lags are {lags_value}"
                f"<|<end_prompt>|>"
            )

            prompts.append(prompt)

        return prompts

    def get_task_description(self):
        if self.task == "forecasting":
            self.task_description = f"Forecast the next {self.pred_len} steps given the previous {self.seq_len} steps of data."
        elif self.task == "anomaly_detection":
            self.task_description = f"Reconstruct the past {self.seq_len} steps of data as accurately as possible using the following information."
        elif self.task == "semantic_segmentation":
            self.task_description = f"Classify the past {self.seq_len} steps of data as accurately as possible using the following information."
        elif self.task == "segmentation":
            self.task_description = f"Identify the change points in the past {self.seq_len} steps of data to segment the sequence."
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
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
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
