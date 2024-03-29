import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from .layers.embed import DataEmbedding
from einops import rearrange


class GPT4TS(nn.Module):

    supported_tasks = ["forecasting", "imputation", "anomaly_detection", "classification", "semantic_segmentation", "segmentation"]
    supported_modes = ["multivariate", "univariate"]

    def __init__(self, config, dataset):
        super(GPT4TS, self).__init__()
        self.config = config
        self.model_config = self.config.models.gpt4ts
        self.task = self.config.task

        self.d_ff = self.model_config.d_ff
        self.d_model = self.model_config.d_model
        self.gpt_layers = self.model_config.gpt_layers
        self.train_mlp = self.model_config.train_mlp

        self.enc_in = dataset.n_features
        self.c_out = dataset.n_features
        self.num_class = dataset.n_classes if self.task in ["classification", "semantic_segmentation"] else 0
        self.seq_len = self.config.history_len
        if self.task == "forecasting":
            self.pred_len = self.config.pred_len
        else:
            assert self.config.pred_len == self.seq_len
            self.pred_len = 0

        self.patch_size = self.model_config.patching.patch_len
        self.stride = self.model_config.patching.stride
        self.patch_num = (self.seq_len + self.pred_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.enc_in * self.patch_size, self.d_model, "timeF", "h", self.config.training.dropout)

        self.gpt2 = GPT2Model.from_pretrained("gpt2", output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if "ln" in name or "wpe" in name:
                param.requires_grad = True
            elif "mlp" in name and self.train_mlp:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if self.task == "forecasting":
            self.predict_linear_pre = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.predict_linear = nn.Linear(self.patch_size, self.enc_in)
            self.ln = nn.LayerNorm(self.d_ff)
            self.out_layer = nn.Linear(self.d_ff, self.c_out)
        if self.task == "imputation":
            self.ln_proj = nn.LayerNorm(self.d_model)
            self.out_layer = nn.Linear(self.d_model, self.c_out, bias=True)
        if self.task == "anomaly_detection":
            self.ln_proj = nn.LayerNorm(self.d_ff)
            self.out_layer = nn.Linear(self.d_ff, self.c_out, bias=True)
        if self.task == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(0.1)
            self.ln_proj = nn.LayerNorm(self.d_model * self.patch_num)
            self.out_layer = nn.Linear(self.d_model * self.patch_num, self.num_class)
        if self.task == "semantic_segmentation":
            self.ln_proj = nn.LayerNorm(self.d_ff)
            n_output = self.num_class if self.num_class > 2 else 1
            self.out_layer = nn.Linear(self.d_ff, n_output, bias=True)
        if self.task == "segmentation":
            assert self.config.tasks.segmentation.mode == "boundary-prediction"
            self.ln_proj = nn.LayerNorm(self.d_ff)
            self.out_layer = nn.Linear(self.d_ff, 1, bias=True)

    def forward(self, inputs):
        x_enc = inputs["x_enc"]
        x_dec = inputs.get("x_dec", None)
        x_mark_enc = inputs.get("x_mark_enc", None)
        x_mark_dec = inputs.get("x_mark_dec", None)
        mask = inputs.get("mask", None)

        if self.task == "forecasting":
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
        elif self.task == "imputation":
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        elif self.task == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        elif self.task == "classification":
            return self.classification(x_enc, x_mark_enc)
        elif self.task == "semantic_segmentation":
            return self.semantic_segmentation(x_enc, x_mark_enc)
        elif self.task == "segmentation":
            return self.segmentation(x_enc, x_mark_enc)
        else:
            raise ValueError("Task name is not valid")

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state

        outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(outputs)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
        enc_out = torch.nn.functional.pad(enc_out, (0, 768-enc_out.shape[-1]))

        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = self.out_layer(dec_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        return dec_out

    def anomaly_detection(self, x_enc):
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        seg_num = 25
        x_enc = rearrange(x_enc, "b (n s) m -> b n s m", s=seg_num)
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = rearrange(x_enc, "b n s m -> b (n s) m")

        enc_out = torch.nn.functional.pad(x_enc, (0, 768-x_enc.shape[-1]))

        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state

        outputs = outputs[:, :, :self.d_ff]
        dec_out = self.out_layer(outputs)

        # De-Normalization from Non-stationary Transformer
        dec_out = rearrange(dec_out, "b (n s) m -> b n s m", s=seg_num)
        dec_out = dec_out * (stdev[:, :, 0, :].unsqueeze(2).repeat(1, 1, seg_num, 1))
        dec_out = dec_out + (means[:, :, 0, :].unsqueeze(2).repeat(1, 1, seg_num, 1))
        dec_out = rearrange(dec_out, "b n s m -> b (n s) m")

        return dec_out

    def classification(self, x_enc):
        B, L, M = x_enc.shape
        input_x = rearrange(x_enc, "b l m -> b m l")
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, "b m n p -> b n (p m)")

        outputs = self.enc_embedding(input_x, None)

        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.dropout(outputs)
        outputs = self.out_layer(outputs)

        return outputs

    def semantic_segmentation(self, x_enc, x_mark_enc=None):
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = torch.nn.functional.pad(enc_out, (0, 768-enc_out.shape[-1]))

        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = self.out_layer(dec_out)
        dec_out = dec_out.squeeze(-1)

        if not self.training:
            if self.num_class > 2:
                dec_out = dec_out.reshape(B, self.pred_len, self.num_class)
                dec_out = F.softmax(dec_out, dim=-1)
            else:
                dec_out = torch.sigmoid(dec_out)

        return dec_out

    def segmentation(self, x_enc, x_mark_enc=None):
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = torch.nn.functional.pad(enc_out, (0, 768-enc_out.shape[-1]))

        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = self.out_layer(dec_out)
        dec_out = dec_out.squeeze(-1)

        if not self.training:
            dec_out = torch.sigmoid(dec_out)

        return dec_out
