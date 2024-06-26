import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from .layers.embed import DataEmbedding
from .layers.Conv_Blocks import Inception_Block_V1


class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    supported_tasks = [
        "forecasting",
        "reconstruction",
        "anomaly_detection",
        "imputation",
        "classification",
        "semantic_segmentation",
        "segmentation",
    ]
    supported_modes = ["multivariate"]

    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.model_config = self.config.models.timesnet
        self.task_name = self.config.task

        self.seq_len = self.config.history_len
        if self.task_name == "forecasting":
            self.pred_len = self.config.pred_len
        else:
            assert self.config.pred_len == self.seq_len
            self.pred_len = 0

        self.enc_in = dataset.n_features
        self.c_out = dataset.n_features
        self.num_class = (
            dataset.n_classes
            if self.task_name in ["classification", "semantic_segmentation"]
            else 0
        )

        self.embed = "timeF"
        self.freq = "s"

        self.dropout = self.config.training.dropout

        self.model = nn.ModuleList(
            [TimesBlock(self.config) for _ in range(self.model_config.e_layers)]
        )
        self.enc_embedding = DataEmbedding(
            self.enc_in,
            self.model_config.d_model,
            self.embed,
            self.freq,
            self.dropout,
        )
        self.layer = self.model_config.e_layers
        self.layer_norm = nn.LayerNorm(self.model_config.d_model)

        if self.task_name == "forecasting":
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                self.model_config.d_model, self.c_out, bias=True
            )
        if self.task_name in ["imputation", "reconstruction", "anomaly_detection"]:
            self.projection = nn.Linear(
                self.model_config.d_model, self.c_out, bias=True
            )
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(self.dropout)
            self.projection = nn.Linear(
                self.model_config.d_model * self.seq_len, self.num_class
            )
        if self.task_name == "semantic_segmentation":
            n_outputs = self.num_class if self.num_class > 2 else 1
            self.projection = nn.Linear(self.model_config.d_model, n_outputs)
        if self.task_name == "segmentation":
            self.projection = nn.Linear(self.model_config.d_model, 1)
            self.seg_mode = self.config.tasks.segmentation.mode

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def semantic_segmentation(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)
        dec_out = dec_out.squeeze(-1)

        if not self.training:
            if self.num_class > 2:
                dec_out = F.softmax(dec_out, dim=-1)
            else:
                dec_out = F.sigmoid(dec_out)

        return dec_out

    def segmentation(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)
        dec_out = dec_out.squeeze(-1)

        if not self.training and self.seg_mode == "boundary-prediction":
            dec_out = F.sigmoid(dec_out)

        return dec_out

    def forward(self, inputs):
        x_enc = inputs["x_enc"]
        x_dec = inputs.get("x_dec", None)
        x_mark_enc = inputs.get("x_mark_enc", None)
        x_mark_dec = inputs.get("x_mark_dec", None)
        mask = inputs.get("mask", None)

        if self.task_name == "forecasting":
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == "anomaly_detection" or self.task_name == "reconstruction":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == "semantic_segmentation":
            dec_out = self.semantic_segmentation(x_enc)
            return dec_out
        if self.task_name == "segmentation":
            dec_out = self.segmentation(x_enc)
            return dec_out
        return None


class TimesBlock(nn.Module):
    def __init__(self, config):
        super(TimesBlock, self).__init__()
        model_config = config.models.timesnet

        self.seq_len = config.history_len
        self.pred_len = config.pred_len if config.task == "forecasting" else 0
        self.k = model_config.top_k

        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(
                model_config.d_model,
                model_config.d_ff,
                num_kernels=model_config.num_kernels,
            ),
            nn.GELU(),
            Inception_Block_V1(
                model_config.d_ff,
                model_config.d_model,
                num_kernels=model_config.num_kernels,
            ),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        period_weight = period_weight.to(x.dtype)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]],
                    device=x.device, dtype=x.dtype,
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # reshape
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x.float(), dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]
