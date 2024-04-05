import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.embed import DataEmbedding
from .layers.AutoCorrelation import AutoCorrelationLayer
from .layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from .layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from .layers.Autoformer_EncDec import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    my_Layernorm,
    series_decomp,
)


class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    supported_tasks = [
        "forecasting",
        "anomaly_detection",
        "reconstruction",
        "imputation",
        "classification",
        "semantic_segmentation",
        "segmentation",
    ]
    supported_modes = ["multivariate"]

    def __init__(self, config, dataset):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super().__init__()
        self.config = config
        self.model_config = config.models.fedformer

        self.task_name = self.config.task
        self.seq_len = self.config.history_len
        self.label_len = self.model_config.label_len
        self.pred_len = self.config.pred_len

        self.version = self.model_config.version
        self.mode_select = self.model_config.mode_select
        self.modes = self.model_config.modes

        self.e_layers = self.model_config.e_layers
        self.d_layers = self.model_config.d_layers
        self.d_model = self.model_config.d_model
        self.n_heads = self.model_config.n_heads
        self.d_ff = self.model_config.d_ff
        self.moving_avg = self.model_config.moving_avg
        self.activation = self.model_config.activation

        self.embed = "timeF"
        self.freq = "s"

        self.enc_in = dataset.n_features
        self.dec_in = dataset.n_features
        self.c_out = dataset.n_features
        self.num_class = dataset.n_classes if self.task_name in ["classification", "semantic_segmentation"] else 0

        # Decomp
        self.decomp = series_decomp(self.moving_avg)
        self.enc_embedding = DataEmbedding(
            self.enc_in,
            self.d_model,
            self.embed,
            self.freq,
            self.config.training.dropout,
        )
        self.dec_embedding = DataEmbedding(
            self.dec_in,
            self.d_model,
            self.embed,
            self.freq,
            self.config.training.dropout,
        )

        if self.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=self.d_model, L=1, base="legendre"
            )
            decoder_self_att = MultiWaveletTransform(
                ich=self.d_model, L=1, base="legendre"
            )
            decoder_cross_att = MultiWaveletCross(
                in_channels=self.d_model,
                out_channels=self.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                ich=self.d_model,
                base="legendre",
                activation="tanh",
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=self.d_model,
                out_channels=self.d_model,
                seq_len=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
            )
            decoder_self_att = FourierBlock(
                in_channels=self.d_model,
                out_channels=self.d_model,
                seq_len=self.seq_len // 2 + self.pred_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=self.d_model,
                out_channels=self.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
                num_heads=self.n_heads,
            )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
                        self.d_model,
                        self.n_heads,
                    ),
                    self.d_model,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.config.training.dropout,
                    activation=self.activation,
                )
                for l in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att, self.d_model, self.n_heads
                    ),
                    AutoCorrelationLayer(
                        decoder_cross_att, self.d_model, self.n_heads
                    ),
                    self.d_model,
                    self.c_out,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.config.training.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True),
        )

        if self.task_name == "imputation":
            self.projection = nn.Linear(
                self.d_model, self.c_out, bias=True
            )
        if self.task_name == "anomaly_detection" or self.task_name == "reconstruction":
            self.projection = nn.Linear(
                self.d_model, self.c_out, bias=True
            )
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(self.config.training.dropout)
            self.projection = nn.Linear(
                self.d_model * self.seq_len, self.num_class
            )
        if self.task_name == "semantic_segmentation":
            self.act = F.gelu
            out_size = (
                self.pred_len * self.num_class
                if self.num_class > 2
                else self.pred_len
            )
            self.projection = nn.Linear(self.d_model * self.seq_len, out_size)
        if self.task_name == "segmentation":
            self.act = F.gelu
            self.projection = nn.Linear(
                self.d_model * self.seq_len, self.seq_len
            )
            self.seg_mode = self.config.tasks.segmentation.mode

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)  # x - moving_avg, moving_avg
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len :, :], (0, 0, 0, self.pred_len)
        )
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def semantic_segmentation(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        enc_out = self.act(enc_out)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)

        if self.num_class > 2:
            output = output.reshape(output.shape[0], self.pred_len, self.num_class)

        if not self.training:
            if self.num_class > 2:
                output = F.softmax(output, dim=-1)
            else:
                output = F.sigmoid(output)

        return output

    def segmentation(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        enc_out = self.act(enc_out)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)

        if not self.training and self.seg_mode == "boundary-prediction":
            output = F.sigmoid(output)

        return output

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
