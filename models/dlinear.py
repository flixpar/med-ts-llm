import torch
import torch.nn as nn
import torch.nn.functional as F


class DLinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    supported_tasks = ["forecasting", "anomaly_detection", "imputation", "classification", "semantic_segmentation", "segmentation"]
    supported_modes = ["multivariate"]

    def __init__(self, config, dataset):
        super(DLinear, self).__init__()

        self.task_name = config.task

        self.individual = config.models.dlinear.individual
        self.channels = dataset.n_features
        self.decompsition = series_decomp(config.models.dlinear.moving_avg)

        self.seq_len = config.history_len
        if self.task_name in ["classification", "anomaly_detection", "imputation", "semantic_segmentation", "segmentation"]:
            self.pred_len = self.seq_len
        else:
            self.pred_len = config.pred_len

        if self.task_name in ["classification", "semantic_segmentation"]:
            self.n_classes = dataset.n_classes
        else:
            self.n_classes = 0

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(config.training.dropout)
            self.projection = nn.Linear(self.channels * self.seq_len, dataset.n_classes)
        elif self.task_name == "semantic_segmentation":
            out_size = self.pred_len * self.n_classes if self.n_classes > 2 else self.pred_len
            self.projection = nn.Linear(self.channels * self.seq_len, out_size)
        elif self.task_name == "segmentation":
            self.projection = nn.Linear(self.channels * self.seq_len, self.seq_len)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc) # (batch_size, seq_length, d_model)
        output = enc_out.reshape(enc_out.shape[0], -1) # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def semantic_segmentation(self, x_enc):
        enc_out = self.encoder(x_enc)
        enc_out = F.gelu(enc_out)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)
        if not self.training:
            if self.n_classes > 2:
                output = output.reshape(output.shape[0], self.pred_len, self.n_classes)
                output = F.softmax(output, dim=-1)
            else:
                output = F.sigmoid(output)
        return output

    def segmentation(self, x_enc):
        enc_out = self.encoder(x_enc)
        enc_out = F.gelu(enc_out)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)
        if not self.training:
            output = F.sigmoid(output)
        return output

    def forward(self, x_enc, x_dec):
        if self.task_name == "forecasting":
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        elif self.task_name == "imputation":
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        elif self.task_name == "anomaly_detection":
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        elif self.task_name == "classification":
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        elif self.task_name == "semantic_segmentation":
            dec_out = self.semantic_segmentation(x_enc)
            return dec_out
        elif self.task_name == "segmentation":
            dec_out = self.segmentation(x_enc)
            return dec_out
        else:
            raise ValueError(f"Invalid task name for DLinear: {self.task_name}")


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
