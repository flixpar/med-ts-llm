import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig
transformers.logging.set_verbosity_error()

from peft import LoraConfig, TaskType
from peft import get_peft_model

from .layers.embed import PatchEmbedding
from .layers.RevIN import RevIN

from utils import dict_to_object

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TimeLLM(nn.Module):

    supported_tasks = ["forecasting", "reconstruction", "anomaly_detection", "semantic_segmentation", "segmentation", "pretraining"]
    supported_modes = ["univariate", "multivariate"]

    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.model_config = self.config.models.timellm

        self.device = None

        self.pred_len = self.config.pred_len
        self.seq_len = self.config.history_len

        self.task = self.config.task
        self.task_description = self.get_task_description(dataset)
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

        if self.task in ["forecasting", "reconstruction", "anomaly_detection", "pretraining"]:
            self.n_outputs_per_step = self.n_features
        elif self.task == "semantic_segmentation":
            self.n_outputs_per_step = self.n_classes if self.n_classes > 2 else 1
        elif self.task == "segmentation":
            self.n_outputs_per_step = 1
            assert self.config.tasks.segmentation.mode in ["boundary-prediction", "steps-to-boundary"]
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
                pass
            case "merge-end":
                self.feature_weighting = nn.Linear(self.n_features * self.n_outputs_per_step, self.n_outputs_per_step)
            case "weighted-average":
                self.feature_weighting = nn.Linear(self.n_features, 1)
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

        self.embedding_downsample_mode = self.model_config.embedding_downsample_mode
        if self.embedding_downsample_mode == "linear":
            self.embedding_downsample_layer = nn.Linear(self.d_llm, self.d_ff)
        elif self.embedding_downsample_mode == "average":
            assert self.d_llm % self.d_ff == 0

        if not self.llm_enabled:
            self.llm_replacement = nn.Sequential(
                nn.Linear(self.d_llm, self.d_llm),
                nn.GELU(),
                nn.Linear(self.d_llm, self.d_ff),
                nn.LayerNorm(self.d_ff),
            )

        n_params_total = sum(p.numel() for p in self.parameters())
        n_params_llm = sum(p.numel() for p in self.llm.parameters()) if self.llm_enabled else 0
        n_params_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total number of parameters: {n_params_total:,}")
        print(f"Number of trainable parameters: {n_params_trainable:,}")
        print(f"Number of parameters in LLM: {n_params_llm:,}")

        layer_names = {
            "norm": self.normalize_layers,
            "mapping": self.mapping_layer,
            "patch_emb": self.patch_embedding,
            "reprogramming": self.reprogramming_layer,
            "output": self.output_projection,
        }
        n_params_layers = {k: sum(p.numel() for p in v.parameters()) for k, v in layer_names.items()}
        for k, v in n_params_layers.items():
            print(f"Number of parameters in {k} layer: {v:,}")

    def setup_llm(self):
        self.llm_enabled = self.model_config.llm.enabled
        self.llm_id = self.model_config.llm.llm
        self.llm_layers = self.model_config.llm.llm_layers

        cache_dir = self.config.get("paths", {}).get("llm_path")
        if cache_dir == "" or cache_dir == "none":
            cache_dir = None

        trust_remote_code = (self.llm_id != "microsoft/phi-2")

        llm_config = AutoConfig.from_pretrained(
            self.llm_id,
            cache_dir = cache_dir,
            trust_remote_code = trust_remote_code,
        )
        if self.llm_layers > 0 and self.llm_layers < llm_config.num_hidden_layers:
            llm_config.num_hidden_layers = self.llm_layers
        llm_config.output_hidden_states = True

        match self.config.setup.dtype:
            case "bfloat16" | "bf16":
                model_dtype = torch.bfloat16
            case "float16" | "half" | "fp16" | "16" | 16:
                model_dtype = torch.float16
            case "float32" | "float" | "fp32" | "32" | 32 | "mixed":
                model_dtype = torch.float32
            case x:
                raise ValueError(f"Invalid dtype selection: {x}")

        attn_implementation = "flash_attention_2" if (model_dtype in [torch.float16, torch.bfloat16]) else "sdpa"
        attn_implementation = attn_implementation if ("mamba" not in self.llm_id) and () else "eager"

        if self.model_config.llm.load_in_4bit or self.model_config.llm.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit = self.model_config.llm.load_in_4bit,
                load_in_8bit = self.model_config.llm.load_in_8bit,

                llm_int8_has_fp16_weight = True,
                llm_int8_skip_modules = ["mamba"],

                bnb_4bit_compute_dtype = model_dtype,
            )
        else:
            quantization_config = None

        llm = AutoModel.from_pretrained(
            self.llm_id,
            config = llm_config,
            quantization_config = quantization_config,
            torch_dtype = model_dtype,
            attn_implementation = attn_implementation,
            cache_dir = cache_dir,
            trust_remote_code = trust_remote_code,
            device_map = "auto",
            # device_map = self.device,
        )

        if "lora" in self.model_config and self.model_config.lora.enabled and self.llm_enabled:
            print("Setting up LoRA...")
            self.lora_enabled = True
            lora_cfg = self.model_config.lora
            assert lora_cfg.layers == "auto"
            peft_config = LoraConfig(
                task_type = TaskType.FEATURE_EXTRACTION,
                inference_mode = False,
                r = lora_cfg.rank,
                lora_alpha = lora_cfg.alpha,
                init_lora_weights = lora_cfg.get("init", True),
                lora_dropout = lora_cfg.get("dropout", 0.0),
                use_rslora = lora_cfg.get("rslora", True),
            )
            llm = get_peft_model(llm, peft_config)
            llm.print_trainable_parameters()
        else:
            self.lora_enabled = False

        tokenizer = AutoTokenizer.from_pretrained(
            self.llm_id,
            cache_dir = cache_dir,
            trust_remote_code = trust_remote_code,
        )

        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            tokenizer.add_special_tokens({"pad_token": pad_token})
            tokenizer.pad_token = pad_token

        self.word_embeddings = llm.get_input_embeddings().weight
        if self.word_embeddings.size(0) > 100_000:
            inds = torch.linspace(0, self.word_embeddings.size(0)-1, 100_000, dtype=torch.long)
            self.word_embeddings = nn.Parameter(self.word_embeddings[inds,:])

        self.vocab_size = self.word_embeddings.shape[0]
        self.d_llm = llm_config.hidden_size

        if self.llm_enabled:
            self.llm = llm
            self.tokenizer = tokenizer

            if not self.lora_enabled:
                for param in self.llm.parameters():
                    param.requires_grad = False

    def state_dict(self):
        state_dict = super().state_dict()

        if self.llm_enabled:
            llm_keys = [k for k in state_dict.keys() if k[:4] == "llm."]
            for k in llm_keys:
                del state_dict[k]

        if "word_embeddings" in state_dict:
            del state_dict["word_embeddings"]

        return state_dict

    def forward(self, inputs):
        pred = self.predict(inputs)

        if not self.training:
            if self.task == "semantic_segmentation":
                if self.n_classes > 2:
                    pred = F.softmax(pred, dim=-1)
                else:
                    pred = F.sigmoid(pred)
            elif self.task == "segmentation":
                if self.config.tasks.segmentation.mode == "boundary-prediction":
                    pred = F.sigmoid(pred)

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

        n_patches = enc_out.size(1)
        if self.covariate_mode == "concat":
            enc_out = enc_out.reshape(bs, n_features, n_patches, self.d_patch)
            enc_out = enc_out.permute(0, 2, 1, 3) # [bs, n_patches, n_features, d_patch]
            enc_out = enc_out.reshape(bs, n_patches, n_features * self.d_patch)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings) # [bs * n_features, n_patches, d_llm]

        if self.covariate_mode == "add":
            enc_out = enc_out.reshape(bs, n_features, n_patches, self.d_llm)
            enc_out = enc_out.mean(dim=1)                           # [bs, n_patches, d_llm]
        elif self.covariate_mode == "weighted-average":
            enc_out = enc_out.reshape(bs, n_features, n_patches, self.d_llm)
            enc_out = enc_out.permute(0, 2, 3, 1)                     # [bs, n_patches, d_llm, n_features]
            enc_out = self.feature_weighting(enc_out)                 # [bs, n_patches, d_llm, 1]
            enc_out = enc_out.squeeze(-1)                             # [bs, n_patches, d_llm]
        elif self.covariate_mode == "interleave":
            enc_out = enc_out.reshape(bs, n_features, -1, self.d_llm) # [bs, n_features, n_patches, d_llm]
            enc_out = enc_out.permute(0, 2, 1, 3)                     # [bs, n_patches, n_features, d_llm]
            enc_out = enc_out.reshape(bs, -1, self.d_llm)             # [bs, n_patches * n_features, d_llm]

        if self.llm_enabled:
            prompt = self.build_prompt(inputs)
            prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
            prompt_embeddings = self.llm.get_input_embeddings()(prompt.to(x_enc.device)) # [bs, n_tok, d_llm]
            prompt_embeddings = prompt_embeddings.to(enc_out.dtype)

            if self.covariate_mode == "independent" or self.covariate_mode == "merge-end":
                prompt_embeddings = prompt_embeddings.repeat_interleave(n_features, dim=0)

            if self.llm.config.is_encoder_decoder:
                dec_out = self.llm(inputs_embeds=prompt_embeddings, decoder_inputs_embeds=enc_out).last_hidden_state  # [bs, n_tok + n_patches, d_llm]
            else:
                llm_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
                dec_out = self.llm(inputs_embeds=llm_enc_out).last_hidden_state  # [bs, n_tok + n_patches, d_llm]
            dec_out = dec_out.to(x_enc.dtype)
        else:
            dec_out = self.llm_replacement(enc_out)

        dec_out = dec_out[:, -self.n_patches:, :]
        match self.embedding_downsample_mode:
            case "truncate":
                dec_out = dec_out[:, :, :self.d_ff]
            case "linear":
                dec_out = self.embedding_downsample_layer(dec_out)
            case "average":
                dec_out = dec_out.reshape(bs, self.n_patches, self.d_ff, -1)
                dec_out = dec_out.mean(dim=-1)
            case _:
                raise ValueError(f"Unknown embedding downsample mode {self.embedding_downsample}")

        # dec_out = dec_out[:, -self.n_patches:, :self.d_ff]
        dec_out = dec_out.permute(0, 2, 1).contiguous() # [bs, d_ff, n_patches]
        dec_out = self.output_projection(dec_out)       # [bs, pred_len * n_features]

        if self.covariate_mode == "independent":
            dec_out = dec_out.view(bs, self.n_features, self.pred_len, self.n_outputs_per_step)
            dec_out = dec_out.mean(dim=1)
        elif self.covariate_mode == "merge-end":
            dec_out = dec_out.view(bs, self.n_features, self.pred_len, self.n_outputs_per_step)
            dec_out = dec_out.permute(0, 2, 3, 1).reshape(bs, self.pred_len, -1).contiguous() # [bs, pred_len, n_features*n_outputs_per_step]
            dec_out = self.feature_weighting(dec_out) # [bs, pred_len, n_outputs_per_step]
        else:
            dec_out = dec_out.view(bs, self.pred_len, self.n_outputs_per_step)

        if self.task in ["forecasting", "reconstruction", "anomaly_detection", "pretraining"]:
            dec_out = self.normalize_layers(dec_out, "denorm")
        else:
            dec_out = dec_out.squeeze(-1)

        return dec_out

    def build_prompt(self, inputs):
        bs = inputs["x_enc"].size(0)

        cfg = self.model_config.get("prompting")
        if cfg is None:
            cfg = {"dataset": True, "clip": True, "input_stats": True, "task": True, "input_stats_dim": 0, "input_stats_select": "all"}
            cfg = dict_to_object(cfg)

        if not (cfg.dataset or cfg.clip or cfg.input_stats or cfg.task):
            return [""] * bs

        if cfg.dataset:
            dataset_prompt = f"Dataset: {self.dataset_description}"
        else:
            dataset_prompt = ""

        if cfg.clip:
            clip_prompts = inputs.get("descriptions", [""] * bs)
        else:
            clip_prompts = [""] * bs

        if cfg.input_stats:
            input_stats_prompts = self.build_input_stats_prompt(cfg, inputs)
        else:
            input_stats_prompts = [""] * bs

        if cfg.task:
            task_prompt = f"Task: {self.task_description}"
        else:
            task_prompt = ""

        bos = self.tokenizer.bos_token if self.tokenizer.bos_token is not None else ""

        prompts = []
        for b in range(bs):
            parts = [
                bos,
                dataset_prompt,
                clip_prompts[b],
                input_stats_prompts[b],
                task_prompt,
                "Time series:",
            ]
            prompt = " ".join([p for p in parts if p])
            prompt = "<|start_prompt|>" + prompt + "<|end_prompt|>"
            prompts.append(prompt)

        return prompts

    def build_input_stats_prompt(self, cfg, inputs):
        xs = inputs["x_enc"].detach() # [bs, seq_len, n_features]
        if xs.ndim == 2:
            xs = xs.unsqueeze(-1)

        assert cfg.input_stats_select == "all"

        def fmt_list(xs):
            return "[" + ", ".join(xs) + "]"

        def fmt_float(x):
            if isinstance(x, list):
                return fmt_list([fmt_float(v) for v in x])
            return f"{x:.3f}"

        def fmt_trend(x):
            match x:
                case True:
                    return "upward"
                case False:
                    return "downward"
                case [*xs]:
                    return fmt_list([fmt_trend(x) for x in xs])
                case _:
                    return x

        if cfg.input_stats_dim == "all":
            prompt_insert = "per feature"
            s = "s"
        else:
            d = cfg.input_stats_dim
            prompt_insert = f"feature {d}"
            xs = xs[:, :, d]
            s = ""

        with torch.no_grad():
            min_values = torch.min(xs, dim=1).values.tolist()
            max_values = torch.max(xs, dim=1).values.tolist()
            medians = torch.median(xs.float(), dim=1).values.tolist()
            trends = (xs.diff(dim=1).sum(dim=1) > 0).tolist()
            lags = calcute_lags(xs.float(), self.n_lags).tolist()

        prompts = []
        for b in range(xs.size(0)):
            prompt = (
                f"Input statistics ({prompt_insert}): "
                f"min value{s} = {fmt_float(min_values[b])}, "
                f"max value{s} = {fmt_float(max_values[b])}, "
                f"median value{s} = {fmt_float(medians[b])}, "
                f"the trend of input is {fmt_trend(trends[b])}, "
                f"the top {self.n_lags} lags are {lags[b]}."
            )
            prompts.append(prompt)

        return prompts

    def get_task_description(self, dataset):
        if getattr(dataset, "task_description", None) is not None:
            self.task_description = dataset.task_description
            return self.task_description

        if self.task == "forecasting" or self.task == "pretraining":
            self.task_description = f"Forecast the next {self.pred_len} steps given the previous {self.seq_len} steps of data."
        elif self.task == "anomaly_detection" or self.task == "reconstruction":
            self.task_description = f"Reconstruct the past {self.seq_len} steps of data as accurately as possible using the following information."
        elif self.task == "semantic_segmentation":
            self.task_description = f"Classify the past {self.seq_len} steps of data as accurately as possible using the following information."
        elif self.task == "segmentation":
            self.task_description = f"Identify the change points in the past {self.seq_len} steps of data to segment the sequence."
        else:
            raise ValueError(f"Task {self.task} is not supported.")

        return self.task_description

    def load_pretrained(self, saved_state):
        if "word_embeddings" in saved_state:
            del saved_state["word_embeddings"]
        if "output_projection.linear.bias" in saved_state:
            del saved_state["output_projection.linear.bias"]
        if "output_projection.linear.weight" in saved_state:
            del saved_state["output_projection.linear.weight"]

        incompat_keys = self.load_state_dict(saved_state, strict=False)
        assert len(incompat_keys.unexpected_keys) == 0, f"Unexpected keys in model state: {incompat_keys.unexpected_keys}"

        loaded_keys = list(saved_state.keys())
        return loaded_keys


def calcute_lags(x, n_lags=5):
    x = x.permute(0, 2, 1).contiguous() if x.ndim == 3 else x.unsqueeze(1)
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
