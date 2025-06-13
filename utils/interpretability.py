import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ModelInterpretabilityMixin:
    """Mixin providing interpretability utilities for MedTsLLM."""

    def __init__(self):
        self.interpretability_enabled = False
        self.capture_llm_attention = False
        self.capture_gradients = False
        self._hooks = []
        self._llm_attentions = []
        self._stored_cross_attention = None

    def enable_interpretability(self, capture_llm_attention=True, capture_gradients=True):
        self.interpretability_enabled = True
        self.capture_llm_attention = capture_llm_attention
        self.capture_gradients = capture_gradients
        if capture_llm_attention:
            self._register_llm_hooks()

    def disable_interpretability(self):
        self.interpretability_enabled = False
        self._remove_hooks()
        self._llm_attentions.clear()
        self._stored_cross_attention = None

    # Hook registration -------------------------------------------------
    def _register_llm_hooks(self):
        if not hasattr(self, "llm"):
            return
        if hasattr(self.llm, "h"):  # GPT style
            modules = [layer.attn for layer in self.llm.h]
        elif hasattr(self.llm, "encoder"):
            modules = [layer.attention for layer in self.llm.encoder.layer]
        else:
            return
        for module in modules:
            handle = module.register_forward_hook(self._capture_llm_attention)
            self._hooks.append(handle)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _capture_llm_attention(self, module, inputs, output):
        if isinstance(output, tuple) and len(output) > 1:
            attn = output[-1]
        else:
            attn = getattr(module, "attn_weights", None)
        if attn is not None:
            self._llm_attentions.append(attn.detach().cpu())

    # Accessors ---------------------------------------------------------
    def get_cross_attention_maps(self):
        return self._stored_cross_attention

    def get_patch_importance_scores(self, inputs, method="attention"):
        if method == "attention":
            if not self.interpretability_enabled:
                self.enable_interpretability(capture_llm_attention=False, capture_gradients=False)
                with torch.no_grad():
                    self(inputs)
                self.disable_interpretability()
            attn = self.get_cross_attention_maps()
            if attn is None:
                return None
            return attn.mean(dim=(1, 2))
        elif method == "gradient":
            x = inputs["x_enc"].clone().detach().requires_grad_(True)
            cloned = {k: v for k, v in inputs.items()}
            cloned["x_enc"] = x
            pred = self(cloned)
            loss = pred.sum()
            loss.backward()
            grad = x.grad.abs().mean(dim=-1)
            patches = grad.unfold(1, self.patch_len, self.stride)
            scores = patches.mean(dim=-1)
            return scores
        else:
            raise ValueError(f"Unknown importance method: {method}")

    def generate_interpretability_report(self, inputs, save_path=None):
        cross = self.get_cross_attention_maps()
        scores = {"attention": self.get_patch_importance_scores(inputs, method="attention")}
        viz = InterpretabilityVisualizer()
        if cross is not None:
            heatmap = cross.mean(dim=1)[0]  # [L,S]
        else:
            heatmap = None
        fig = viz.create_clinical_dashboard(heatmap, scores, save_path=save_path)
        return {"figure": fig, "cross_attention": cross, "scores": scores}


class InterpretabilityVisualizer:
    """Simple visualization utilities for interpretability."""

    def plot_cross_attention_heatmap(self, attn, text_tokens=None, save_path=None):
        if attn is None:
            return None
        fig = go.Figure(data=go.Heatmap(z=attn.cpu().numpy(), colorscale="Viridis"))
        if text_tokens is not None:
            fig.update_xaxes(ticktext=text_tokens, tickvals=list(range(len(text_tokens))))
        if save_path:
            fig.write_html(save_path)
        return fig

    def plot_temporal_importance(self, scores, save_path=None):
        fig = go.Figure()
        for name, vals in scores.items():
            if vals is None:
                continue
            fig.add_trace(go.Scatter(y=vals[0].cpu().numpy(), mode="lines", name=name))
        if save_path:
            fig.write_html(save_path)
        return fig

    def create_clinical_dashboard(self, attn, scores, patient_info="", save_path=None):
        if attn is None:
            attn = torch.zeros(1, 1)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=("Cross Attention", "Patch Importance"))
        fig.add_trace(go.Heatmap(z=attn.cpu().numpy(), colorscale="Viridis"), row=1, col=1)
        for name, vals in scores.items():
            if vals is None:
                continue
            fig.add_trace(go.Scatter(y=vals[0].cpu().numpy(), mode="lines", name=name), row=2, col=1)
        fig.update_layout(title=patient_info)
        if save_path:
            fig.write_html(save_path)
        return fig
