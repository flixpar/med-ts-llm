import torch
from models.medtsllm import ReprogrammingLayer


def test_reprogramming_attention_shapes():
    layer = ReprogrammingLayer(d_model=4, n_heads=2, d_keys=4, d_llm=4)
    target = torch.randn(1, 3, 4)
    source = torch.randn(5, 4)
    out, attn = layer(target, source, source, store_attention=True)
    assert attn.shape == (1, 2, 3, 5)
    assert out.shape[:2] == (1, 3)


def test_forward_compatibility():
    layer = ReprogrammingLayer(d_model=4, n_heads=2, d_keys=4, d_llm=4)
    target = torch.randn(2, 3, 4)
    source = torch.randn(5, 4)
    out1 = layer(target, source, source)
    out2, attn = layer(target, source, source, store_attention=True)
    assert torch.allclose(out1, out2, atol=1e-6)
