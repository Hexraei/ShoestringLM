import torch

from shoestring.config import ModelConfig
from shoestring.model import ShoestringLM


def test_main_config_under_50m_cap():
    model = ShoestringLM(ModelConfig.from_json("configs/shoestring_30m.json"))
    total = sum(p.numel() for p in model.parameters())
    assert total <= 50_000_000
    assert 25_000_000 < total  # not trivially small either


def test_forward_shapes_and_tied_head():
    cfg = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=32)
    model = ShoestringLM(cfg)
    x = torch.randint(0, 128, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 128)
    assert loss is not None and loss.item() > 0


def test_generate_extends_sequence():
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2, d_ff=64, max_seq_len=16)
    model = ShoestringLM(cfg)
    out = model.generate(torch.tensor([[1, 5, 6]]), max_new=4, top_k=10)
    assert out.shape[1] == 7
