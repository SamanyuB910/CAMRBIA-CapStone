import pytest

# NOTE: torch is imported inside the fixtures, not here. A module-level import
# would fail collection of the whole tests/ directory on a torch-less machine,
# taking the CPU-only C5/C6 tests (test_stats, test_figures) down with it.


def tiny_qwen3_5_config():
    """A tiny random-weight Qwen3.5 text config: same module classes as the
    real model (hybrid linear/full attention, SwiGLU MLP, (1+w) RMSNorm)."""
    from transformers import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=512,
        max_position_embeddings=512,
        full_attention_interval=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )


@pytest.fixture(scope="session")
def tiny_qwen():
    """Tiny fp32 Qwen3_5ForCausalLM with random weights, eval mode."""
    import torch
    from transformers import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(tiny_qwen3_5_config()).float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture()
def tiny_batch(tiny_qwen):
    import torch

    torch.manual_seed(1)
    return torch.randint(0, tiny_qwen.config.vocab_size, (2, 24))
