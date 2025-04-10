import pytest
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, GPT2LMHeadModel


@pytest.fixture
def gpt2_model():
    return HookedTransformer.from_pretrained("gpt2", device="cpu")


@pytest.fixture
def gpt2_tokenizer(gpt2_model: HookedTransformer):
    return gpt2_model.tokenizer


@pytest.fixture
def fake_mistral_tokenizer():
    # This seems to work the same as the mistral tokenizer, but we can't load the
    # real mistral tokenizer in tests since it's gated.
    return AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")


@pytest.fixture
def gpt2_l4_sae() -> SAE:
    return SAE.from_pretrained(
        "gpt2-small-res-jb", "blocks.4.hook_resid_pre", device="cpu"
    )[0]


@pytest.fixture
def gpt2_l4_sae_sparsity() -> torch.Tensor:
    sparsity = SAE.from_pretrained(
        "gpt2-small-res-jb", "blocks.4.hook_resid_pre", device="cpu"
    )[2]
    assert sparsity is not None
    return sparsity


@pytest.fixture
def gpt2_l5_sae() -> SAE:
    return SAE.from_pretrained(
        "gpt2-small-res-jb", "blocks.5.hook_resid_pre", device="cpu"
    )[0]


@pytest.fixture
def gpt2_hf_model(gpt2_model: HookedTransformer):
    model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="cpu")
    return model, gpt2_model.tokenizer
