from sae_lens import SAE, ActivationsStore
from transformer_lens import HookedTransformer

from sae_bench.evals.mdl.main import _get_filtered_buffer


def test_get_filtered_buffer_concatenates_n_llm_batches(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    store_batch_size_prompts = 2
    n_batches = 3
    context_size = 128

    activations_store = ActivationsStore.from_sae(
        gpt2_model,
        gpt2_l4_sae,
        context_size=context_size,
        store_batch_size_prompts=store_batch_size_prompts,
        dataset="roneneldan/TinyStories",
        device="cpu",
    )

    buffer = _get_filtered_buffer(activations_store, n_batches=n_batches)

    expected_rows = n_batches * store_batch_size_prompts * context_size
    assert buffer.shape == (expected_rows, gpt2_l4_sae.cfg.d_in)
