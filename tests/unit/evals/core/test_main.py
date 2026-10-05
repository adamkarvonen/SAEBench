from unittest.mock import MagicMock

import pytest
import torch

from sae_bench.evals.core.main import get_sparsity_and_variance_metrics


def _evaluate(
    activations: torch.Tensor,
    error: torch.Tensor,
    batch_size: int,
    keep: torch.Tensor | None = None,
) -> dict[str, float]:
    if keep is None:
        keep = torch.ones(activations.shape[0], dtype=torch.bool)
    batches = list(activations.split(batch_size))
    tokens = [mask.long().unsqueeze(0) for mask in keep.split(batch_size)]
    width = activations.shape[-1]
    sae = MagicMock()
    sae.device = "cpu"
    sae.cfg.hook_name = "blocks.0.hook_resid_pre"
    sae.cfg.hook_head_index = None
    sae.cfg.hook_layer = 0
    sae.cfg.d_sae = 2 * width
    sae.encode.side_effect = lambda x: torch.cat([x.relu(), (-x).relu()], dim=-1)
    sae.decode.side_effect = lambda z: z[..., :width] - z[..., width:] + error
    model = MagicMock()
    model.run_with_cache.side_effect = [
        (None, {sae.cfg.hook_name: batch.unsqueeze(0)}) for batch in batches
    ]
    store = MagicMock()
    store.get_batch_tokens.side_effect = tokens

    metrics, _ = get_sparsity_and_variance_metrics(
        sae,
        model,
        store,
        n_batches=len(batches),
        compute_l2_norms=False,
        compute_sparsity_metrics=False,
        compute_variance_metrics=True,
        compute_featurewise_density_statistics=False,
        eval_batch_size_prompts=1,
        model_kwargs={},
        ignore_tokens={0},
    )
    return metrics


def _global_r2(x: torch.Tensor, error: torch.Tensor) -> float:
    reconstruction = x + error
    x = x.double()
    residual = x - reconstruction.double()
    return (1 - residual.square().sum() / (x - x.mean(dim=0)).square().sum()).item()


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("scale,shift", [(1.0, 0.0), (2.0, 0.0), (1.0, 10.0)])
def test_explained_variance_matches_global_r2(dtype, scale, shift):
    activations = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=dtype).repeat(4, 1)
    activations = activations * scale + shift
    error = torch.tensor(0.5 * scale, dtype=dtype)

    metrics = _evaluate(activations, error, batch_size=8)

    assert metrics["explained_variance"] == pytest.approx(
        _global_r2(activations, error)
    )
    assert metrics["explained_variance"] == pytest.approx(0.75)


@pytest.mark.parametrize("batch_size", [3, 6])
@pytest.mark.parametrize(
    "keep",
    [[True, True, True, True, False, False], [False, False, False, True, True, True]],
)
def test_explained_variance_weights_retained_tokens(batch_size, keep):
    activations = torch.tensor(
        [[-2.0, 3.0], [0.0, 5.0], [2.0, 7.0], [10.0, -3.0], [12.0, -1.0], [14.0, 1.0]],
        dtype=torch.float64,
    )
    error = torch.tensor([0.5, -1.0], dtype=torch.float64)
    mask = torch.tensor(keep)

    metrics = _evaluate(activations, error, batch_size, mask)

    assert metrics["explained_variance"] == pytest.approx(
        _global_r2(activations[mask], error)
    )


def test_explained_variance_promotes_before_low_precision_arithmetic():
    activations = torch.tensor(
        [[-300.0, -300.0], [300.0, 300.0]], dtype=torch.float16
    ).repeat(4, 1)
    error = torch.tensor(150.0, dtype=torch.float16)

    metrics = _evaluate(activations, error, batch_size=4)

    assert metrics["explained_variance"] == pytest.approx(
        _global_r2(activations, error)
    )
    assert metrics["explained_variance"] == pytest.approx(0.75)


def test_corrected_variance_preserves_legacy_and_mse():
    activations = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float64).repeat(
        4, 1
    )
    error = torch.tensor(0.5, dtype=torch.float64)

    metrics = _evaluate(activations, error, batch_size=4)

    assert metrics["explained_variance_legacy"] == pytest.approx(0.75)
    # Preserve the existing MSE convention; changing it is a separate fix.
    assert metrics["mse"] == pytest.approx(0.125)
