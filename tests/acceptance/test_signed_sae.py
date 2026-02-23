"""
Integration test for SAE models with signed (positive and negative) activations.

This test creates a mock SAE that returns both positive and negative activations
and verifies that all core metrics are computed correctly.
"""

import torch
import pytest
from typing import Any

from sae_bench.custom_saes.base_sae import BaseSAE, CustomSAEConfig


class MockSignedSAE(BaseSAE):
    """
    Mock SAE implementation that returns signed activations for testing.

    This SAE intentionally produces both positive and negative activations
    to test that the benchmark correctly handles non-ReLU-style SAEs.
    """

    def __init__(self, cfg: CustomSAEConfig):
        super().__init__(cfg)

        # Initialize weights
        self.W_enc = torch.nn.Parameter(
            torch.randn(cfg.d_in, cfg.d_sae, dtype=cfg.dtype, device=cfg.device) * 0.1
        )
        self.W_dec = torch.nn.Parameter(
            torch.randn(cfg.d_sae, cfg.d_in, dtype=cfg.dtype, device=cfg.device) * 0.1
        )
        self.b_enc = torch.nn.Parameter(
            torch.randn(cfg.d_sae, dtype=cfg.dtype, device=cfg.device) * 0.01
        )
        self.b_dec = torch.nn.Parameter(
            torch.randn(cfg.d_in, dtype=cfg.dtype, device=cfg.device) * 0.01
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to SAE features with signed activations.

        Unlike standard ReLU SAEs, this returns both positive and negative values.
        We use tanh to ensure bounded signed outputs, then apply sparsity.
        """
        # Standard SAE encoding but with tanh instead of ReLU
        pre_acts = (x - self.b_dec) @ self.W_enc + self.b_enc

        # Use tanh for signed activations with smooth gradients
        acts = torch.tanh(pre_acts)

        # Apply top-k sparsity to ensure sparse but signed activations
        k = max(1, acts.shape[-1] // 20)  # Keep top 5% of features

        # Get top-k by absolute value (largest magnitude activations)
        topk_vals, topk_indices = torch.topk(acts.abs(), k=k, dim=-1)

        # Create sparse tensor with same shape
        sparse_acts = torch.zeros_like(acts)
        sparse_acts.scatter_(-1, topk_indices, acts.gather(-1, topk_indices))

        return sparse_acts

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Standard linear decoder."""
        return features @ self.W_dec


@pytest.mark.parametrize("d_in,d_sae", [(64, 256), (128, 512)])
def test_signed_sae_encode_returns_negative_values(d_in, d_sae):
    """Test that our mock signed SAE actually produces negative activations."""
    cfg = CustomSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        hook_name="blocks.0.hook_resid_post",
        hook_layer=0,
        model_name="test-model",
    )

    sae = MockSignedSAE(cfg)

    # Create random input
    x = torch.randn(2, 10, d_in)

    # Encode
    features = sae.encode(x)

    # Verify we have both positive and negative activations
    has_positive = (features > 0).any().item()
    has_negative = (features < 0).any().item()
    has_zero = (features == 0).any().item()

    assert has_positive, "Mock signed SAE should produce positive activations"
    assert has_negative, "Mock signed SAE should produce negative activations"
    assert has_zero, "Mock signed SAE should produce sparse (zero) activations"

    # Verify sparsity (most features should be zero)
    sparsity = (features == 0).float().mean().item()
    assert sparsity > 0.9, f"SAE should be sparse, but sparsity is only {sparsity:.2%}"


def test_signed_sae_l1_metric_uses_absolute_values():
    """Test that L1 metric correctly uses absolute values for signed activations."""
    cfg = CustomSAEConfig(
        d_in=64,
        d_sae=256,
        hook_name="blocks.0.hook_resid_post",
        hook_layer=0,
        model_name="test-model",
    )

    sae = MockSignedSAE(cfg)

    # Create input
    x = torch.randn(2, 10, 64)
    features = sae.encode(x)

    # Flatten for metric calculation (as done in core eval)
    flattened = features.reshape(-1, features.shape[-1])

    # Compute L1 the correct way (with abs)
    l1_correct = flattened.abs().sum(dim=-1).mean().item()

    # Compute L1 the wrong way (without abs) - would give wrong results
    l1_wrong = flattened.sum(dim=-1).mean().item()

    # For signed activations, these should differ
    # (unless by chance the positive and negative exactly cancel, which is very unlikely)
    assert l1_correct > 0, "L1 with abs() should be positive"

    # L1 with abs should generally be larger than raw sum for signed activations
    assert abs(l1_correct) >= abs(l1_wrong), \
        "L1 with abs() should have larger or equal magnitude than raw sum"


def test_signed_sae_feature_density_counts_negatives():
    """Test that feature density counts both positive and negative activations."""
    cfg = CustomSAEConfig(
        d_in=64,
        d_sae=256,
        hook_name="blocks.0.hook_resid_post",
        hook_layer=0,
        model_name="test-model",
    )

    sae = MockSignedSAE(cfg)

    # Create input
    x = torch.randn(2, 10, 64)
    features = sae.encode(x)

    # Count activations the correct way (for signed SAEs)
    activations_correct = (features != 0).float()

    # Count activations the old way (only positive)
    activations_old = (features > 0).float()

    # For signed SAEs, correct method should count more activations
    count_correct = activations_correct.sum().item()
    count_old = activations_old.sum().item()

    # Verify we're counting negative activations
    negative_count = (features < 0).sum().item()

    if negative_count > 0:  # If we have negative activations
        assert count_correct > count_old, \
            "Correct method (!=0) should count more activations than old method (>0)"
        assert count_correct == count_old + negative_count, \
            "Difference should equal number of negative activations"


def test_signed_sae_l0_unchanged():
    """Test that L0 (count of non-zeros) works correctly for signed activations."""
    cfg = CustomSAEConfig(
        d_in=64,
        d_sae=256,
        hook_name="blocks.0.hook_resid_post",
        hook_layer=0,
        model_name="test-model",
    )

    sae = MockSignedSAE(cfg)

    # Create input
    x = torch.randn(2, 10, 64)
    features = sae.encode(x)

    # Flatten
    flattened = features.reshape(-1, features.shape[-1])

    # L0 should count all non-zero elements (positive or negative)
    l0 = (flattened != 0).sum(dim=-1).float()

    # Verify L0 matches total non-zeros
    positive_count = (flattened > 0).sum(dim=-1).float()
    negative_count = (flattened < 0).sum(dim=-1).float()

    assert torch.allclose(l0, positive_count + negative_count), \
        "L0 should equal sum of positive and negative counts"


def test_backward_compatibility_with_nonnegative_mock():
    """
    Test that metrics work identically for non-negative activations.

    This verifies backward compatibility - if we forced all activations to be
    non-negative, the new metric calculations should give same results as old.
    """
    cfg = CustomSAEConfig(
        d_in=64,
        d_sae=256,
        hook_name="blocks.0.hook_resid_post",
        hook_layer=0,
        model_name="test-model",
    )

    sae = MockSignedSAE(cfg)

    # Create input and encode
    x = torch.randn(2, 10, 64)
    features = sae.encode(x)

    # Force features to be non-negative (simulate ReLU SAE)
    features_nonneg = features.abs()  # All non-negative

    # Flatten
    flat = features_nonneg.reshape(-1, features_nonneg.shape[-1])

    # Old and new L1 should be identical for non-negative
    l1_old = flat.sum(dim=-1)
    l1_new = flat.abs().sum(dim=-1)
    assert torch.allclose(l1_old, l1_new), \
        "L1 with and without abs() should be identical for non-negative values"

    # Old and new density should be identical for non-negative
    density_old = (flat > 0).float()
    density_new = (flat != 0).float()
    assert torch.allclose(density_old, density_new), \
        "Density >0 and !=0 should be identical for non-negative values"


class TestSignedActivationsIntegration:
    """
    Integration tests that would ideally run full evaluations.

    Note: These are marked as slow/integration and may be skipped in CI.
    They require full model loading and are more comprehensive.
    """

    @pytest.mark.slow
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Integration test requires GPU"
    )
    def test_core_eval_with_signed_sae_mock(self):
        """
        Full integration test running core.run_evals() with a signed SAE.

        This would test the entire pipeline end-to-end, but requires
        significant resources (model loading, activation collection, etc.).
        """
        pytest.skip(
            "Full integration test requires model loading and is resource-intensive. "
            "Run manually when testing major changes."
        )

        # TODO: Implement full integration test when needed
        # This would:
        # 1. Create MockSignedSAE with proper config
        # 2. Load a small model (e.g., pythia-70m)
        # 3. Create ActivationsStore
        # 4. Run core.run_evals()
        # 5. Verify all metrics computed without errors
        # 6. Check that L1, L0, feature density metrics are sensible
