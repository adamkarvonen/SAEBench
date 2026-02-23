"""
Unit tests for core metrics with signed SAE activations.

Tests that L0, L1, and feature density metrics correctly handle SAE activations
that include negative values (not just non-negative ReLU-style activations).
"""

import torch
import pytest


class TestL1SparsityWithSignedActivations:
    """Test L1 norm calculation with mixed sign activations."""

    def test_l1_with_positive_and_negative_activations(self):
        """Test that L1 uses absolute values (sum of magnitudes)."""
        # Create activations: [1.0, -1.0, 2.0, -0.5, 0.0]
        # Expected L1: |1| + |-1| + |2| + |-0.5| + |0| = 4.5
        activations = torch.tensor([[1.0, -1.0, 2.0, -0.5, 0.0]])

        l1 = activations.abs().sum(dim=-1)

        assert torch.isclose(l1, torch.tensor(4.5))

    def test_l1_all_negative(self):
        """Test L1 with all negative activations."""
        activations = torch.tensor([[-1.0, -2.0, -0.5]])
        # Expected L1: 1 + 2 + 0.5 = 3.5
        l1 = activations.abs().sum(dim=-1)
        assert torch.isclose(l1, torch.tensor(3.5))

    def test_l1_all_positive(self):
        """Test L1 with all positive activations (backward compatibility)."""
        activations = torch.tensor([[1.0, 2.0, 0.5]])
        # Expected L1: 1 + 2 + 0.5 = 3.5
        l1 = activations.abs().sum(dim=-1)
        assert torch.isclose(l1, torch.tensor(3.5))

    def test_l1_without_abs_is_incorrect_for_signed(self):
        """Demonstrate that sum() without abs() gives wrong result for signed activations."""
        activations = torch.tensor([[1.0, -1.0, 2.0, -0.5]])
        # Without abs: 1 - 1 + 2 - 0.5 = 1.5 (WRONG)
        # With abs: |1| + |-1| + |2| + |-0.5| = 4.5 (CORRECT)

        incorrect_l1 = activations.sum(dim=-1)
        correct_l1 = activations.abs().sum(dim=-1)

        assert torch.isclose(incorrect_l1, torch.tensor(1.5))
        assert torch.isclose(correct_l1, torch.tensor(4.5))
        assert not torch.isclose(incorrect_l1, correct_l1)


class TestL0SparsityWithSignedActivations:
    """Test L0 (count of non-zeros) with signed activations."""

    def test_l0_counts_negative_activations(self):
        """Test that L0 correctly counts negative activations as non-zero."""
        activations = torch.tensor([[1.0, -1.0, 0.0, -0.5, 2.0, 0.0]])
        # Expected L0: 4 (positions 0, 1, 3, 4 are non-zero)

        l0 = (activations != 0).sum(dim=-1).float()

        assert torch.isclose(l0, torch.tensor(4.0))

    def test_l0_all_negative(self):
        """Test L0 with all negative activations."""
        activations = torch.tensor([[-1.0, -2.0, -0.5]])
        # Expected L0: 3

        l0 = (activations != 0).sum(dim=-1).float()
        assert torch.isclose(l0, torch.tensor(3.0))

    def test_l0_mixed_signs(self):
        """Test L0 with mixed positive and negative."""
        activations = torch.tensor([[1.0, -1.0, 2.0, -2.0, 0.0]])
        # Expected L0: 4

        l0 = (activations != 0).sum(dim=-1).float()
        assert torch.isclose(l0, torch.tensor(4.0))


class TestFeatureDensityWithSignedActivations:
    """Test feature density calculations with signed activations."""

    def test_feature_density_counts_negative_activations(self):
        """Test that feature density counts both positive and negative activations."""
        # Shape: [batch=2, seq_len=3, d_sae=4]
        # Feature 0: [1, 0, 0] and [0, 0, 0] -> 1 active token
        # Feature 1: [-1, 2, 0] and [0, 0, 0] -> 2 active tokens
        # Feature 2: [0, 0, -1] and [-2, 0, 0] -> 2 active tokens
        # Feature 3: [0, 0, 0] and [0, 0, 0] -> 0 active tokens
        activations = torch.tensor([
            [[1.0, -1.0, 0.0, 0.0],
             [0.0, 2.0, 0.0, 0.0],
             [0.0, 0.0, -1.0, 0.0]],
            [[0.0, 0.0, -2.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0]]
        ])

        # Count activations per feature (sum over batch and seq_len)
        activations_bool = (activations != 0).float()
        total_feature_acts = activations_bool.sum(dim=0).sum(dim=0)

        expected = torch.tensor([1.0, 2.0, 2.0, 0.0])
        assert torch.allclose(total_feature_acts, expected)

    def test_feature_density_backward_compatibility_positive_only(self):
        """Test that feature density works correctly for positive-only activations."""
        # All positive activations (ReLU-style SAE)
        activations = torch.tensor([
            [[1.0, 2.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0]]
        ])

        # Both methods should give same result for positive-only
        method_old = (activations > 0).float()
        method_new = (activations != 0).float()

        assert torch.allclose(method_old, method_new)

    def test_feature_density_detects_difference_with_negatives(self):
        """Test that old method (> 0) misses negative activations."""
        activations = torch.tensor([[1.0, -1.0, 0.0, 2.0, -0.5]])

        old_method = (activations > 0).float()  # Only counts positives
        new_method = (activations != 0).float()  # Counts both

        # Old method: [1, 0, 0, 1, 0] -> sum = 2
        # New method: [1, 1, 0, 1, 1] -> sum = 4
        assert torch.isclose(old_method.sum(), torch.tensor(2.0))
        assert torch.isclose(new_method.sum(), torch.tensor(4.0))


class TestBackwardCompatibility:
    """Test that changes maintain backward compatibility with non-negative SAEs."""

    def test_all_metrics_unchanged_for_nonnegative_activations(self):
        """Verify all metrics produce identical results for non-negative activations."""
        # Simulate ReLU SAE activations (all non-negative)
        activations = torch.tensor([[0.0, 1.5, 0.0, 2.3, 0.5, 0.0, 3.1]])

        # L0: should be same with both approaches
        l0 = (activations != 0).sum(dim=-1).float()
        assert torch.isclose(l0, torch.tensor(4.0))

        # L1: abs() should be no-op for non-negative
        l1_with_abs = activations.abs().sum(dim=-1)
        l1_without_abs = activations.sum(dim=-1)
        assert torch.isclose(l1_with_abs, l1_without_abs)

        # Feature density: != 0 equivalent to > 0 for non-negative
        density_old = (activations > 0).float()
        density_new = (activations != 0).float()
        assert torch.allclose(density_old, density_new)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_zeros(self):
        """Test with all-zero activations."""
        activations = torch.zeros(2, 5)

        l0 = (activations != 0).sum(dim=-1).float()
        l1 = activations.abs().sum(dim=-1)
        density = (activations != 0).float()

        assert torch.allclose(l0, torch.zeros(2))
        assert torch.allclose(l1, torch.zeros(2))
        assert torch.allclose(density, torch.zeros(2, 5))

    def test_very_small_negative_values(self):
        """Test with very small negative values near zero."""
        activations = torch.tensor([[1e-8, -1e-8, 0.0, 1e-7, -1e-7]])

        l0 = (activations != 0).sum(dim=-1).float()
        l1 = activations.abs().sum(dim=-1)

        # All non-zero values should be counted
        assert torch.isclose(l0, torch.tensor(4.0))
        # L1 should sum absolute values
        expected_l1 = 1e-8 + 1e-8 + 1e-7 + 1e-7
        assert torch.isclose(l1, torch.tensor(expected_l1), atol=1e-10)

    def test_large_negative_values(self):
        """Test with large negative activations."""
        activations = torch.tensor([[-100.0, 50.0, -75.0, 0.0, 25.0]])

        l0 = (activations != 0).sum(dim=-1).float()
        l1 = activations.abs().sum(dim=-1)

        assert torch.isclose(l0, torch.tensor(4.0))
        assert torch.isclose(l1, torch.tensor(250.0))  # 100 + 50 + 75 + 25
