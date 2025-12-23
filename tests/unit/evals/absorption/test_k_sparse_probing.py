import numpy as np
import pandas as pd
import torch
from sae_lens import SAE

from sae_bench.evals.absorption.k_sparse_probing import (
    KSparseProbe,
    _get_sae_acts,
    eval_probe_and_sae_k_sparse_raw_scores,
    train_k_sparse_probes,
    train_sparse_multi_probe,
)
from sae_bench.evals.absorption.probing import LinearProbe
from sae_bench.evals.absorption.vocab import LETTERS


def test_train_sparse_multi_probe_results_in_many_zero_weights():
    torch.set_grad_enabled(True)
    x = torch.rand(1000, 500)
    y = torch.randint(2, (1000, 3))
    probe1 = train_sparse_multi_probe(x, y, l1_decay=0.015, device=torch.device("cpu"))
    probe2 = train_sparse_multi_probe(x, y, l1_decay=1.0, device=torch.device("cpu"))

    probe1_zero_weights = (probe1.weights.abs() < 1e-5).sum()
    probe2_zero_weights = (probe2.weights.abs() < 1e-5).sum()

    assert probe1_zero_weights > 0
    assert probe2_zero_weights > 0
    assert probe2_zero_weights > probe1_zero_weights


def test_train_k_sparse_probes_returns_reasonable_values(gpt2_l4_sae: SAE):
    torch.set_grad_enabled(True)
    train_labels = [("aaa", 0), ("bbb", 1), ("ccc", 2)]
    train_activations = torch.randn(3, 768)
    probes = train_k_sparse_probes(
        gpt2_l4_sae,
        train_labels,
        train_activations,
        ks=[1, 2, 3],
    )
    assert probes.keys() == {1, 2, 3}
    for k, k_probes in probes.items():
        assert k_probes.keys() == {0, 1, 2}
        for probe in k_probes.values():
            assert probe.weight.shape == (k,)
            assert probe.feature_ids.shape == (k,)
            assert probe.k == k


def test_get_sae_acts(gpt2_l4_sae: SAE):
    token_act = torch.randn(768)
    sae_acts = _get_sae_acts(gpt2_l4_sae, token_act.unsqueeze(0)).squeeze()
    assert sae_acts.shape == (24576,)


def test_get_sae_acts_gives_same_results_batched_and_not_batched(gpt2_l4_sae: SAE):
    token_acts = torch.randn(10, 768)
    sae_acts_unbatched = _get_sae_acts(gpt2_l4_sae, token_acts, batch_size=1)
    sae_acts_batched = _get_sae_acts(gpt2_l4_sae, token_acts, batch_size=5)
    assert torch.allclose(sae_acts_unbatched, sae_acts_batched, atol=1e-3)


def test_eval_probe_and_sae_k_sparse_raw_scores_gives_sane_results(gpt2_l4_sae: SAE):
    torch.set_grad_enabled(True)
    fake_probe = LinearProbe(768, 26)
    eval_data = [(letter, i) for i, letter in enumerate(LETTERS)]
    eval_activations = torch.randn(26, 768)
    k_sparse_probes = train_k_sparse_probes(
        gpt2_l4_sae,
        eval_data,
        eval_activations,
        ks=[1, 2, 3],
    )
    df = eval_probe_and_sae_k_sparse_raw_scores(
        gpt2_l4_sae,
        fake_probe,
        k_sparse_probes,
        eval_data,
        eval_activations,
    )
    expected_columns = [
        "token",
        "answer_letter",
    ]
    for letter in LETTERS:
        expected_columns.append(f"score_probe_{letter}")
        for k in [1, 2, 3]:
            expected_columns.append(f"score_sparse_sae_{letter}_k_{k}")
            expected_columns.append(f"sum_sparse_sae_{letter}_k_{k}")
            expected_columns.append(f"sparse_sae_{letter}_k_{k}_acts")
    assert set(df.columns.values.tolist()) == set(expected_columns)


def test_eval_probe_and_sae_k_sparse_raw_scores_matches_previous_implementation_results(
    gpt2_l4_sae: SAE,
):
    @torch.inference_mode()
    def _prev_eval_probe_and_sae_k_sparse_raw_scores(
        sae: SAE,
        probe: LinearProbe,
        k_sparse_probes: dict[int, dict[int, KSparseProbe]],
        eval_labels: list[tuple[str, int]],  # list of (token, letter number) pairs
        eval_activations: torch.Tensor,  # n_vocab X d_model
    ) -> pd.DataFrame:
        probe = probe.to("cpu")

        # using a generator to avoid storing all the rows in memory
        def row_generator():
            for token_act, (token, answer_idx) in zip(eval_activations, eval_labels):
                probe_scores = probe(token_act).tolist()
                row: dict[str, float | str | int | np.ndarray] = {
                    "token": token,
                    "answer_letter": LETTERS[answer_idx],
                }
                sae_acts = (
                    _get_sae_acts(sae, token_act.unsqueeze(0).to(sae.device))
                    .float()
                    .cpu()
                ).squeeze()
                for letter_i, (letter, probe_score) in enumerate(
                    zip(LETTERS, probe_scores)
                ):
                    row[f"score_probe_{letter}"] = probe_score
                    for k, k_probes in k_sparse_probes.items():
                        k_probe = k_probes[letter_i]
                        k_probe_score = k_probe(sae_acts)
                        sparse_acts = sae_acts[k_probe.feature_ids]
                        row[f"score_sparse_sae_{letter}_k_{k}"] = k_probe_score.item()
                        row[f"sum_sparse_sae_{letter}_k_{k}"] = sparse_acts.sum().item()
                        row[f"sparse_sae_{letter}_k_{k}_acts"] = sparse_acts.numpy()
                yield row

        return pd.DataFrame(row_generator())

    torch.set_grad_enabled(True)
    fake_probe = LinearProbe(768, 26)
    eval_data = [(letter, i) for i, letter in enumerate(LETTERS)] * 10
    eval_activations = torch.randn(len(eval_data), 768)
    k_sparse_probes = train_k_sparse_probes(
        gpt2_l4_sae,
        eval_data,
        eval_activations,
        ks=[1, 2, 3],
    )
    new_df = eval_probe_and_sae_k_sparse_raw_scores(
        gpt2_l4_sae,
        fake_probe,
        k_sparse_probes,
        eval_data,
        eval_activations,
    )
    prev_df = _prev_eval_probe_and_sae_k_sparse_raw_scores(
        gpt2_l4_sae,
        fake_probe,
        k_sparse_probes,
        eval_data,
        eval_activations,
    )
    pd.testing.assert_frame_equal(
        new_df, prev_df, check_exact=False, rtol=1e-3, atol=1e-3, check_dtype=False
    )
