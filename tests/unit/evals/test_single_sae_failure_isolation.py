from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import sae_bench.evals.absorption.main as absorption_main
import sae_bench.evals.autointerp.main as autointerp_main
import sae_bench.evals.core.main as core_main
import sae_bench.evals.mdl.main as mdl_main
import sae_bench.evals.meta_structure.main as meta_structure_main
import sae_bench.evals.ravel.main as ravel_main
import sae_bench.evals.scr_and_tpp.main as scr_and_tpp_main
import sae_bench.evals.sparse_probing.main as sparse_probing_main
import sae_bench.evals.sparse_probing_sae_probes.main as sae_probes_main
import sae_bench.evals.unlearning.main as unlearning_main
import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.mdl.eval_config import MDLEvalConfig
from sae_bench.evals.meta_structure.eval_config import MetaStructureEvalConfig
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
from sae_bench.evals.sparse_probing_sae_probes.eval_config import (
    SparseProbingSaeProbesEvalConfig,
)
from sae_bench.evals.unlearning.eval_config import UnlearningEvalConfig


def _make_fake_sae(
    model_name: str = "gpt2", hook_name: str = "blocks.4.hook_resid_pre"
):
    sae = MagicMock()
    sae.W_enc = torch.zeros((2, 2), dtype=torch.float32)
    sae.cfg = SimpleNamespace(
        model_name=model_name,
        hook_name=hook_name,
        hook_layer=4,
        model_from_pretrained_kwargs={},
        metadata=SimpleNamespace(sae_lens_id="custom_id"),
        to_dict=lambda: {"model_name": model_name, "hook_name": hook_name},
    )
    sae.to.return_value = sae
    return sae


def test_format_sae_label():
    assert general_utils.format_sae_label("layer_0/width_16k/average_l0_50") == (
        "layer_0/width_16k/average_l0_50"
    )
    fake_sae = _make_fake_sae()
    assert general_utils.format_sae_label(fake_sae) == "custom_id"

    plain_obj = SimpleNamespace()
    assert general_utils.format_sae_label(plain_obj) == "custom_sae"


def test_cleanup_cuda_guards_cuda_availability():
    with (
        patch("sae_bench.sae_bench_utils.general_utils.gc.collect") as mock_gc,
        patch(
            "sae_bench.sae_bench_utils.general_utils.torch.cuda.is_available",
            return_value=False,
        ),
        patch(
            "sae_bench.sae_bench_utils.general_utils.torch.cuda.empty_cache"
        ) as mock_empty_cache,
    ):
        general_utils.cleanup_cuda()
        mock_gc.assert_called_once()
        mock_empty_cache.assert_not_called()

    with (
        patch("sae_bench.sae_bench_utils.general_utils.gc.collect") as mock_gc,
        patch(
            "sae_bench.sae_bench_utils.general_utils.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "sae_bench.sae_bench_utils.general_utils.torch.cuda.empty_cache"
        ) as mock_empty_cache,
    ):
        general_utils.cleanup_cuda()
        mock_gc.assert_called_once()
        mock_empty_cache.assert_called_once()


def _make_three_sae_load_side_effect(ok_result_path: Path):
    """SAE 1 fails to load, SAE 2 loads then fails mid-eval, SAE 3 succeeds via pre-existing result."""
    ok_result_path.parent.mkdir(parents=True, exist_ok=True)
    ok_result_path.write_text("{}")

    sae_2 = _make_fake_sae()
    sae_3 = _make_fake_sae()

    def _side_effect(sae_release: str, sae_object_or_id: object, device: str):
        if sae_object_or_id == "bad_load_sae":
            raise RuntimeError("Failed to download 1m SAE weights")
        if sae_object_or_id == "oom_eval_sae":
            return ("oom_eval_sae", sae_2, torch.zeros(4))
        return ("good_sae", sae_3, torch.zeros(4))

    return _side_effect


@pytest.mark.parametrize(
    "eval_kind",
    [
        "absorption",
        "autointerp",
        "mdl",
        "meta_structure",
        "ravel",
        "scr_and_tpp",
        "sparse_probing",
        "sparse_probing_sae_probes",
        "unlearning",
    ],
)
def test_run_eval_continues_when_single_sae_fails(eval_kind: str, tmp_path: Path):
    release = "test_release"
    selected_saes = [
        (release, "bad_load_sae"),
        (release, "oom_eval_sae"),
        (release, "good_sae"),
    ]
    output_dir = tmp_path / eval_kind

    if eval_kind == "scr_and_tpp":
        expected_good_path = Path(
            general_utils.get_results_filepath(
                str(output_dir / "scr"), release, "good_sae"
            )
        )
    else:
        expected_good_path = Path(
            general_utils.get_results_filepath(str(output_dir), release, "good_sae")
        )

    load_side_effect = _make_three_sae_load_side_effect(expected_good_path)

    with (
        patch(
            "sae_bench.sae_bench_utils.general_utils.load_and_format_sae",
            side_effect=load_side_effect,
        ) as mock_load,
        patch("sae_bench.sae_bench_utils.general_utils.cleanup_cuda") as mock_cleanup,
    ):
        if eval_kind == "absorption":
            with (
                patch.object(
                    absorption_main.HookedTransformer,
                    "from_pretrained_no_processing",
                    return_value=MagicMock(),
                ),
                patch.object(
                    absorption_main,
                    "run_k_sparse_probing_experiment",
                    side_effect=RuntimeError("CUDA out of memory"),
                ) as mock_step,
            ):
                res = absorption_main.run_eval(
                    AbsorptionEvalConfig(model_name="gpt2", llm_dtype="float32"),
                    selected_saes,
                    device="cpu",
                    output_path=str(output_dir),
                )
                mock_step.assert_called_once()
                assert res == {}
        elif eval_kind == "autointerp":
            with (
                patch.object(
                    autointerp_main.HookedTransformer,
                    "from_pretrained_no_processing",
                    return_value=MagicMock(),
                ),
                patch.object(
                    autointerp_main,
                    "run_eval_single_sae",
                    side_effect=RuntimeError("CUDA out of memory"),
                ) as mock_step,
            ):
                res = autointerp_main.run_eval(
                    AutoInterpEvalConfig(model_name="gpt2", llm_dtype="float32"),
                    selected_saes,
                    device="cpu",
                    api_key="dummy",
                    output_path=str(output_dir),
                    artifacts_path=str(tmp_path / "artifacts"),
                )
                mock_step.assert_called_once()
                assert res == {}
        elif eval_kind == "mdl":
            with (
                patch.object(
                    mdl_main.HookedTransformer,
                    "from_pretrained_no_processing",
                    return_value=MagicMock(),
                ),
                patch.object(
                    mdl_main,
                    "run_eval_single_sae",
                    side_effect=RuntimeError("CUDA out of memory"),
                ) as mock_step,
            ):
                res = mdl_main.run_eval(
                    MDLEvalConfig(model_name="gpt2", llm_dtype="float32"),
                    selected_saes,
                    device="cpu",
                    output_path=str(output_dir),
                )
                mock_step.assert_called_once()
                assert "custom_eval_config" in res
        elif eval_kind == "meta_structure":
            with patch.object(
                meta_structure_main,
                "run_eval_single_sae",
                side_effect=RuntimeError("CUDA out of memory"),
            ) as mock_step:
                res = meta_structure_main.run_eval(
                    MetaStructureEvalConfig(),
                    selected_saes,
                    device="cpu",
                    output_path=str(output_dir),
                )
                mock_step.assert_called_once()
                assert res == {}
        elif eval_kind == "ravel":
            with (
                patch.object(
                    ravel_main.AutoModelForCausalLM,
                    "from_pretrained",
                    return_value=MagicMock(),
                ),
                patch.object(
                    ravel_main.AutoTokenizer,
                    "from_pretrained",
                    side_effect=RuntimeError("CUDA out of memory"),
                ) as mock_step,
            ):
                res = ravel_main.run_eval(
                    RAVELEvalConfig(model_name="gemma-2-2b", llm_dtype="float32"),
                    selected_saes,
                    device="cpu",
                    output_path=str(output_dir),
                    artifacts_path=str(tmp_path / "artifacts"),
                )
                mock_step.assert_called_once()
                assert res == {}
        elif eval_kind == "scr_and_tpp":
            with (
                patch.object(
                    scr_and_tpp_main.HookedTransformer,
                    "from_pretrained_no_processing",
                    return_value=MagicMock(),
                ),
                patch.object(
                    scr_and_tpp_main,
                    "run_eval_single_sae",
                    side_effect=RuntimeError("CUDA out of memory"),
                ) as mock_step,
            ):
                res = scr_and_tpp_main.run_eval(
                    ScrAndTppEvalConfig(
                        model_name="gpt2", llm_dtype="float32", perform_scr=True
                    ),
                    selected_saes,
                    device="cpu",
                    output_path=str(output_dir),
                    clean_up_activations=True,
                    artifacts_path=str(tmp_path / "artifacts"),
                )
                mock_step.assert_called_once()
                assert res == {}
        elif eval_kind == "sparse_probing":
            with (
                patch.object(
                    sparse_probing_main.HookedTransformer,
                    "from_pretrained_no_processing",
                    return_value=MagicMock(),
                ),
                patch.object(
                    sparse_probing_main,
                    "run_eval_single_sae",
                    side_effect=RuntimeError("CUDA out of memory"),
                ) as mock_step,
            ):
                res = sparse_probing_main.run_eval(
                    SparseProbingEvalConfig(model_name="gpt2", llm_dtype="float32"),
                    selected_saes,
                    device="cpu",
                    output_path=str(output_dir),
                    clean_up_activations=True,
                    artifacts_path=str(tmp_path / "artifacts"),
                )
                mock_step.assert_called_once()
                assert res == {}
        elif eval_kind == "sparse_probing_sae_probes":
            with patch.object(
                sae_probes_main,
                "run_sae_evals",
                side_effect=RuntimeError("CUDA out of memory"),
            ) as mock_step:
                res = sae_probes_main.run_eval(
                    SparseProbingSaeProbesEvalConfig(
                        model_name="gpt2",
                        include_llm_baseline=False,
                        results_path=str(tmp_path / "sae_probes_results"),
                        model_cache_path=str(tmp_path / "model_cache"),
                    ),
                    selected_saes,
                    device="cpu",
                    output_path=str(output_dir),
                )
                mock_step.assert_called_once()
                assert res == {}
        elif eval_kind == "unlearning":
            with (
                patch.object(
                    unlearning_main.HookedTransformer,
                    "from_pretrained_no_processing",
                    return_value=MagicMock(),
                ),
                patch.object(
                    unlearning_main,
                    "run_eval_single_sae",
                    side_effect=RuntimeError("CUDA out of memory"),
                ) as mock_step,
            ):
                res = unlearning_main.run_eval(
                    UnlearningEvalConfig(
                        model_name="gemma-2-2b-it", llm_dtype="float32"
                    ),
                    selected_saes,
                    device="cpu",
                    output_path=str(output_dir),
                    clean_up_artifacts=True,
                    artifacts_path=str(tmp_path / "artifacts"),
                )
                mock_step.assert_called_once()
                assert res == {}

        assert mock_load.call_count == 3
        # mdl calls cleanup_cuda once per SAE plus once after the loop
        expected_cleanups = 4 if eval_kind == "mdl" else 3
        assert mock_cleanup.call_count == expected_cleanups


def test_core_multiple_evals_isolates_sae_and_model_load_failures(tmp_path: Path):
    release = "test_release"
    selected_saes = [
        (release, "bad_load_sae"),
        (release, "bad_model_sae"),
        (release, "oom_sae"),
        (release, "good_sae"),
    ]

    sae_bad_model = _make_fake_sae(model_name="broken-model")
    sae_oom = _make_fake_sae(model_name="gpt2")
    sae_good = _make_fake_sae(model_name="gpt2")

    def _load_sae_side_effect(sae_release: str, sae_object_or_id: object, device: str):
        if sae_object_or_id == "bad_load_sae":
            raise RuntimeError("Missing SAE checkpoint")
        if sae_object_or_id == "bad_model_sae":
            return ("bad_model_sae", sae_bad_model, torch.zeros(4))
        if sae_object_or_id == "oom_sae":
            return ("oom_sae", sae_oom, torch.zeros(4))
        return ("good_sae", sae_good, torch.zeros(4))

    fake_model = MagicMock()
    fake_model.tokenizer = MagicMock()

    def _load_model_side_effect(model_name: str, *args, **kwargs):
        if model_name == "broken-model":
            raise RuntimeError("Model download failed")
        return fake_model

    eval_call_count = 0

    def _single_eval_side_effect(*args, **kwargs):
        nonlocal eval_call_count
        eval_call_count += 1
        if eval_call_count == 1:
            raise RuntimeError("CUDA out of memory on 1m SAE")
        return (
            {
                "model_behavior_preservation": {"kl_div_score": 0.1},
                "model_performance_preservation": {
                    "ce_loss_score": 0.9,
                    "ce_loss_with_ablation": 5.0,
                    "ce_loss_with_sae": 2.1,
                    "ce_loss_without_sae": 2.0,
                },
                "reconstruction_quality": {
                    "explained_variance": 0.85,
                    "mse": 0.02,
                    "cossim": 0.95,
                },
                "shrinkage": {
                    "l2_norm_in": 1.0,
                    "l2_norm_out": 0.98,
                    "l2_ratio": 0.98,
                    "relative_reconstruction_bias": 1.0,
                },
                "sparsity": {"l0": 20.0, "l1": 15.0},
                "token_stats": {
                    "total_tokens_evaluated_reconstruction": 128,
                    "total_tokens_evaluated_eval_sparsity_variance": 128,
                },
            },
            {},
        )

    with (
        patch(
            "sae_bench.sae_bench_utils.general_utils.load_and_format_sae",
            side_effect=_load_sae_side_effect,
        ) as mock_load_sae,
        patch("sae_bench.sae_bench_utils.general_utils.time.sleep", return_value=None),
        patch.object(
            core_main.HookedTransformer,
            "from_pretrained_no_processing",
            side_effect=_load_model_side_effect,
        ) as mock_load_model,
        patch.object(core_main.ActivationsStore, "from_sae", return_value=MagicMock()),
        patch.object(core_main, "run_evals", side_effect=_single_eval_side_effect),
        patch.object(
            core_main, "save_single_eval_result", return_value="saved.json"
        ) as mock_save,
        patch("sae_bench.sae_bench_utils.general_utils.cleanup_cuda") as mock_cleanup,
    ):
        results = core_main.multiple_evals(
            selected_saes=selected_saes,
            n_eval_reconstruction_batches=1,
            n_eval_sparsity_variance_batches=1,
            output_folder=str(tmp_path / "core_out"),
            device="cpu",
            dtype="float32",
        )

    assert mock_load_sae.call_count == 4
    # 6 attempts (1 initial + 5 backoff retries) for broken-model, plus 1 call for gpt2 reused across oom_sae & good_sae
    assert mock_load_model.call_count == 7
    assert mock_cleanup.call_count == 4
    assert len(results) == 1
    assert results[0]["sae_id"] == "good_sae"
    mock_save.assert_called_once()


def test_ravel_train_mdas_config_assertion_is_not_swallowed(tmp_path: Path):
    with patch.object(
        ravel_main.AutoModelForCausalLM, "from_pretrained", return_value=MagicMock()
    ):
        with pytest.raises(AssertionError):
            ravel_main.run_eval(
                RAVELEvalConfig(
                    model_name="gemma-2-2b", llm_dtype="float32", train_mdas=True
                ),
                [("rel", "sae_1"), ("rel", "sae_2")],
                device="cpu",
                output_path=str(tmp_path / "ravel_out"),
            )
