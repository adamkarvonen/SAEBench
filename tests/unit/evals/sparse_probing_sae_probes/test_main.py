import json
from pathlib import Path

from sae_lens import SAE

from sae_bench.evals.sparse_probing_sae_probes.eval_config import (
    SparseProbingSaeProbesEvalConfig,
)
from sae_bench.evals.sparse_probing_sae_probes.eval_output import (
    SparseProbingSaeProbesEvalOutput,
)
from sae_bench.evals.sparse_probing_sae_probes.main import run_eval


def test_run_eval_without_baselines(gpt2_l4_sae: SAE, tmp_path: Path):
    output_path = tmp_path / "test_output"
    artifacts_path = tmp_path / "test_artifacts"
    model_cache_path = tmp_path / "model_cache"
    config = SparseProbingSaeProbesEvalConfig(
        model_name="gpt2",
        include_llm_baseline=False,
        model_cache_path=str(model_cache_path),
        results_path=str(artifacts_path),
        dataset_names=["118_us_state_CA", "119_us_state_TX"],
    )
    results_dict = run_eval(
        config,
        [("gpt2_l4_sae", gpt2_l4_sae)],
        device="cpu",
        output_path=str(output_path),
    )

    assert isinstance(results_dict, dict)
    assert len(results_dict) == 1
    assert "gpt2_l4_sae_custom_sae" in results_dict

    result_data = results_dict["gpt2_l4_sae_custom_sae"]
    assert isinstance(result_data, dict)
    assert result_data["eval_type_id"] == "sparse_probing_sae_probes"
    assert result_data["sae_lens_release_id"] == "gpt2_l4_sae"
    assert result_data["sae_lens_id"] == "custom_sae"

    expected_output_file = output_path / "gpt2_l4_sae_custom_sae_eval_results.json"
    assert expected_output_file.exists(), "Main output JSON file should exist"

    with open(expected_output_file) as f:
        output_data = json.load(f)

    assert result_data["eval_type_id"] == output_data["eval_type_id"]
    assert result_data["sae_lens_release_id"] == output_data["sae_lens_release_id"]
    assert result_data["eval_result_metrics"] == output_data["eval_result_metrics"]

    eval_output = SparseProbingSaeProbesEvalOutput(**output_data)

    assert eval_output.eval_type_id == "sparse_probing_sae_probes"
    assert eval_output.sae_lens_release_id == "gpt2_l4_sae"
    assert eval_output.sae_lens_id == "custom_sae"
    assert eval_output.eval_config.model_name == "gpt2"

    assert eval_output.eval_result_metrics.sae.sae_top_1_test_accuracy is not None
    assert 0 <= eval_output.eval_result_metrics.sae.sae_top_1_test_accuracy <= 1
    assert eval_output.eval_result_metrics.sae.sae_top_1_test_auc is not None
    assert 0 <= eval_output.eval_result_metrics.sae.sae_top_1_test_auc <= 1
    assert eval_output.eval_result_metrics.sae.sae_top_1_test_f1 is not None
    assert 0 <= eval_output.eval_result_metrics.sae.sae_top_1_test_f1 <= 1
    assert eval_output.eval_result_metrics.sae.sae_top_2_test_accuracy is not None
    assert 0 <= eval_output.eval_result_metrics.sae.sae_top_2_test_accuracy <= 1
    assert eval_output.eval_result_metrics.sae.sae_top_5_test_accuracy is not None
    assert 0 <= eval_output.eval_result_metrics.sae.sae_top_5_test_accuracy <= 1

    assert eval_output.eval_result_metrics.llm.llm_test_accuracy is None
    assert eval_output.eval_result_metrics.llm.llm_test_auc is None
    assert eval_output.eval_result_metrics.llm.llm_test_f1 is None

    assert eval_output.sae_metrics_by_k is not None
    assert set(eval_output.sae_metrics_by_k.keys()) == {1, 2, 5}

    assert len(eval_output.eval_result_details) == 2
    dataset_names = {detail.dataset_name for detail in eval_output.eval_result_details}
    assert dataset_names == {"118_us_state_CA", "119_us_state_TX"}

    for detail in eval_output.eval_result_details:
        assert detail.sae_top_1_test_accuracy is not None
        assert 0 <= detail.sae_top_1_test_accuracy <= 1
        assert detail.sae_top_1_test_auc is not None
        assert 0 <= detail.sae_top_1_test_auc <= 1
        assert detail.sae_top_1_test_f1 is not None
        assert 0 <= detail.sae_top_1_test_f1 <= 1
        assert detail.llm_test_accuracy is None
        assert detail.llm_test_auc is None
        assert detail.llm_test_f1 is None

    sae_probes_results_dir = (
        artifacts_path / "gpt2_l4_sae_custom_sae" / "sae_probes_gpt2" / "normal_setting"
    )
    assert sae_probes_results_dir.exists()
    json_files = list(sae_probes_results_dir.glob("*.json"))
    assert len(json_files) >= 2


def test_run_eval_with_baselines(gpt2_l4_sae: SAE, tmp_path: Path):
    output_path = tmp_path / "test_output"
    artifacts_path = tmp_path / "test_artifacts"
    model_cache_path = tmp_path / "model_cache"
    config = SparseProbingSaeProbesEvalConfig(
        model_name="gpt2",
        include_llm_baseline=True,
        model_cache_path=str(model_cache_path),
        results_path=str(artifacts_path),
        dataset_names=["118_us_state_CA"],
    )
    results_dict = run_eval(
        config,
        [("gpt2_l4_sae", gpt2_l4_sae)],
        device="cpu",
        output_path=str(output_path),
    )

    assert isinstance(results_dict, dict)
    assert len(results_dict) == 1
    assert "gpt2_l4_sae_custom_sae" in results_dict

    result_data = results_dict["gpt2_l4_sae_custom_sae"]
    assert isinstance(result_data, dict)
    assert "eval_result_metrics" in result_data
    assert "llm" in result_data["eval_result_metrics"]
    assert result_data["eval_result_metrics"]["llm"]["llm_test_accuracy"] is not None

    expected_output_file = output_path / "gpt2_l4_sae_custom_sae_eval_results.json"
    assert expected_output_file.exists()

    with open(expected_output_file) as f:
        output_data = json.load(f)

    assert result_data["eval_type_id"] == output_data["eval_type_id"]
    assert result_data["sae_lens_release_id"] == output_data["sae_lens_release_id"]
    assert result_data["eval_result_metrics"] == output_data["eval_result_metrics"]

    eval_output = SparseProbingSaeProbesEvalOutput(**output_data)

    assert eval_output.eval_result_metrics.llm.llm_test_accuracy is not None
    assert 0 <= eval_output.eval_result_metrics.llm.llm_test_accuracy <= 1
    assert eval_output.eval_result_metrics.llm.llm_test_auc is not None
    assert 0 <= eval_output.eval_result_metrics.llm.llm_test_auc <= 1
    assert eval_output.eval_result_metrics.llm.llm_test_f1 is not None
    assert 0 <= eval_output.eval_result_metrics.llm.llm_test_f1 <= 1

    assert len(eval_output.eval_result_details) == 1
    detail = eval_output.eval_result_details[0]
    assert detail.dataset_name == "118_us_state_CA"
    assert detail.llm_test_accuracy is not None
    assert 0 <= detail.llm_test_accuracy <= 1
    assert detail.llm_test_auc is not None
    assert 0 <= detail.llm_test_auc <= 1
    assert detail.llm_test_f1 is not None
    assert 0 <= detail.llm_test_f1 <= 1
    assert detail.sae_top_1_test_accuracy is not None
    assert detail.sae_top_1_test_auc is not None
    assert detail.sae_top_1_test_f1 is not None

    baseline_results_dir = artifacts_path / "baseline_results_gpt2" / "normal_setting"
    assert baseline_results_dir.exists()
    baseline_json_files = list(baseline_results_dir.glob("*.json"))
    assert len(baseline_json_files) >= 1


def test_run_eval_with_custom_ks(gpt2_l4_sae: SAE, tmp_path: Path):
    output_path = tmp_path / "test_output"
    artifacts_path = tmp_path / "test_artifacts"
    model_cache_path = tmp_path / "model_cache"
    custom_ks = [3, 7, 15]
    config = SparseProbingSaeProbesEvalConfig(
        model_name="gpt2",
        include_llm_baseline=True,
        model_cache_path=str(model_cache_path),
        results_path=str(artifacts_path),
        dataset_names=["118_us_state_CA"],
        ks=custom_ks,
    )
    results_dict = run_eval(
        config,
        [("gpt2_l4_sae", gpt2_l4_sae)],
        device="cpu",
        output_path=str(output_path),
    )

    assert isinstance(results_dict, dict)
    assert len(results_dict) == 1

    expected_output_file = output_path / "gpt2_l4_sae_custom_sae_eval_results.json"
    assert expected_output_file.exists()

    with open(expected_output_file) as f:
        output_data = json.load(f)

    eval_output = SparseProbingSaeProbesEvalOutput(**output_data)

    assert eval_output.eval_result_metrics.llm.llm_test_accuracy is not None
    assert 0 <= eval_output.eval_result_metrics.llm.llm_test_accuracy <= 1
    assert eval_output.eval_result_metrics.llm.llm_test_auc is not None
    assert 0 <= eval_output.eval_result_metrics.llm.llm_test_auc <= 1
    assert eval_output.eval_result_metrics.llm.llm_test_f1 is not None
    assert 0 <= eval_output.eval_result_metrics.llm.llm_test_f1 <= 1

    assert "sae_metrics_by_k" in output_data
    sae_metrics_by_k = eval_output.sae_metrics_by_k
    assert sae_metrics_by_k is not None
    assert set(sae_metrics_by_k.keys()) == {3, 7, 15}

    for k in custom_ks:
        metrics = sae_metrics_by_k[k]
        assert "test_accuracy" in metrics
        assert "test_auc" in metrics
        assert "test_f1" in metrics
        assert 0 <= metrics["test_accuracy"] <= 1
        assert 0 <= metrics["test_auc"] <= 1
        assert 0 <= metrics["test_f1"] <= 1

    detail = eval_output.eval_result_details[0]
    assert detail.dataset_name == "118_us_state_CA"
    assert detail.llm_test_accuracy is not None
    assert 0 <= detail.llm_test_accuracy <= 1
    assert detail.llm_test_auc is not None
    assert 0 <= detail.llm_test_auc <= 1
    assert detail.llm_test_f1 is not None
    assert 0 <= detail.llm_test_f1 <= 1
    assert detail.sae_metrics_by_k is not None
    assert set(detail.sae_metrics_by_k.keys()) == {3, 7, 15}

    for k in custom_ks:
        metrics = detail.sae_metrics_by_k[k]
        assert "test_accuracy" in metrics
        assert 0 <= metrics["test_accuracy"] <= 1
