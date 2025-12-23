import json
import os

import pytest

import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.evals.meta_structure.eval_config import MetaStructureEvalConfig
from sae_bench.evals.meta_structure.main import run_eval
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex

TEST_RELEASE = "sae_bench_pythia70m_sweep_standard_ctx128_0712"
TEST_SAE_NAME = "blocks.4.hook_resid_post__trainer_10"
EXPECTED_RESULTS_PATH = (
    "tests/acceptance/test_data/meta_structure/pythia70m_tr10_meta_structure.json"
)


def _load_metrics(data: dict) -> dict:
    metrics_block = data.get("eval_result_metrics", {})
    metrics = metrics_block.get("meta_structure") or metrics_block.get("meta_sae")
    assert metrics, "Missing meta-structure metrics block"
    return metrics


def test_meta_structure_eval_matches_fixture(tmp_path):
    """Runs the meta-structure eval and compares key metrics to a stored fixture."""
    with open(EXPECTED_RESULTS_PATH) as f:
        expected = json.load(f)
    expected_metrics = _load_metrics(expected)

    device = general_utils.setup_environment()

    config = MetaStructureEvalConfig()
    selected_saes = get_saes_from_regex(TEST_RELEASE, TEST_SAE_NAME)
    assert selected_saes, "No SAEs selected with provided regex patterns"

    output_dir = tmp_path / "meta_structure"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_eval(
        config=config,
        selected_saes=selected_saes,
        device=device,
        output_path=str(output_dir),
        force_rerun=True,
    )

    result_path = general_utils.get_results_filepath(
        str(output_dir), TEST_RELEASE, TEST_SAE_NAME
    )
    assert os.path.exists(result_path), "Meta-structure eval result file missing"

    with open(result_path) as f:
        actual = json.load(f)
    actual_metrics = _load_metrics(actual)

    assert pytest.approx(
        expected_metrics["decoder_fraction_variance_explained"], rel=1e-2,
    ) == actual_metrics["decoder_fraction_variance_explained"]
    assert pytest.approx(
        expected_metrics["final_reconstruction_mse"], rel=1e-2,
    ) == actual_metrics["final_reconstruction_mse"]
