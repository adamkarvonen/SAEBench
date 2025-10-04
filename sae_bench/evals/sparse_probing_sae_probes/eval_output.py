from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from sae_bench.evals.base_eval_output import (
    DEFAULT_DISPLAY,
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)
from sae_bench.evals.sparse_probing_sae_probes.eval_config import (
    SparseProbingSaeProbesEvalConfig,
)

EVAL_TYPE_ID_SPARSE_PROBING_SAE_PROBES = "sparse_probing_sae_probes"


@dataclass
class SaeProbesLlmMetrics(BaseMetrics):
    llm_test_accuracy: float | None = Field(
        default=None,
        title="LLM Test Accuracy",
        description="Linear probe accuracy when training on the full LLM residual stream",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    llm_test_auc: float | None = Field(
        default=None,
        title="LLM Test AUC",
        description="Linear probe AUC when training on the full LLM residual stream",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    llm_test_f1: float | None = Field(
        default=None,
        title="LLM Test F1",
        description="Linear probe F1 score when training on the full LLM residual stream",
        json_schema_extra=DEFAULT_DISPLAY,
    )


@dataclass
class SaeProbesSaeMetrics(BaseMetrics):
    sae_test_accuracy: float | None = Field(
        default=None,
        title="SAE Test Accuracy",
        description="Linear probe accuracy when trained on all SAE latents",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    sae_top_1_test_accuracy: float | None = Field(
        default=None, json_schema_extra=DEFAULT_DISPLAY
    )
    sae_top_1_test_auc: float | None = Field(
        default=None, json_schema_extra=DEFAULT_DISPLAY
    )
    sae_top_1_test_f1: float | None = Field(
        default=None, json_schema_extra=DEFAULT_DISPLAY
    )
    sae_top_2_test_accuracy: float | None = Field(
        default=None, json_schema_extra=DEFAULT_DISPLAY
    )
    sae_top_2_test_auc: float | None = Field(default=None)
    sae_top_2_test_f1: float | None = Field(default=None)
    sae_top_5_test_accuracy: float | None = Field(
        default=None, json_schema_extra=DEFAULT_DISPLAY
    )
    sae_top_5_test_auc: float | None = Field(default=None)
    sae_top_5_test_f1: float | None = Field(default=None)
    sae_top_10_test_accuracy: float | None = Field(default=None)
    sae_top_10_test_auc: float | None = Field(default=None)
    sae_top_10_test_f1: float | None = Field(default=None)
    sae_top_20_test_accuracy: float | None = Field(default=None)
    sae_top_20_test_auc: float | None = Field(default=None)
    sae_top_20_test_f1: float | None = Field(default=None)
    sae_top_50_test_accuracy: float | None = Field(default=None)
    sae_top_50_test_auc: float | None = Field(default=None)
    sae_top_50_test_f1: float | None = Field(default=None)
    sae_top_100_test_accuracy: float | None = Field(default=None)
    sae_top_100_test_auc: float | None = Field(default=None)
    sae_top_100_test_f1: float | None = Field(default=None)


@dataclass
class SaeProbesMetricCategories(BaseMetricCategories):
    llm: SaeProbesLlmMetrics = Field(
        title="LLM",
        description="LLM metrics",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    sae: SaeProbesSaeMetrics = Field(
        title="SAE",
        description="SAE metrics",
        json_schema_extra=DEFAULT_DISPLAY,
    )


@dataclass
class SaeProbesResultDetail(BaseResultDetail):
    dataset_name: str = Field(title="Dataset Name", description="Dataset name")
    llm_test_accuracy: float | None = Field(default=None)
    llm_test_auc: float | None = Field(default=None)
    llm_test_f1: float | None = Field(default=None)
    sae_test_accuracy: float | None = Field(default=None)
    sae_top_1_test_accuracy: float | None = Field(default=None)
    sae_top_1_test_auc: float | None = Field(default=None)
    sae_top_1_test_f1: float | None = Field(default=None)
    sae_top_2_test_accuracy: float | None = Field(default=None)
    sae_top_2_test_auc: float | None = Field(default=None)
    sae_top_2_test_f1: float | None = Field(default=None)
    sae_top_5_test_accuracy: float | None = Field(default=None)
    sae_top_5_test_auc: float | None = Field(default=None)
    sae_top_5_test_f1: float | None = Field(default=None)
    sae_top_10_test_accuracy: float | None = Field(default=None)
    sae_top_10_test_auc: float | None = Field(default=None)
    sae_top_10_test_f1: float | None = Field(default=None)
    sae_top_20_test_accuracy: float | None = Field(default=None)
    sae_top_20_test_auc: float | None = Field(default=None)
    sae_top_20_test_f1: float | None = Field(default=None)
    sae_top_50_test_accuracy: float | None = Field(default=None)
    sae_top_50_test_auc: float | None = Field(default=None)
    sae_top_50_test_f1: float | None = Field(default=None)
    sae_top_100_test_accuracy: float | None = Field(default=None)
    sae_top_100_test_auc: float | None = Field(default=None)
    sae_top_100_test_f1: float | None = Field(default=None)
    sae_metrics_by_k: dict[int, dict[str, float]] | None = Field(
        default=None,
        title="SAE Metrics by K",
        description="Per-dataset metrics for arbitrary k values. Maps k -> {test_accuracy, test_auc, test_f1}",
    )


@dataclass(config=ConfigDict(title="Sparse Probing (sae-probes)"))
class SparseProbingSaeProbesEvalOutput(
    BaseEvalOutput[
        SparseProbingSaeProbesEvalConfig,
        SaeProbesMetricCategories,
        SaeProbesResultDetail,
    ]
):
    """
    Wraps sae-probes sparse probing benchmark and collates per-dataset JSONs into a single SAEBench output.
    """

    eval_config: SparseProbingSaeProbesEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: SaeProbesMetricCategories
    eval_result_details: list[SaeProbesResultDetail] = Field(
        default_factory=list,
        title="Per-Dataset Sparse Probing Results",
        description="Per-dataset probe accuracies aggregated from sae-probes output.",
    )
    sae_metrics_by_k: dict[int, dict[str, float]] | None = Field(
        default=None,
        title="SAE Metrics by K",
        description="SAE metrics for arbitrary k values. Maps k -> {test_accuracy, test_auc, test_f1}",
    )
    eval_type_id: str = Field(default=EVAL_TYPE_ID_SPARSE_PROBING_SAE_PROBES)
