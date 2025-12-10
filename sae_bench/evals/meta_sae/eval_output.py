from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from sae_bench.evals.base_eval_output import (
    DEFAULT_DISPLAY,
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)
from sae_bench.evals.meta_sae.eval_config import MetaSAEEvalConfig

EVAL_TYPE_ID_META_SAE = "meta_sae_decoder_variance"


@dataclass
class MetaSAEMetrics(BaseMetrics):
    decoder_fraction_variance_explained: float = Field(
        title="Decoder Fraction Variance Explained",
        description="Fraction of variance in the base SAE decoder reconstructed by the meta-SAE.",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    train_time_seconds: float = Field(
        title="Training Time (s)",
        description="Wall-clock seconds spent fitting the meta-SAE.",
    )
    final_reconstruction_mse: float = Field(
        title="Final Reconstruction MSE",
        description="Mean squared reconstruction error on the decoder matrix after training.",
    )


@dataclass
class MetaSAEMetricCategories(BaseMetricCategories):
    meta_sae: MetaSAEMetrics = Field(
        title="Meta-SAE",
        description="Metrics for the meta-SAE decoder variance evaluation.",
        json_schema_extra=DEFAULT_DISPLAY,
    )


@dataclass(config=ConfigDict(title="Meta-SAE Decoder Variance"))
class MetaSAEEvalOutput(
    BaseEvalOutput[MetaSAEEvalConfig, MetaSAEMetricCategories, BaseResultDetail]
):
    """
    Evaluation measuring how well a BatchTopK meta-SAE compresses the decoder matrix of a base SAE.
    """

    eval_config: MetaSAEEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: MetaSAEMetricCategories
    eval_result_details: list[BaseResultDetail] | None = None
    eval_type_id: str = Field(
        default=EVAL_TYPE_ID_META_SAE,
        title="Eval Type ID",
        description="The type of the evaluation",
    )
