from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from sae_bench.evals.base_eval_output import (
    DEFAULT_DISPLAY,
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)
from sae_bench.evals.meta_structure.eval_config import MetaStructureEvalConfig

EVAL_TYPE_ID_META_STRUCTURE = "meta_structure_decoder_variance"


@dataclass
class MetaStructureMetrics(BaseMetrics):
    decoder_fraction_variance_explained: float = Field(
        title="Decoder Fraction Variance Explained",
        description="Fraction of variance in the base SAE decoder reconstructed by the meta-structure model.",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    train_time_seconds: float = Field(
        title="Training Time (s)",
        description="Wall-clock seconds spent fitting the meta-structure model.",
    )
    final_reconstruction_mse: float = Field(
        title="Final Reconstruction MSE",
        description="Mean squared reconstruction error on the decoder matrix after training.",
    )


@dataclass
class MetaStructureMetricCategories(BaseMetricCategories):
    meta_structure: MetaStructureMetrics = Field(
        title="Meta-Structure",
        description="Metrics for the meta-structure decoder variance evaluation.",
        json_schema_extra=DEFAULT_DISPLAY,
    )


@dataclass(config=ConfigDict(title="Meta-Structure Decoder Variance"))
class MetaStructureEvalOutput(
    BaseEvalOutput[
        MetaStructureEvalConfig, MetaStructureMetricCategories, BaseResultDetail
    ]
):
    """
    Evaluation measuring how well a BatchTopK meta-structure compresses the decoder matrix of a base SAE.
    """

    eval_config: MetaStructureEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: MetaStructureMetricCategories
    eval_result_details: list[BaseResultDetail] | None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    eval_type_id: str = Field(
        default=EVAL_TYPE_ID_META_STRUCTURE,
        title="Eval Type ID",
        description="The type of the evaluation",
    )
