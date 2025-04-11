from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from sae_bench.evals.base_eval_output import (
    DEFAULT_DISPLAY,
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig

EVAL_TYPE_ID_RAVEL = "ravel"


@dataclass
class RAVELMetricResults(BaseMetrics):
    disentanglement_score: float = Field(
        title="Disentanglement Score",
        description="Mean of cause and isolation scores across RAVEL datasets.",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    cause_score: float = Field(
        title="Cause Score",
        description="Cause score: Patching attribute-related SAE latents. High cause accuracy indicates that the SAE latents are related to the attribute.",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    isolation_score: float = Field(
        title="Isolation Score",
        description="Isolation score: Patching SAE latents related to another attribute. High isolation accuracy indicates that latents related to another attribute are not related to this attribute.",
        json_schema_extra=DEFAULT_DISPLAY,
    )


@dataclass
class RAVELMetricCategories(BaseMetricCategories):
    ravel: RAVELMetricResults = Field(
        title="RAVEL",
        description="RAVEL metrics",
        json_schema_extra=DEFAULT_DISPLAY,
    )


@dataclass(config=ConfigDict(title="RAVEL"))
class RAVELEvalOutput(
    BaseEvalOutput[RAVELEvalConfig, RAVELMetricCategories, BaseResultDetail]
):
    # This will end up being the description of the eval in the UI.
    """
    An evaluation using SAEs for targeted modification of language model output. We leverage the RAVEL dataset of entity-attribute pairs. After filtering for known pairs, we identify attribute-related SAE latents and deterimine the effect on model predictions with activation patching experiments.
    """

    eval_config: RAVELEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: RAVELMetricCategories
    eval_type_id: str = Field(default=EVAL_TYPE_ID_RAVEL)
