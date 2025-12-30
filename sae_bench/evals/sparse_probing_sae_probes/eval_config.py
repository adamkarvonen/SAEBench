from pydantic import Field
from pydantic.dataclasses import dataclass
from sae_probes import DATASETS

from sae_bench.evals.base_eval_output import BaseEvalConfig


@dataclass
class SparseProbingSaeProbesEvalConfig(BaseEvalConfig):
    model_name: str = Field(
        default="",
        title="Model Name",
        description="TransformerLens model name used by sae-probes (e.g., 'gemma-2-2b').",
    )

    dataset_names: list[str] = Field(
        default_factory=lambda: [*DATASETS],
        title="Dataset Names",
        description="List of dataset names.",
    )

    reg_type: str = Field(
        default="l1",
        title="Regularization Type",
        description="Regularization used for sparse probing selection in sae-probes ('l1' or 'l2').",
    )

    setting: str = Field(
        default="normal",
        title="Data Balance Setting",
        description="sae-probes benchmark setting: 'normal', 'scarcity', or 'imbalance'.",
    )

    ks: list[int] = Field(
        default_factory=lambda: [1, 2, 5],
        title="K Values",
        description="List of K values (number of SAE features) to evaluate.",
    )

    binarize: bool = Field(
        default=False,
        title="Binarize Latents",
        description="Whether to binarize SAE latents during probing (sae-probes option).",
    )

    results_path: str = Field(
        default="artifacts/sparse_probing_sae_probes",
        title="sae-probes Results Root",
        description="Directory where sae-probes will save its per-dataset JSONs.",
    )

    model_cache_path: str | None = Field(
        default="artifacts/sparse_probing_sae_probes--model_acts_cache",
        title="Model Activations Cache",
        description="Optional path where sae-probes will cache generated model activations.",
    )

    include_llm_baseline: bool = Field(
        default=True,
        title="Include LLM Baseline",
        description="If True, also run sae-probes baselines on model residual stream and aggregate.",
    )

    baseline_method: str = Field(
        default="logreg",
        title="Baseline Method",
        description="sae-probes baseline method (e.g., 'logreg').",
    )
