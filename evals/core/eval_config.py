from pydantic.dataclasses import dataclass
from pydantic import Field
from evals.base_eval_output import DEFAULT_DISPLAY, BaseEvalConfig


# Define the eval config, which inherits from BaseEvalConfig, and include fields with titles and descriptions.
@dataclass
class CoreEvalConfig(BaseEvalConfig):
    batch_size_prompts: int = Field(
        default=16,
        title="Batch Size Prompts",
        description="Batch size for evaluation prompts",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    n_eval_reconstruction_batches: int = Field(
        default=10,
        title="Reconstruction Batches",
        description="Number of evaluation batches for reconstruction metrics",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    n_eval_sparsity_variance_batches: int = Field(
        default=1,
        title="Sparsity Variance Batches",
        description="Number of evaluation batches for sparsity and variance metrics",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    dataset: str = Field(
        default="Skylion007/openwebtext",
        title="Dataset",
        description="Dataset to evaluate on",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    context_size: int = Field(
        default=128,
        title="Context Length",
        description="Context length to evaluate on",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    compute_kl: bool = Field(
        default=False,
        title="Compute KL",
        description="Compute KL divergence",
    )
    compute_ce_loss: bool = Field(
        default=False,
        title="Compute CE Loss",
        description="Compute cross-entropy loss",
    )
    compute_l2_norms: bool = Field(
        default=False,
        title="Compute L2 Norms",
        description="Compute L2 norms",
    )
    compute_sparsity_metrics: bool = Field(
        default=False,
        title="Compute Sparsity Metrics",
        description="Compute sparsity metrics",
    )
    compute_variance_metrics: bool = Field(
        default=False,
        title="Compute Variance Metrics",
        description="Compute variance metrics",
    )
    compute_featurewise_density_statistics: bool = Field(
        default=False,
        title="Compute Featurewise Density Statistics",
        description="Compute featurewise density statistics",
    )
    compute_featurewise_weight_based_metrics: bool = Field(
        default=False,
        title="Compute Featurewise Weight-Based Metrics",
        description="Compute featurewise weight-based metrics",
    )
