from pydantic import Field
from pydantic.dataclasses import dataclass

from sae_bench.evals.base_eval_output import BaseEvalConfig


# Define the eval config, which inherits from BaseEvalConfig, and include fields with titles and descriptions.
@dataclass
class AbsorptionEvalConfig(BaseEvalConfig):
    model_name: str = Field(
        default="",
        title="Model Name",
        description="Model name. Must be set with a command line argument. For this eval, we currently recommend to only use models >= 2B parameters.",
    )

    random_seed: int = Field(
        default=42,
        title="Random Seed",
        description="Random seed",
    )
    f1_jump_threshold: float = Field(
        default=0.03,
        title="F1 Jump Threshold",
        description="F1 jump threshold",
    )
    max_k_value: int = Field(
        default=10,
        title="Max K Value",
        description="Max k value",
    )

    # double-check token_pos matches prompting_template for other tokenizers
    prompt_template: str = Field(
        default="{word} has the first letter:",
        title="Prompt Template",
        description="Prompt template",
    )
    prompt_token_pos: int = Field(
        default=-6,
        title="Prompt Token Position",
        description="Prompt token position",
    )

    llm_batch_size: int = Field(
        default=10,
        title="LLM Batch Size",
        description="LLM batch size. This is set by default in the main script, or it can be set with a command line argument.",
    )
    llm_dtype: str = Field(
        default="float32",
        title="LLM Data Type",
        description="LLM data type. This is set by default in the main script, or it can be set with a command line argument.",
    )
    k_sparse_probe_l1_decay: float = Field(
        default=0.01,
        title="K-Sparse Probe L1 Decay",
        description="L1 decay for k-sparse probes.",
    )
    k_sparse_probe_batch_size: int = Field(
        default=4096,
        title="K-Sparse Probe Batch Size",
        description="Batch size for k-sparse probes.",
    )
    k_sparse_probe_num_epochs: int = Field(
        default=50,
        title="K-Sparse Probe Number of Epochs",
        description="Number of epochs for k-sparse probes.",
    )
    min_GT_probe_f1: float = Field(
        default=0.6,
        title="Minimum ground truth probe F1 score",
        description="The minimum ground truth probe F1 score for the first-letter feature to be considered present",
    )
    min_feats_for_eval: int = Field(
        default=20,
        ge=1,
        title="Minumum features for evaluation",
        description="The minimum number of first-letter features (as detected by the GT probes) needed to evaluate absorption",
    )
    eval_k_sparse_probe_batch_size: int = Field(
        default=24,
        title="Eval K-Sparse Probe Batch Size",
        description="Batch size for evaluating k-sparse probes.",
    )
    precalc_k_sparse_probe_sae_acts: bool = Field(
        default=False,
        title="Precalc SAE Acts when training k-sparse probes",
        description="Precalculate SAE acts when training k-sparse probes. This speeds up the training process, but takes up more memory. For smaller SAEs, this is highly recommended.",
    )
