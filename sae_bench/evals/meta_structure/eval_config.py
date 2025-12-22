from pydantic import Field
from pydantic.dataclasses import dataclass

from sae_bench.evals.base_eval_output import BaseEvalConfig


@dataclass
class MetaStructureEvalConfig(BaseEvalConfig):
    random_seed: int = Field(
        default=42,
        title="Random Seed",
        description="Random seed used when training the meta-structure model.",
    )
    width_ratio: float = Field(
        default=0.25,
        title="Meta-SAE Width Ratio",
        description="Meta-SAE width as a fraction of the base SAE width (d_sae * width_ratio).",
    )
    k: int = Field(
        default=4,
        title="BatchTopK k",
        description="Average number of active latents to maintain during meta-structure training.",
    )
    train_steps: int = Field(
        default=1500,
        title="Training Steps",
        description="Number of training steps to run when fitting the meta-structure model.",
    )
    train_batch_size: int = Field(
        default=1024,
        title="Training Batch Size",
        description="Number of decoder rows to use per meta-structure training batch.",
    )
    learning_rate: float = Field(
        default=5e-4,
        title="Learning Rate",
        description="Learning rate for the meta-structure Adam optimizer.",
    )
    weight_decay: float = Field(
        default=0.0,
        title="Weight Decay",
        description="Weight decay to apply while training the meta-structure model.",
    )
    autocast: bool = Field(
        default=True,
        title="Use Autocast",
        description="Whether to enable torch.autocast during meta-structure training.",
    )
    eval_batch_size: int = Field(
        default=4096,
        title="Eval Batch Size",
        description="Batch size used to compute decoder variance explained.",
    )
    dead_feature_window: int = Field(
        default=2000,
        title="Dead Feature Window",
        description="Steps before a latent is considered dead for auxiliary losses.",
    )
    feature_sampling_window: int = Field(
        default=200,
        title="Feature Sampling Window",
        description="Steps between sparsity statistics resets during training.",
    )
    dtype: str = Field(
        default="float32",
        title="Meta-Structure DType",
        description="Datatype to use for meta-structure training.",
    )
