from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import List, Optional
from sae_bench.evals.base_eval_output import BaseEvalConfig

@dataclass
class DASConfig(BaseEvalConfig):
    # RAVEL dataset parameters
    entity_attribute_selection: dict[str, list[str]] = Field(
        default={
            "city": ['Country', 'Continent']
        },
        title="Selection of entity and attribute classes",
        description="Subset of the RAVEL datset to be evaluated. Each key is an entity class, and the value is a list of at least two attribute classes.",
    )
    chosen_attributes: List[str] = Field(
        default=["Country", "Continent"],
        title="Chosen Attributes",
        description="Attributes to use for dataset",
    )
    force_dataset_recompute: bool = Field(
        default=False,
        title="Force Dataset Recompute",
        description="Force recomputation of dataset",
    )
    n_samples_per_attribute_class: int = Field(
        default=100,
        title="Samples per Attribute Class",
        description="Number of samples per attribute class",
    )
    top_n_entities: int = Field(
        default=100,
        title="Top N Entities",
        description="Number of top entities to use for dataset",
    )
    top_n_templates: int = Field(
        default=100,
        title="Top N Templates",
        description="Number of top templates to use for dataset",
    )
    full_dataset_downsample: int = Field(
        default=10000,
        title="Full Dataset Downsample",
        description="Downsample the full dataset to this size.",
    )
    num_pairs_per_attribute: int = Field(
        default=100,
        title="Number of Pairs per Attribute",
        description="Number of pairs per attribute",
    )

    # DASTraining parameters
    layer_intervened: int = Field(
        default=12,
        title="Intervention Layer",
        description="Layer number where intervention is performed",
    )
    learning_rate: float = Field(
        default=1e-3,
        title="Learning Rate",
        description="Learning rate for the optimizer",
    )
    num_epochs: int = Field(
        default=10,
        title="Number of Epochs",
        description="Number of training epochs",
    )
    early_stop_patience: int = Field(
        default=10,
        title="Early Stopping Patience",
        description="Number of epochs to wait before early stopping",
    )
    batch_size: int = Field(
        default=32,
        title="Batch Size",
        description="Batch size for training",
    )
    d_subspace: int = Field(
        default=256,
        title="Subspace Dimension",
        description="Dimension of the learned subspace",
    )
    # token_length_allowed: int = Field(
    #     default=64,
    #     title="Maximum Token Length",
    #     description="Maximum number of tokens allowed in input",
    # )


    # Model, device and dtype settings
    model_name: str = Field(
        default="gemma-2-2b",
        title="Model Name",
        description="Name of the model to use",
    )
    model_dir: Optional[str] = Field(
        default=None,
        title="Cache Directory",
        description="Directory to save model cache",
    )
    device: str = Field(
        default="cuda:0",
        title="Device",
        description="Device to run the model on",
    )
    llm_dtype: str = Field(
        default="bfloat16",
        title="Data Type",
        description="Data type for model parameters",
    )

    # Validation settings
    validation_frequency: int = Field(
        default=1,
        title="Validation Frequency",
        description="Number of epochs between validation runs",
    )
    validation_batch_size: int = Field(
        default=64,
        title="Validation Batch Size",
        description="Batch size for validation",
    )

    # Logging and checkpointing
    log_frequency: int = Field(
        default=100,
        title="Log Frequency",
        description="Number of steps between logging",
    )
    checkpoint_frequency: int = Field(
        default=5,
        title="Checkpoint Frequency",
        description="Number of epochs between checkpoints",
    )
    checkpoint_dir: str = Field(
        default="checkpoints",
        title="Checkpoint Directory",
        description="Directory to save checkpoints",
    )
    artifact_dir: str = Field(
        default="/share/u/can/SAEBench/artifacts/ravel",
        title="Artifact Directory",
        description="Directory to save artifacts",
    )

    # Debug mode settings
    debug_mode: bool = Field(
        default=False,
        title="Debug Mode",
        description="Enable debug mode with smaller dataset and more frequent logging",
    )

    def __post_init__(self):
        if self.debug_mode:
            self.batch_size = 4
            self.validation_batch_size = 8
            self.log_frequency = 10
            self.num_epochs = 2 