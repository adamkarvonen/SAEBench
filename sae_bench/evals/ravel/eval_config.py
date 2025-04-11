from pydantic import Field
from pydantic.dataclasses import dataclass

from sae_bench.evals.base_eval_output import BaseEvalConfig

DEBUG_MODE = False


@dataclass
class RAVELEvalConfig(BaseEvalConfig):
    # Dataset
    entity_attribute_selection: dict[str, list[str]] = Field(
        default={
            "city": ["Country", "Continent", "Language"],
            "nobel_prize_winner": ["Country of Birth", "Field", "Gender"],
        },
        title="Selection of entity and attribute classes",
        description="Subset of the RAVEL datset to be evaluated. Each key is an entity class, and the value is a list of at least two attribute classes.",
    )
    top_n_entities: int = Field(
        default=500,
        title="Number of distinct entities in the dataset",
        description="Number of entities in the dataset, filtered by prediction accuracy over attributes / templates.",
    )
    top_n_templates: int = Field(
        default=90,
        title="Number of distinct templates in the dataset",
        description="Number of templates in the dataset, filtered by prediction accuracy over entities.",
    )
    full_dataset_downsample: int | None = Field(
        default=None,
        title="Full Dataset Downsample",
        description="Downsample the full dataset to this size.",
    )
    num_pairs_per_attribute: int = Field(
        default=5000,
        title="Number of Pairs per Attribute",
        description="Number of pairs per attribute",
    )
    train_test_split: float = Field(
        default=0.7,
        title="Train Test Split",
        description="Fraction of dataset to use for training.",
    )
    force_dataset_recompute: bool = Field(
        default=False,
        title="Force Dataset Recompute",
        description="Force recomputation of the dataset, ie. generating model predictions for attribute values, evaluating, and downsampling.",
    )

    # Language model and SAE
    model_name: str = Field(
        default="gemma-2-2b",
        title="Model Name",
        description="Model name",
    )
    llm_dtype: str = Field(
        default="bfloat16",
        title="LLM Data Type",
        description="LLM data type",
    )
    llm_batch_size: int = Field(
        default=2048,
        title="LLM Batch Size",
        description="LLM batch size, inference only",
    )

    learning_rate: float = Field(
        default=1e-3,
        title="Learning Rate",
        description="Learning rate for the MDBM",
    )
    num_epochs: int = Field(
        default=2,
        title="Number of Epochs",
        description="Number of training epochs",
    )
    train_mdas: bool = Field(
        default=False,
        title="Train MDAS",
        description="If True, we completely ignore the SAE and train an MDAS instead.",
    )
    # Intervention
    n_generated_tokens: int = Field(
        default=6,
        title="Number of Generated Tokens",
        description="Number of tokens to generate for each intervention. 8 was used in the RAVEL paper",
    )

    # Misc
    random_seed: int = Field(
        default=42,
        title="Random Seed",
        description="Random seed",
    )
    artifact_dir: str = Field(
        default="artifacts/ravel",
        title="Artifact Directory",
        description="Directory to save artifacts",
    )

    if DEBUG_MODE:
        num_pairs_per_attribute = 500
        top_n_entities = 500
        top_n_templates = 90

        llm_batch_size = 10
