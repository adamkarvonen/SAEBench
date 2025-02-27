from pydantic.dataclasses import dataclass
from pydantic import Field
from sae_bench.evals.base_eval_output import BaseEvalConfig
from typing import List

DEBUG_MODE = True


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
    n_samples_per_attribute_class: int = Field(
        default=1000,
        title="Number of Samples per Attribute Class",
        description="Number of samples per attribute class. If None, all samples are used.",
    )
    top_n_entities: int = Field(
        default=500,
        title="Number of distinct entities in the dataset",
        description="Number of entities in the dataset, filtered by prediction accuracy over attributes / templates.",
    )
    top_n_templates: int = Field(
        default=500,
        title="Number of distinct templates in the dataset",
        description="Number of templates in the dataset, filtered by prediction accuracy over entities.",
    )
    full_dataset_downsample: int = Field(
        default=101,
        title="Full Dataset Downsample",
        description="Downsample the full dataset to this size.",
    )
    num_pairs_per_attribute: int = Field(
        default=500,
        title="Number of Pairs per Attribute",
        description="Number of pairs per attribute",
    )
    train_test_split: float = Field(
        default=0.5,
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
    model_dir: str = Field(
        default="None",
        title="Model Directory",
        description="Model directory for cached hf model",
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
    sae_batch_size: int = Field(
        default=125,
        title="SAE Batch Size",
        description="SAE batch size, inference only",
    )

    max_samples_per_attribute: int = Field(
        default=1024,
        title="Max Samples per Attribute",
        description="Indirect definition of probe training datset size, which contains half target attribute and half balanced mix of non-target attributes.",
    )

    learning_rate: float = Field(
        default=1e-3,
        title="Learning Rate",
        description="Learning rate for the MDBM",
    )
    num_epochs: int = Field(
        default=5,
        title="Number of Epochs",
        description="Number of training epochs",
    )

    # Intervention
    n_interventions: int = Field(
        default=128,
        title="Number of Interventions",
        description="Number of interventions per attribute feature threshold, ie. number of experiments to compute a single cause / isolation score.",
    )
    n_generated_tokens: int = Field(
        default=6,
        title="Number of Generated Tokens",
        description="Number of tokens to generate for each intervention. 8 was used in the RAVEL paper",
    )
    inv_batch_size: int = Field(
        default=16,
        title="Intervention Batch Size",
        description="Intervention batch size, inference only",
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
        n_samples_per_attribute_class = 500
        top_n_entities = 500
        top_n_templates = 500

        n_interventions = 500
        llm_batch_size = 10
