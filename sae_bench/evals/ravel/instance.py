"""
RAVEL Entity Prompt Data Module

This module provides functionality for handling and processing entity prompt data
for the RAVEL evaluation benchmark.
"""

import json
import os
import random
from copy import deepcopy
from dataclasses import dataclass

from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
from sae_bench.evals.ravel.generation import generate_batched
from sae_bench.evals.ravel.validation import evaluate_completion


@dataclass
class AttributePrompt:
    """Represents an attribute_type with its associated prompt templates."""

    attribute_type: str
    templates: list[str]


@dataclass
class Prompt:
    """Represents a single prompt with its associated data."""

    text: str  # Template with inserted entity label.
    template: str  # The template string with %s placeholder for entity label.
    attribute_type: str  # The abstract attribute type, eg. "Country".
    attribute_label: str  # The concrete attribute label, eg. "Finland".
    entity_label: str  # The entity label, eg. "Helsinki".
    context_split: str  # The context split, "train"/"val".
    entity_split: str  # The entity split, "train"/"val".
    input_ids: list[int] | None = None  # Tokenized text.
    final_entity_token_pos: int | None = (
        None  # Position of the final entity token in the input_ids, as counted from the end (negative index)
    )
    attention_mask: list[int] | None = None
    attribute_generation: str | None = (
        None  # Given the text, the generated next tokens which may contain the attribute label, decoded to string.
    )
    first_generated_token_id: int | None = (
        None  # The first generated token id from attribute_generation.
    )
    is_correct: bool | None = (
        None  # Whether the attribute generation contains the attribute label.
    )


def get_instance_name(
    entity_type: str,
    model_name: str,
    downsample: int | None = None,
    top_n_entities: int | None = None,
) -> str:
    model_name_str = model_name.replace("/", "--")
    instance_name = f"{entity_type}_{model_name_str}_downsampled-{downsample}"
    if top_n_entities:
        instance_name += f"_top-{top_n_entities}-entities_filtered_dataset.json"
    return instance_name


class RAVELInstance:
    """
    The dataset for the RAVEL Benchmark is created in two steps:
    1. Create a RAVELInstance object from the raw RAVEL dataset files. This will contain all (num_templates x num_entities) prompts.
        We'll have to check whether the model correctly answers these prompts, and only want to keep the entities with the most correctly answered prompts.
        Therefore, we'll have to generate completions and evaluate correctness for all prompts, once for each model.
        Optionally, we can downsample the dataset before generating completions. This risks loosing entities with low coverage, but is faster.
        This can take a while, so we'll save the RAVELInstance object json after each model and host on huggingface.
    2. Create a RAVELFilteredDataset object from the RAVELInstance object. This will contain a filtered subset of the prompts, padded to the max prompt length.
        Filtering:
        - Only keep entities with the most correctly answered prompts.
        - Only keep templates with the most correctly answered prompts.
        - Pad prompts to the max prompt length.
        - Save as json.
    """

    def __init__(self):
        self.prompts = []  # list of Prompt objects
        self.entityLBL_attrTYP_attrLBL = {}  # entity label -> attribute type -> attribute label
        self.template_splits = {}  # template -> 'train'/'val'
        self.entity_splits = {}  # entity -> 'train'/'val'
        self.attribute_type_to_templates = {}  # attribute type -> (templates x entities) Prompt objects
        self.config = {}

        # If this exists, we only tokenize prompts for these attribute types.
        self.attribute_types: list[str] | None = None

    @classmethod
    def create_from_files(
        cls,
        config: RAVELEvalConfig,
        entity_type: str,
        data_dir: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        model: PreTrainedModel,
        model_name: str,
        attribute_types: list[str] | None = None,
        downsample: int | None = None,
    ) -> "RAVELInstance":
        instance = cls()
        instance.attribute_types = attribute_types

        instance.initialize_config(entity_type, model_name, downsample)
        save_path = os.path.join(
            config.artifact_dir,
            f"{instance.config['instance_name']}_unfiltered_full_instance.json",
        )

        if os.path.exists(save_path):
            print(f"Loading instance from {save_path}.")
            return instance.load(save_path)

        print("Loading files.")
        instance.load_files(entity_type, data_dir)

        print("Tokenizing prompts.")
        instance.build_and_tokenize_prompts(tokenizer)

        # Optional: Downsample to fewer prompts.
        if downsample:
            print(f"Downsample to {downsample} prompts.")
            instance.downsample_(downsample)

        print("Generate completions.")
        instance.generate_completions(
            model,
            tokenizer,
            max_new_tokens=config.n_generated_tokens,
            llm_batch_size=config.llm_batch_size,
        )

        print("Evaluate correctness.")
        instance.evaluate_correctness()

        print("Filter correct completions.")
        instance.filter_correct_()

        print("Save filtered dataset.")
        instance.save_as_instance(save_path)
        return instance

    def initialize_config(
        self, entity_type: str, model_name: str, downsample: int | None = None
    ):
        instance_name = get_instance_name(entity_type, model_name, downsample)
        self.config = {
            "entity_type": entity_type,
            "model_name": model_name,
            "downsample": downsample,
            "instance_name": instance_name,
        }
        return self.config

    def build_and_tokenize_prompts(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> list[Prompt]:
        """
        Load the full RAVEL dataset from files.
        """
        # Tokenize prompts from (template x entity) combinations.
        for entity_label in tqdm(
            self.entityLBL_attrTYP_attrLBL,
            total=len(self.entityLBL_attrTYP_attrLBL),
            desc="Tokenizing prompts",
        ):
            for attribute_type, templates in self.attribute_type_to_templates.items():
                if self.attribute_types and attribute_type not in self.attribute_types:
                    continue
                for template in templates:
                    text = template % entity_label
                    encoded = tokenizer.encode(text)
                    if isinstance(
                        encoded[0], list
                    ):  # TODO: actually check this and remove this check.
                        raise ValueError(
                            "Batch dimension not supported. Please adapt tokenization"
                        )

                    remainder = template.split("%s")[1]
                    encoded_remainder = tokenizer.encode(remainder)
                    final_pos = -len(encoded_remainder)

                    self.prompts.append(
                        Prompt(
                            text=text,
                            template=template,
                            attribute_type=attribute_type,
                            attribute_label=self.entityLBL_attrTYP_attrLBL[
                                entity_label
                            ][attribute_type],
                            entity_label=entity_label,
                            context_split=self.template_splits[template],
                            entity_split=self.entity_splits[entity_label],
                            input_ids=encoded,
                            final_entity_token_pos=final_pos,
                        )
                    )
        return self.prompts

    def load_files(
        self,
        entity_type: str,
        data_dir: str,
    ) -> None:
        # Define file paths and names
        base_dir = os.path.join(data_dir, "base")
        os.makedirs(base_dir, exist_ok=True)

        required_files = [
            f"ravel_{entity_type}_attribute_to_prompts.json",
            f"ravel_{entity_type}_prompt_to_split.json",
            f"ravel_{entity_type}_entity_attributes.json",
            f"ravel_{entity_type}_entity_to_split.json",
        ]

        # Check if any file is missing
        if any(not os.path.exists(os.path.join(base_dir, f)) for f in required_files):
            print("Downloading RAVEL dataset from HuggingFace...")
            snapshot_download(
                repo_id="adamkarvonen/ravel_prompts",
                repo_type="dataset",
                local_dir=base_dir,
                local_dir_use_symlinks=False,
                allow_patterns="*.json",
            )

        # Load data files
        with open(
            os.path.join(
                data_dir, "base", f"ravel_{entity_type}_entity_attributes.json"
            )
        ) as f:
            self.entityLBL_attrTYP_attrLBL = json.load(f)
        with open(
            os.path.join(data_dir, "base", f"ravel_{entity_type}_prompt_to_split.json")
        ) as f:
            self.template_splits = json.load(f)
        with open(
            os.path.join(data_dir, "base", f"ravel_{entity_type}_entity_to_split.json")
        ) as f:
            self.entity_splits = json.load(f)
        with open(
            os.path.join(
                data_dir, "base", f"ravel_{entity_type}_attribute_to_prompts.json"
            )
        ) as f:
            self.attribute_type_to_templates = json.load(f)

    def downsample_(self, n: int) -> None:
        sampled_keys = random.sample(list(range(len(self.prompts))), n)
        sampled_prompts = [self.prompts[k] for k in sampled_keys]
        self._filter_data_(sampled_prompts)

    def __len__(self) -> int:
        return len(self.prompts)

    def get_prompts_by_split(self, context_split: str) -> list[Prompt]:
        """Return all prompts with the given context split."""
        return [
            prompt for prompt in self.prompts if prompt.context_split == context_split
        ]

    def get_entities(self, split: str | None = None) -> list[str]:
        """Return all entities with the given split."""
        if split is None:
            return list(self.entity_splits.keys())
        return [
            entity_label
            for entity_label, entity_split in self.entity_splits.items()
            if entity_split == split
        ]

    def get_attributes(self) -> list[str]:
        """Return all attribute types."""
        return list(self.attribute_type_to_templates.keys())

    def get_prompt_by_text(self, text: str) -> Prompt | None:
        """Return the unique prompt with the given text, if available."""
        return next((p for p in self.prompts if p.text == text), None)

    def get_prompts_by_template(self, template: str) -> list[Prompt]:
        """Return all prompts with the given template."""
        return [p for p in self.prompts if p.template == template]

    def get_prompts_by_attribute(
        self, attribute: str, n_samples: int | None = None
    ) -> list[Prompt]:
        """Return all prompts with the given attribute type."""
        prompts = [p for p in self.prompts if p.attribute_type == attribute]
        if n_samples:
            if n_samples > len(prompts):
                print(
                    f"Warning: Requested {n_samples} samples but only {len(prompts)} available"
                )
            return prompts[:n_samples]
        return prompts

    def get_prompts_by_entity(self, entity_label: str) -> list[Prompt]:
        """Return all prompts with the given entity label."""
        return [p for p in self.prompts if p.entity_label == entity_label]

    def generate_completions(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        max_new_tokens: int,
        llm_batch_size: int = 32,
        **kwargs,
    ) -> None:
        """Generate completions for all prompts."""

        token_ids = [p.input_ids for p in self.prompts]
        attention_masks = None  # Attention masks are computed per batch dependent on padding within generate_batched.
        completions, first_token_ids = (
            generate_batched(  # TODO: Add tokenization to this function.
                model,
                tokenizer,
                input_ids_BL=token_ids,  # type: ignore
                attention_mask_BL=attention_masks,
                max_new_tokens=max_new_tokens,
                llm_batch_size=llm_batch_size,
                return_first_generated_token=True,
                **kwargs,
            )
        )

        for prompt, completion, first_token_id in zip(
            self.prompts, completions, first_token_ids
        ):
            prompt.attribute_generation = completion
            prompt.first_generated_token_id = first_token_id  # type: ignore

    def _filter_data_(self, filtered_prompts: list[Prompt]) -> None:
        """Filter the data based on the filtered prompts."""
        filtered_entity_labels = set(p.entity_label for p in filtered_prompts)
        filtered_attribute_types = set(p.attribute_type for p in filtered_prompts)
        filtered_templates = set(p.template for p in filtered_prompts)

        filtered_entity_label_to_attribute_type = {
            e: attrTYP_attrLBL
            for e, attrTYP_attrLBL in self.entityLBL_attrTYP_attrLBL.items()
            if e
            in filtered_entity_labels  # NOTE attributes listed do not necessarily have a prompt the model can answer correctly.
        }
        filtered_template_splits = {
            t: split
            for t, split in self.template_splits.items()
            if t in filtered_templates
        }
        filtered_entity_splits = {
            e: split
            for e, split in self.entity_splits.items()
            if e in filtered_entity_labels
        }
        filtered_attribute_prompts = {
            attribute_type: [t for t in templates if t in filtered_templates]
            for attribute_type, templates in self.attribute_type_to_templates.items()
            if attribute_type in filtered_attribute_types
        }

        # Update the instance attributes.
        self.prompts = filtered_prompts
        self.entityLBL_attrTYP_attrLBL = filtered_entity_label_to_attribute_type
        self.template_splits = filtered_template_splits
        self.entity_splits = filtered_entity_splits
        self.attribute_type_to_templates = filtered_attribute_prompts

    def filter_correct_(self):
        correct_prompts = [p for p in self.prompts if p.is_correct]
        self._filter_data_(correct_prompts)

    def evaluate_correctness(self):
        """Evaluate whether the generated completion contains the expected attribute label."""
        for prompt in self.prompts:
            if prompt.attribute_generation is not None:
                prompt.is_correct = evaluate_completion(
                    text=prompt.text,
                    expected_label=prompt.attribute_label,
                    completion=prompt.attribute_generation,
                )

    def get_accuracy_stats(self):
        """Get accuracy stats for all prompts."""
        stats = {}
        for prompt in self.prompts:
            if prompt.is_correct is not None:
                key = (prompt.entity_label, prompt.template)
                if key not in stats:
                    stats[key] = {"correct": 0, "total": 0}
                stats[key]["total"] += 1
                if prompt.is_correct:
                    stats[key]["correct"] += 1
        return stats

    def calculate_average_accuracy(self):
        """Calculate the average accuracy of the model."""
        correct = sum(1 for p in self.prompts if p.is_correct)
        total = len(self.prompts)
        return correct / total if total > 0 else 0

    def filter_prompts_by_template_format(self):
        return {
            text: p
            for text, p in self.prompts.items()  # type: ignore
            if p.template.count("%s") == 1
        }

    def filter_top_entities(self, top_n_entities=400):
        stats = self.get_accuracy_stats()

        # Get top entities
        entity_scores = {}
        for (entity, _), stat in stats.items():
            entity_scores[entity] = entity_scores.get(entity, 0) + stat["correct"]
        kept_entities = set(
            sorted(entity_scores, key=lambda x: entity_scores[x], reverse=True)[
                :top_n_entities
            ]
        )

        filtered_prompts = [p for p in self.prompts if p.entity_label in kept_entities]
        return self._filter_data_(filtered_prompts)

    def save_as_instance(self, save_path: str):
        """Save the RAVELInstance object to a json file."""
        ravel_instance_dict = {
            "prompts": [p.__dict__ for p in self.prompts],
            "entityLBL_attrTYP_attrLBL": self.entityLBL_attrTYP_attrLBL,
            "template_splits": self.template_splits,
            "entity_splits": self.entity_splits,
            "attribute_type_to_templates": self.attribute_type_to_templates,
            "config": self.config,
        }
        with open(save_path, "w") as f:
            json.dump(ravel_instance_dict, f)
        return ravel_instance_dict

    @classmethod
    def load(cls, load_path: str):
        """Load the RAVELInstance object from a json file."""
        with open(load_path) as f:
            ravel_instance_dict = json.load(f)
        fresh_instance = cls()
        fresh_instance.prompts = [Prompt(**p) for p in ravel_instance_dict["prompts"]]
        fresh_instance.entityLBL_attrTYP_attrLBL = ravel_instance_dict[
            "entityLBL_attrTYP_attrLBL"
        ]
        fresh_instance.template_splits = ravel_instance_dict["template_splits"]
        fresh_instance.entity_splits = ravel_instance_dict["entity_splits"]
        fresh_instance.attribute_type_to_templates = ravel_instance_dict[
            "attribute_type_to_templates"
        ]
        fresh_instance.config = ravel_instance_dict["config"]
        return fresh_instance

    def create_and_save_filtered_dataset(
        self,
        artifact_dir: str,
        top_n_entities: int,
    ) -> "RAVELFilteredDataset":
        """Create and save the filtered dataset."""
        self.filter_top_entities(top_n_entities)

        config = deepcopy(self.config)
        config["top_n_entities"] = top_n_entities
        config["instance_name"] = (
            config["instance_name"] + f"_top-{top_n_entities}-entities"
        )
        prompt_dict = {
            "prompts": [p.__dict__ for p in self.prompts],
            "config": config,
        }

        filtered_dataset_path = os.path.join(
            artifact_dir, f"{config['instance_name']}_filtered_dataset.json"
        )
        with open(filtered_dataset_path, "w") as f:
            json.dump(prompt_dict, f)

        return RAVELFilteredDataset.from_dict(prompt_dict)


class RAVELFilteredDataset:
    def __init__(self, prompts: list[Prompt], config: dict):
        self.prompts = prompts
        self.config = config

    def get_prompts_by_attribute(self, attribute: str) -> list[Prompt]:
        return [p for p in self.prompts if p.attribute_type == attribute]

    def get_prompts_by_entity(self, entity: str) -> list[Prompt]:
        return [p for p in self.prompts if p.entity_label == entity]

    def get_prompts_by_template(self, template: str) -> list[Prompt]:
        return [p for p in self.prompts if p.template == template]

    def get_prompts_by_context_split(self, split: str) -> list[Prompt]:
        return [p for p in self.prompts if p.context_split == split]

    def get_prompts_by_entity_split(self, split: str) -> list[Prompt]:
        return [p for p in self.prompts if p.entity_split == split]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

    def __iter__(self):
        return iter(self.prompts)

    def __contains__(self, item):
        return item in self.prompts

    def __repr__(self):
        return f"RAVELFilteredDataset(prompts={self.prompts})"

    def __str__(self):
        return f"RAVELFilteredDataset(prompts={self.prompts})"

    def save(self, artifact_dir: str):
        prompt_dict = {
            "prompts": [p.__dict__ for p in self.prompts],
            "config": self.config,
        }
        save_path = os.path.join(
            artifact_dir,
            f"{self.config['instance_name']}_top-{self.config['top_n_entities']}-entities_filtered_dataset.json",
        )
        with open(save_path, "w") as f:
            json.dump(prompt_dict, f)

    @classmethod
    def from_dict(cls, prompt_dict: dict):
        return cls(
            prompts=[Prompt(**p) for p in prompt_dict["prompts"]],
            config=prompt_dict["config"],
        )

    @classmethod
    def load(cls, load_path: str):
        with open(load_path) as f:
            prompt_dict = json.load(f)
        return cls.from_dict(prompt_dict)


if __name__ == "__main__":
    import sae_bench.sae_bench_utils.general_utils as general_utils

    # Load model and tokenizer
    config = RAVELEvalConfig()
    device = "cuda:0"

    LLM_NAME_MAP = {
        "gemma-2-2b": "google/gemma-2-2b",
    }
    config.model_name = LLM_NAME_MAP[config.model_name]
    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)
    config.llm_batch_size = 32
    config.full_dataset_downsample = None

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=device,
        torch_dtype=llm_dtype,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Create full RAVELInstance, no downsample, generate completions, filter for correct completions, save.
    entity_type = list(config.entity_attribute_selection.keys())[0]
    attribute_types = config.entity_attribute_selection[entity_type]
    print("Loading and tokenizing full dataset")
    full_dataset = RAVELInstance.create_from_files(
        config=config,
        entity_type=entity_type,
        tokenizer=tokenizer,
        data_dir=config.artifact_dir,
        model=model,
        model_name=config.model_name,
        attribute_types=attribute_types,
        downsample=config.full_dataset_downsample,
    )

    # Test loading the full dataset.
    instance_filename = (
        full_dataset.config["instance_name"] + "_unfiltered_full_instance.json"
    )
    instance_path = os.path.join(config.artifact_dir, instance_filename)
    full_dataset = RAVELInstance.load(instance_path)

    # Create filtered dataset.
    filtered_dataset = full_dataset.create_and_save_filtered_dataset(
        artifact_dir=config.artifact_dir,
        top_n_entities=config.top_n_entities,
    )

    # Test loading the filtered dataset.
    filtered_dataset_filename = (
        filtered_dataset.config["instance_name"] + "_filtered_dataset.json"
    )
    filtered_dataset_path = os.path.join(config.artifact_dir, filtered_dataset_filename)
    filtered_dataset = RAVELFilteredDataset.load(filtered_dataset_path)
