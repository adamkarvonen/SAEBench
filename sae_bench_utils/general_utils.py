import pandas as pd
import re
import os
import torch
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory


def str_to_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str.lower())
    if dtype is None:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. Supported dtypes: {list(dtype_map.keys())}"
        )
    return dtype


def setup_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    return device


def find_gemmascope_average_l0_sae_names(
    layer_num: int, gemmascope_release_name: str = "gemma-scope-2b-pt-res", width_num: str = "16k"
) -> list[str]:
    df = pd.DataFrame.from_records(
        {k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}
    ).T
    filtered_df = df[df.release == gemmascope_release_name]
    name_to_id_map = filtered_df.saes_map.item()

    pattern = rf"layer_{layer_num}/width_{width_num}/average_l0_\d+"

    matching_keys = [key for key in name_to_id_map.keys() if re.match(pattern, key)]

    return matching_keys


def get_sparsity_penalty(config: dict) -> float:
    trainer_class = config["trainer"]["trainer_class"]
    if trainer_class == "TrainerTopK":
        return config["trainer"]["k"]
    elif trainer_class == "PAnnealTrainer":
        return config["trainer"]["sparsity_penalty"]
    else:
        return config["trainer"]["l1_penalty"]


def average_results_dictionaries(
    results_dict: dict[str, dict[str, float]], dataset_names: list[str]
) -> dict[str, float]:
    """If we have multiple dicts of results from separate datasets, get an average performance over all datasets.
    Results_dict is dataset -> dict of metric_name : float result"""
    averaged_results = {}
    aggregated_results = {}

    for dataset_name in dataset_names:
        dataset_results = results_dict[f"{dataset_name}_results"]

        for metric_name, metric_value in dataset_results.items():
            if metric_name not in aggregated_results:
                aggregated_results[metric_name] = []

            aggregated_results[metric_name].append(metric_value)

    averaged_results = {}
    for metric_name, values in aggregated_results.items():
        average_value = sum(values) / len(values)
        averaged_results[metric_name] = average_value

    return averaged_results
