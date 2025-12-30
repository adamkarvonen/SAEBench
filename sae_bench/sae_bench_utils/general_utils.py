import functools
import os
import random
import re
import time
import warnings
from typing import Any, Callable

import pandas as pd
import torch
from sae_lens import SAE
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens.saes.sae import SAEMetadata


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


def dtype_to_str(dtype: torch.dtype) -> str:
    return dtype.__str__().split(".")[1]


def filter_keywords(
    sae_locations: list[str],
    exclude_keywords: list[str],
    include_keywords: list[str],
    case_sensitive: bool = False,
) -> list[str]:
    """
    Filter a list of locations based on exclude and include keywords.

    Args:
        sae_locations: List of location strings to filter
        exclude_keywords: List of keywords to exclude
        include_keywords: List of keywords that must be present
        case_sensitive: Whether to perform case-sensitive filtering

    Returns:
        List of filtered locations that match the criteria
    """
    if not case_sensitive:
        exclude = [k.lower() for k in exclude_keywords]
        include = [k.lower() for k in include_keywords]
    else:
        exclude = exclude_keywords
        include = include_keywords

    filtered_locations = []

    for location in sae_locations:
        location_lower = location.lower()

        # Check if any exclude keywords are present
        should_exclude = any(keyword in location_lower for keyword in exclude)

        # Check if all include keywords are present
        has_all_includes = all(keyword in location_lower for keyword in include)

        # Add location if it passes both criteria
        if not should_exclude and has_all_includes:
            filtered_locations.append(location)

    return filtered_locations


def filter_with_regex(filenames: list[str], regex_list: list[str]) -> list[str]:
    """
    Filters a list of filenames, returning those that match at least one of the given regex patterns.

    Args:
        filenames (list of str): The list of filenames to filter.
        regex_list (list of str): A list of regular expressions to match.

    Returns:
        list of str: Filenames that match at least one regex.
    """
    # Compile all regex patterns for efficiency
    compiled_regexes = [re.compile(pattern) for pattern in regex_list]

    # Filter filenames that match any of the compiled regex patterns
    matching_filenames = [
        filename
        for filename in filenames
        if any(regex.search(filename) for regex in compiled_regexes)
    ]

    return matching_filenames


def setup_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    return device


@torch.no_grad()
def check_decoder_norms(W_dec: torch.Tensor) -> bool:
    """
    It's important to check that the decoder weights are normalized.
    """
    norms = torch.norm(W_dec, dim=1).to(dtype=W_dec.dtype, device=W_dec.device)

    # In bfloat16, it's common to see errors of (1/256) in the norms
    tolerance = 1e-2 if W_dec.dtype in [torch.bfloat16, torch.float16] else 1e-5

    if torch.allclose(norms, torch.ones_like(norms), atol=tolerance):
        return True
    else:
        max_diff = torch.max(torch.abs(norms - torch.ones_like(norms)))
        warnings.warn(
            f"Decoder weights are not normalized. Max diff: {max_diff.item()}. Refer to base_sae.py and relu_sae.py for more info."
        )
        return False


def load_and_format_sae(
    sae_release_or_unique_id: str, sae_object_or_sae_lens_id: str | SAE, device: str
) -> tuple[str, SAE, torch.Tensor | None] | None:
    """Handle both pretrained SAEs (identified by string) and custom SAEs (passed as objects)"""
    if isinstance(sae_object_or_sae_lens_id, str):
        sae, _, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
            release=sae_release_or_unique_id,
            sae_id=sae_object_or_sae_lens_id,
            device=device,
        )
        sae_id = sae_object_or_sae_lens_id
        try:
            sae.fold_W_dec_norm()
        except NotImplementedError:
            print(
                f"Failed to fold W_dec norm for {sae_release_or_unique_id}_{sae_object_or_sae_lens_id}"
            )
    else:
        sae = sae_object_or_sae_lens_id
        sae_id = "custom_sae"
        sparsity = None
        check_decoder_norms(sae.W_dec.data)

    _standardize_sae_cfg(sae.cfg)

    return sae_id, sae, sparsity


def get_results_filepath(output_path: str, sae_release: str, sae_id: str) -> str:
    sae_result_file = f"{sae_release}_{sae_id}_eval_results.json"
    sae_result_file = sae_result_file.replace("/", "_")
    sae_result_path = os.path.join(output_path, sae_result_file)

    return sae_result_path


def find_gemmascope_average_l0_sae_names(
    layer_num: int,
    gemmascope_release_name: str = "gemma-scope-2b-pt-res",
    width_num: str = "16k",
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


def retry_with_exponential_backoff(
    retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: type[Exception] | tuple[type[Exception], ...] = Exception,
) -> Callable:
    """
    Decorator for retrying a function with exponential backoff.

    Args:
        retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delay
        exceptions: Exception(s) to catch and retry on
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None

            for retry_count in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if retry_count == retries:
                        print(f"Failed after {retries} retries: {str(e)}")
                        raise

                    # Calculate delay with optional jitter
                    current_delay = min(
                        delay * (exponential_base**retry_count), max_delay
                    )
                    if jitter:
                        current_delay *= 1 + random.random() * 0.1  # 10% jitter

                    print(
                        f"Attempt {retry_count + 1}/{retries} failed: {str(e)}. "
                        f"Retrying in {current_delay:.2f} seconds..."
                    )
                    time.sleep(current_delay)

            if last_exception:
                raise last_exception
            return None

        return wrapper

    return decorator


def _get_cfg_meta_field(cfg: Any, field: str) -> Any | None:
    # SAELens v6 moves some cfg properties to `cfg.metadata` that were previously on the cfg object
    if hasattr(cfg, field):
        return getattr(cfg, field)
    if hasattr(cfg, "metadata") and hasattr(cfg.metadata, field):
        return getattr(cfg.metadata, field)
    return None


def _standardize_sae_cfg(cfg: Any):
    """
    Helper to standardize the SAE cfg object so both SAEBench SAEs and SAELens v6 SAEs can be used interchangeably.
    """
    hook_name = _get_cfg_meta_field(cfg, "hook_name")
    hook_layer = _get_cfg_meta_field(cfg, "hook_layer")
    if hook_name is not None and hook_layer is None:
        match = re.search(r"\d+", str(hook_name))
        if match:
            hook_layer = int(match.group(0))
    context_size = _get_cfg_meta_field(cfg, "context_size")
    dataset_trust_remote_code = _get_cfg_meta_field(cfg, "dataset_trust_remote_code")
    model_name = _get_cfg_meta_field(cfg, "model_name")
    hook_head_index = _get_cfg_meta_field(cfg, "hook_head_index")
    exclude_special_tokens = _get_cfg_meta_field(cfg, "exclude_special_tokens")
    model_from_pretrained_kwargs = (
        _get_cfg_meta_field(cfg, "model_from_pretrained_kwargs") or {}
    )
    prepend_bos = _get_cfg_meta_field(cfg, "prepend_bos")
    if hook_layer is None:
        raise ValueError("Cound not determine Hook layer from SAE cfg")
    if hook_name is None:
        raise ValueError("Cound not determine Hook name from SAE cfg")
    if model_name is None:
        raise ValueError("Cound not determine Model name from SAE cfg")

    # for SAELens v6, these fields are on the cfg.metadata object
    # for backwards compatibility, we also set them on the cfg object
    cfg.hook_layer = hook_layer
    cfg.hook_name = hook_name
    cfg.context_size = context_size
    cfg.dataset_trust_remote_code = dataset_trust_remote_code
    cfg.model_name = model_name
    cfg.hook_head_index = hook_head_index
    cfg.model_from_pretrained_kwargs = model_from_pretrained_kwargs
    cfg.prepend_bos = prepend_bos
    cfg.exclude_special_tokens = exclude_special_tokens

    # In SAELens v6, sae.architecture is a function rather than a raw string, so we need a new field that stores the string version
    cfg.architecture_str = (
        cfg.architecture() if callable(cfg.architecture) else cfg.architecture
    )

    # If the SAE doesn't have a SAELens v6 metadata field, add it
    metadata = cfg.metadata if hasattr(cfg, "metadata") else SAEMetadata()
    metadata.hook_layer = hook_layer
    metadata.hook_name = hook_name
    metadata.context_size = context_size
    metadata.dataset_trust_remote_code = dataset_trust_remote_code
    metadata.model_name = model_name
    metadata.hook_head_index = hook_head_index
    metadata.model_from_pretrained_kwargs = model_from_pretrained_kwargs
    metadata.prepend_bos = prepend_bos
    metadata.exclude_special_tokens = exclude_special_tokens

    cfg.metadata = metadata
