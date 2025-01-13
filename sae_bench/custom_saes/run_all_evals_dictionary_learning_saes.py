import os
from typing import Any, Optional
from tqdm import tqdm
import torch
from huggingface_hub import snapshot_download
import json

import sae_bench.evals.absorption.main as absorption
import sae_bench.evals.autointerp.main as autointerp
import sae_bench.evals.core.main as core
import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
import sae_bench.evals.sparse_probing.main as sparse_probing
import sae_bench.evals.unlearning.main as unlearning
import sae_bench.sae_bench_utils.general_utils as general_utils

import sae_bench.custom_saes.base_sae as base_sae
import sae_bench.custom_saes.relu_sae as relu_sae
import sae_bench.custom_saes.jumprelu_sae as jumprelu_sae
import sae_bench.custom_saes.topk_sae as topk_sae
import sae_bench.custom_saes.batch_topk_sae as batch_topk_sae
import sae_bench.custom_saes.gated_sae as gated_sae

MODEL_CONFIGS = {
    "pythia-70m-deduped": {
        "batch_size": 512,
        "dtype": "float32",
        "layers": [3, 4],
        "d_model": 512,
    },
    "pythia-160m-deduped": {
        "batch_size": 256,
        "dtype": "float32",
        "layers": [8],
        "d_model": 768,
    },
    "gemma-2-2b": {
        "batch_size": 32,
        "dtype": "bfloat16",
        "layers": [5, 12, 19],
        "d_model": 2304,
    },
}

output_folders = {
    "absorption": "eval_results/absorption",
    "autointerp": "eval_results/autointerp",
    "core": "eval_results/core",
    "scr": "eval_results/scr",
    "tpp": "eval_results/tpp",
    "sparse_probing": "eval_results/sparse_probing",
    "unlearning": "eval_results/unlearning",
}


TRAINER_LOADERS = {
    "MatroyshkaBatchTopKTrainer": batch_topk_sae.load_dictionary_learning_matroyshka_batch_topk_sae,
    "BatchTopKTrainer": batch_topk_sae.load_dictionary_learning_batch_topk_sae,
    "TopKTrainer": topk_sae.load_dictionary_learning_topk_sae,
    "StandardTrainerAprilUpdate": relu_sae.load_dictionary_learning_relu_sae,
    "StandardTrainer": relu_sae.load_dictionary_learning_relu_sae,
    "PAnnealTrainer": relu_sae.load_dictionary_learning_relu_sae,
    "JumpReluTrainer": jumprelu_sae.load_dictionary_learning_jump_relu_sae,
    "GatedSAETrainer": gated_sae.load_dictionary_learning_gated_sae,
}


def get_all_hf_repo_autoencoders(
    repo_id: str, download_location: str = "downloaded_saes"
) -> list[str]:
    download_location = os.path.join(download_location, repo_id.replace("/", "_"))
    config_dir = snapshot_download(
        repo_id,
        allow_patterns=["*config.json"],
        local_dir=download_location,
        force_download=False,
    )

    config_locations = []

    for root, _, files in os.walk(config_dir):
        for file in files:
            if file == "config.json":
                config_locations.append(os.path.join(root, file))

    repo_locations = []

    for config in config_locations:
        repo_location = config.split(f"{download_location}/")[1].split("/config.json")[
            0
        ]
        repo_locations.append(repo_location)

    return repo_locations


def load_dictionary_learning_sae(
    repo_id: str,
    location: str,
    model_name,
    device: str,
    dtype: torch.dtype,
    layer: Optional[int] = None,
    download_location: str = "downloaded_saes",
) -> base_sae.BaseSAE:
    download_location = os.path.join(download_location, repo_id.replace("/", "_"))

    config_file = f"{download_location}/{location}/config.json"

    with open(config_file, "r") as f:
        config = json.load(f)

    trainer_class = config["trainer"]["trainer_class"]

    location = f"{location}/ae.pt"

    sae = TRAINER_LOADERS[trainer_class](
        repo_id=repo_id,
        filename=location,
        layer=layer,
        model_name=model_name,
        device=device,
        dtype=dtype,
    )
    return sae


def verify_saes_load(
    repo_id: str,
    sae_locations: list[str],
    model_name: str,
    device: str,
    dtype: torch.dtype,
):
    """Verify that all SAEs load correctly. Useful to check this before a big evaluation run."""
    for sae_location in sae_locations:
        sae = load_dictionary_learning_sae(
            repo_id=repo_id,
            location=sae_location,
            layer=None,
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
        del sae


def run_evals(
    repo_id: str,
    model_name: str,
    sae_locations: list[str],
    llm_batch_size: int,
    llm_dtype: str,
    device: str,
    eval_types: list[str],
    random_seed: int,
    api_key: Optional[str] = None,
    force_rerun: bool = False,
):
    """Run selected evaluations for the given model and SAEs."""

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    # Mapping of eval types to their functions and output paths
    eval_runners = {
        "absorption": (
            lambda selected_saes, is_final: absorption.run_eval(
                absorption.AbsorptionEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/absorption",
                force_rerun,
            )
        ),
        "autointerp": (
            lambda selected_saes, is_final: autointerp.run_eval(
                autointerp.AutoInterpEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                api_key,
                "eval_results/autointerp",
                force_rerun,
            )
        ),
        # TODO: Do a better job of setting num_batches and batch size
        # The core run_eval() interface isn't well suited for custom SAEs, so we have to do this instead.
        # It isn't ideal, but it works.
        # TODO: Don't hardcode magic numbers
        "core": (
            lambda selected_saes, is_final: core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=16,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset="Skylion007/openwebtext",
                context_size=128,
                output_folder="eval_results/core",
                verbose=True,
                dtype=llm_dtype,
            )
        ),
        "scr": (
            lambda selected_saes, is_final: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    perform_scr=True,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results",  # We add scr or tpp depending on perform_scr
                force_rerun,
                clean_up_activations=is_final,
                save_activations=True,
            )
        ),
        "tpp": (
            lambda selected_saes, is_final: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    perform_scr=False,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results",  # We add scr or tpp depending on perform_scr
                force_rerun,
                clean_up_activations=is_final,
                save_activations=True,
            )
        ),
        "sparse_probing": (
            lambda selected_saes, is_final: sparse_probing.run_eval(
                sparse_probing.SparseProbingEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/sparse_probing",
                force_rerun,
                clean_up_activations=is_final,
                save_activations=True,
            )
        ),
        "unlearning": (
            lambda selected_saes, is_final: unlearning.run_eval(
                unlearning.UnlearningEvalConfig(
                    model_name="gemma-2-2b-it",
                    random_seed=random_seed,
                    llm_dtype=llm_dtype,
                    llm_batch_size=llm_batch_size
                    // 8,  # 8x smaller batch size for unlearning due to longer sequences
                ),
                selected_saes,
                device,
                "eval_results/unlearning",
                force_rerun,
            )
        ),
    }

    for eval_type in eval_types:
        if eval_type not in eval_runners:
            raise ValueError(f"Unsupported eval type: {eval_type}")

    verify_saes_load(
        repo_id,
        sae_locations,
        model_name,
        device,
        general_utils.str_to_dtype(llm_dtype),
    )

    # Run selected evaluations
    for eval_type in tqdm(eval_types, desc="Evaluations"):
        if eval_type == "autointerp" and api_key is None:
            print("Skipping autointerp evaluation due to missing API key")
            continue
        if eval_type == "unlearning":
            if not os.path.exists("./evals/unlearning/data/bio-forget-corpus.jsonl"):
                print(
                    "Skipping unlearning evaluation due to missing bio-forget-corpus.jsonl"
                )
                continue

        print(f"\n\n\nRunning {eval_type} evaluation\n\n\n")

        for i, sae_location in enumerate(sae_locations):
            is_final = False
            if i == len(sae_locations) - 1:
                is_final = True

            sae = load_dictionary_learning_sae(
                repo_id=repo_id,
                location=sae_location,
                layer=None,
                model_name=model_name,
                device=device,
                dtype=general_utils.str_to_dtype(llm_dtype),
            )
            unique_sae_id = sae_location.replace("/", "_")
            unique_sae_id = f"{repo_id.split('/')[1]}_{unique_sae_id}"
            selected_saes = [(unique_sae_id, sae)]

            os.makedirs(output_folders[eval_type], exist_ok=True)
            eval_runners[eval_type](selected_saes, is_final)

            del sae


if __name__ == "__main__":
    RANDOM_SEED = 42

    device = general_utils.setup_environment()

    # Select your eval types here.
    eval_types = [
        "absorption",
        "core",
        "scr",
        "tpp",
        "sparse_probing",
        "autointerp",
        # "unlearning",
    ]

    repos = [
        (
            "adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0104",
            "pythia-160m-deduped",
        ),
        ("adamkarvonen/saebench_gemma-2-2b_width-2pow14_date-0104", "gemma-2-2b"),
    ]

    for repo_id, model_name in repos:
        print(f"\n\n\nEvaluating {model_name} with {repo_id}\n\n\n")

        # model_name = "gemma-2-2b"
        llm_batch_size = MODEL_CONFIGS[model_name]["batch_size"]
        str_dtype = MODEL_CONFIGS[model_name]["dtype"]
        torch_dtype = general_utils.str_to_dtype(str_dtype)

        sae_locations = get_all_hf_repo_autoencoders(repo_id)

        sae_locations = general_utils.filter_keywords(
            sae_locations, exclude_keywords=["checkpoints"], include_keywords=[]
        )

        # Note: Unlearning is not recommended for models with < 2B parameters and we recommend an instruct tuned model
        # Unlearning will also require requesting permission for the WMDP dataset (see unlearning/README.md)
        # Absorption not recommended for models < 2B parameters

        if "autointerp" in eval_types:
            try:
                with open("openai_api_key.txt") as f:
                    api_key = f.read().strip()
            except FileNotFoundError:
                raise Exception("Please create openai_api_key.txt with your API key")
        else:
            api_key = None

        run_evals(
            repo_id=repo_id,
            model_name=model_name,
            sae_locations=sae_locations,
            llm_batch_size=llm_batch_size,
            llm_dtype=str_dtype,
            device=device,
            eval_types=eval_types,
            api_key=api_key,
            random_seed=RANDOM_SEED,
        )
