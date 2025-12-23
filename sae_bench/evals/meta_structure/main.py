import argparse
import math
import os
import random
import time
from collections.abc import Iterator

import torch
from sae_lens import SAE
from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.training.sae_trainer import SAETrainer

import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.evals.meta_structure.eval_config import MetaStructureEvalConfig
from sae_bench.evals.meta_structure.eval_output import (
    EVAL_TYPE_ID_META_STRUCTURE,
    MetaStructureEvalOutput,
    MetaStructureMetricCategories,
    MetaStructureMetrics,
)
from sae_bench.sae_bench_utils import (
    get_eval_uuid,
    get_sae_bench_version,
    get_sae_lens_version,
)
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex


def _decoder_data_provider(
    decoder: torch.Tensor, batch_size: int
) -> Iterator[torch.Tensor]:
    """Yield batches of decoder rows indefinitely."""
    n_rows = decoder.shape[0]
    while True:
        indices = torch.randint(0, n_rows, (batch_size,), device=decoder.device)
        yield decoder[indices]


def _train_meta_structure_on_decoder(
    decoder: torch.Tensor,
    meta_width: int,
    config: MetaStructureEvalConfig,
    device: str,
) -> tuple[BatchTopKTrainingSAE, float]:
    """Train a BatchTopK meta-structure model on the decoder matrix."""
    meta_cfg = BatchTopKTrainingSAEConfig(
        d_in=decoder.shape[1],
        d_sae=meta_width,
        k=config.k,
        device=device,
        dtype=config.dtype,
        rescale_acts_by_decoder_norm=True,
    )
    meta_structure = BatchTopKTrainingSAE(meta_cfg)

    trainer_cfg = SAETrainerConfig(
        n_checkpoints=0,
        checkpoint_path=None,
        save_final_checkpoint=False,
        total_training_samples=config.train_steps * config.train_batch_size,
        device=device,
        autocast=config.autocast,
        lr=config.learning_rate,
        lr_end=config.learning_rate,
        lr_scheduler_name="constant",
        lr_warm_up_steps=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_decay_steps=0,
        n_restart_cycles=0,
        train_batch_size_samples=config.train_batch_size,
        dead_feature_window=config.dead_feature_window,
        feature_sampling_window=config.feature_sampling_window,
        logger=LoggingConfig(log_to_wandb=False),
    )

    data_provider = _decoder_data_provider(decoder, config.train_batch_size)
    trainer = SAETrainer(
        cfg=trainer_cfg, sae=meta_structure, data_provider=data_provider
    )
    if config.weight_decay != 0.0:
        for group in trainer.optimizer.param_groups:
            group["weight_decay"] = config.weight_decay

    start_time = time.time()
    trained_sae = trainer.fit()
    train_time = time.time() - start_time

    return trained_sae, train_time


@torch.no_grad()
def _decoder_variance_metrics(
    meta_structure: BatchTopKTrainingSAE,
    decoder: torch.Tensor,
    eval_batch_size: int,
) -> tuple[float, float]:
    """Return (fraction_variance_explained, mse)."""
    meta_structure.eval()
    total_var = 0.0
    residual = 0.0

    decoder_mean = decoder.mean(dim=0, keepdim=True)

    for start in range(0, decoder.shape[0], eval_batch_size):
        batch = decoder[start : start + eval_batch_size]
        recon = meta_structure(batch)
        residual += torch.sum((batch - recon) ** 2).item()
        total_var += torch.sum((batch - decoder_mean) ** 2).item()

    mse = residual / decoder.numel()
    if total_var == 0:
        return 0.0, mse
    return 1 - residual / total_var, mse


def run_eval_single_sae(
    sae_release: str,
    sae_id: str,
    sae: SAE,
    config: MetaStructureEvalConfig,
    device: str,
) -> MetaStructureEvalOutput:
    dtype = general_utils.str_to_dtype(config.dtype)
    sae = sae.to(device=device, dtype=dtype)
    decoder = sae.W_dec.detach().to(device=device, dtype=dtype).contiguous()

    meta_width = max(math.ceil(decoder.shape[0] * config.width_ratio), config.k)

    meta_structure, train_time = _train_meta_structure_on_decoder(
        decoder=decoder, meta_width=meta_width, config=config, device=device
    )
    variance_explained, mse = _decoder_variance_metrics(
        meta_structure=meta_structure,
        decoder=decoder,
        eval_batch_size=config.eval_batch_size,
    )

    metrics = MetaStructureMetricCategories(
        meta_structure=MetaStructureMetrics(
            decoder_fraction_variance_explained=variance_explained,
            train_time_seconds=train_time,
            final_reconstruction_mse=mse,
        )
    )

    return MetaStructureEvalOutput(
        eval_config=config,
        eval_id=get_eval_uuid(),
        datetime_epoch_millis=int(time.time() * 1000),
        eval_result_metrics=metrics,
        eval_result_details=[],
        sae_bench_commit_hash=get_sae_bench_version(),
        sae_lens_id=sae_id,
        sae_lens_release_id=sae_release,
        sae_lens_version=get_sae_lens_version(),
        sae_cfg_dict=sae.cfg.to_dict(),
    )


def run_eval(
    config: MetaStructureEvalConfig,
    selected_saes: list[tuple[str, str]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
) -> dict[str, float]:
    results: dict[str, float] = {}
    os.makedirs(output_path, exist_ok=True)

    for sae_release, sae_id_or_path in selected_saes:
        sae_id, sae, _ = general_utils.load_and_format_sae(
            sae_release, sae_id_or_path, device
        )  # type: ignore

        sae_result_path = general_utils.get_results_filepath(
            output_path, sae_release, sae_id
        )

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Skipping {sae_release}_{sae_id} as results already exist.")
            continue

        eval_output = run_eval_single_sae(
            sae_release=sae_release,
            sae_id=sae_id,
            sae=sae,
            config=config,
            device=device,
        )
        eval_output.to_json_file(sae_result_path, indent=2)
        results[f"{sae_release}_{sae_id}"] = (
            eval_output.eval_result_metrics.meta_structure.decoder_fraction_variance_explained
        )

    return results


def _arg_parser() -> argparse.ArgumentParser:
    default_config = MetaStructureEvalConfig()
    parser = argparse.ArgumentParser(
        description="Run meta-structure decoder variance evaluation"
    )
    parser.add_argument(
        "--sae_regex_pattern",
        type=str,
        required=True,
        help="Regex pattern to select SAE releases.",
    )
    parser.add_argument(
        "--sae_id_pattern",
        type=str,
        required=True,
        help="Regex pattern to select SAEs within a release.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="eval_results/meta_structure",
        help="Where to store evaluation JSON outputs.",
    )
    parser.add_argument(
        "--width_ratio",
        type=float,
        default=default_config.width_ratio,
        help="Meta-structure width as a fraction of the base SAE width.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=default_config.k,
        help="Average number of active latents to maintain (BatchTopK k).",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=default_config.train_steps,
        help="Number of training steps for the meta-structure.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=default_config.train_batch_size,
        help="Training batch size when sampling decoder rows.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=default_config.learning_rate,
        help="Learning rate for meta-structure training.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=default_config.weight_decay,
        help="Weight decay for the meta-structure optimizer.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=default_config.eval_batch_size,
        help="Batch size used to compute decoder variance explained.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=default_config.dtype,
        choices=["float16", "float32", "bfloat16", "float64"],
        help="Datatype for meta-structure training.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=default_config.random_seed,
        help="Random seed for meta-structure training.",
    )
    parser.add_argument(
        "--autocast",
        default=default_config.autocast,
        action=argparse.BooleanOptionalAction,
        help="Enable or disable torch.autocast during training.",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Force rerun even if results already exist.",
    )
    return parser


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_config(args: argparse.Namespace) -> MetaStructureEvalConfig:
    return MetaStructureEvalConfig(
        random_seed=args.random_seed,
        width_ratio=args.width_ratio,
        k=args.k,
        train_steps=args.train_steps,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_batch_size=args.eval_batch_size,
        dtype=args.dtype,
        autocast=args.autocast,
    )


if __name__ == "__main__":
    """
    Example:
    python sae_bench/evals/meta_structure/main.py \
        --sae_regex_pattern ".*" \
        --sae_id_pattern ".*" \
        --output_folder eval_results/meta_structure
    """
    args = _arg_parser().parse_args()
    _set_random_seed(args.random_seed)
    device = general_utils.setup_environment()

    config = create_config(args)
    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_id_pattern)
    assert len(selected_saes) > 0, (
        "No SAEs matched the provided regex patterns. Did you mistype them?"
    )

    print(f"Running {EVAL_TYPE_ID_META_STRUCTURE} for {len(selected_saes)} SAEs...")
    start_time = time.time()
    run_eval(
        config=config,
        selected_saes=selected_saes,
        device=device,
        output_path=args.output_folder,
        force_rerun=args.force_rerun,
    )
    print(f"Completed in {time.time() - start_time:.2f} seconds.")
