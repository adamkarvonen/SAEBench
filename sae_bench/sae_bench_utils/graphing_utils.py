import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from copy import deepcopy

from scipy import stats
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from typing import Optional, Dict, Any
from collections import defaultdict

# create a dictionary mapping trainer types to marker shapes

TRAINER_MARKERS = {
    "standard": "o",
    "standard_april_update": "o",
    "jumprelu": "X",
    "topk": "^",
    "batch_topk": "s",
    "p_anneal": "*",
    "matroyshka_batch_topk": "*",
    "gated": "d",
}

TRAINER_COLORS = {
    "standard": "blue",
    "standard_april_update": "blue",
    "jumprelu": "orange",
    "topk": "green",
    "batch_topk": "purple",
    "p_anneal": "red",
    "matroyshka_batch_topk": "brown",
    "gated": "purple",
}


TRAINER_LABELS = {
    "standard": "Standard",
    "standard_april_update": "Standard",
    "jumprelu": "JumpReLU",
    "topk": "TopK",
    "batch_topk": "Batch TopK",
    "p_anneal": "Standard with P-Annealing",
    "matroyshka_batch_topk": "Matroyshka Batch TopK",
    "gated": "Gated",
}


# default text size
plt.rcParams.update({"font.size": 20})


def get_best_results(
    results_dict: dict[str, dict[str, float]], results_path: str, ks: list[int]
) -> dict[str, dict[str, float]]:
    best_results = {}
    for sae, data in results_dict.items():
        best_results[sae] = 0
        for k in ks:
            custom_metric, _ = get_custom_metric_key_and_name(results_path, k)
            if custom_metric in data:
                best_results[sae] = max(best_results[sae], data[custom_metric])
            else:
                print(f"Custom metric {custom_metric} not found for {sae}")

    for sae in best_results.keys():
        results_dict[sae]["best_custom_metric"] = best_results[sae]

    return results_dict


def get_single_figure(
    selected_saes: list[tuple[str, str]],
    results_path: str,
    core_results_path: str,
    image_base_name: str,
    k: Optional[int] = None,
    trainer_markers: Optional[dict[str, str]] = None,
    title: Optional[str] = None,
    title_prefix: str = "",
    plot_type: bool = True,
    baseline_sae: Optional[tuple[str, str]] = None,
    baseline_label: Optional[str] = None,
):
    eval_results = get_eval_results(selected_saes, results_path)
    core_results = get_core_results(selected_saes, core_results_path)

    for sae in eval_results:
        eval_results[sae].update(core_results[sae])

    custom_metric, custom_metric_name = get_custom_metric_key_and_name(results_path, k)

    if baseline_sae:
        baseline_results = get_eval_results([baseline_sae], results_path)
        baseline_id = f"{baseline_sae[0]}_{baseline_sae[1]}"
        baseline_results[baseline_id].update(
            get_core_results([baseline_sae], core_results_path)[baseline_id]
        )
        baseline_value = baseline_results[baseline_id][custom_metric]
        assert baseline_label, "Please provide a label for the baseline"
    else:
        baseline_value = None
        assert baseline_label is None, "Please do not provide a label for the baseline"

    if not title:
        title = f"{title_prefix}L0 vs {custom_metric_name}"

    if plot_type:
        fig = plot_2var_graph(
            eval_results,
            custom_metric,
            y_label=custom_metric_name,
            title=title,
            output_filename=f"{image_base_name}_2var_sae_type.png",
            trainer_markers=trainer_markers,
            return_fig=True,
            baseline_value=baseline_value,
            baseline_label=baseline_label,
        )
    else:
        fig = plot_2var_graph_dict_size(
            eval_results,
            custom_metric,
            y_label=custom_metric_name,
            title=title,
            output_filename=f"{image_base_name}_2var_dict_size.png",
            return_fig=True,
            baseline_value=baseline_value,
            baseline_label=baseline_label,
        )

    return fig


def plot_results(
    eval_filenames: list[str],
    core_filenames: list[str],
    eval_type: str,
    image_base_name: str,
    k: Optional[int] = None,
    trainer_markers: Optional[dict[str, str]] = None,
    trainer_colors: Optional[dict[str, str]] = None,
    title_prefix: str = "",
    return_fig: bool = False,
    baseline_sae_path: Optional[str] = None,
    baseline_label: Optional[str] = None,
):
    eval_results = get_eval_results(eval_filenames)
    core_results = get_core_results(core_filenames)

    for sae in eval_results:
        eval_results[sae].update(core_results[sae])

    custom_metric, custom_metric_name = get_custom_metric_key_and_name(eval_type, k)

    if baseline_sae_path:
        baseline_results = get_eval_results([baseline_sae_path])

        baseline_filename = os.path.basename(baseline_sae_path)
        baseline_results_key = baseline_filename.replace("_eval_results.json", "")

        core_baseline_filename = baseline_sae_path.replace(eval_type, "core")

        baseline_results[baseline_results_key].update(
            get_core_results([core_baseline_filename])[baseline_results_key]
        )

        baseline_value = baseline_results[baseline_results_key][custom_metric]
        assert baseline_label, "Please provide a label for the baseline"
    else:
        baseline_value = None
        assert baseline_label is None, "Please do not provide a label for the baseline"

    title_3var = f"{title_prefix}L0 vs Loss Recovered vs {custom_metric_name}"
    title_2var = f"{title_prefix}L0 vs {custom_metric_name}"

    plot_3var_graph(
        eval_results,
        title_3var,
        custom_metric,
        colorbar_label="Custom Metric",
        output_filename=f"{image_base_name}_3var.png",
        trainer_markers=trainer_markers,
    )
    fig = plot_2var_graph(
        eval_results,
        custom_metric,
        y_label=custom_metric_name,
        title=title_2var,
        output_filename=f"{image_base_name}_2var_sae_type.png",
        trainer_markers=trainer_markers,
        trainer_colors=trainer_colors,
        baseline_value=baseline_value,
        baseline_label=baseline_label,
    )

    plot_2var_graph_dict_size(
        eval_results,
        custom_metric,
        y_label=custom_metric_name,
        title=title_2var,
        output_filename=f"{image_base_name}_2var_dict_size.png",
        baseline_value=baseline_value,
        baseline_label=baseline_label,
        trainer_markers=trainer_markers,
    )

    if return_fig:
        return fig


def plot_best_of_ks_results(
    selected_saes: list[tuple[str, str]],
    results_path: str,
    core_results_path: str,
    image_base_name: str,
    ks: list[int],
    trainer_markers: Optional[dict[str, str]] = None,
    title_prefix: str = "",
    baseline_sae: Optional[tuple[str, str]] = None,
    baseline_label: Optional[str] = None,
):
    dummy_k = 0

    eval_results = get_eval_results(selected_saes, results_path)
    core_results = get_core_results(selected_saes, core_results_path)

    for sae in eval_results:
        eval_results[sae].update(core_results[sae])

    custom_metric, custom_metric_name = get_custom_metric_key_and_name(results_path, dummy_k)

    custom_metric = "best_custom_metric"
    custom_metric_name = custom_metric_name.replace(f"Top {dummy_k}", f"Best of {ks}")
    eval_results = get_best_results(eval_results, results_path, ks)

    if baseline_sae:
        baseline_results = get_eval_results([baseline_sae], results_path)
        baseline_id = f"{baseline_sae[0]}_{baseline_sae[1]}"
        baseline_results[baseline_id].update(
            get_core_results([baseline_sae], core_results_path)[baseline_id]
        )

        baseline_results = get_best_results(baseline_results, results_path, ks)
        baseline_value = baseline_results[baseline_id]["best_custom_metric"]
        assert baseline_label, "Please provide a label for the baseline"
    else:
        baseline_value = None
        assert baseline_label is None, "Please do not provide a label for the baseline"

    title_3var = f"{title_prefix}L0 vs Loss Recovered vs {custom_metric_name}"
    title_2var = f"{title_prefix}L0 vs {custom_metric_name}"

    plot_3var_graph(
        eval_results,
        title_3var,
        custom_metric,
        colorbar_label="Custom Metric",
        output_filename=f"{image_base_name}_3var.png",
        trainer_markers=trainer_markers,
    )
    plot_2var_graph(
        eval_results,
        custom_metric,
        y_label=custom_metric_name,
        title=title_2var,
        output_filename=f"{image_base_name}_2var_sae_type.png",
        trainer_markers=trainer_markers,
        baseline_value=baseline_value,
        baseline_label=baseline_label,
    )

    plot_2var_graph_dict_size(
        eval_results,
        custom_metric,
        y_label=custom_metric_name,
        title=title_2var,
        output_filename=f"{image_base_name}_2var_dict_size.png",
        baseline_value=baseline_value,
        baseline_label=baseline_label,
    )


def get_custom_metric_key_and_name(eval_path: str, k: Optional[int] = None) -> tuple[str, str]:
    if "tpp" in eval_path:
        custom_metric = f"tpp_threshold_{k}_total_metric"
        custom_metric_name = f"TPP Top {k} Metric"
    elif "scr" in eval_path:
        custom_metric = f"scr_metric_threshold_{k}"
        custom_metric_name = f"SCR Top {k} Metric"
    elif "sparse_probing" in eval_path:
        custom_metric = f"sae_top_{k}_test_accuracy"
        custom_metric_name = f"Sparse Probing Top {k} Test Accuracy"
    elif "absorption" in eval_path:
        custom_metric = "mean_absorption_score"
        custom_metric_name = "Mean Absorption Score"
    elif "autointerp" in eval_path:
        custom_metric = "autointerp_score"
        custom_metric_name = "Autointerp Score"
    elif "unlearning" in eval_path:
        custom_metric = "unlearning_score"
        custom_metric_name = "Unlearning Score"
    elif "core" in eval_path:
        custom_metric = "ce_loss_score"
        custom_metric_name = "Loss Recovered"
    else:
        raise ValueError("Please add the correct key for the custom metric")

    return custom_metric, custom_metric_name


def get_sae_bench_train_tokens(filename: str) -> int:
    """This is for SAE Bench internal use. The SAE cfg does not contain the number of training tokens, so we need to hardcode it."""

    if "sae_bench" not in filename:
        raise ValueError("This function is only for SAE Bench releases")

    batch_size = 2048

    if "step" not in filename:
        steps = 244140
        return steps * batch_size
    else:
        match = re.search(r"step_(\d+)", filename)
        if match:
            step = int(match.group(1))
            return step * batch_size
        else:
            raise ValueError("No step match found")


def get_d_sae_string(d_sae: int) -> str:
    rounded_d_sae = round(d_sae / 1000) * 1000

    # TODO: Temp SAE Bench fix
    if rounded_d_sae == 66000:
        rounded_d_sae = 65000

    if rounded_d_sae >= 1_000_000 and rounded_d_sae <= 1_060_000:
        return "1M"
    else:
        return f"{rounded_d_sae//1000}k"


def get_eval_results(eval_filenames: list[str]) -> dict[str, dict]:
    """eval_filenames is assumed to be a list of filenames of this format:
    {sae_release}_{sae_id}_eval_results.json"""
    eval_results = {}
    for filepath in eval_filenames:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        with open(filepath, "r") as f:
            single_sae_results = json.load(f)

        filename = os.path.basename(filepath)
        results_key = filename.replace("_eval_results.json", "")

        if "tpp" in filepath:
            eval_results[results_key] = single_sae_results["eval_result_metrics"]["tpp_metrics"]
        elif "scr" in filepath:
            eval_results[results_key] = single_sae_results["eval_result_metrics"]["scr_metrics"]
        elif "absorption" in filepath:
            eval_results[results_key] = single_sae_results["eval_result_metrics"]["mean"]
        elif "autointerp" in filepath:
            eval_results[results_key] = single_sae_results["eval_result_metrics"]["autointerp"]
        elif "sparse_probing" in filepath:
            eval_results[results_key] = single_sae_results["eval_result_metrics"]["sae"]
        elif "unlearning" in filepath:
            eval_results[results_key] = single_sae_results["eval_result_metrics"]["unlearning"]
        elif "core" in filepath:
            # core has nested evaluation metrics, so we flatten them out here
            core_results = single_sae_results["eval_result_metrics"]
            eval_results[results_key] = {}
            for parent_key, child_dict in core_results.items():
                for metric_key, value in child_dict.items():
                    eval_results[results_key][metric_key] = value
        else:
            raise ValueError("Please add the correct key for the eval results")

        eval_results[results_key]["eval_config"] = single_sae_results["eval_config"]

        sae_config = single_sae_results["sae_cfg_dict"]

        eval_results[results_key]["sae_class"] = sae_config["architecture"]

        eval_results[results_key]["d_sae"] = get_d_sae_string(sae_config["d_sae"])

        if "sae_bench" in filename:
            eval_results[results_key]["train_tokens"] = get_sae_bench_train_tokens(filename)
        else:
            eval_results[results_key]["train_tokens"] = 1e-6

    return eval_results


def get_core_results(core_filenames: list[str]) -> dict:
    core_results = {}
    for filepath in core_filenames:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        with open(filepath, "r") as f:
            single_sae_results = json.load(f)

        l0 = single_sae_results["eval_result_metrics"]["sparsity"]["l0"]
        ce_score = single_sae_results["eval_result_metrics"]["model_performance_preservation"][
            "ce_loss_score"
        ]

        filename = os.path.basename(filepath)
        results_key = filename.replace("_eval_results.json", "")
        core_results[results_key] = {"l0": l0, "ce_loss_score": ce_score}
    return core_results


def find_eval_results_files(folders: list[str]) -> list[str]:
    """
    Recursively explores the given list of folder names and finds all file paths
    containing 'eval_results.json'. Returns a list of the full file paths.

    Args:
        folders (list[str]): A list of folder names to explore.

    Returns:
        list[str]: A list of full file paths containing 'eval_results.json'.
    """
    result_files = []

    for folder in folders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if "eval_results.json" in file:
                    result_files.append(os.path.join(root, file))

    return result_files


def update_trainer_markers_and_colors(
    results: dict[str, dict[str, Any]],
    trainer_markers: dict[str, str],
    trainer_colors: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    trainer_markers = deepcopy(trainer_markers)
    trainer_colors = deepcopy(trainer_colors)

    available_markers = list(set(trainer_markers.values()))
    available_colors = list(set(trainer_colors.values()))

    existing_trainers = set(trainer_markers.keys())
    all_trainers = {v["sae_class"] for v in results.values()}
    new_trainers = all_trainers - existing_trainers

    for trainer in new_trainers:
        trainer_markers[trainer] = available_markers.pop(0)
        trainer_colors[trainer] = available_colors.pop(0)

    return trainer_markers, trainer_colors


def plot_3var_graph(
    results: dict[str, dict[str, float]],
    title: str,
    custom_metric: str,
    xlims: Optional[tuple[float, float]] = None,
    ylims: Optional[tuple[float, float]] = None,
    colorbar_label: str = "Average Diff",
    output_filename: Optional[str] = None,
    legend_location: str = "lower right",
    x_axis_key: str = "l0",
    y_axis_key: str = "ce_loss_score",
    trainer_markers: Optional[dict[str, str]] = None,
):
    if not trainer_markers:
        trainer_markers = TRAINER_MARKERS

    trainer_markers, _ = update_trainer_markers_and_colors(results, trainer_markers, TRAINER_COLORS)

    # Extract data from results
    l0_values = [data[x_axis_key] for data in results.values()]
    frac_recovered_values = [data[y_axis_key] for data in results.values()]
    custom_metric_values = [data[custom_metric] for data in results.values()]

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a normalize object for color scaling
    norm = Normalize(vmin=min(custom_metric_values), vmax=max(custom_metric_values))

    handles, labels = [], []

    for trainer, marker in trainer_markers.items():
        # Filter data for this trainer
        trainer_data = {k: v for k, v in results.items() if v["sae_class"] == trainer}

        if not trainer_data:
            continue  # Skip this trainer if no data points

        l0_values = [data[x_axis_key] for data in trainer_data.values()]
        frac_recovered_values = [data[y_axis_key] for data in trainer_data.values()]
        custom_metric_values = [data[custom_metric] for data in trainer_data.values()]

        # Plot data points
        scatter = ax.scatter(
            l0_values,
            frac_recovered_values,
            c=custom_metric_values,
            cmap="viridis",
            marker=marker,
            s=100,
            label=trainer,
            norm=norm,
            edgecolor="black",
        )

        # custom legend stuff
        _handle, _ = scatter.legend_elements(prop="sizes")
        _handle[0].set_markeredgecolor("black")
        _handle[0].set_markerfacecolor("white")
        _handle[0].set_markersize(10)
        if marker == "d":
            _handle[0].set_markersize(13)
        handles += _handle

        if trainer in TRAINER_LABELS:
            trainer_label = TRAINER_LABELS[trainer]
        else:
            trainer_label = trainer.capitalize()

        labels.append(trainer_label)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, label=colorbar_label)

    # Set labels and title
    ax.set_xlabel("L0 (Sparsity)")
    ax.set_ylabel("Loss Recovered (Fidelity)")
    ax.set_title(title)

    ax.legend(handles, labels, loc=legend_location)

    # Set axis limits
    if xlims:
        ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)

    plt.tight_layout()

    # Save and show the plot
    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")
    plt.show()


def plot_interactive_3var_graph(
    results: dict[str, dict[str, float]],
    custom_color_metric: str,
    xlims: Optional[tuple[float, float]] = None,
    y_lims: Optional[tuple[float, float]] = None,
    output_filename: Optional[str] = None,
    x_axis_key: str = "l0",
    y_axis_key: str = "ce_loss_score",
    title: str = "",
):
    # Extract data from results
    ae_paths = list(results.keys())
    l0_values = [data[x_axis_key] for data in results.values()]
    frac_recovered_values = [data[y_axis_key] for data in results.values()]

    custom_metric_value = [data[custom_color_metric] for data in results.values()]

    dict_size = [data["d_sae"] for data in results.values()]

    # Create the scatter plot
    fig = go.Figure()

    # Add trace
    fig.add_trace(
        go.Scatter(
            x=l0_values,
            y=frac_recovered_values,
            mode="markers",
            marker=dict(
                size=10,
                color=custom_metric_value,  # Color points based on frac_recovered
                colorscale="Viridis",  # You can change this colorscale
                showscale=True,
            ),
            text=[
                f"AE Path: {ae}<br>L0: {l0:.4f}<br>Frac Recovered: {fr:.4f}<br>Custom Metric: {ad:.4f}<br>Dict Size: {d}"
                for ae, l0, fr, ad, d in zip(
                    ae_paths,
                    l0_values,
                    frac_recovered_values,
                    custom_metric_value,
                    dict_size,
                )
            ],
            hoverinfo="text",
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="L0 (Sparsity)",
        yaxis_title="Loss Recovered (Fidelity)",
        hovermode="closest",
    )

    # Set axis limits
    if xlims:
        fig.update_xaxes(range=xlims)
    if y_lims:
        fig.update_yaxes(range=y_lims)

    # Save and show the plot
    if output_filename:
        fig.write_html(output_filename)

    fig.show()


def plot_2var_graph(
    results: dict[str, dict[str, float]],
    custom_metric: str,
    title: str = "L0 vs Custom Metric",
    y_label: str = "Custom Metric",
    xlims: Optional[tuple[float, float]] = None,
    ylims: Optional[tuple[float, float]] = None,
    output_filename: Optional[str] = None,
    legend_location: str = "lower right",
    baseline_value: Optional[float] = None,
    baseline_label: Optional[str] = None,
    x_axis_key: str = "l0",
    trainer_markers: Optional[dict[str, str]] = None,
    trainer_colors: Optional[dict[str, str]] = None,
    return_fig: bool = False,
):
    if not trainer_markers:
        trainer_markers = TRAINER_MARKERS

    if not trainer_colors:
        trainer_colors = TRAINER_COLORS

    trainer_markers, trainer_colors = update_trainer_markers_and_colors(
        results, trainer_markers, trainer_colors
    )

    # Extract data from results
    l0_values = [data[x_axis_key] for data in results.values()]
    custom_metric_values = [data[custom_metric] for data in results.values()]

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    handles, labels = [], []

    for trainer, marker in trainer_markers.items():
        # Filter data for this trainer
        trainer_data = {k: v for k, v in results.items() if v["sae_class"] == trainer}

        if not trainer_data:
            continue  # Skip this trainer if no data points

        l0_values = [data[x_axis_key] for data in trainer_data.values()]
        custom_metric_values = [data[custom_metric] for data in trainer_data.values()]

        # Plot data points
        scatter = ax.scatter(
            l0_values,
            custom_metric_values,
            marker=marker,
            s=100,
            label=trainer,
            color=trainer_colors[trainer],
            edgecolor="black",
        )

        # Create custom legend handle with both marker and color
        legend_handle = plt.scatter(
            [], [], marker=marker, s=100, color=trainer_colors[trainer], edgecolor="black"
        )
        handles.append(legend_handle)

        if trainer in TRAINER_LABELS:
            trainer_label = TRAINER_LABELS[trainer]
        else:
            trainer_label = trainer.capitalize()
        labels.append(trainer_label)

    # Set labels and title
    ax.set_xlabel("L0 (Sparsity)")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if baseline_value is not None:
        ax.axhline(baseline_value, color="red", linestyle="--", label=baseline_label)
        labels.append(baseline_label)
        handles.append(Line2D([0], [0], color="red", linestyle="--", label=baseline_label))

    ax.legend(handles, labels, loc=legend_location)

    # Set axis limits
    if xlims:
        ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)

    plt.tight_layout()

    # Save and show the plot
    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")

    if return_fig:
        return fig

    plt.show()


def plot_2var_graph_dict_size(
    results: dict[str, dict[str, float]],
    custom_metric: str,
    title: str = "L0 vs Custom Metric",
    y_label: str = "Custom Metric",
    xlims: Optional[tuple[float, float]] = None,
    ylims: Optional[tuple[float, float]] = None,
    output_filename: Optional[str] = None,
    legend_location: str = "lower right",
    baseline_value: Optional[float] = None,
    baseline_label: Optional[str] = None,
    x_axis_key: str = "l0",
    return_fig: bool = False,
    trainer_markers: Optional[dict[str, str]] = None,
):
    if not trainer_markers:
        trainer_markers = TRAINER_MARKERS

    # Extract data
    l0_values = [data[x_axis_key] for data in results.values()]
    custom_metric_values = [data[custom_metric] for data in results.values()]
    dict_sizes = [data["d_sae"] for data in results.values()]

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define possible dict sizes and their markers
    possible_sizes = ["4k", "16k", "65k", "131k", "1M"]

    # Create color map with more exaggerated differences for 3 colors
    colors = [plt.cm.Reds(x) for x in [0.1, 0.5, 0.9]]  # Light, medium, dark red

    # Get unique dict sizes present in the data while preserving order from possible_sizes
    unique_sizes = [
        size for size in possible_sizes if size in set(v["d_sae"] for v in results.values())
    ]

    assert len(unique_sizes) <= len(colors), "Too many unique dictionary sizes for color map"

    size_to_color = {size: colors[i] for i, size in enumerate(unique_sizes)}

    # Iterate over each unique dictionary size
    handles, labels = [], []

    for dict_size in unique_sizes:
        # Filter data points for the current dictionary size
        size_data = {k: v for k, v in results.items() if v["d_sae"] == dict_size}

        # Get values for l0 and custom metric for this dictionary size
        l0_values = [data[x_axis_key] for data in size_data.values()]
        custom_metric_values = [data[custom_metric] for data in size_data.values()]
        sae_classes = [data["sae_class"] for data in size_data.values()]

        # Plot data points with the assigned marker and color
        for l0, metric, sae_class in zip(l0_values, custom_metric_values, sae_classes):
            marker = trainer_markers[sae_class]
            scatter = ax.scatter(
                l0,
                metric,
                marker=marker,
                s=100,
                color=size_to_color[dict_size],
                edgecolor="black",
            )

        # Collect legend handles and labels
        _handle = plt.scatter(
            [], [], marker="o", s=100, color=size_to_color[dict_size], edgecolor="black"
        )
        handles.append(_handle)
        labels.append(f"SAE Width: {dict_size}")

    # Set labels and title
    ax.set_xlabel("L0 (Sparsity)")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if baseline_value:
        ax.axhline(baseline_value, color="red", linestyle="--", label=baseline_label)
        labels.append(baseline_label)
        handles.append(Line2D([0], [0], color="red", linestyle="--", label=baseline_label))

    ax.legend(handles, labels, loc=legend_location)

    # Set axis limits
    if xlims:
        ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)

    plt.tight_layout()

    # Save and show the plot
    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")

    if return_fig:
        return fig
    plt.show()


def plot_steps_vs_average_diff(
    results_dict: dict,
    steps_key: str = "train_tokens",
    avg_diff_key: str = "average_diff",
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    output_filename: Optional[str] = None,
):
    # Initialize a defaultdict to store data for each trainer
    trainer_data = defaultdict(lambda: {"train_tokens": [], "metric_scores": []})

    all_steps = set()

    # Extract data from the dictionary
    for key, value in results_dict.items():
        # Extract trainer number from the key
        trainer = key.split("/")[-1].split("_")[1]  # Assuming format like "trainer_1_..."
        layer = key.split("/")[-2].split("_")[-2]

        if "topk_ctx128" in key:
            trainer_type = "TopK SAE"
        elif "standard_ctx128" in key:
            trainer_type = "Standard SAE"
        else:
            raise ValueError(f"Unknown trainer type in key: {key}")

        step = int(value[steps_key])
        avg_diff = value[avg_diff_key]

        trainer_key = f"{trainer_type} Layer {layer} Trainer {trainer}"

        trainer_data[trainer_key]["train_tokens"].append(step)
        trainer_data[trainer_key]["metric_scores"].append(avg_diff)
        all_steps.add(step)

    # Calculate average across all trainers
    average_trainer_data = {"train_tokens": [], "metric_scores": []}
    for step in sorted(all_steps):
        step_diffs = []
        for data in trainer_data.values():
            if step in data["train_tokens"]:
                idx = data["train_tokens"].index(step)
                step_diffs.append(data["metric_scores"][idx])
        if step_diffs:
            average_trainer_data["train_tokens"].append(step)
            average_trainer_data["metric_scores"].append(np.mean(step_diffs))

    # Add average_trainer_data to trainer_data
    trainer_data["Average"] = average_trainer_data

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot data for each trainer
    for trainer_key, data in trainer_data.items():
        steps = data["train_tokens"]
        metric_scores = data["metric_scores"]

        # Sort the data by steps to ensure proper ordering
        sorted_data = sorted(zip(steps, metric_scores))
        steps, metric_scores = zip(*sorted_data)

        # Find the maximum step value for this trainer
        max_step = max(steps)

        # Convert steps to percentages of max_step
        step_percentages = [step / max_step * 100 for step in steps]

        # Plot the line for this trainer
        if trainer_key == "Average":
            plt.plot(
                step_percentages,
                metric_scores,
                marker="o",
                label=trainer_key,
                linewidth=3,
                color="red",
                zorder=10,
            )  # Emphasized average line
        else:
            plt.plot(
                step_percentages,
                metric_scores,
                marker="o",
                label=trainer_key,
                alpha=0.3,
                linewidth=1,
            )  # More transparent individual lines

    # log scale
    # plt.xscale("log")

    # if not title:
    #     title = f'{steps_key.capitalize()} vs {avg_diff_key.replace("_", " ").capitalize()}'

    if not y_label:
        y_label = avg_diff_key.replace("_", " ").capitalize()

    plt.xlabel("Training Progess (%)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)  # More transparent grid

    if len(trainer_data) < 50 and False:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    # Adjust layout to prevent clipping of tick-labels
    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")

    # Show the plot
    plt.show()


def plot_correlation_heatmap(
    plotting_results: dict[str, dict[str, float]],
    metric_names: list[str],
    ae_names: Optional[list[str]] = None,
    title: str = "Metric Correlation Heatmap",
    output_filename: str = None,
    figsize: tuple = (12, 10),
    cmap: str = "coolwarm",
    annot: bool = True,
):
    # If ae_names is not provided, use all ae_names from plotting_results
    if ae_names is None:
        ae_names = list(plotting_results.keys())

    # If metric_names is not provided, use all metric names from the first ae_name
    # if metric_names is None:
    #     metric_names = list(plotting_results[ae_names[0]].keys())

    # Create a DataFrame from the plotting_results
    data = []
    for ae in ae_names:
        row = [plotting_results[ae].get(metric, np.nan) for metric in metric_names]
        data.append(row)

    df = pd.DataFrame(data, index=ae_names, columns=metric_names)

    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, vmin=-1, vmax=1, center=0)

    plt.title(title)
    plt.tight_layout()

    # Save the plot if output_filename is provided
    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")

    plt.show()


def plot_correlation_scatter(
    plotting_results: dict[str, dict[str, float]],
    metric_x: str,
    metric_y: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    ae_names: Optional[list[str]] = None,
    title: str = "Metric Comparison Scatter Plot",
    output_filename: Optional[str] = None,
    figsize: tuple = (10, 8),
):
    # If ae_names is not provided, use all ae_names from plotting_results
    if ae_names is None:
        ae_names = list(plotting_results.keys())

    # Extract x and y values for the specified metrics
    x_values = [plotting_results[ae].get(metric_x, float("nan")) for ae in ae_names]
    y_values = [plotting_results[ae].get(metric_y, float("nan")) for ae in ae_names]

    # Remove any NaN values
    valid_data = [
        (x, y, ae)
        for x, y, ae in zip(x_values, y_values, ae_names)
        if not (np.isnan(x) or np.isnan(y))
    ]
    if not valid_data:
        print("No valid data points after removing NaN values.")
        return

    x_values, y_values, valid_ae_names = zip(*valid_data)

    # Convert to numpy arrays
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Calculate correlation coefficients
    r, p_value = stats.pearsonr(x_values, y_values)
    r_squared = r**2

    # Create the scatter plot
    plt.figure(figsize=figsize)
    scatter = sns.scatterplot(x=x_values, y=y_values, label="SAE", color="blue")

    if x_label is None:
        x_label = metric_x
    if y_label is None:
        y_label = metric_y

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Add a trend line
    sns.regplot(x=x_values, y=y_values, scatter=False, color="red", label=f"r = {r:.4f}")

    plt.legend()

    plt.tight_layout()

    # Save the plot if output_filename is provided
    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")

    plt.show()

    # Print correlation coefficients
    print(f"Pearson correlation coefficient (r): {r:.4f}")
    print(f"Coefficient of determination (r²): {r_squared:.4f}")
    print(f"P-value: {p_value:.4f}")


def plot_training_steps(
    results_dict: dict,
    metric_key: str,
    steps_key: str = "train_tokens",
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    output_filename: Optional[str] = None,
    break_fraction: float = 0.15,  # Parameter to control break position
):
    # Initialize a defaultdict to store data for each trainer
    trainer_data = defaultdict(lambda: {"train_tokens": [], "metric_scores": []})
    all_steps = set()
    all_trainers = set()

    # Extract data from the dictionary
    for key, value in results_dict.items():
        trainer = key.split("_trainer_")[-1].split("_")[0]
        trainer_class = value["sae_class"]
        step = int(value[steps_key])
        metric_scores = value[metric_key]
        trainer_key = f"{trainer_class} Trainer {trainer}"

        trainer_data[trainer_key]["train_tokens"].append(step)
        trainer_data[trainer_key]["metric_scores"].append(metric_scores)
        # trainer_data[trainer_key]["l0"] = value["l0"] # currently not shown in plot
        trainer_data[trainer_key]["sae_class"] = trainer_class
        all_steps.add(step)
        all_trainers.add(trainer_class)

    # Calculate average across all trainers
    average_trainer_data = {"train_tokens": [], "metric_scores": []}
    for step in sorted(all_steps):
        step_diffs = [
            data["metric_scores"][data["train_tokens"].index(step)]
            for data in trainer_data.values()
            if step in data["train_tokens"]
        ]
        if step_diffs:
            average_trainer_data["train_tokens"].append(step)
            average_trainer_data["metric_scores"].append(np.mean(step_diffs))
    trainer_data["Average"] = average_trainer_data

    # Create the plot with broken axis
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(15, 6),
        gridspec_kw={"width_ratios": [break_fraction, 1 - break_fraction]},
    )
    fig.subplots_adjust(wspace=0.01)  # Adjust space between axes

    # Calculate break point based on data
    steps_break_point = min([s for s in all_steps if s > 0]) / 2
    break_point = steps_break_point  # / max(all_steps) * 100  # Convert to percentage

    for trainer_key, data in trainer_data.items():
        steps = data["train_tokens"]
        metric_scores = data["metric_scores"]

        if trainer_key == "Average":
            color, trainer_class = "black", "Average"
        elif data["sae_class"] == "standard" or data["sae_class"] == "Vanilla":
            color, trainer_class = "red", data["sae_class"]
        elif data["sae_class"] == "topk":
            color, trainer_class = "blue", data["sae_class"]
        else:
            raise ValueError(f"Trainer type not recognized for {trainer_key}")

        sorted_data = sorted(zip(steps, metric_scores))
        steps, metric_scores = zip(*sorted_data)

        ax1.plot(
            steps,
            metric_scores,
            marker="o",
            label=trainer_class,
            linewidth=4 if trainer_key == "Average" else 2,
            color=color,
            alpha=1 if trainer_key == "Average" else 0.3,
            zorder=10 if trainer_key == "Average" else 1,
        )
        ax2.plot(
            steps,
            metric_scores,
            marker="o",
            label=trainer_class,
            linewidth=4 if trainer_key == "Average" else 2,
            color=color,
            alpha=1 if trainer_key == "Average" else 0.3,
            zorder=10 if trainer_key == "Average" else 1,
        )

    # Set up the broken axis
    ax1.set_xlim(-break_point / 4, break_point)
    # ax2.set_xlim(break_point, 100)
    ax2.set_xscale("log")

    # Hide the spines between ax1 and ax2
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    # Add break lines
    d = 0.015  # Size of diagonal lines
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False, lw=4)

    ax1.plot((1, 1), (-d, +d), **kwargs)  # top-right vertical
    ax1.plot((1, 1), (1 - d, 1 + d), **kwargs)  # bottom-right vertical
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((0, 0), (-d, +d), **kwargs)  # top-left vertical
    ax2.plot((0, 0), (1 - d, 1 + d), **kwargs)  # bottom-left vertical

    # Set labels and title
    if not y_label:
        y_label = metric_key.replace("_", " ").capitalize()
    ax1.set_ylabel(y_label)
    fig.text(0.5, 0.01, "Training Steps", ha="center", va="center")
    fig.suptitle(title)

    # Adjust x-axis ticks
    # ax1.set_xticks([0])
    # ax1.set_xticklabels(['0%'])
    # ax2.set_xticks([0.1, 1, 10, 100])
    # ax2.set_xticklabels([f'0.1%', '1%', '10%', '100%'])

    # Add grid
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Add custom legend
    legend_elements = []
    legend_elements.append(Line2D([0], [0], color="black", lw=3, label="Average"))
    if "standard" in all_trainers or "Vanilla" in all_trainers:
        legend_elements.append(Line2D([0], [0], color="red", lw=3, label="Standard"))
    if "topk" in all_trainers:
        legend_elements.append(Line2D([0], [0], color="blue", lw=3, label="TopK"))
    ax2.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")

    plt.show()


def get_sae_class_archived(sae_cfg: dict, sae_release) -> str:
    """For results pre Jan 2025"""
    if "sae_bench" in sae_release and "panneal" in sae_release:
        return "p_anneal"

    if sae_cfg["activation_fn_str"] == "topk":
        return "topk"

    return sae_cfg["architecture"]


def get_sae_bench_train_tokens_archived(sae_release: str, sae_id: str) -> int:
    """For results pre Jan 2025.
    This is for SAE Bench internal use. The SAE cfg does not contain the number of training tokens, so we need to hardcode it."""

    if "sae_bench" not in sae_release:
        raise ValueError("This function is only for SAE Bench releases")

    if "pythia" in sae_release:
        batch_size = 4096
    else:
        batch_size = 2048

    if "step" not in sae_id:
        if "pythia" in sae_release:
            steps = 48828
        elif "2pow14" in sae_release:
            steps = 146484
        elif "2pow12" or "2pow16" in sae_release:
            steps = 97656
        else:
            raise ValueError(f"sae release {sae_release} not recognized")

        return steps * batch_size
    else:
        match = re.search(r"step_(\d+)", sae_id)
        if match:
            step = int(match.group(1))
            return step * batch_size
        else:
            raise ValueError("No step match found")
