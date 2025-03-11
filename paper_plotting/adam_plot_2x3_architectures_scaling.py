# %%
# Plot 2x2 architectures
#
# for Gemma2-2B, Layer 12, 65k

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy import stats

import sae_bench.sae_bench_utils.general_utils as general_utils
import sae_bench.sae_bench_utils.matryoshka_graphing_utils as graphing_utils

model_name = "gemma-2-2b"
layer = 12


def aggregate_eval_results_by_width(
    eval_results: dict, custom_metric: str, l0_range: tuple[float, float] = (10, 200)
) -> dict[str, dict]:
    """
    Aggregate eval results by dictionary width, averaging over L0 values in specified range.

    Args:
        eval_results: Dictionary of SAE evaluation results
        custom_metric: Metric key to aggregate
        l0_range: Inclusive range of L0 values to include (min, max)

    Returns:
        Dictionary of aggregated results with keys:
        {
            "sae_class": str,
            "d_sae": int,  # Parsed dictionary size
            custom_metric: float,  # Mean metric value
            "count": int  # Number of points averaged
        }
    """

    def parse_d_sae(d_str: str) -> int:
        """Convert d_sae string (e.g. '4k', '16k') to integer"""
        d_str = d_str.lower().replace(" ", "")
        if "k" in d_str:
            return int(float(d_str.replace("k", "")) * 1000)
        elif "m" in d_str:
            return int(float(d_str.replace("m", "")) * 1_000_000)
        else:
            return int(d_str)

    aggregated = {}

    for sae_id, sae_data in eval_results.items():
        # Filter by L0 range
        l0 = sae_data.get("l0")
        if l0 is None or not (l0_range[0] <= l0 <= l0_range[1]):
            continue

        # Parse dictionary size
        try:
            d_sae = parse_d_sae(sae_data["d_sae"])
        except (KeyError, ValueError):
            continue

        # Get metric value
        metric_value = sae_data.get(custom_metric)
        if metric_value is None:
            continue

        # Create group key
        sae_class = sae_data["sae_class"]
        group_key = (sae_class, d_sae)

        # Initialize group if needed
        if group_key not in aggregated:
            aggregated[group_key] = {
                "sae_class": sae_class,
                "d_sae": d_sae,
                "values": [],
                "count": 0,
            }

        # Add to group
        aggregated[group_key]["values"].append(metric_value)
        aggregated[group_key]["count"] += 1

    # Compute means and format output
    output = {}
    for (sae_class, d_sae), group in aggregated.items():
        if len(group["values"]) == 0:
            continue

        output_key = f"{sae_class}_{d_sae}"
        output[output_key] = {
            "sae_class": sae_class,
            "d_sae": d_sae,
            custom_metric: np.mean(group["values"]),
            "count": group["count"],
        }

    return output


def plot_2var_graph(
    results: dict[str, dict[str, float]],
    custom_metric: str,
    title: str = "L0 vs Custom Metric",
    y_label: str = "Custom Metric",
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    output_filename: str | None = None,
    baseline_value: float | None = None,
    baseline_label: str | None = None,
    x_axis_key: str = "l0",
    trainer_markers: dict[str, str] | None = None,
    trainer_colors: dict[str, str] | None = None,
    return_fig: bool = False,
    connect_points: bool = False,
    passed_ax: plt.Axes | None = None,
    legend_mode: str = "show_outside",  # show_outside, show_inside, hide
    show_grid: bool = True,
    bold_x0: bool = False,
    max_l0: float | None = None,
    highlighted_class: str | None = None,
):
    if not trainer_markers:
        trainer_markers = graphing_utils.TRAINER_MARKERS

    if not trainer_colors:
        trainer_colors = graphing_utils.TRAINER_COLORS

    for k, v in results.items():
        if v["sae_class"] == "matroyshka_batch_topk":
            raise ValueError(
                "Matroyshka found in results, please rename to matryoshka_batch_topk"
            )

    # Filter results if max_l0 is provided
    if max_l0 is not None:
        results = {k: v for k, v in results.items() if v[x_axis_key] <= max_l0}

    # Verify highlighted_class exists if provided
    if highlighted_class is not None:
        assert any(v["sae_class"] == highlighted_class for v in results.values()), (
            f"Highlighted class {highlighted_class} not found in results {results}"
        )

    trainer_markers, trainer_colors = graphing_utils.update_trainer_markers_and_colors(
        results, trainer_markers, trainer_colors
    )

    # Create the scatter plot with extra width for legend
    if passed_ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        ax = passed_ax
        assert return_fig is False, "Cannot return fig if ax is provided"

    handles, labels = [], []

    for trainer, marker in trainer_markers.items():
        # Filter data for this trainer
        trainer_data = {k: v for k, v in results.items() if v["sae_class"] == trainer}

        if not trainer_data:
            continue  # Skip this trainer if no data points

        l0_values = [data[x_axis_key] for data in trainer_data.values()]
        custom_metric_values = [data[custom_metric] for data in trainer_data.values()]

        # Sort points by l0 values for proper line connection
        if connect_points and len(l0_values) > 1:
            points = sorted(zip(l0_values, custom_metric_values))
            l0_values = [p[0] for p in points]
            custom_metric_values = [p[1] for p in points]

            # Add connecting line with transparency based on highlighted_class
            line_alpha = (
                1.0
                if trainer == highlighted_class
                else 0.2
                if highlighted_class
                else 0.5
            )
            line_width = 2.0 if trainer == highlighted_class else 1.0

            ax.plot(
                l0_values,
                custom_metric_values,
                color=trainer_colors[trainer],
                linestyle="-",
                alpha=line_alpha,
                linewidth=line_width,
                zorder=1,  # Ensure lines are plotted behind points
            )

        # Plot data points
        point_alpha = (
            1.0 if trainer == highlighted_class else 0.5 if highlighted_class else 1.0
        )
        ax.scatter(
            l0_values,
            custom_metric_values,
            marker=marker,
            s=100,
            label=trainer,
            color=trainer_colors[trainer],
            edgecolor="black",
            alpha=point_alpha,
            zorder=2,  # Ensure points are plotted on top of lines
        )

        # Create custom legend handle with both marker and color
        legend_handle = plt.scatter(
            [],
            [],
            marker=marker,
            s=100,
            color=trainer_colors[trainer],
            edgecolor="black",
            alpha=point_alpha,
        )
        handles.append(legend_handle)

        if trainer in graphing_utils.TRAINER_LABELS:
            trainer_label = graphing_utils.TRAINER_LABELS[trainer]
        else:
            trainer_label = trainer.capitalize()
        labels.append(trainer_label)

    # Set labels and title
    ax.set_xlabel("Dictionary Size")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # x log
    ax.set_xscale("log")

    custom_ticks = [4000, 16000, 65000]
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(["4k", "16k", "65k"])

    # Add grid if requested
    if show_grid:
        ax.grid(True, alpha=0.3)  # Semi-transparent grid

    if baseline_value is not None:
        ax.axhline(baseline_value, color="red", linestyle="--", label=baseline_label)
        labels.append(baseline_label)
        handles.append(
            Line2D([0], [0], color="red", linestyle="--", label=baseline_label)
        )

    # Place legend outside the plot on the right
    if legend_mode == "show_outside":
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    elif legend_mode == "show_inside":
        ax.legend(handles, labels)
    elif legend_mode == "hide":
        pass
    else:
        raise ValueError(
            f"Invalid legend mode: {legend_mode}. Must be one of: show_outside, show_inside, hide"
        )

    # Set axis limits
    if xlims:
        ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)
    if bold_x0:
        ax.axhline(0, color="black", linestyle="-", linewidth=2, zorder=-1)

    plt.tight_layout()

    # Save and show the plot
    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")

    if return_fig:
        return fig
    elif passed_ax is None:
        plt.show()


selection = {
    # "Gemma-Scope Gemma-2-2B Width Series": [
    #     r"gemma-scope-2b-pt-res_layer_{layer}_width_(16k|65k|1m)",
    # ],
    # "Gemma-Scope Gemma-2-9B Width Series": [
    #     r"gemma-scope-9b-pt-res_layer_{layer}_width_(16k|131k|1m)",
    # ],
    # "SAE Bench Gemma-2-2B Width Series": [
    #     r"saebench_gemma-2-2b_width-2pow12_date-0108.*(batch|Batch).*(?!.*step)(?!.*Standard).*",
    #     r"saebench_gemma-2-2b_width-2pow14_date-0108.*(batch|Batch).*(?!.*step)(?!.*Standard).*",
    #     r"saebench_gemma-2-2b_width-2pow16_date-0108.*(batch|Batch).*(?!.*step)(?!.*Standard).*",
    # ],
    # "SAE Bench Gemma-2-2B Matryoshka Width Series": [
    #     r"matroyshka_gemma-2-2b-16k-v2_MatroyshkaBatchTopKTrainer_notemp.*",
    #     r"matroyshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_temp1000.*",
    # ],
    # "SAE Bench Gemma-2-2B 4K Width Series": [
    #     r"saebench_gemma-2-2b_width-2pow12_date-0108(?!.*step).*",
    # ],
    "SAE Bench Gemma-2-2B Scaling Width Series": [
        r"saebench_gemma-2-2b_width-2pow12_date-0108(?!.*step).*",
        r"saebench_gemma-2-2b_width-2pow14_date-0108(?!.*step).*",
        r"saebench_gemma-2-2b_width-2pow16_date-0108(?!.*step).*",
    ],
    # "SAE Bench Pythia-70M SAE Type Series": [
    #     r"sae_bench_pythia70m_sweep.*_ctx128_.*blocks\.({layer})\.hook_resid_post__trainer_.*",
    # ],
}

highlight_matryoshka_options = [
    True,
    False,
]  # Whether to highlight matryoshka architectures in plots

for selection_title, highlight_matryoshka in itertools.product(
    selection.keys(), highlight_matryoshka_options
):
    sae_regex_patterns = selection[selection_title]

    sae_regex_patterns = selection[selection_title]
    for i, pattern in enumerate(sae_regex_patterns):
        sae_regex_patterns[i] = pattern.format(layer=layer)

    results_folders = ["./graphing_eval_results_0125"]

    baseline_folder = results_folders[0]

    eval_types = ["core", "autointerp", "absorption", "scr", "sparse_probing"]
    eval_types = ["core", "autointerp", "absorption", "scr", "ravel"]
    title_prefix = f"{selection_title} Layer {layer}\n"

    # TODO: Add other ks, try mean over multiple ks
    ks_lookup = {
        "scr": 20,
        "tpp": 20,
        "sparse_probing": 1,
    }

    baseline_type = "pca_sae"
    include_baseline = True

    # # Naming the image save path
    if highlight_matryoshka:
        image_path = "./images_paper_2x2_matryoshka"
    else:
        image_path = "./images_paper_2x2"
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    image_name = f"plot_2x3_{selection_title.replace(' ', '_').lower()}_layer_{layer}"

    # %%

    fig = plt.figure(figsize=(20, 12))
    # Create a 2x3 subplot figure
    gs = fig.add_gridspec(2, 3)
    axes = []
    axes.append(fig.add_subplot(gs[0, 0]))  # row 1, col 1
    axes.append(fig.add_subplot(gs[0, 1]))  # row 1, col 2
    axes.append(fig.add_subplot(gs[0, 2]))  # row 2, col 1
    axes.append(fig.add_subplot(gs[1, 0]))  # row 2, col 2
    axes.append(fig.add_subplot(gs[1, 1]))  # row 3, col 1
    legend_ax = fig.add_subplot(gs[1, 2])  # row 4, col 2 for legend
    legend_ax.axis("off")  # Hide the axis

    # Loop through eval types and create subplot for each
    for idx, eval_type in tqdm(enumerate(eval_types)):
        if eval_type in ks_lookup:
            k = ks_lookup[eval_type]
        else:
            k = -1

        # Load data
        eval_folders = []
        core_folders = []

        for results_folder in results_folders:
            eval_folders.append(f"{results_folder}/{eval_type}")
            core_folders.append(f"{results_folder}/core")

        eval_filenames = graphing_utils.find_eval_results_files(eval_folders)
        core_filenames = graphing_utils.find_eval_results_files(core_folders)

        filtered_eval_filenames = general_utils.filter_with_regex(
            eval_filenames, sae_regex_patterns
        )
        filtered_core_filenames = general_utils.filter_with_regex(
            core_filenames, sae_regex_patterns
        )

        eval_results = graphing_utils.get_eval_results(filtered_eval_filenames)
        core_results = graphing_utils.get_core_results(filtered_core_filenames)

        for sae in eval_results:
            eval_results[sae].update(core_results[sae])

        custom_metric, custom_metric_name = (
            graphing_utils.get_custom_metric_key_and_name(eval_type, k)
        )

        if include_baseline:
            if model_name == "gemma-2-2b":
                baseline_sae_path = (
                    f"{model_name}_layer_{layer}_pca_sae_custom_sae_eval_results.json"
                )
                baseline_sae_path = os.path.join(
                    baseline_folder, eval_type, baseline_sae_path
                )
                baseline_label = "PCA Baseline"
        else:
            baseline_sae_path = None
            baseline_label = None
            baseline_sae_path = None

        if baseline_sae_path:
            baseline_results = graphing_utils.get_eval_results([baseline_sae_path])

            baseline_filename = os.path.basename(baseline_sae_path)
            baseline_results_key = baseline_filename.replace("_eval_results.json", "")

            core_baseline_filename = baseline_sae_path.replace(eval_type, "core")

            baseline_results[baseline_results_key].update(
                graphing_utils.get_core_results([core_baseline_filename])[
                    baseline_results_key
                ]
            )

            baseline_value = baseline_results[baseline_results_key][custom_metric]
            assert baseline_label, "Please provide a label for the baseline"
        else:
            baseline_value = None
            assert baseline_label is None, (
                "Please do not provide a label for the baseline"
            )

        # Plot
        title_2var = f"{title_prefix}L0 vs {custom_metric_name}"
        title_2var = None

        legend_mode = "hide"

        if custom_metric == "mean_absorption_fraction_score":
            for sae_name in eval_results:
                score = eval_results[sae_name][custom_metric]
                eval_results[sae_name][custom_metric] = 1 - score
            if baseline_value:
                baseline_value = 1 - baseline_value

        if highlight_matryoshka:
            highlighted_class = "matryoshka_batch_topk"
        else:
            highlighted_class = None

        aggregated_results = aggregate_eval_results_by_width(
            eval_results, custom_metric, l0_range=(40, 200)
        )

        ax = plot_2var_graph(
            aggregated_results,
            custom_metric,
            y_label=custom_metric_name,
            title=title_2var,
            baseline_value=baseline_value,
            baseline_label=baseline_label,
            passed_ax=axes[idx],
            legend_mode=legend_mode,
            connect_points=True,
            highlighted_class=highlighted_class,
            x_axis_key="d_sae",
        )

        # After plotting, get the lines and labels directly from the axis
        lines, labels = axes[idx].get_legend_handles_labels()

        new_labels = []

        for label in labels:
            new_labels.append(graphing_utils.TRAINER_LABELS.get(label, label))

        labels = new_labels

        if idx == 4 and lines and labels:  # Only for the last plot
            # Create legend in the legend axis
            legend_ax.legend(
                lines, labels, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize="large"
            )

    plt.tight_layout()
    print(f"Saving image to {os.path.join(image_path, image_name)}")
    plt.savefig(os.path.join(image_path, image_name))
