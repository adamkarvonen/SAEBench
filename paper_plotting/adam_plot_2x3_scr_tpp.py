# %%
# Plot 2x2 architectures
#
# for Gemma2-2B, Layer 12, 65k

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools


import sae_bench.sae_bench_utils.general_utils as general_utils
import sae_bench.sae_bench_utils.matryoshka_graphing_utils as graphing_utils

include_baseline = True

model_name = "gemma-2-2b"
layer = 12
selection = {
    "SAE Bench Gemma-2-2B 65K Architecture Series": [
        r"saebench_gemma-2-2b_width-2pow16_date-0108(?!.*step).*",
    ],
}

highlight_matryoshka_options = [
    False,
]  # Whether to highlight matryoshka architectures in plots

for selection_title, highlight_matryoshka in itertools.product(
    selection.keys(), highlight_matryoshka_options
):
    sae_regex_patterns = selection[selection_title]
    for i, pattern in enumerate(sae_regex_patterns):
        sae_regex_patterns[i] = pattern.format(layer=layer)

    results_folders = ["./graphing_eval_results_0125"]

    baseline_folder = results_folders[0]

    eval_types = [
        "scr",
        "tpp",
        "sparse_probing",
    ]
    title_prefix = f"{selection_title} Layer {layer}\n"

    # TODO: Add other ks, try mean over multiple ks
    ks_lookup = {
        "scr": [5, 10, 20, 50, 100, 500],
        "tpp": [5, 10, 20, 50, 100, 500],
        "sparse_probing": [1, 2, 5],
    }

    # # Naming the image save path
    if highlight_matryoshka:
        raise ValueError("Highlighting matryoshka not supported")
    else:
        image_path = "./images_paper_2x2"
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # Store all lines and labels for later
    all_lines = []
    all_labels = []

    # Loop through eval types and create subplot for each
    for _, eval_type in tqdm(enumerate(eval_types)):
        # Create a 4x2 subplot figure

        if eval_type != "sparse_probing":
            final_idx = 5
            fig = plt.figure(figsize=(20, 24))

            # Create a special layout for 7 plots and a legend
            gs = fig.add_gridspec(4, 2)
            axes = []
            axes.append(fig.add_subplot(gs[0, 0]))  # row 1, col 1
            axes.append(fig.add_subplot(gs[0, 1]))  # row 1, col 2
            axes.append(fig.add_subplot(gs[1, 0]))  # row 2, col 1
            axes.append(fig.add_subplot(gs[1, 1]))  # row 2, col 2
            axes.append(fig.add_subplot(gs[2, 0]))  # row 3, col 1
            axes.append(fig.add_subplot(gs[2, 1]))  # row 3, col 2
            legend_ax = fig.add_subplot(gs[3, 0])  # row 4, col 2 for legend
        else:
            final_idx = 2
            # create a 2x2 subplot figure
            fig = plt.figure(figsize=(12, 12))
            # Create a special layout for 3 plots and a legend
            gs = fig.add_gridspec(2, 2)
            axes = []
            axes.append(fig.add_subplot(gs[0, 0]))  # row 1, col 1
            axes.append(fig.add_subplot(gs[0, 1]))  # row 1, col 2
            axes.append(fig.add_subplot(gs[1, 0]))  # row 2, col 1
            # axes.append(fig.add_subplot(gs[1, 1]))  # row 2, col 2
            legend_ax = fig.add_subplot(gs[1, 1])  # row 4, col 2 for legend
        legend_ax.axis("off")  # Hide the axis

        image_name = (
            f"plot_2x4_{selection_title.replace(' ', '_').lower()}_layer_{layer}_{eval_type}"
        )

        if eval_type in ks_lookup:
            ks = ks_lookup[eval_type]
        else:
            raise ValueError(f"Need to add k for {eval_type}")

        for idx, k in enumerate(ks):
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

            custom_metric, custom_metric_name = graphing_utils.get_custom_metric_key_and_name(
                eval_type, k
            )

            if eval_type == "sparse_probing":
                custom_metric_name = custom_metric_name.replace("Sparse ", "")

            # Plot
            title_2var = f"{title_prefix}L0 vs {custom_metric_name}"
            title_2var = None

            # Set legend_mode to "show" only for last plot
            legend_mode = "show_outside" if idx == 2 else "hide"
            legend_mode = "hide"

            if custom_metric == "mean_absorption_fraction_score":
                for sae_name in eval_results:
                    score = eval_results[sae_name][custom_metric]
                    eval_results[sae_name][custom_metric] = 1 - score

            if highlight_matryoshka:
                max_l0 = 400.0
                highlighted_class = "matryoshka_batch_topk"
            else:
                max_l0 = None
                highlighted_class = None

            if include_baseline:
                baseline_sae_path = (
                    f"{model_name}_layer_{layer}_pca_sae_custom_sae_eval_results.json"
                )
                baseline_sae_path = os.path.join(baseline_folder, eval_type, baseline_sae_path)
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
                    graphing_utils.get_core_results([core_baseline_filename])[baseline_results_key]
                )

                baseline_value = baseline_results[baseline_results_key][custom_metric]
                assert baseline_label, "Please provide a label for the baseline"
            else:
                baseline_value = None
                assert baseline_label is None, "Please do not provide a label for the baseline"

            graphing_utils.plot_2var_graph(
                eval_results,
                custom_metric,
                y_label=custom_metric_name,
                title=title_2var,
                passed_ax=axes[idx],
                legend_mode=legend_mode,
                connect_points=True,
                max_l0=max_l0,
                highlighted_class=highlighted_class,
                baseline_value=baseline_value,
                baseline_label=baseline_label,
            )

            # After plotting, get the lines and labels directly from the axis
            lines, labels = axes[idx].get_legend_handles_labels()

            new_labels = []

            for label in labels:
                new_labels.append(graphing_utils.TRAINER_LABELS.get(label, label))

            labels = new_labels

            if idx == final_idx and lines and labels:  # Only for the last plot
                # Create legend in the legend axis
                legend_ax.legend(
                    lines, labels, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize="large"
                )

        plt.tight_layout()
        plt.savefig(os.path.join(image_path, image_name))
