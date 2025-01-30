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

model_name = "gemma-2-2b"
layer = 12
selection = {
    "Gemma-Scope Gemma-2-2B Width Series": [
        r"(gemma-scope-2b-pt-res).*layer_{layer}(?!.*canonical)",
    ],
    "Gemma-Scope Gemma-2-9B Width Series": [
        r"(gemma-scope-9b-pt-res).*layer_{layer}(?!.*canonical)",
    ],
}


gemma_2b_layers = [5, 12, 19]
gemma_9b_layers = [9, 20, 31]

for selection_title in selection.keys():
    if "2-2B" in selection_title:
        layers = gemma_2b_layers
        include_baseline = True
    elif "2-9B" in selection_title:
        layers = gemma_9b_layers
        include_baseline = False
    else:
        raise ValueError("Unknown model")

    for layer in layers:
        sae_regex_patterns = selection[selection_title].copy()
        for i, pattern in enumerate(sae_regex_patterns):
            sae_regex_patterns[i] = pattern.format(layer=layer)

        results_folders = ["./graphing_eval_results_old"]

        baseline_folder = results_folders[0]

        eval_types = [
            "core",
            "autointerp",
            "absorption",
            "sparse_probing",
            "scr",
            "tpp",
            # "tpp",
            "unlearning",
        ]
        title_prefix = f"{selection_title} Layer {layer}\n"

        # TODO: Add other ks, try mean over multiple ks
        ks_lookup = {
            "scr": 10,
            "tpp": 10,
            "sparse_probing": 1,
        }

        baseline_type = "pca_sae"
        # # Naming the image save path
        image_path = "./images_paper_2x2"
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        image_name = f"plot_2x4_{selection_title.replace(' ', '_').lower()}_layer_{layer}"

        # Create a 4x2 subplot figure
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
        axes.append(fig.add_subplot(gs[3, 0]))  # row 4, col 1
        legend_ax = fig.add_subplot(gs[3, 1])  # row 4, col 2 for legend
        legend_ax.axis("off")  # Hide the axis

        # Store all lines and labels for later
        all_lines = []
        all_labels = []

        # Loop through eval types and create subplot for each
        for idx, eval_type in tqdm(enumerate(eval_types)):
            if eval_type == "unlearning" and "gemma" not in selection_title.lower():
                continue
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

            custom_metric, custom_metric_name = graphing_utils.get_custom_metric_key_and_name(
                eval_type, k
            )

            if custom_metric_name == "Mean Absorption Fraction Score":
                custom_metric_name = "Mean Absorption Score"
                custom_metric = "mean_absorption_score"

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

            # Plot
            title_2var = f"{title_prefix}L0 vs {custom_metric_name}"
            title_2var = None

            legend_mode = "hide"

            if custom_metric == "mean_absorption_score":
                for sae_name in eval_results:
                    score = eval_results[sae_name][custom_metric]
                    eval_results[sae_name][custom_metric] = 1 - score
                if baseline_value:
                    baseline_value = 1 - baseline_value

            graphing_utils.plot_2var_graph_dict_size(
                eval_results,
                custom_metric,
                y_label=custom_metric_name,
                title=title_2var,
                baseline_value=baseline_value,
                baseline_label=baseline_label,
                passed_ax=axes[idx],
                legend_mode=legend_mode,
                connect_points=True,
            )

            # After plotting, get the lines and labels directly from the axis
            lines, labels = axes[idx].get_legend_handles_labels()

            if idx == 6 and lines and labels:  # Only for the last plot
                # Create legend in the legend axis
                legend_ax.legend(
                    lines, labels, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize="large"
                )

        plt.tight_layout()
        plt.savefig(os.path.join(image_path, image_name))
