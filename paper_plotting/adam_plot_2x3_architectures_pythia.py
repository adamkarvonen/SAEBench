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

model_name = "pythia-160m"
layer = 8
selection = {
    "SAE Bench Pythia-160M 4K Architecture Series": [
        r"saebench_pythia-160m-deduped_width-2pow12_date-0108.*",
    ],
    "SAE Bench Pythia-160M 16K Architecture Series": [
        r"saebench_pythia-160m-deduped_width-2pow14_date-0108.*",
    ],
    "SAE Bench Pythia-160M 65K Architecture Series": [
        r"saebench_pythia-160m-deduped_width-2pow16_date-0108.*",
    ],
}

highlight_matryoshka_options = [
    False,
    True,
]  # Whether to highlight matryoshka architectures in plots

for selection_title, highlight_matryoshka in itertools.product(
    selection.keys(), highlight_matryoshka_options
):
    assert "pythia" in selection_title.lower(), "This script is only for Pythia models"

    sae_regex_patterns = selection[selection_title]
    for i, pattern in enumerate(sae_regex_patterns):
        sae_regex_patterns[i] = pattern.format(layer=layer)

    results_folders = ["./graphing_eval_results_0125"]

    baseline_folder = results_folders[0]

    eval_types = [
        "core",
        "autointerp",
        "absorption",
        "sparse_probing",
        "scr",
        "tpp",
    ]
    title_prefix = f"{selection_title} Layer {layer}\n"

    # TODO: Add other ks, try mean over multiple ks
    ks_lookup = {
        "scr": 20,
        "tpp": 20,
        "sparse_probing": 1,
    }

    # # Naming the image save path
    if highlight_matryoshka:
        image_path = "./images_paper_2x2_matryoshka"
    else:
        image_path = "./images_paper_2x2"
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    image_name = f"plot_2x4_{selection_title.replace(' ', '_').lower()}_layer_{layer}"

    # Create a 4x2 subplot figure
    fig = plt.figure(figsize=(20, 24))

    # Create a special layout for 7 plots and a legend
    gs = fig.add_gridspec(3, 2)
    axes = []
    axes.append(fig.add_subplot(gs[0, 0]))  # row 1, col 1
    axes.append(fig.add_subplot(gs[0, 1]))  # row 1, col 2
    axes.append(fig.add_subplot(gs[1, 0]))  # row 2, col 1
    axes.append(fig.add_subplot(gs[1, 1]))  # row 2, col 2
    axes.append(fig.add_subplot(gs[2, 0]))  # row 3, col 1
    axes.append(fig.add_subplot(gs[2, 1]))  # row 3, col 2

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

        # Plot
        title_2var = f"{title_prefix}L0 vs {custom_metric_name}"
        title_2var = ""

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

        graphing_utils.plot_2var_graph(
            eval_results,
            custom_metric,
            y_label=custom_metric_name,
            title=title_2var,
            passed_ax=axes[idx],
            legend_mode="hide",
            connect_points=True,
            max_l0=max_l0,
            highlighted_class=highlighted_class,
        )

        # After plotting, get the lines and labels directly from the axis
        lines, labels = axes[idx].get_legend_handles_labels()

        new_labels = []

        for label in labels:
            new_labels.append(graphing_utils.TRAINER_LABELS.get(label, label))

        labels = new_labels

    # Add a new section to create a horizontal legend at the bottom of the figure
    # Collect all unique lines and labels from all subplots
    all_lines = []
    all_labels = []
    for ax in axes:
        lines, labels = ax.get_legend_handles_labels()
        for line, label in zip(lines, labels):
            # Convert label to display name if available
            display_label = graphing_utils.TRAINER_LABELS.get(label, label)
            # Check if this label is already in our collection
            if display_label not in all_labels:
                all_lines.append(line)
                all_labels.append(display_label)
    
    # Create a slim horizontal legend at the bottom
    fig.subplots_adjust(bottom=0.15)  # Make room for the legend at the bottom
    fig.legend(
        all_lines, 
        all_labels, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.02), 
        ncol=min(4, len(all_lines)),  # Adjust number of columns as needed
        fontsize='medium',
        frameon=True,
        borderaxespad=0.
    )

    plt.tight_layout(rect=(0, 0.1, 1, 1))  # Adjust the rect to make room for the legend
    plt.savefig(os.path.join(image_path, image_name))
