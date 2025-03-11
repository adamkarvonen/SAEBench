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
    "SAE Bench Gemma-2-2B 16K Architecture Series": [
        r"saebench_gemma-2-2b_width-2pow14_date-0108(?!.*step).*",
    ],
    "SAE Bench Gemma-2-2B 65K Architecture Series": [
        r"saebench_gemma-2-2b_width-2pow16_date-0108(?!.*step).*",
    ],
    # "SAE Bench Pythia-70M SAE Type Series": [
    #     r"sae_bench_pythia70m_sweep.*_ctx128_.*blocks\.({layer})\.hook_resid_post__trainer_.*",
    # ],
}

highlight_matryoshka_options = [
    # True,
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

    eval_types = [
        "core",
        "autointerp",
        "absorption",
        "scr",
        "sparse_probing",
        "ravel",
    ]
    title_prefix = f"{selection_title} Layer {layer}\n"

    # TODO: Add other ks, try mean over multiple ks
    ks_lookup = {
        "scr": 20,
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
    axes.append(fig.add_subplot(gs[1, 2]))  # row 3, col 2
    # Remove the dedicated legend subplot
    # legend_ax = fig.add_subplot(gs[1, 2])  # row 4, col 2 for legend
    # legend_ax.axis("off")  # Hide the axis

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

        custom_metric, custom_metric_name = graphing_utils.get_custom_metric_key_and_name(
            eval_type, k
        )

        if include_baseline:
            if model_name == "gemma-2-2b":
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

        if custom_metric == "mean_absorption_fraction_score":
            for sae_name in eval_results:
                score = eval_results[sae_name][custom_metric]
                eval_results[sae_name][custom_metric] = 1 - score
            if baseline_value:
                baseline_value = 1 - baseline_value

        if highlight_matryoshka:
            max_l0 = 400.0
            highlighted_class = "matryoshka_batch_topk"
        else:
            max_l0 = None
            highlighted_class = None

        ax = graphing_utils.plot_2var_graph(
            eval_results,
            custom_metric,
            y_label=custom_metric_name,
            title=title_2var,
            baseline_value=baseline_value,
            baseline_label=baseline_label,
            passed_ax=axes[idx],
            legend_mode=legend_mode,
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

        # Remove the legend creation in the last plot's section
        # if idx == 4 and lines and labels:  # Only for the last plot
        #     # Create legend in the legend axis
        #     legend_ax.legend(
        #         lines, labels, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize="large"
        #     )

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
    print(f"Saving image to {os.path.join(image_path, image_name)}")
    plt.savefig(os.path.join(image_path, image_name))
