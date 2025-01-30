# %%
# Plot 2x2 architectures
#
# for Gemma2-2B, Layer 12, 65k

import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import sae_bench.sae_bench_utils.general_utils as general_utils
import sae_bench.sae_bench_utils.graphing_utils as graphing_utils

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
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    ],
    "SAE Bench Gemma-2-2B 65K Architecture Series": [
        r"saebench_gemma-2-2b_width-2pow16_date-0108(?!.*step).*",
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_temp1000.*",
    ],
    # "SAE Bench Pythia-70M SAE Type Series": [
    #     r"sae_bench_pythia70m_sweep.*_ctx128_.*blocks\.({layer})\.hook_resid_post__trainer_.*",
    # ],
}

for selection_title in selection:
    sae_regex_patterns = selection[selection_title]
    for i, pattern in enumerate(sae_regex_patterns):
        sae_regex_patterns[i] = pattern.format(layer=layer)

    results_folders = ["./graphing_eval_results_0122", "./matroyshka_eval_results_0117"]

    baseline_folder = results_folders[0]

    eval_types = [
        "tpp",
        "sparse_probing",
        "unlearning",
    ]
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
    image_path = "./images_paper_2x2"
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    image_name = f"plot_2x2_{selection_title.replace(' ', '_').lower()}_layer_{layer}_tpp_sparse_unlearning"  # Create a 2x2 subplot figure

    # Create a 2x2 subplot figure
    fig = plt.figure(figsize=(20, 12))

    # Create a special layout for 3 plots and a legend
    gs = fig.add_gridspec(2, 2)
    axes = []
    axes.append(fig.add_subplot(gs[0, 0]))  # top left
    axes.append(fig.add_subplot(gs[0, 1]))  # top right
    axes.append(fig.add_subplot(gs[1, 0]))  # bottom left
    legend_ax = fig.add_subplot(gs[1, 1])  # bottom right for legend
    legend_ax.axis("off")  # Hide the axis

    # Store all lines and labels for later
    all_lines = []
    all_labels = []

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
            if model_name != "gemma-2-9b":
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

        # Set legend_mode to "show" only for last plot
        legend_mode = "show_outside" if idx == 2 else "hide"
        legend_mode = "hide"

        if custom_metric == "mean_absorption_fraction_score":
            for sae_name in eval_results:
                score = eval_results[sae_name][custom_metric]
                eval_results[sae_name][custom_metric] = 1 - score
            baseline_value = 1 - baseline_value

        graphing_utils.plot_2var_graph(
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

        new_labels = []

        for label in labels:
            new_labels.append(graphing_utils.TRAINER_LABELS.get(label, label))

        labels = new_labels

        if idx == 2 and lines and labels:  # Only for the last plot
            # Create legend in the legend axis
            legend_ax.legend(
                lines, labels, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize="large"
            )

    plt.tight_layout()
    plt.savefig(os.path.join(image_path, image_name))
