# %%
# Plot 2x2 architectures
#
# for Gemma2-2B, Layer 12, 65k

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from huggingface_hub import snapshot_download

import sae_bench.sae_bench_utils.general_utils as general_utils
import sae_bench.sae_bench_utils.matryoshka_graphing_utils as graphing_utils

model_name = "gemma-2-2b"
layer = 12
selections = {
    "65k loss scaling": [
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_temp1_.*",
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_temp1000.*",
        r"matryoshka_0121_MatryoshkaBatchTopKTrainer_gemma_batch_topk_65k.*",
    ],
    "65k fixed groups": [
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_temp1000.*",
        r"matryoshka_0121_MatryoshkaBatchTopKTrainer_gemma_batch_topk_65k.*",
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_10_fixed_groups.*",
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_3_fixed_groups.*",
    ],
    "65k stop grads": [
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_temp1000.*",
        r"matryoshka_0121_MatryoshkaBatchTopKTrainer_gemma_batch_topk_65k.*",
        r"matryoshka_0121_MatryoshkaBatchTopKTrainer_gemma_stop_grads_65k.*",
    ],
    "16k stop_grads": [
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_stop_grads_v2.*",
        r"matryoshka_gemma-2-2b-16k-v2_BatchTopKTrainer_baseline.*",
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    ],
    "16k loss_scaling": [
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_temp_1_.*",
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_temp_2_.*",
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_temp_3_.*",
        r"matryoshka_gemma-2-2b-16k-v2_BatchTopKTrainer_baseline.*",
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    ],
    "16k_fixed_groups": [
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_3_fixed_groups.*",
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_10_fixed_groups.*",
        r"matryoshka_gemma-2-2b-16k-v2_BatchTopKTrainer_baseline.*",
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    ],
    "16k_group_sizes": [
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
        r"matryoshka_gemma-2-2b-16k-v2_BatchTopKTrainer_baseline.*",
        r"matryoshka_0121_MatryoshkaBatchTopKTrainer_gemma_sixteenths_16k.*",
    ],
    "16k_random_groups": [
        r"matryoshka_gemma-2-2b-16k-v2-random-10_matryoshka.*",
        r"matryoshka_gemma-2-2b-16k-v2-random-5_matryoshka.*",
        r"matryoshka_gemma-2-2b-16k-v2-random-3_matryoshka.*",
        r"matryoshka_gemma-2-2b-16k-v2_BatchTopKTrainer_baseline.*",
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    ],
    "16k_fixed_vs_random_3": [
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_3_fixed_groups.*",
        r"matryoshka_gemma-2-2b-16k-v2-random-3_matryoshka.*",
        r"matryoshka_gemma-2-2b-16k-v2_BatchTopKTrainer_baseline.*",
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    ],
    "16k_fixed_vs_random_5": [
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
        r"matryoshka_gemma-2-2b-16k-v2-random-5_matryoshka.*",
        r"matryoshka_gemma-2-2b-16k-v2_BatchTopKTrainer_baseline.*",
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    ],
    "16k_fixed_vs_random_10": [
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_10_fixed_groups.*",
        r"matryoshka_gemma-2-2b-16k-v2-random-10_matryoshka.*",
        r"matryoshka_gemma-2-2b-16k-v2_BatchTopKTrainer_baseline.*",
        r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    ],
}


def customize_class(sae_dict: dict) -> dict:
    for sae_name in sae_dict:
        if (
            "matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_stop_grads_v2_google_gemma"
            in sae_name
        ):
            sae_dict[sae_name]["sae_class"] = "Stop Gradients"
        elif "matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_temp_1" in sae_name:
            sae_dict[sae_name]["sae_class"] = "Weighted Temperature 1"
        elif "matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_temp_2" in sae_name:
            sae_dict[sae_name]["sae_class"] = "Weighted Temperature 2"
        elif "matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_temp_3" in sae_name:
            sae_dict[sae_name]["sae_class"] = "Weighted Temperature 3"
        elif "matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_3_fixed_groups" in sae_name:
            sae_dict[sae_name]["sae_class"] = "Three Fixed Groups"
        elif "matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_10_fixed_groups" in sae_name:
            sae_dict[sae_name]["sae_class"] = "Ten Fixed Groups"
        elif "matryoshka_0121_MatryoshkaBatchTopKTrainer_gemma_sixteenths_16k" in sae_name:
            sae_dict[sae_name]["sae_class"] = "Sixteenths Groups"
        elif "matryoshka_0121_MatryoshkaBatchTopKTrainer_gemma_stop_grads_65k" in sae_name:
            sae_dict[sae_name]["sae_class"] = "Stop Gradients"
        elif (
            "matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_3_fixed_groups" in sae_name
        ):
            sae_dict[sae_name]["sae_class"] = "Three Fixed Groups"
        elif (
            "matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_10_fixed_groups"
            in sae_name
        ):
            sae_dict[sae_name]["sae_class"] = "Ten Fixed Groups"
        elif "matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_temp1_" in sae_name:
            sae_dict[sae_name]["sae_class"] = "Weighted Matryoshka"
        elif (
            "matryoshka_gemma-2-2b-16k-v2-random-3_matryoshka_google_gemma-2-2b_random" in sae_name
        ):
            sae_dict[sae_name]["sae_class"] = "Three Random Groups"
        elif (
            "matryoshka_gemma-2-2b-16k-v2-random-5_matryoshka_google_gemma-2-2b_random" in sae_name
        ):
            sae_dict[sae_name]["sae_class"] = "Five Random Groups"
        elif (
            "matryoshka_gemma-2-2b-16k-v2-random-10_matryoshka_google_gemma-2-2b_random" in sae_name
        ):
            sae_dict[sae_name]["sae_class"] = "Ten Random Groups"
        elif "10_fixed" in sae_name:
            sae_dict[sae_name]["sae_class"] = "Ten Fixed Groups"
        # else:
        #     raise ValueError(f"Unknown class for {sae_name}")

    return sae_dict


results_folders = ["./matryoshka_ablations_0125"]


hf_repo_id = "adamkarvonen/matryoshka_ablation_results_0125"
local_dir = results_folders[0]
if not os.path.exists(local_dir) or True:
    # raise ValueError(f"Local directory {local_dir} does not exist")
    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        repo_id=hf_repo_id,
        local_dir=local_dir,
        # force_download=True,
        repo_type="dataset",
        ignore_patterns=[
            "*autointerp_with_generations*",
            "*core_with_feature_statistics*",
        ],  # These use significant disk space / download time and are not needed for graphing
    )

eval_types = [
    "core",
    "autointerp",
    "absorption",
    "sparse_probing",
    "scr",
    "tpp",
]
# TODO: Add other ks, try mean over multiple ks
ks_lookup = {
    "scr": 20,
    "tpp": 20,
    "sparse_probing": 1,
}

for selection_title in selections.keys():
    sae_regex_patterns = selections[selection_title]
    title_prefix = f"{selection_title} Layer {layer}\n"

    image_path = "./images_paper_2x2_matryoshka_ablations"
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    image_name = f"ablations_plot_2x4_{selection_title.replace(' ', '_').lower()}_layer_{layer}"

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
    legend_ax = fig.add_subplot(gs[3, 0])  # row 4, col 2 for legend
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

        if selection_title == "16k_random_groups":
            max_l0 = 200.0
        else:
            max_l0 = 400.0
        highlighted_class = None

        eval_results = customize_class(eval_results)

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
        )

        # After plotting, get the lines and labels directly from the axis
        lines, labels = axes[idx].get_legend_handles_labels()

        new_labels = []

        for label in labels:
            new_labels.append(graphing_utils.TRAINER_LABELS.get(label, label))

        labels = new_labels

        if idx == 5 and lines and labels:  # Only for the last plot
            # Create legend in the legend axis
            legend_ax.legend(
                lines, labels, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize="large"
            )

    plt.tight_layout()
    plt.savefig(os.path.join(image_path, image_name))
