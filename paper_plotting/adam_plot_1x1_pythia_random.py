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
import sae_bench.sae_bench_utils.random_graphing_utils as graphing_utils

model_name = "pythia-1b"
layer = 10
selections = {
    # "65k loss scaling": [
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_temp1_.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_temp1000.*",
    #     r"matryoshka_0121_MatryoshkaBatchTopKTrainer_gemma_batch_topk_65k.*",
    # ],
    # "65k fixed groups": [
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_temp1000.*",
    #     r"matryoshka_0121_MatryoshkaBatchTopKTrainer_gemma_batch_topk_65k.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_10_fixed_groups.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_3_fixed_groups.*",
    # ],
    # "65k stop grads": [
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_65k_temp1000.*",
    #     r"matryoshka_0121_MatryoshkaBatchTopKTrainer_gemma_batch_topk_65k.*",
    #     r"matryoshka_0121_MatryoshkaBatchTopKTrainer_gemma_stop_grads_65k.*",
    # ],
    # "16k stop_grads": [
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_stop_grads_v2.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_BatchTopKTrainer_baseline.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    # ],
    # "16k loss_scaling": [
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_temp_1_.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_temp_2_.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_temp_3_.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_BatchTopKTrainer_baseline.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    # ],
    # "16k_fixed_groups": [
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_3_fixed_groups.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_10_fixed_groups.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_BatchTopKTrainer_baseline.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    # ],
    "random": [
        r".*",
    ],
    # "16k_random_groups": [
    #     r"matryoshka_gemma-2-2b-16k-v2-random-10_matryoshka.*",
    #     r"matryoshka_gemma-2-2b-16k-v2-random-5_matryoshka.*",
    #     r"matryoshka_gemma-2-2b-16k-v2-random-3_matryoshka.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_BatchTopKTrainer_baseline.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    # ],
    # "16k_fixed_vs_random_3": [
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_3_fixed_groups.*",
    #     r"matryoshka_gemma-2-2b-16k-v2-random-3_matryoshka.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_BatchTopKTrainer_baseline.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    # ],
    # "16k_fixed_vs_random_5": [
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    #     r"matryoshka_gemma-2-2b-16k-v2-random-5_matryoshka.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_BatchTopKTrainer_baseline.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    # ],
    # "16k_fixed_vs_random_10": [
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_10_fixed_groups.*",
    #     r"matryoshka_gemma-2-2b-16k-v2-random-10_matryoshka.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_BatchTopKTrainer_baseline.*",
    #     r"matryoshka_gemma-2-2b-16k-v2_MatryoshkaBatchTopKTrainer_notemp.*",
    # ],
}


def customize_class(sae_dict: dict) -> dict:
    for sae_name in sae_dict:
        if "trained" in sae_name:
            sae_dict[sae_name]["sae_class"] = "Pythia-1B (Trained)"
        elif "random" in sae_name:
            sae_dict[sae_name]["sae_class"] = "Pythia-1B Step0 (Random)"

    return sae_dict


results_folders = [
    "./random_results/random_pythia-1b_eval_results",
    "./random_results/trained_pythia-1b_eval_results",
    # "./residual_results/residual_pythia-1b_eval_results",
]

baseline_folder = "./random_results/residual_pythia-1b_eval_results"
baseline_type = "residual_stream"

# hf_repo_id = "adamkarvonen/matryoshka_ablation_results_0125"
local_dir = results_folders[0]
# if not os.path.exists(local_dir) or True:
#     # raise ValueError(f"Local directory {local_dir} does not exist")
#     os.makedirs(local_dir, exist_ok=True)

#     snapshot_download(
#         repo_id=hf_repo_id,
#         local_dir=local_dir,
#         # force_download=True,
#         repo_type="dataset",
#         ignore_patterns=[
#             "*autointerp_with_generations*",
#             "*core_with_feature_statistics*",
#         ],  # These use significant disk space / download time and are not needed for graphing
#     )
eval_types = [
    "core",
    "autointerp",
    # "absorption",
    "sparse_probing",
    "scr",
    "tpp",
]
# TODO: Add other ks, try mean over multiple ks
ks_lookup = {
    "scr": [5, 10, 20, 50, 100],
    "tpp": [5, 10, 20, 50, 100],
    "sparse_probing": [1, 2, 5],
}

import random

random.seed(0)


for selection_title in selections.keys():
    sae_regex_patterns = selections[selection_title]
    title_prefix = f"{selection_title} Layer {layer}\n"

    image_path = "./images_paper_1x1_pythia"
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # Loop through eval types and create subplot for each
    for eval_type in tqdm(eval_types):
        if eval_type in ks_lookup:
            ks = ks_lookup[eval_type]
        else:
            ks = [-1]

        for k in ks:
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

            # Create new figure for each plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)

            baseline_sae_path = (
                f"{model_name}_layer_{layer}_identity_sae_custom_sae_eval_results.json"
            )
            baseline_sae_path = os.path.join(
                baseline_folder, eval_type, baseline_sae_path
            )
            baseline_label = "Residual Stream Baseline"

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

            # Plot
            title_2var = f""

            # if custom_metric == "mean_absorption_fraction_score":
            #     for sae_name in eval_results:
            #         score = eval_results[sae_name][custom_metric]
            #         eval_results[sae_name][custom_metric] = 1 - score

            max_l0 = 400.0
            highlighted_class = None

            eval_results = customize_class(eval_results)
            legend_mode = "show_outside"
            legend_mode = "hide"

            graphing_utils.plot_2var_graph(
                eval_results,
                custom_metric,
                y_label=custom_metric_name,
                title=title_2var,
                baseline_value=baseline_value,
                baseline_label=baseline_label,
                passed_ax=ax,
                legend_mode=legend_mode,
                connect_points=True,
                max_l0=max_l0,
                highlighted_class=highlighted_class,
            )

            # Save individual plot
            image_name = f"plot_1x1_{selection_title.replace(' ', '_').lower()}_layer_{layer}_{eval_type}"
            if k != -1:
                image_name += f"_k{k}"
            plt.tight_layout()
            plt.savefig(os.path.join(image_path, f"{image_name}.png"))
            plt.close()


def create_combined_plot(image_path, selection_title, layer):
    """
    Create a 2x3 grid of plots with legend for a given selection and layer.
    To be called after generating all individual plots.
    """
    from PIL import Image
    import os

    # Define which evaluation types to include
    plot_types = [
        ("core", None),
        ("autointerp", None),
        ("sparse_probing", "k1"),
        ("tpp", "k20"),
        ("scr", "k20"),
    ]

    # Generate file names based on the pattern from your plot generation code
    image_files = []
    for eval_type, k_suffix in plot_types:
        base_name = f"plot_1x1_{selection_title.replace(' ', '_').lower()}_layer_{layer}_{eval_type}"
        if k_suffix:
            base_name += f"_{k_suffix}"
        base_name += ".png"
        if os.path.exists(os.path.join(image_path, base_name)):
            image_files.append(base_name)

    # Ensure we have exactly 5 plots plus legend
    if len(image_files) < 5:
        print(f"Warning: Only found {len(image_files)} plots, expected 5")
        return

    # Open images and legend
    images = [
        Image.open(os.path.join(image_path, f)).convert("RGB") for f in image_files[:5]
    ]
    legend = Image.open(os.path.join(image_path, "legend.png")).convert("RGB")

    # Get dimensions from the first image
    img_width, img_height = images[0].size

    # Create the combined image
    grid_width = img_width * 3
    grid_height = img_height * 2
    grid_image = Image.new("RGB", (grid_width, grid_height), "white")

    # Place the plots
    for idx, img in enumerate(images):
        row = idx // 3
        col = idx % 3
        grid_image.paste(img, (col * img_width, row * img_height))

    # Place the legend in bottom right, vertically centered
    legend_width, legend_height = legend.size
    vertical_offset = img_height + (img_height - legend_height) // 2
    grid_image.paste(legend, (2 * img_width, vertical_offset))

    # Save the combined plot
    output_name = (
        f"combined_plot_{selection_title.replace(' ', '_').lower()}_layer_{layer}.png"
    )
    grid_image.save(os.path.join(image_path, output_name))
    print(f"Created combined plot: {output_name}")


# Add this at the end of your main script, after the plot generation loops:
for selection_title in selections.keys():
    create_combined_plot(image_path, selection_title, layer)

# %%
