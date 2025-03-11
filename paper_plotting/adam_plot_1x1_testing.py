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

selections = {
    # "SAE Bench Gemma-2-2B 16K Architecture Series": [
    #     r"saebench_gemma-2-2b_width-2pow14_date-0108(?!.*step).*",
    # ],
    "SAE Bench Gemma-2-2B 65K Architecture Series": [
        r"saebench_gemma-2-2b_width-2pow16_date-0108(?!.*step).*",
    ],
}


def generate_legend_image(
    eval_results,
    custom_metric,
    custom_metric_name,
    max_l0,
    highlighted_class,
    image_path,
):
    """
    Create a standalone legend image by drawing a dummy plot with the legend enabled,
    then extracting the legend and saving it as legend.png.
    """
    # Create a temporary figure and axis for the dummy plot
    fig_dummy, ax_dummy = plt.subplots(figsize=(4, 4))

    # Call your plotting function with a legend mode that ensures the legend is created.
    # We set connect_points to False here since we only care about the legend.
    graphing_utils.plot_2var_graph(
        eval_results,
        custom_metric,
        y_label=custom_metric_name,
        title="",  # no title needed
        passed_ax=ax_dummy,
        legend_mode="show_outside",  # ensure the legend is drawn
        connect_points=False,
        max_l0=max_l0,
        highlighted_class=highlighted_class,
    )

    # Retrieve the legend from the axis
    legend = ax_dummy.get_legend()
    if legend is None:
        print(
            "No legend found. Check that your plotting function is generating a legend."
        )
        plt.close(fig_dummy)
        return

    # Create a new figure just for the legend
    fig_legend = plt.figure(figsize=(3, 3))
    # Add the legend to this figure using the same handles and labels
    fig_legend.legend(
        handles=legend.legend_handles,
        labels=[text.get_text() for text in legend.get_texts()],
        loc="center",
        frameon=False,
    )
    # Hide the axes for a cleaner image
    fig_legend.gca().set_axis_off()
    plt.tight_layout()

    # Save the legend image
    legend_path = os.path.join(image_path, "legend.png")
    fig_legend.savefig(legend_path, bbox_inches="tight")

    # Clean up
    plt.close(fig_dummy)
    plt.close(fig_legend)

    print(f"Legend saved at {legend_path}")


def customize_class(sae_dict: dict) -> dict:
    for sae_name in sae_dict:
        if "95.0" in sae_name:
            sae_dict[sae_name]["sae_class"] = "KL Finetune"
        elif "0.0" in sae_name:
            sae_dict[sae_name]["sae_class"] = "E2E"
        elif "475000" in sae_name:
            sae_dict[sae_name]["sae_class"] = "MSE Checkpoint"

    return sae_dict


results_folders = [
    # "./eval_results_gemma_relu_saebench",
    # "./eval_results_gemma_topk_finetune",
    # "./eval_results_from_scratch",
    "./graphing_eval_results_0125",
    # "./eval_results_2pow16",
]

eval_types = [
    "core",
    # "autointerp",
    # "absorption",
    # "sparse_probing",
    # "scr",
    # "tpp",
    # "unlearning",
    # "ravel",
]
# TODO: Add other ks, try mean over multiple ks
ks_lookup = {
    "scr": [5, 10, 20, 50, 100],
    "tpp": [5, 10, 20, 50, 100],
    "sparse_probing": [1, 2, 5],
}

import random

random.seed(0)

layer = 0

legend_created = False

for selection_title in selections.keys():
    sae_regex_patterns = selections[selection_title]
    title_prefix = f"{selection_title}\n"

    image_path = "./images_paper_1x1_testing"
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
            fig = plt.figure(figsize=(16, 8))
            ax = fig.add_subplot(111)

            ax.set_yscale("log")

            # Plot
            title_2var = f""

            # if custom_metric == "mean_absorption_fraction_score":
            #     for sae_name in eval_results:
            #         score = eval_results[sae_name][custom_metric]
            #         eval_results[sae_name][custom_metric] = 1 - score

            max_l0 = None
            highlighted_class = None

            eval_results = customize_class(eval_results)
            legend_mode = "show_outside"
            # legend_mode = "hide"

            if not legend_created:
                generate_legend_image(
                    eval_results,
                    custom_metric,
                    custom_metric_name,
                    max_l0,
                    highlighted_class,
                    image_path,
                )
                legend_created = True

            graphing_utils.plot_2var_graph(
                eval_results,
                custom_metric,
                y_label=custom_metric_name,
                title=title_2var,
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
        ("absorption", None),
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
