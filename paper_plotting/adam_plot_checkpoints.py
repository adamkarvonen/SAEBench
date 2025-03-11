import itertools
import os

import sae_bench.sae_bench_utils.general_utils as general_utils
import sae_bench.sae_bench_utils.graphing_utils as graphing_utils

selections = {
    # "Gemma-Scope Gemma-2-2B Width Series": [
    #     r"gemma-scope-2b-pt-res_layer_{layer}_width_(16k|65k|1m)",
    # ],
    # "Gemma-Scope Gemma-2-9B Width Series": [
    #     r"gemma-scope-9b-pt-res_layer_{layer}_width_(16k|131k|1m)",
    # ],
    "SAE Bench Gemma-2-2B Checkpoint Series": [
        r"saebench_gemma-2-2b_width-2pow14_date-0108.*(_Standard|_TopK).*",
    ],
}

results_folders = ["./graphing_eval_results_0125"]

eval_type = "absorption"

eval_types = [
    "scr",
    "tpp",
    "sparse_probing",
    "absorption",
    "core",
    "autointerp",
    "unlearning",
    "ravel",
]

combinations = list(itertools.product(eval_types, selections.keys()))

ks_lookup = {
    "scr": [5, 10, 20, 50, 100, 500],
    "tpp": [5, 10, 20, 50, 100, 500],
    "sparse_probing": [1, 2, 5],
}

ks_lookup = {
    "scr": [20],
    "tpp": [20],
    "sparse_probing": [1],
}

image_path = "./images_checkpoints"


for eval_type, selection in combinations:
    if eval_type in ks_lookup:
        ks = ks_lookup[eval_type]
    else:
        ks = [-1]

    layers = [12]
    model_name = "gemma-2-2b"

    layer_ks_combinations = list(itertools.product(layers, ks))

    for layer, k in layer_ks_combinations:
        sae_regex_patterns = selections[selection]

        for i, pattern in enumerate(sae_regex_patterns):
            sae_regex_patterns[i] = pattern.format(layer=layer)

        prefix = f"{selection} Layer {layer}\n"

        image_base_folder = os.path.join(image_path, eval_type)

        if not os.path.exists(image_base_folder):
            os.makedirs(image_base_folder)

        image_base_name = os.path.join(
            image_base_folder, f"{selection.replace(' ', '_').lower()}_layer_{layer}"
        )

        if eval_type in ks_lookup:
            image_base_name = f"{image_base_name}_topk_{k}"

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

            if "batch" in sae:
                print(sae)
                raise ValueError("Stop here")
            if "batch" in eval_results[sae]["sae_class"]:
                print(sae)
                raise ValueError("Stop here")

        custom_metric, custom_metric_name = (
            graphing_utils.get_custom_metric_key_and_name(eval_type, k)
        )

        graphing_utils.plot_training_steps(
            eval_results,
            custom_metric,
            title="",
            y_label=custom_metric_name,
            output_filename=f"{image_base_name}_tokens_vs_diff.png",
        )
