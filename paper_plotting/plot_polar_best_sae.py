# %%
# Plot 2x2 architectures
# 
# for Gemma2-2B, Layer 12, 65k

import os
import matplotlib.pyplot as plt
import numpy as np

import sae_bench.sae_bench_utils.general_utils as general_utils
import sae_bench.sae_bench_utils.graphing_utils as graphing_utils
from collections import defaultdict
from sae_bench.sae_bench_utils.graphing_utils import TRAINER_MARKERS, TRAINER_COLORS

model_name = "gemma-2-2b"
layer = 12

sae_regex_patterns_65k = [
    r"saebench_gemma-2-2b_width-2pow16_date-0108(?!.*step).*",
    r"65k_temp1000.*" # matryoshka 65k
]

selection_title = "SAE Bench Gemma-2-2B Width Diff Series"

results_folders = ["./graphing_eval_results_0122", "./matroyshka_eval_results_0117"]

baseline_folder = results_folders[0]

eval_types = [
    "core",
    "autointerp",
    "absorption",
    "scr",
    "tpp",
    "unlearning",
    "sparse_probing",
]

arch_types = {
    "standard": "_Standard",
    "jumprelu": "_JumpRelu",
    "topk": "_TopK",
    "p_anneal": "_PAnneal",
    "gated": "_GatedSAE",
    "batch_topk": "_BatchTopK",
    "matryoshka_batch_topk": "_MatryoshkaBatchTopK",
}

title_prefix = f"{selection_title} Layer {layer}\n"

# TODO: Add other ks, try mean over multiple ks
ks_lookup = {
    "scr": 10,
    "tpp": 10,
    "sparse_probing": 1,
}

baseline_type = "pca_sae"
include_baseline = False


def convert_to_1_minus_score(eval_results, custom_metric, baseline_value=None):
    for sae_name in eval_results:
        score = eval_results[sae_name][custom_metric]
        eval_results[sae_name][custom_metric] = 1 - score
    if baseline_value:
        baseline_value = 1 - baseline_value
    return eval_results, baseline_value

# # Naming the image save path
image_path = "./images_paper"
if not os.path.exists(image_path):
    os.makedirs(image_path)
image_name = f"plot_best_sparsity_{selection_title.replace(' ', '_').lower()}_layer_{layer}"

# Loop through eval types and create subplot for each
sorted_scores = {}
sorted_sparsities = {}
for idx, eval_type in enumerate(eval_types):
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

    filtered_eval_filenames_65k = general_utils.filter_with_regex(
        eval_filenames, sae_regex_patterns_65k
    )
    filtered_core_filenames_65k = general_utils.filter_with_regex(
        core_filenames, sae_regex_patterns_65k
    )

    eval_results_65k = graphing_utils.get_eval_results(filtered_eval_filenames_65k)
    core_results_65k = graphing_utils.get_core_results(filtered_core_filenames_65k)
    
    # Add core results to eval results
    for sae in eval_results_65k:
        eval_results_65k[sae].update(core_results_65k[sae])

    custom_metric, custom_metric_name = graphing_utils.get_custom_metric_key_and_name(eval_type, k)

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


    print(f'custom_metric: {custom_metric}')

    # Convert to 1 - score
    if custom_metric == "absorption":
        eval_results_65k, baseline_value = convert_to_1_minus_score(eval_results_65k, custom_metric, baseline_value)

    # sort results by arch_type and trainer
    sorted_scores[eval_type] = defaultdict(list)
    sorted_sparsities[eval_type] = defaultdict(list)
    for arch_type in arch_types.keys():
        # match all keys in eval_results_65k that contain the arch_type
        matching_keys = [key for key in eval_results_65k if arch_types[arch_type] in key]
        assert len(matching_keys) == 6, f"Expected 6 keys for arch_type {arch_type}, but got {len(matching_keys)}"
        for trainer_id in range(6):
            matching_trainer_key = [key for key in matching_keys if f"trainer_{trainer_id}" in key]
            assert len(matching_trainer_key) == 1, f"Expected 1 key for trainer {trainer_id}, but got {len(matching_trainer_key)}"
            matching_trainer_key = matching_trainer_key[0]
            metric_score = eval_results_65k[matching_trainer_key][custom_metric]
            sorted_scores[eval_type][arch_type].append(metric_score)
            sorted_sparsities[eval_type][arch_type].append(eval_results_65k[matching_trainer_key]["l0"])

"""
Example for sorted_scores
{'core': defaultdict(<class 'list'>, {'standard': [0.9967105263157895, 0.9950657894736842, 0.993421052631579, 0.9884868421052632, 0.9851973684210527, 0.975328947368421], 'jump_relu': [0.975328947368421, 0.9851973684210527, 0.9917763157894737, 0.9950657894736842, 0.9967105263157895, 0.9983552631578947], 'topk': [0.9769736842105263, 0.9868421052631579, 0.9917763157894737, 0.9950657894736842, 0.9967105263157895, 0.9983552631578947], 'batch_topk': [0.9786184210526315, 0.9868421052631579, 0.9917763157894737, 0.9950657894736842, 0.9967105263157895, 0.9983552631578947], 'p_anneal': [0.9983552631578947, 0.9950657894736842, 0.993421052631579, 0.9901315789473685, 0.9868421052631579, 0.9819078947368421], 'matryoshka_batch_topk': [0.9703947368421053, 0.9835526315789473, 0.9901315789473685, 0.993421052631579, 0.9967105263157895, 0.9983552631578947], 'gated_sae': [1.0, 0.9983552631578947, 0.9967105263157895, 0.9950657894736842, 0.9901315789473685, 0.9868421052631579]}), 'autointerp': defaultdict(<class 'list'>, {'standard': [0.7919347286224365, 0.8075096607208252, 0.8309140801429749, 0.8475329279899597, 0.8475759029388428, 0.8649587035179138], 'jump_relu': [0.8895031213760376, 0.872667670249939, 0.8654568791389465, 0.8596034049987793, 0.8568515181541443, 0.841472327709198], 'topk': [0.8699482679367065, 0.8769230246543884, 0.8795981407165527, 0.872011661529541, 0.8619096279144287, 0.8757379651069641], 'batch_topk': [0.8820684552192688, 0.8864583969116211, 0.8779608607292175, 0.8756966590881348, 0.8717816472053528, 0.8613831996917725], 'p_anneal': [0.7867143154144287, 0.8257142901420593, 0.8279285430908203, 0.847112774848938, 0.856423556804657, 0.8487352728843689], 'matryoshka_batch_topk': [0.838951587677002, 0.8654502034187317, 0.8732141852378845, 0.8743153214454651, 0.8621331453323364, 0.8504949808120728], 'gated_sae': [0.8227351903915405, 0.8230255246162415, 0.8262194991111755, 0.8332365155220032, 0.8477309346199036, 0.8611152172088623]}), 'absorption': defaultdict(<class 'list'>, {'standard': [0.35621240115111547, 0.39142577386522914, 0.41163700384282514, 0.4803653814104993, 0.5019036205197468, 0.4908199501191287], 'jump_relu': [0.5242425646145429, 0.49480236651039733, 0.34314115092084496, 0.15057060272177722, 0.08776290891749315, 0.06958956167129975], 'topk': [0.49006402119553844, 0.4901183659374549, 0.3463697076385002, 0.14103043509215804, 0.04734311064592846, 0.05669425553281834], 'batch_topk': [0.565391642382547, 0.5118451969624243, 0.25486289586443966, 0.07672715474974164, 0.062100846472795956, 0.12466222745959571], 'p_anneal': [0.12032020278171812, 0.04815889005548093, 0.04830015378452148, 0.2170900003507592, 0.38392941016685034, 0.40393008368151057], 'matryoshka_batch_topk': [0.24018728075669346, 0.06251758565433889, 0.037431080622489406, 0.06011385609811043, 0.10375120549184169, 0.18727472705406786], 'gated_sae': [0.1900568104612271, 0.1703959986148162, 0.21657126780765878, 0.3903925795996471, 0.5634058517247414, 0.5738053735462033]}), 'scr': defaultdict(<class 'list'>, {'standard': [0.14247619541303724, 0.1322597763382832, 0.1411871888990302, 0.13879724659879122, 0.12313563364352617, 0.11166515385438797], 'jump_relu': [0.16904193123332642, 0.20189024127225283, 0.20058198936833324, 0.19785954313418164, 0.21814504640514387, 0.24293656636017213], 'topk': [0.1750385836033523, 0.2034137415516456, 0.2265061975883445, 0.23297210207649782, 0.2432079819643102, 0.1444073780055609], 'batch_topk': [0.18667359411786785, 0.1867551073003066, 0.2423896287009217, 0.24967404475394162, 0.24327554382782854, 0.198082360857876], 'p_anneal': [0.2760125728806488, 0.2657187583049849, 0.24614071070810215, 0.1934430157899953, 0.209136140778353, 0.1972963329572508], 'matryoshka_batch_topk': [0.27629129630132204, 0.33313318592320534, 0.3408630820051422, 0.3579795913319417, 0.30762075304110087, 0.2803938207686883], 'gated_sae': [0.14574655399945938, 0.15502279189649928, 0.20333639308163587, 0.22173742430453108, 0.2047871760679777, 0.1966383586084079]}), 'tpp': defaultdict(<class 'list'>, {'standard': [0.032374992966651917, 0.034199999272823335, 0.03480000644922257, 0.027799998223781586, 0.02352500855922699, 0.015075004100799561], 'jump_relu': [0.05260000079870224, 0.08732499629259109, 0.09617500454187393, 0.14205001741647721, 0.1525000140070915, 0.2459499970078468], 'topk': [0.04477500319480896, 0.07855000644922255, 0.1141000121831894, 0.14702500849962236, 0.29650001376867297, 0.31947503238916397], 'batch_topk': [0.04807500094175339, 0.07625000476837158, 0.11697501242160797, 0.13140000700950621, 0.32002500742673873, 0.3298750340938568], 'p_anneal': [0.2915750175714493, 0.1351250007748604, 0.11264999955892563, 0.0901500016450882, 0.07662499696016312, 0.06105000227689743], 'matryoshka_batch_topk': [0.05604999661445617, 0.10612501055002213, 0.19960001558065416, 0.274600014090538, 0.3772000283002853, 0.3241250365972519], 'gated_sae': [0.20192501097917556, 0.13859999924898148, 0.12477501034736634, 0.11057500243186952, 0.10422500222921371, 0.08337499350309371]}), 'unlearning': defaultdict(<class 'list'>, {'standard': [0.04878050088882446, 0.06003749370574951, 0.030018746852874756, 0.20262664556503296, 0.17823642492294312, 0.04127579927444458], 'jump_relu': [0.08442777395248413, 0.052532851696014404, 0.10506564378738403, 0.08067542314529419, 0.022514045238494873, 0.052532851696014404], 'topk': [0.09193247556686401, 0.11257034540176392, 0.106941819190979, 0.09193247556686401, 0.0675421953201294, 0.056285202503204346], 'batch_topk': [0.06941837072372437, 0.07504689693450928, 0.054409027099609375, 0.1257035732269287, 0.09943711757659912, 0.04690432548522949], 'p_anneal': [0.0337710976600647, 0.0863039493560791, 0.07504689693450928, 0.19512194395065308, 0.09005630016326904, 0.12195122241973877], 'matryoshka_batch_topk': [0.09193247556686401, 0.18574106693267822, 0.03564727306365967, 0.052532851696014404, 0.08442777395248413, 0.026266396045684814], 'gated_sae': [0.03752344846725464, 0.030018746852874756, 0.056285202503204346, 0.1332082748413086, 0.054409027099609375, 0.08818012475967407]}), 'sparse_probing': defaultdict(<class 'list'>, {'standard': [0.78811875, 0.79645, 0.7856000000000001, 0.79291875, 0.7804375, 0.76226875], 'jump_relu': [0.73244375, 0.74340625, 0.7309375000000001, 0.73920625, 0.7638562499999999, 0.7443624999999999], 'topk': [0.7370125000000001, 0.7356375, 0.7524500000000001, 0.7460125000000001, 0.7221624999999999, 0.7696875], 'batch_topk': [0.7397937499999999, 0.7436375, 0.7392875, 0.7728312499999999, 0.71235625, 0.77760625], 'p_anneal': [0.7861812499999999, 0.76044375, 0.741425, 0.7578437499999999, 0.7672875, 0.7366562499999999], 'matryoshka_batch_topk': [0.7341187499999999, 0.7971874999999999, 0.8013499999999999, 0.7862, 0.7560937500000001, 0.78125625], 'gated_sae': [0.704425, 0.7161875, 0.74008125, 0.73871875, 0.7470875, 0.75066875]})}
"""



# Select the best sparsity for each arch_type

selected_arch_types = ['matryoshka_batch_topk', 'batch_topk']

cumulative_scores = {}
for arch_type in selected_arch_types:
    cumulative_scores[arch_type] = np.zeros(6)

for eval_type in sorted_scores.keys():
    for arch_type in selected_arch_types:
        cumulative_scores[arch_type] += sorted_scores[eval_type][arch_type]

top_score_idxs = {}
for arch_type in selected_arch_types:
    top_score_idxs[arch_type] = np.argmax(cumulative_scores[arch_type])

print(top_score_idxs)

top_idx_scores = {}
for arch_type in selected_arch_types:
    top_idx_scores[arch_type] = {}
    for eval_type in sorted_scores.keys():
        top_idx_scores[arch_type][eval_type] = sorted_scores[eval_type][arch_type][top_score_idxs[arch_type]]


# Normalize scores for each metric to [0,1] range based on min/max of all scores
normalized_scores = {}
for eval_type in eval_types:
    # Get min and max scores across all architectures and all sparsities
    all_scores = []
    for arch_type in arch_types.keys():
        all_scores.extend(sorted_scores[eval_type][arch_type])
    min_score = min(all_scores)
    max_score = max(all_scores)
    score_range = max_score - min_score
    
    # Skip normalization if all scores are identical
    if score_range == 0:
        normalized_scores[eval_type] = {arch_type: 1.0 for arch_type in arch_types.keys()}
        continue
        
    # Normalize scores for each architecture
    normalized_scores[eval_type] = {}
    for arch_type in selected_arch_types:
        normalized_scores[eval_type][arch_type] = (top_idx_scores[arch_type][eval_type] - min_score) / score_range


# Create a polar plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='polar')

# Get number of metrics and angle for each metric
num_metrics = len(eval_types)
angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
angles += angles[:1] # Complete the circle

# Plot data for each architecture type
for arch_type in selected_arch_types:
    # Get values for this architecture
    values = [normalized_scores[eval_type][arch_type] for eval_type in eval_types]
    values += values[:1] # Complete the circle
    
    # Plot
    ax.plot(angles, values, 
            color=TRAINER_COLORS.get(arch_type, 'black'),
            linewidth=2,
            label=arch_type)
    ax.fill(angles, values, 
            color=TRAINER_COLORS.get(arch_type, 'black'),
            alpha=0.25)

# Fix axis to go in the right order and start at 12 o'clock
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines for each angle and label
ax.set_xticks(angles[:-1])
ax.set_xticklabels(eval_types)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.title("Performance Across Evaluation Metrics")
plt.tight_layout()

# Save figure
plt.savefig(os.path.join(image_path, "polar_best_sae.png"), bbox_inches="tight")
plt.close()

