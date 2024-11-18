# SAE Bench

## Table of Contents
- [Installation](#installation)
- [Overview](#overview)
- [Running Evaluations](#running-evaluations)
- [Custom SAE Usage](#custom-sae-usage)
- [Training Your Own SAEs](#training-your-own-saes)
- [Graphing Results](#graphing-results)

### Installation
Set up a virtual environment with python >= 3.10.

```
git clone https://github.com/adamkarvonen/SAEBench.git
cd SAEBench
pip install -e .
```

All evals can be ran with current batch sizes on Gemma-2-2B on a 24GB VRAM GPU (e.g. a RTX 3090). By default, some evals cache LLM activations, which can require up to 100 GB of disk space. However, this can be disabled.

## Overview

SAE Bench is a comprehensive suite of 8 evaluations for Sparse Autoencoder (SAE) models:
- **Feature Absorption**
- **AutoInterp**
- **L0 / Loss Recovered**
- **RAVEL**
- **Spurious Correlation Removal (SCR)**
- **Targeted Probe Pertubation (TPP)**
- **Sparse Probing**
- **Unlearning** (requires access to the WMDP dataset, see README)

### Supported Models and SAEs
- **SAE Lens Pretrained Models**: Supports evaluations on any SAE Lens pretrained model.
- **Custom SAEs**: Supports any general SAE object with `encode()` and `decode()` methods (see [Custom SAE Usage](#custom-sae-usage)).

## Running Evaluations
Each evaluation has an example command located in its respective `main.py` file. Here's how to run a sparse probing evaluation on a single SAE Bench Pythia-70M SAE:

```
python evals/sparse_probing/main.py \
    --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
    --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
    --model_name pythia-70m-deduped
```

The results will be saved to the eval_results/sparse_probing directory.

We use regex patterns to select SAE Lens SAEs. For more examples of regex patterns, refer to `sae_regex_selection.ipynb`.

Every eval folder contains an `eval_config.py`, which contains all relevant hyperparamters for that evaluation. The values are currently set to the default recommended values.

For a tutorial of using SAE Lens SAEs, including calculating L0 and Loss Recovered and getting a set of tokens from The Pile, refer to this notebook: https://github.com/jbloomAus/SAELens/blob/main/tutorials/basic_loading_and_analysing.ipynb

## Custom SAE Usage

Our goal is to have first class support for custom SAEs as the field is rapidly evolving. Our evaluations can run on any SAE object with encode(), decode(), and a few config values. For example custom SAE implementations and more info, refer to the `custom_saes/README.md`.

There are two ways to evaluate custom SAEs:

1. **Using Evaluation Templates**: 
   - Use the secondary `if __name__ == "__main__"` block in each `main.py`
   - Results are saved in SAE Bench format for easy visualization
   - Compatible with provided plotting tools

2. **Direct Function Calls**:
   - Use `run_eval_single_sae()` in each `main.py`
   - Simpler interface requiring only model, SAE, and config values
   - Graphing will require manual formatting

The script `run_all_evals_custom_saes()` will run approach 1 on all SAE Bench evals. We currently have a suite of SAE Bench SAEs on layers 3 and 4 of Pythia-70M and layers 5, 12, and 19 of Gemma-2-2B, each trained on 200M tokens with checkpoints at various points. These SAEs can serve as baselines for any new custom SAEs. We also have baseline eval results, saved at TODO.

## Training Your Own SAEs

You can replicate the training of our SAEs using scripts provided [here](https://github.com/canrager/dictionary_training/), or implement your own SAE, or make a change to one of our SAE implementations. Once you train your new version, you can benchmark against our existing SAEs for a true apples to apples comparison.

## Graphing Results

To graph the results, refer to `graphing.ipynb`, which can graph the generated SAE Bench data. Note that many graphs plot SAEs by L0 and / or Loss Recovered. To obtain these scores, run `evals/core/main.py`.