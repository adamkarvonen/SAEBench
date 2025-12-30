This eval implements the k-sparse probing benchmark from the paper [Are Sparse Autoencoders Useful? A Case Study in Sparse Probing](https://arxiv.org/pdf/2502.16681), which runs k-sparse probing on over 140 datasets. This eval wraps the standalone `sae-probes` python package, putting results in SAEBench format. For further customization of the eval, refer to the [sae-probes documentation](https://github.com/sae-probes/sae-probes).

## Usage

### Basic Usage

Run the eval from the command line:

```bash
python sae_bench/evals/sparse_probing_sae_probes/main.py \
    --model_name gpt2 \
    --sae_regex_pattern "gpt2-small-res-jb" \
    --sae_block_pattern "blocks.4.hook_resid_pre"
```

### Configuration Options

- `--model_name`: Name of the model (e.g., `gpt2`, `pythia-70m`)
- `--sae_regex_pattern`: Regex pattern to match SAE releases
- `--sae_block_pattern`: Regex pattern to match SAE hook points
- `--ks`: List of k values for sparse probing (default: `[1, 2, 5]`)
  - Example: `--ks 1 2 5 10 20`
- `--reg_type`: Regularization type for probing (`l1` or `l2`, default: `l1`)
- `--setting`: Data balance setting (`normal`, `scarcity`, or `imbalance`, default: `normal`)
- `--binarize`: Whether to binarize probe targets (flag, default: False)
- `--results_path`: Directory where sae-probes writes intermediate JSONs (default: `artifacts/sparse_probing_sae_probes`)
- `--model_cache_path`: Optional directory to cache model activations for faster re-runs (default: `artifacts/sparse_probing_sae_probes--model_acts_cache`)
- `--output_folder`: Where to save SAEBench output files (default: `eval_results/sparse_probing_sae_probes`)
- `--force_rerun`: Force re-running the eval even if results exist (flag)

### Programmatic Usage

```python
from sae_bench.evals.sparse_probing_sae_probes.eval_config import SparseProbingSaeProbesEvalConfig
from sae_bench.evals.sparse_probing_sae_probes.main import run_eval
from sae_lens import SAE

# Configure the eval
config = SparseProbingSaeProbesEvalConfig(
    model_name="gpt2",
    dataset_names=["118_us_state_CA", "119_us_state_TX"],  # Subset of datasets
    ks=[1, 2, 5, 10],  # Custom k values
    include_llm_baseline=True,  # Compare against LLM residual stream baseline
    results_path="artifacts/sparse_probing_sae_probes",
    model_cache_path="cache/models",
)

# Load your SAE
sae = SAE.from_pretrained("gpt2-small-res-jb", "blocks.4.hook_resid_pre")[0]

# Run the eval
results = run_eval(
    config=config,
    selected_saes=[("my_sae_release", sae)],
    device="cuda",
    output_path="eval_results/sparse_probing_sae_probes",
)
```

### Output Structure

The eval produces a JSON file with the following structure:

```json
{
  "eval_type_id": "sparse_probing_sae_probes",
  "eval_result_metrics": {
    "llm": {
      "llm_test_accuracy": 0.85,
      "llm_test_auc": 0.92,
      "llm_test_f1": 0.83
    },
    "sae": {
      "sae_top_1_test_accuracy": 0.78,
      "sae_top_1_test_auc": 0.85,
      "sae_top_1_test_f1": 0.76,
      "sae_top_2_test_accuracy": 0.81,
      ...
    }
  },
  "sae_metrics_by_k": {
    "1": {"test_accuracy": 0.78, "test_auc": 0.85, "test_f1": 0.76},
    "2": {"test_accuracy": 0.81, "test_auc": 0.87, "test_f1": 0.79},
    ...
  },
  "eval_result_details": [
    {
      "dataset_name": "118_us_state_CA",
      "llm_test_accuracy": 0.90,
      "sae_top_1_test_accuracy": 0.82,
      "sae_metrics_by_k": {
        "1": {"test_accuracy": 0.82, ...},
        ...
      }
    },
    ...
  ]
}
```

**Key Metrics:**

- **LLM metrics**: Baseline performance using full LLM residual stream (all dimensions)
- **SAE top-k metrics**: Performance using only k SAE latents with highest probe weights
- **sae_metrics_by_k**: Flexible dictionary supporting arbitrary k values
- **eval_result_details**: Per-dataset breakdown of all metrics

### Custom K Values

By default, the eval runs with k=[1, 2, 5]. You can specify custom k values:

```bash
python sae_bench/evals/sparse_probing_sae_probes/main.py \
    --model_name gpt2 \
    --sae_regex_pattern "gpt2-small-res-jb" \
    --sae_block_pattern "blocks.4.hook_resid_pre" \
    --ks 3 7 15 25 50
```

Results will be available in:

- Individual hardcoded fields (e.g., `sae_top_1_test_accuracy`) for standard k values
- `sae_metrics_by_k` dictionary for all k values (including custom ones)

### Dataset Selection

By default, the eval runs on all 140+ datasets from sae-probes. To run on a subset:

```python
config = SparseProbingSaeProbesEvalConfig(
    model_name="gpt2",
    dataset_names=["118_us_state_CA", "119_us_state_TX", "120_us_state_NY"],
    # ... other config
)
```

See the [sae-probes datasets](https://github.com/sae-probes/sae-probes#available-datasets) for the full list.

### Including LLM Baselines

To compare SAE performance against full LLM residual stream baselines:

```python
config = SparseProbingSaeProbesEvalConfig(
    model_name="gpt2",
    include_llm_baseline=True,  # Enables baseline comparison
    baseline_method="logreg",   # Method for baseline probe (default)
    # ... other config
)
```

This adds LLM baseline metrics to the output, allowing you to compare how well k SAE latents perform versus using all LLM dimensions.

### Caching model activations for Faster Iteration

Set `model_cache_path` to cache model activations across runs if you expect to rerun this eval for lots of different SAEs on the same model / layers. Set this to `None` to disable caching.
