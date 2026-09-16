# AutoInterp

The command-line entrypoint reads the provider's API key from `openai_api_key.txt` at `SAEBench/`. Python callers pass the key through the existing `api_key` argument to `run_eval`.

## Judge provider

By default, explanation generation and scoring use `gpt-4o-mini` through the OpenAI SDK. To use another OpenAI-compatible Chat Completions endpoint, set the judge fields on the config passed to `run_eval`:

```python
config = AutoInterpEvalConfig(
    model_name="pythia-70m-deduped",  # The model whose SAE is being evaluated
    judge_model_name="local-model",  # The provider's model identifier
    judge_base_url="http://localhost:8000/v1",
)
```

The command-line entrypoint exposes the same settings as `--judge_model_name` and `--judge_base_url`. `judge_base_url=None` keeps the SDK's default URL behavior, including `OPENAI_BASE_URL` if set. Supply the selected provider's API key; for a local server without authentication, the SDK still requires a non-empty value (for example, `local`).

Both phases use the selected judge, and its configuration is saved in each result's `eval_config`. Use a separate output folder or `force_rerun=True` when changing judges, because existing results are skipped by default. Keep credentials out of the base URL since it is saved in results. The endpoint must support the existing Chat Completions parameters (`messages`, `n`, `max_tokens`, and `stream=False`) and return text completions. Scheduling and SDK error/retry behavior are unchanged; judge quality and score comparability depend on the chosen model.

## File structure

There are 4 Python files in this folder:

- `eval_config.py` - this contains the config class for AutoInterp.
- `main.py` - this contains the main `AutoInterp` class, as well as the functions which are the interface to the rest of the SAEBench codebase.
- `demo.py` - you can run this via `python demo.py` to see an example output & how the function works. It creates & saves a log file (I've left the output of those files in the repo, so you can see what they look like).
- `sae_encode.py` - this contains a temporary replacement for the `encode` method in SAELens, until [my PR](https://github.com/jbloomAus/SAELens/pull/334) is merged. For memory efficiency, this could be used instead of encoding all SAE latents. We are currently using `encode()` to support compatibility with other SAE objects.

## Summary of how it works

### Generation phase

We run a batch through the model & SAE, getting activation values. We take some number of sequences from the top of the activation distribution, and also sample some number of sequences from the rest of the distribution with sample probability proportional to their activation (this is a stand-in for quantile sampling, which should be more compatible with e.g. Gated models which won't have values in all quantiles). We take these sequences and format the activating token using `<<token>>` syntax, then feed them through the model and ask for an explanation.

### Scoring phase

We select some number of top sequences & importance weighting sampled sequences (like the generation phase), but also include some sequences chosen randomly from the rest of the distribution. We'll shuffle these together and give them to the LLM as a numbered list, and we'll ask the LLM to return a comma-separated list of the indices of the sequences which it thinks will activate this feature.
