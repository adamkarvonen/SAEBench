This repo implements David Chanin's feature absorption metric, with the absorption fraction metric added by Demian Till.

The code produces two scores:

- `mean_absorption_fraction_score` captures both full and partial absorption with an arbitrary number of absorbing latents. For a given SAE input, the absorption fraction is essentially the fraction of the SAE reconstruction's projection onto the ground truth probe activation that is not accounted for by the main latents which usually represent the feature in question.
- `mean_full_absorption_score` captures full absorption (not partial absorption) with a single absorbing latent. For a given SAE input, full absorption is judged to occur when the feature is present according to the ground truth probe, the main latents usually representing that feature have zero activation, and another latent compensates with a projection onto the ground truth probe direction which is above a set threshold as a proportion of the ground truth probe activation.

Estimated runtime:

- Pythia-70M: ~1 minute to collect activations / train probes per layer with SAEs, plus ~1 minute per SAE
- Gemma-2-2B: ~30 minutes to collect activations / train probes per layer with SAEs, plus ~10 minutes per SAE

Using Gemma-2-2B, at current batch sizes, I see a peak GPU memory usage of 24 GB. It successfully fits on an RTX 3090.

All configuration arguments and hyperparameters are located in `eval_config.py`. The full eval config is saved to the results json file.

If ran in the current state, `cd` in to `evals/absorption/` and run `python main.py`. It should produce `eval_results/absorption/pythia-70m-deduped_layer_4_eval_results.json`.

`tests/test_absorption.py` contains an end-to-end test of the sparse probing eval. Expected results are in `tests/test_data/absorption_expected_results.json`. Running `pytest -s tests/test_absorption` will verify that the actual results are within the specified tolerance of the expected results.

## Tips on running this eval

- This eval only makes sense if the LLM is large enough to have decent spelling knowledge. It is not recommended to run this eval on LLMs with less than 1B parameters.
- This eval only makes sense if the SAE is wide enough to engage in absorption. We do not recommend this benchmark on SAEs with less than 16k latents.
- This eval will return a very low score for a randomly initialized SAE, so make sure that you are training the SAE fully before evaluating it. If you severely undertrain your SAE, you will get a low absorption score, but this does not mean that your SAE architecture has solved absorption.
- For smaller LLMs, it is recommended to set `precalc_k_sparse_probe_sae_acts` to `True` to speed up the training process. In general, you should set this to `True` and only disable if you run out of GPU memory.
