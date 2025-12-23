# Meta-Structure (Meta-SAE) Evaluation

This evaluation reproduces the composition experiment from Appendix D.1 (“Investigating Composition with Meta-SAEs”) of **Bussmann, Bart, et al. "Learning multi-level features with matryoshka sparse autoencoders." ICML (2025).**. The paper trains a *meta-SAE* on the decoder matrix of a base SAE and reports how much variance the meta-SAE can explain; higher explained variance means more shared structure between decoder directions.

We mirror that setup by training a BatchTopK meta-structure model:
- **Dictionary size**: one-quarter of the base SAE width (`width_ratio=0.25`).
- **Activation budget**: average `k=4` active latents (BatchTopK).
- **Data**: rows of the base SAE decoder (`W_dec`); batches are sampled with replacement.
- **Metric**: `decoder_fraction_variance_explained` over the decoder matrix; MSE is also logged.

## How it works
1. Load the target SAE (pretrained or custom) and its decoder matrix.
2. Instantiate a BatchTopKTrainingSAE with `d_sae = ceil(d_base * width_ratio)` and `k=4`.
3. Train on decoder rows for `train_steps` with batch size `train_batch_size` (defaults: 1500 steps, 1024 batch size, lr=5e-4).
4. Compute variance explained and reconstruction MSE on the full decoder; write a JSON result to the output folder.

## Estimate Runtime
Less than a minute on a 4090.

## Outputs
Each run writes `<release>_<sae_id>_eval_results.json` containing:
- `decoder_fraction_variance_explained`
- `final_reconstruction_mse`
- `train_time_seconds`
- Eval/config metadata for reproducibility

An acceptance fixture for the pythia-70m run above is stored at `tests/acceptance/test_data/meta_structure/pythia70m_tr10_meta_structure.json`, and `tests/acceptance/test_meta_structure.py` verifies new runs against it.
