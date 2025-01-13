import torch
import torch.nn as nn
import einops
import time
from typing import Optional
from transformer_lens import HookedTransformer
from sklearn.decomposition import IncrementalPCA
import math
from tqdm import tqdm
import gc

import sae_bench.custom_saes.custom_sae_config as sae_config
import sae_bench.custom_saes.base_sae as base_sae
import sae_bench.sae_bench_utils.dataset_utils as dataset_utils
import sae_bench.sae_bench_utils.activation_collection as activation_collection


class PCASAE(base_sae.BaseSAE):
    def __init__(
        self,
        d_in: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: Optional[str] = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_in, model_name, hook_layer, device, dtype, hook_name)

        # Additional parameter specific to PCA
        self.mean = nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor):
        centered_acts = x - self.mean
        encoded_acts = centered_acts @ self.W_enc
        return encoded_acts

    def decode(self, feature_acts: torch.Tensor):
        decoded_acts = feature_acts @ self.W_dec
        return decoded_acts + self.mean

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon

    def save_state_dict(self, file_path: str):
        """Save the encoder and decoder to a file."""
        torch.save(
            {
                "W_enc": self.W_enc.data,
                "W_dec": self.W_dec.data,
                "mean": self.mean.data,
            },
            file_path,
        )

    def load_from_file(self, file_path: str):
        """Load the encoder and decoder from a file."""
        state_dict = torch.load(file_path, map_location=self.device)
        self.W_enc.data = state_dict["W_enc"]
        self.W_dec.data = state_dict["W_dec"]
        self.mean.data = state_dict["mean"]
        self.normalize_decoder()
        self.to(dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def normalize_decoder(self):
        norms = torch.norm(self.W_dec, dim=1).to(dtype=self.dtype, device=self.device)

        print("Decoder vectors are not normalized. Normalizing.")

        test_input = torch.randn(10, self.cfg.d_in)
        initial_output = self(test_input)

        self.W_dec.data /= norms[:, None]

        new_norms = torch.norm(self.W_dec, dim=1)
        assert torch.allclose(new_norms, torch.ones_like(new_norms))

        self.W_enc *= norms

        new_output = self(test_input)

        max_diff = torch.abs(initial_output - new_output).max()
        print(f"Max difference in output: {max_diff}")

        # Errors can be relatively large in larger SAEs due to floating point precision
        assert torch.allclose(initial_output, new_output, atol=1e-4)


@torch.no_grad()
def fit_PCA(
    pca: PCASAE,
    model: HookedTransformer,
    tokens_BL: torch.Tensor,
    llm_batch_size: int,
    pca_batch_size: int,
) -> PCASAE:
    # Calculate number of sequences per PCA batch
    sequences_per_batch = pca_batch_size // pca.cfg.context_size
    num_batches = math.ceil(len(tokens_BL) / sequences_per_batch)

    # Initialize incremental PCA
    ipca = IncrementalPCA(n_components=pca.cfg.d_in)

    start_time = time.time()

    # Process tokens in batches
    for batch_idx in tqdm(range(num_batches), desc="Fitting PCA"):
        batch_start = batch_idx * sequences_per_batch
        batch_end = min((batch_idx + 1) * sequences_per_batch, len(tokens_BL))

        tokens_batch = tokens_BL[batch_start:batch_end]

        activations_BLD = activation_collection.get_llm_activations(
            tokens_batch,
            model,
            llm_batch_size,
            pca.cfg.hook_layer,
            pca.cfg.hook_name,
            mask_bos_pad_eos_tokens=False,
        )

        activations_BD = einops.rearrange(activations_BLD, "B L D -> (B L) D")

        if activations_BD.shape[0] <= pca.cfg.d_in:
            print(
                f"Skipping batch {batch_idx} as it has {activations_BLD.shape[0]} sequences, which is less than {pca.cfg.d_in}"
            )
            continue

        # Partial fit on CPU
        ipca.partial_fit(activations_BD.cpu().float().numpy())

    print(f"Incremental PCA fit took {time.time() - start_time:.2f} seconds")

    # Set the learned components
    pca.mean.data = torch.tensor(ipca.mean_, dtype=torch.float32, device="cpu")
    pca.W_enc.data = torch.tensor(ipca.components_, dtype=torch.float32, device="cpu")
    pca.W_dec.data = torch.tensor(ipca.components_.T, dtype=torch.float32, device="cpu")

    pca.save_state_dict(f"pca_{pca.cfg.model_name}_{pca.cfg.hook_name}.pt")

    return pca


@torch.no_grad()
def fit_PCA_gpu(
    pca: PCASAE,
    model: HookedTransformer,
    tokens_BL: torch.Tensor,
    llm_batch_size: int,
    pca_batch_size: int,
) -> PCASAE:
    """Uses CUML for much faster training, requires installing cuml."""
    import cupy as cp
    from cuml.decomposition import IncrementalPCA as cuIPCA

    # Calculate batching
    sequences_per_batch = pca_batch_size // pca.cfg.context_size
    num_batches = math.ceil(len(tokens_BL) / sequences_per_batch)

    # Initialize cuML's incremental PCA
    # Note: cuML's IPCA requires batch_size to be specified
    ipca = cuIPCA(n_components=pca.cfg.d_in, batch_size=min(pca_batch_size, 10000))

    start_time = time.time()

    for batch_idx in tqdm(range(num_batches), desc="Fitting PCA"):
        batch_start = batch_idx * sequences_per_batch
        batch_end = min((batch_idx + 1) * sequences_per_batch, len(tokens_BL))

        tokens_batch = tokens_BL[batch_start:batch_end]

        # Get activations (already on GPU)
        activations_BLD = activation_collection.get_llm_activations(
            tokens_batch,
            model,
            llm_batch_size,
            pca.cfg.hook_layer,
            pca.cfg.hook_name,
            mask_bos_pad_eos_tokens=False,
            show_progress=False,
        )

        # Reshape on GPU
        activations_BD = einops.rearrange(activations_BLD, "B L D -> (B L) D").to(
            dtype=torch.float32
        )

        if activations_BD.shape[0] <= pca.cfg.d_in:
            print(
                f"Skipping batch {batch_idx} as it has {activations_BLD.shape[0]} sequences, which is less than {pca.cfg.d_in}"
            )
            continue

        # Convert to cupy array (zero-copy if already on GPU)
        activations_cupy = cp.asarray(activations_BD.detach())

        # Partial fit using GPU data
        ipca.partial_fit(activations_cupy)

        # Optional: Clear cache periodically
        gc.collect()
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()

    print(f"GPU Incremental PCA fit took {time.time() - start_time:.2f} seconds")

    # Get components back as torch tensors
    components = torch.from_numpy(cp.asnumpy(ipca.components_))
    pca_mean = torch.from_numpy(cp.asnumpy(ipca.mean_))

    # Set the learned components
    pca.mean.data = pca_mean.to(dtype=torch.float32, device="cpu")
    pca.W_enc.data = components.float().to(dtype=torch.float32, device="cpu")
    pca.W_dec.data = components.T.float().to(dtype=torch.float32, device="cpu")

    pca.save_state_dict(f"pca_{pca.cfg.model_name}_{pca.cfg.hook_name}.pt")

    return pca


if __name__ == "__main__":
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    torch.set_grad_enabled(False)

    model_name = "pythia-70m-deduped"
    d_model = 512

    # model_name = "gemma-2-2b"
    # d_model = 2304

    if model_name == "pythia-70m-deduped":
        llm_batch_size = 1024
        pca_batch_size = 400_000
        llm_dtype = torch.float32
        layers = [3, 4]
    elif model_name == "gemma-2-2b":
        llm_batch_size = 128
        pca_batch_size = 100_000
        llm_dtype = torch.bfloat16
        layers = [5, 12, 19]
    else:
        raise ValueError("Invalid model")

    context_size = 128

    dataset_name = "monology/pile-uncopyrighted"
    num_tokens = 200_000_000

    model = HookedTransformer.from_pretrained_no_processing(
        model_name, device=device, dtype=llm_dtype
    )

    tokens_BL = dataset_utils.load_and_tokenize_dataset(
        dataset_name, context_size, num_tokens, model.tokenizer
    )

    for layer in layers:
        pca = PCASAE(model_name, d_model, layer, context_size)
        # pca = fit_PCA(pca, model, tokens_BL, llm_batch_size, pca_batch_size)
        pca = fit_PCA_gpu(pca, model, tokens_BL, llm_batch_size, pca_batch_size)

        pca.load_from_file(f"pca_{model_name}_blocks.{layer}.hook_resid_post.pt")

        pca.to(device=device)

        test_input = torch.randn(1, 128, d_model, device=device, dtype=torch.float32)

        encoded = pca.encode(test_input)

        test_output = pca.decode(encoded)

        print(f"L0: {(encoded != 0).sum() / 128}")

        print(f"Diff: {torch.abs(test_input - test_output).mean()}")

        assert torch.allclose(test_input, test_output, atol=1e-5)
