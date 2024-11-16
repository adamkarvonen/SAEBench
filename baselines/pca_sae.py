import torch
import torch.nn as nn
import einops
import time
from dataclasses import dataclass
from typing import Optional
from transformer_lens import HookedTransformer
from sklearn.decomposition import IncrementalPCA
import math
from tqdm import tqdm

import sae_bench_utils.dataset_utils as dataset_utils
import sae_bench_utils.activation_collection as activation_collection


@dataclass
class SAEConfig:
    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str
    context_size: int = 128  # Can be used for auto-interp
    hook_head_index: Optional[int] = None


class PCASAE(nn.Module):
    def __init__(self, model_name: str, d_model: int, hook_layer: int, context_size: int):
        """Fit a PCA model to the activations of a model and treat it as an SAE.
        NOTE: There is a major footgun here. encode() saves the mean of the activations, which is
        saved and used in decode(). This introduces statefulness.

        I will leave it as is because this is just a simple baseline and I don't want to add complexity
        to the codebase."""
        super().__init__()

        self.W_enc = nn.Parameter(torch.zeros(d_model, d_model))
        self.W_dec = nn.Parameter(torch.zeros(d_model, d_model))
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = torch.float32

        # Mean tensor to store the batch mean during encoding
        self.mean = torch.zeros(d_model, device=self.device, dtype=self.dtype)

        hook_name = f"blocks.{hook_layer}.hook_resid_post"

        # Initialize the configuration dataclass
        self.cfg = SAEConfig(
            model_name,
            d_in=d_model,
            d_sae=d_model,
            hook_name=hook_name,
            hook_layer=hook_layer,
            context_size=context_size,
        )

    def encode(self, input_acts: torch.Tensor):
        """NOTE: There is a major footgun here. encode() saves the mean of the activations, which is
        saved and used in decode(). This introduces statefulness."""
        # Compute the mean across batch and sequence dimensions
        self.mean = input_acts.mean(dim=(0, 1), keepdim=True)
        # Center the data and apply the encoder matrix
        centered_acts = input_acts - self.mean
        encoded_acts = centered_acts @ self.W_enc
        return encoded_acts

    def decode(self, encoded_acts: torch.Tensor):
        """NOTE: There is a major footgun here. encode() saves the mean of the activations, which is
        saved and used in decode(). This introduces statefulness."""
        # Apply the decoder matrix and re-add the mean
        decoded_acts = encoded_acts @ self.W_dec
        return decoded_acts + self.mean

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

    def save_state_dict(self, file_path: str):
        """Save the encoder and decoder to a file."""
        torch.save({"W_enc": self.W_enc.data, "W_dec": self.W_dec.data}, file_path)

    def load_from_file(self, file_path: str):
        """Load the encoder and decoder from a file."""
        state_dict = torch.load(file_path, map_location=self.device)
        self.W_enc.data = state_dict["W_enc"]
        self.W_dec.data = state_dict["W_dec"]

    # required as we have device and dtype class attributes
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Update the device and dtype attributes based on the first parameter
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        # Update device and dtype if they were provided
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self


def fit_PCA(
    pca: PCASAE,
    model: HookedTransformer,
    dataset_name: str,
    num_tokens: int,
    llm_batch_size: int,
    pca_batch_size: int,
) -> PCASAE:
    tokens_BL = dataset_utils.load_and_tokenize_dataset(
        dataset_name, pca.cfg.context_size, num_tokens, model.tokenizer
    )

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

        # Partial fit on CPU
        ipca.partial_fit(activations_BD.cpu().float().numpy())

    print(f"Incremental PCA fit took {time.time() - start_time:.2f} seconds")

    # Set the learned components
    pca.W_enc.data = torch.tensor(ipca.components_, dtype=torch.float32, device="cpu")
    pca.W_dec.data = torch.tensor(ipca.components_.T, dtype=torch.float32, device="cpu")

    pca.save_state_dict(f"pca_{pca.cfg.model_name}_{pca.cfg.hook_name}.pt")

    return pca


if __name__ == "__main__":
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model_name = "pythia-70m-deduped"
    hook_layer = 3
    d_model = 512
    context_size = 128

    dataset_name = "monology/pile-uncopyrighted"

    pca = PCASAE(model_name, d_model, hook_layer, context_size)
    model = HookedTransformer.from_pretrained_no_processing(
        model_name, device=device, dtype=torch.float32
    )

    pca = fit_PCA(pca, model, dataset_name, 2_000_000, 512, 200_000)

    pca.to(device=device)

    test_input = torch.randn(1, 128, d_model, device=device, dtype=torch.float32)

    encoded = pca.encode(test_input)

    test_output = pca.decode(encoded)

    print(f"L0: {(encoded != 0).sum() / 128}")

    print(f"Diff: {torch.abs(test_input - test_output).mean()}")

    assert torch.allclose(test_input, test_output, atol=1e-5)