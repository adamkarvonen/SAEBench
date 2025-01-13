import torch
import torch.nn as nn
from typing import Optional
import sae_bench.custom_saes.custom_sae_config as sae_config
import sae_bench.custom_saes.base_sae as base_sae


class IdentitySAE(base_sae.BaseSAE):
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

        # Override the initialized parameters with identity matrices
        self.W_enc.data = torch.eye(d_in).to(dtype=dtype, device=device)
        self.W_dec.data = torch.eye(d_in).to(dtype=dtype, device=device)

    def encode(self, x: torch.Tensor):
        acts = x @ self.W_enc
        return acts

    def decode(self, feature_acts: torch.Tensor):
        return feature_acts @ self.W_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon


if __name__ == "__main__":
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float32

    model_name = "pythia-70m-deduped"
    hook_layer = 3
    d_model = 512

    identity = IdentitySAE(d_model, model_name, hook_layer, device, dtype)
    test_input = torch.randn(1, 128, d_model, device=device, dtype=dtype)

    encoded = identity.encode(test_input)
    test_output = identity.decode(encoded)

    print(f"L0: {(encoded != 0).sum() / 128}")
    print(f"Diff: {torch.abs(test_input - test_output).mean()}")
    assert torch.equal(test_input, test_output)
