import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import numpy as np
from typing import Optional
import json

import sae_bench.custom_saes.base_sae as base_sae


class GatedSAE(base_sae.BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: Optional[str] = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        self.r_mag = nn.Parameter(torch.zeros(d_sae, dtype=dtype, device=device))
        self.b_mag = nn.Parameter(torch.zeros(d_sae, dtype=dtype, device=device))
        self.gate_bias = nn.Parameter(torch.zeros(d_sae, dtype=dtype, device=device))

        del self.b_enc

    def encode(self, x: torch.Tensor):
        x_enc = (x - self.b_dec) @ self.W_enc

        # Gated network
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).to(dtype=self.W_enc.dtype)

        # Magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.b_mag
        f_mag = torch.nn.functional.relu(pi_mag)

        f = f_gate * f_mag

        return f

    def decode(self, feature_acts: torch.Tensor):
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon


def load_dictionary_learning_gated_sae(
    repo_id: str,
    filename: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: Optional[int] = None,
    local_dir: str = "downloaded_saes",
) -> GatedSAE:
    assert "ae.pt" in filename

    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )

    pt_params = torch.load(path_to_params, map_location=torch.device("cpu"))

    config_filename = filename.replace("ae.pt", "config.json")
    path_to_config = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        force_download=False,
        local_dir=local_dir,
    )

    with open(path_to_config, "r") as f:
        config = json.load(f)

    if layer is not None:
        assert layer == config["trainer"]["layer"]
    else:
        layer = config["trainer"]["layer"]

    # Transformer lens often uses a shortened model name
    assert model_name in config["trainer"]["lm_name"]

    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "decoder_bias": "b_dec",
        "r_mag": "r_mag",
        "gate_bias": "gate_bias",
        "mag_bias": "b_mag",
    }

    # Create a new dictionary with renamed keys
    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

    # due to the way torch uses nn.Linear, we need to transpose the weight matrices
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T

    # Print renamed keys for debugging
    print("Renamed keys in state_dict:", renamed_params.keys())

    sae = GatedSAE(
        d_in=renamed_params["b_dec"].shape[0],
        d_sae=renamed_params["b_mag"].shape[0],
        model_name=model_name,
        hook_layer=layer,
        device=device,
        dtype=dtype,
    )

    sae.load_state_dict(renamed_params)

    sae.to(device=device, dtype=dtype)

    d_sae, d_in = sae.W_dec.data.shape

    assert d_sae >= d_in

    if config["trainer"]["trainer_class"] == "GatedSAETrainer":
        sae.cfg.architecture = "gated"
    else:
        raise ValueError(f"Unknown trainer class: {config['trainer']['trainer_class']}")

    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError(
            "Decoder norms are not normalized. Implement a normalization method."
        )

    return sae


if __name__ == "__main__":
    repo_id = "adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0104"
    filename = "GatedSAETrainer_EleutherAI_pythia-160m-deduped_ctx1024_0104/resid_post_layer_8/trainer_14/ae.pt"
    layer = 8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    model_name = "EleutherAI/pythia-160m-deduped"
    hook_name = f"blocks.{layer}.hook_resid_post"

    sae = load_dictionary_learning_gated_sae(
        repo_id, filename, layer, model_name, device, dtype
    )
    sae.test_sae(model_name)
