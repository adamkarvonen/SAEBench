import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import numpy as np
from typing import Optional
import json

import sae_bench.custom_saes.base_sae as base_sae


class JumpReluSAE(base_sae.BaseSAE):
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

        self.threshold = nn.Parameter(torch.zeros(d_sae, dtype=dtype, device=device))

    def encode(self, x: torch.Tensor):
        pre_acts = x @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, feature_acts: torch.Tensor):
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon


def load_dictionary_learning_jump_relu_sae(
    repo_id: str,
    filename: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: Optional[int] = None,
    local_dir: str = "downloaded_saes",
) -> JumpReluSAE:
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

    sae = JumpReluSAE(
        d_in=pt_params["b_dec"].shape[0],
        d_sae=pt_params["b_enc"].shape[0],
        model_name=model_name,
        hook_layer=layer,
        device=device,
        dtype=dtype,
    )

    sae.load_state_dict(pt_params)

    sae.to(device=device, dtype=dtype)

    d_sae, d_in = sae.W_dec.data.shape

    assert d_sae >= d_in

    if config["trainer"]["trainer_class"] == "JumpReluTrainer":
        sae.cfg.architecture = "jumprelu"
    else:
        raise ValueError(f"Unknown trainer class: {config['trainer']['trainer_class']}")

    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError(
            "Decoder norms are not normalized. Implement a normalization method."
        )

    return sae


def load_gemma_scope_jumprelu_sae(
    repo_id: str,
    filename: str,
    layer: int,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    local_dir: str = "downloaded_saes",
) -> JumpReluSAE:
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )

    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).cpu() for k, v in params.items()}

    d_in = params["W_enc"].shape[0]
    d_sae = params["W_enc"].shape[1]

    assert d_sae >= d_in

    sae = JumpReluSAE(d_in, d_sae, model_name, layer, device, dtype)
    sae.load_state_dict(pt_params)
    sae.to(dtype=dtype, device=device)

    sae.cfg.architecture = "jumprelu"

    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError(
            "Decoder norms are not normalized. Implement a normalization method."
        )

    return sae


if __name__ == "__main__":
    repo_id = "adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0104"
    filename = "JumpReluTrainer_EleutherAI_pythia-160m-deduped_ctx1024_0104/resid_post_layer_8/trainer_32/ae.pt"
    layer = 8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    model_name = "EleutherAI/pythia-160m-deduped"
    hook_name = f"blocks.{layer}.hook_resid_post"

    sae = load_dictionary_learning_jump_relu_sae(
        repo_id, filename, model_name, device, dtype, layer=layer
    )
    sae.test_sae(model_name)


# Gemma-Scope Test
# if __name__ == "__main__":
# layer = 20

# repo_id = "google/gemma-scope-2b-pt-res"
# filename = f"layer_{layer}/width_16k/average_l0_71/params.npz"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype = torch.float32
# model_name = "google/gemma-2-2b"

# sae = load_gemma_scope_jumprelu_sae(repo_id, filename, layer, model_name, device, dtype)
