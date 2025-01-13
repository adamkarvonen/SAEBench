import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import json
from typing import Optional

import sae_bench.custom_saes.base_sae as base_sae


class BatchTopKSAE(base_sae.BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: Optional[str] = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        assert isinstance(k, int) and k > 0
        self.register_buffer("k", torch.tensor(k, dtype=torch.int, device=device))

        # BatchTopK requires a global threshold to use during inference. Must be positive.
        self.use_threshold = True
        self.register_buffer(
            "threshold", torch.tensor(-1.0, dtype=dtype, device=device)
        )

    def encode(self, x: torch.Tensor):
        """Note: x can be either shape (B, F) or (B, L, F)"""
        post_relu_feat_acts_BF = nn.functional.relu(
            (x - self.b_dec) @ self.W_enc + self.b_enc
        )

        if self.use_threshold:
            if self.threshold < 0:
                raise ValueError(
                    "Threshold is not set. The threshold must be set to use it during inference"
                )
            encoded_acts_BF = post_relu_feat_acts_BF * (
                post_relu_feat_acts_BF > self.threshold
            )
            return encoded_acts_BF

        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=tops_acts_BK
        )
        return encoded_acts_BF

    def decode(self, feature_acts: torch.Tensor):
        return (feature_acts @ self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon


def load_dictionary_learning_batch_topk_sae(
    repo_id: str,
    filename: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: Optional[int] = None,
    local_dir: str = "downloaded_saes",
) -> BatchTopKSAE:
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

    k = config["trainer"]["k"]

    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "bias": "b_dec",
        "k": "k",
        "threshold": "threshold",
    }

    # Create a new dictionary with renamed keys
    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

    # due to the way torch uses nn.Linear, we need to transpose the weight matrices
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T

    # Print renamed keys for debugging
    print("Renamed keys in state_dict:", renamed_params.keys())

    sae = BatchTopKSAE(
        d_in=renamed_params["b_dec"].shape[0],
        d_sae=renamed_params["b_enc"].shape[0],
        k=k,
        model_name=model_name,
        hook_layer=layer,
        device=device,
        dtype=dtype,
    )

    sae.load_state_dict(renamed_params)

    sae.to(device=device, dtype=dtype)

    d_sae, d_in = sae.W_dec.data.shape

    assert d_sae >= d_in

    if config["trainer"]["trainer_class"] == "BatchTopKTrainer":
        sae.cfg.architecture = "batch_topk"
    else:
        raise ValueError(f"Unknown trainer class: {config['trainer']['trainer_class']}")

    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError("Decoder vectors are not normalized. Please normalize them")

    return sae


def load_dictionary_learning_matroyshka_batch_topk_sae(
    repo_id: str,
    filename: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: Optional[int] = None,
    local_dir: str = "downloaded_saes",
) -> BatchTopKSAE:
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

    k = config["trainer"]["k"]

    # We currently don't use group sizes, so we remove them to reuse the BatchTopKSAE class
    del pt_params["group_sizes"]

    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    sae = BatchTopKSAE(
        d_in=pt_params["b_dec"].shape[0],
        d_sae=pt_params["b_enc"].shape[0],
        k=k,
        model_name=model_name,
        hook_layer=layer,
        device=device,
        dtype=dtype,
    )

    sae.load_state_dict(pt_params)

    sae.to(device=device, dtype=dtype)

    d_sae, d_in = sae.W_dec.data.shape

    assert d_sae >= d_in

    if config["trainer"]["trainer_class"] == "MatroyshkaBatchTopKTrainer":
        sae.cfg.architecture = "matroyshka_batch_topk"
    else:
        raise ValueError(f"Unknown trainer class: {config['trainer']['trainer_class']}")

    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError("Decoder vectors are not normalized. Please normalize them")

    return sae


if __name__ == "__main__":
    repo_id = "adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0104"
    filename = "BatchTopKTrainer_EleutherAI_pythia-160m-deduped_ctx1024_0104/resid_post_layer_8/trainer_26/ae.pt"
    layer = 8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    model_name = "EleutherAI/pythia-160m-deduped"
    hook_name = f"blocks.{layer}.hook_resid_post"

    sae = load_dictionary_learning_batch_topk_sae(
        repo_id, filename, model_name, device, dtype, layer=layer
    )
    sae.test_sae(model_name)

# Matroyshka BatchTopK SAE

# if __name__ == "__main__":
#     repo_id = "adamkarvonen/matroyshka_pythia_160m_16k"
#     filename = "MatroyshkaBatchTopKTrainer_temp_100_EleutherAI_pythia-160m-deduped_ctx1024_0104/resid_post_layer_8/trainer_2/ae.pt"
#     layer = 8

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     dtype = torch.float32

#     model_name = "EleutherAI/pythia-160m-deduped"
#     hook_name = f"blocks.{layer}.hook_resid_post"

#     sae = load_dictionary_learning_matroyshka_batch_topk_sae(
#         repo_id, filename, model_name, device, dtype, layer=layer
#     )
#     sae.test_sae(model_name)
