from dataclasses import asdict, dataclass


@dataclass
class CustomSAEConfig:
    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str

    # The following are used for the core/main.py SAE evaluation
    # the values aren't important, the fields are just required
    context_size: int = None  # type: ignore # Can be used for auto-interp
    hook_head_index: int | None = None

    # Architecture settings
    architecture: str = ""
    apply_b_dec_to_input: bool = None  # type: ignore
    finetuning_scaling_factor: bool = None  # type: ignore
    activation_fn_str: str = ""
    activation_fn_kwargs = {}
    prepend_bos: bool = True
    normalize_activations: str = "none"

    # Model settings
    dtype: str = ""  # this must be set to e.g. "float32" in core/main.py
    device: str = ""
    model_from_pretrained_kwargs = {}

    # Dataset settings
    dataset_path: str = ""
    dataset_trust_remote_code: bool = True
    seqpos_slice: tuple = (None,)
    training_tokens: int = -100_000

    # Metadata
    sae_lens_training_version: str | None = None
    neuronpedia_id: str | None = None

    def to_dict(self) -> dict:
        cfg_dict = {**asdict(self)}
        # this is added by the standardization process, but can't directly be serialized
        if "metadata" in cfg_dict:
            del cfg_dict["metadata"]
        return cfg_dict
