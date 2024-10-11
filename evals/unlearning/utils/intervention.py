import torch
import einops

from torch import Tensor
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
from contextlib import contextmanager
from functools import partial
from sae_lens import SAE

import numpy as np


def anthropic_clamp_resid_SAE_features(
    resid: Float[Tensor, "batch seq d_model"], 
    hook: HookPoint,
    sae: SAE,
    features_to_ablate: list[int],
    multiplier: float = 1.0,
    random: bool = False,
):
    """
    Given a list of feature indices, this hook function removes feature activations in a manner similar to the one
    used in "Scaling Monosemanticity": https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#appendix-methods-steering
    This version clamps the feature activation to the value(s) specified in multiplier
    """

    if len(features_to_ablate) > 0:

        with torch.no_grad():
            #adjust feature activations with scaling (multiplier = 0 just ablates the feature)
            if isinstance(sae, SAE):
                reconstruction = sae(resid)
                feature_activations = sae.encode(resid)
            # else:
            #     try:
            #         import sys
            #         sys.path.append('/root')

            #         from dictionary_learning import AutoEncoder
            #         from dictionary_learning.trainers.top_k import AutoEncoderTopK

            #         if isinstance(sae, (AutoEncoder, AutoEncoderTopK)):
            #             reconstruction = sae(resid)
            #             feature_activations = sae.encode(resid)
            #     except:
            #         raise ValueError("sae must be an instance of SparseAutoencoder or SAE")

            error = resid - reconstruction

            non_zero_features = feature_activations[:, :, features_to_ablate] > 0
            
            
            if not random:
                    
                if isinstance(multiplier, float) or isinstance(multiplier, int):
                    feature_activations[:, :, features_to_ablate] = torch.where(non_zero_features, -multiplier, feature_activations[:, :, features_to_ablate])
                else:
                    feature_activations[:, :, features_to_ablate] = torch.where(non_zero_features,
                                                                                -multiplier.unsqueeze(dim=0).unsqueeze(dim=0),
                                                                                feature_activations[:, :, features_to_ablate])    
                
            # set the next feature id's activations to the multiplier only if the previous feature id's
            # activations are positive
            else:
                assert isinstance(multiplier, float) or isinstance(multiplier, int)
                
                next_features_to_ablate = [(f + 1) % feature_activations.shape[-1] for f in features_to_ablate]
                feature_activations[:, :, next_features_to_ablate] = torch.where(
                    feature_activations[:, :, features_to_ablate] > 0,
                    -multiplier,
                    feature_activations[:, :, next_features_to_ablate]
                )   
                
            try:
                modified_reconstruction = einops.einsum(feature_activations, sae.W_dec, "... d_sae, d_sae d_in -> ... d_in")\
                    + sae.b_dec
            except:
                # SAEBench doesn't have W_dec and b_dec
                modified_reconstruction = sae.decode(feature_activations)
            
            # Unscale outputs if needed:
            # if sae.input_scaling_factor is not None:
            #     modified_reconstruction = modified_reconstruction / sae.input_scaling_factor
            resid = error + modified_reconstruction
        return resid