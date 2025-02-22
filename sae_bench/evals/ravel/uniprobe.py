# TODO Urgent: Doublecheck BOS token handling
# TODO Doublecheck the dataset selection used in the paper for training the probe

import torch
import numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from transformers import BatchEncoding
from typing import List, Dict, Any
from nnsight import LanguageModel
from sae_bench.evals.ravel.instance import Prompt

def collect_activations(
    model: LanguageModel,
    attr_prompts: List[Prompt],
    layer: int,
    device: str,
    llm_batch_size: int = 512,
    tracer_kwargs: Dict[str, Any] = {"scan": False, "validate": False},
):
    """Collect activations for a batch of prompts at entity positions.
    
    Args:
        model: The transformer model
        attr_prompts: List of prompts containing entity positions
        layer: Layer to collect activations from
        device: Device to run computation on
        llm_batch_size: Batch size for processing
        tracer_kwargs: Additional arguments for the tracer
        
    Returns:
        torch.Tensor: Collected activations of shape [num_prompts x hidden_dim]
    """
    # Prepare inputs
    input_ids, attn_masks, entity_pos = [], [], []
    for p in attr_prompts:
        input_ids.append(p.input_ids)
        attn_masks.append(p.attention_mask) 
        entity_pos.append(p.final_entity_token_pos)
    input_ids = torch.stack(input_ids).to(device)
    attn_masks = torch.stack(attn_masks).to(device)
    entity_positions = torch.tensor(entity_pos)

    batch_activations = []
    for batch_begin in range(0, len(input_ids), llm_batch_size):
        input_ids_BL = input_ids[batch_begin : batch_begin + llm_batch_size]
        attn_masks_BL = attn_masks[batch_begin : batch_begin + llm_batch_size]
        entity_pos_B = entity_positions[batch_begin : batch_begin + llm_batch_size]
        encoding_BL = BatchEncoding(
            {
                "input_ids": input_ids_BL,
                "attention_mask": attn_masks_BL,
            }
        )

        # Get activations
        with torch.no_grad():
            with model.trace(encoding_BL, **tracer_kwargs):
                llm_act_BLD = model.model.layers[layer].output[0]
                batch_arange = torch.arange(llm_act_BLD.shape[0])
                llm_act_BD = llm_act_BLD[batch_arange, entity_pos_B]
                llm_act_BD = llm_act_BD.save()
            batch_activations.append(llm_act_BD)
    
    return torch.cat(batch_activations, dim=0)


def get_attribute_activations_nnsight(
    model,
    dataset,
    all_attributes,
    max_samples_per_attribute=1024,
    layer=11,
    llm_batch_size=512,
    tracer_kwargs={"scan": False, "validate": False},
    device="cpu",
):
    """
    Get model activations for attribute classification.

    Returns:
        Tuple of (activations tensor, binary labels list)
    """
    # Randomly sample prompts for each attribute
    all_attribute_activations_BD = {}
    for attr in all_attributes:
        attr_prompts = dataset.get_prompts_by_attribute(attr, n_samples=max_samples_per_attribute)
        print(f"Number of prompts for {attr}: {len(attr_prompts)}")

        attribute_activations = collect_activations(
            model=model,
            attr_prompts=attr_prompts,
            layer=layer,
            device=device,
            llm_batch_size=llm_batch_size,
            tracer_kwargs=tracer_kwargs
        )

        all_attribute_activations_BD[attr] = attribute_activations
    return all_attribute_activations_BD


def prepare_attribute_probe_data(
    chosen_attribute: str,
    all_attribute_activations_BD: Dict[str, torch.Tensor],
    max_samples_per_attribute: int = 1024,
):

    # Select balanced samples
    balanced_attribute_acts = all_attribute_activations_BD[chosen_attribute]
    n_chosen = len(all_attribute_activations_BD[chosen_attribute])
    for attr, acts in all_attribute_activations_BD.items():
        if attr == chosen_attribute:
            continue
        n_samples = min(max_samples_per_attribute, len(acts))
        balanced_attribute_acts = torch.cat((balanced_attribute_acts, acts[:n_samples]), dim=0)

    # Shuffle and create labels
    n_total = balanced_attribute_acts.shape[0]
    shuffled_indices = torch.randperm(n_total)
    balanced_attribute_acts = balanced_attribute_acts[shuffled_indices]
    labels = torch.zeros(n_total)
    labels[:n_chosen] = 1
    labels = labels[shuffled_indices]

    return balanced_attribute_acts, labels


# from ravel code
def select_features_with_classifier(sae, inputs, labels, coeff):
    coeff_to_select_features = {}
    if coeff[0] == 0.0:
        coeff = coeff[1:]
        coeff_to_select_features[0.0] = []

    for c in tqdm(coeff):
        with torch.no_grad():
            X_transformed = sae.encode(inputs).to(dtype=torch.float32).cpu().numpy()
            lsvc = LinearSVC(C=c, penalty="l1", dual=False, max_iter=5000, tol=0.01).fit(
                X_transformed, labels
            )
            selector = SelectFromModel(lsvc, prefit=True)
            kept_dim = np.where(selector.get_support())[0]
            coeff_to_select_features[float(c)] = kept_dim
    
    return coeff_to_select_features


def run_feature_selection_probe(
    model,
    sae,
    dataset,
    all_attributes,
    coeffs=[0.01, 0.1, 10, 100, 1000],
    max_samples_per_attribute=1024,
    layer=11,
    llm_batch_size=512,
) -> Dict[str, Dict[float, List[int]]]:

    # Cache activations
    all_attribute_activations_BD = get_attribute_activations_nnsight(
        model,
        dataset,
        all_attributes,
        max_samples_per_attribute=max_samples_per_attribute,
        layer=layer,
        llm_batch_size=llm_batch_size,
    )

    # Select features for each attribute by trainin a linear probe on all latents at once
    selected_features = {}  # {attr: {c: [kept_dims]}}
    for attr in tqdm(all_attributes, desc="Running feature selection probe for each attribute"):
        balanced_attribute_acts, labels = prepare_attribute_probe_data(
            attr, all_attribute_activations_BD, max_samples_per_attribute=max_samples_per_attribute
        )
        selected_features[attr] = select_features_with_classifier(
            sae, balanced_attribute_acts, labels, coeff=coeffs
        )

    return selected_features
