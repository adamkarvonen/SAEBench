#%% 
# Setup Python environment
import torch
import numpy as np
from tqdm import tqdm
from typing import Union
from sae_lens import SAE
import nnsight
from nnsight import LanguageModel
import pickle as pkl
import gc
import matplotlib.pyplot as plt

from sae_bench.evals.ravel.uniprobe import get_attribute_activations_nnsight, prepare_attribute_probe_data, collect_activations
from sae_bench.custom_saes.base_sae import BaseSAE
from sae_bench.evals.ravel.intervention import get_prompt_pairs, format_prompt_batch
from sae_bench.evals.ravel.uniprobe import collect_activations

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

#%%
# Load model
device = "cuda:0"
llm_dtype = torch.bfloat16
model = LanguageModel(
    "google/gemma-2-2b",
    device_map=device,
    torch_dtype=llm_dtype,
    dispatch=True,
    cache_dir="/share/u/can/models",
    **{'low_cpu_mem_usage': True,'attn_implementation': 'eager'},
)
model.requires_grad_(True)

#%% 
# Load SAE 

from sae_lens import SAE
sae_release = "sae_bench_gemma-2-2b_topk_width-2pow14_date-1109"
sae_id = "blocks.12.hook_resid_post__trainer_5"


if isinstance(sae_id, str):
    sae = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )[0]
else:
    sae = sae_id
    sae_id = "custom_sae"

sae = sae.to(device=device, dtype=llm_dtype)
sae.requires_grad_(False)

#%%
# Load full RAVEL instance
filename = "/share/u/can/SAEBench/artifacts/ravel/google_gemma-2-2b/city_instance.pkl"
dataset = pkl.load(open(filename, "rb"))

def load_model(config):
    """Load and configure the language model."""
    model = LanguageModel(
        "google/gemma-2-2b",
        device_map=config["device"],
        torch_dtype=config["llm_dtype"],
        dispatch=True,
        cache_dir="/share/u/can/models",
        **{'low_cpu_mem_usage': True,'attn_implementation': 'eager'},
    )
    model.requires_grad_(True)
    return model

def prepare_dbm_dataset(dataset, config, model):
    """Prepare dataset for DBM training."""
    num_dbm_steps = (config["num_pairs_per_attribute"] // config["dbm_batch_size"]) * config["num_epochs"]
    temperatures = torch.linspace(config["temperature_start"], config["temperature_end"], num_dbm_steps)
    
    dbm_dataset = {}
    for attribute in config["attributes"]:
        cause_base_prompts, cause_source_prompts = get_prompt_pairs(
            dataset=dataset, 
            base_attribute=attribute, 
            source_attribute=attribute, 
            n_interventions=config["num_pairs_per_attribute"] #//2
        )
        # other_attributes = [a for a in config["attributes"] if a != attribute]
        # TODO: for other attributes, get cause-effect pairs, assuming there's one other attribute
        # iso_base_prompts, iso_source_prompts = get_prompt_pairs(
        #     dataset=dataset, 
        #     base_attribute=other_attributes[0], 
        #     source_attribute=attribute, 
        #     n_interventions=config["num_pairs_per_attribute"] //2
        # )
        # cause_iso_shuffle = torch.randperm(config["num_pairs_per_attribute"])
        # base_prompts = cause_base_prompts + iso_base_prompts
        # source_prompts = cause_source_prompts + iso_source_prompts
        # prompt_labels = [1] * len(cause_base_prompts) + [-1] * len(iso_base_prompts)


        base_prompts = cause_base_prompts
        source_prompts = cause_source_prompts
        prompt_labels = [1] * len(cause_base_prompts)

        # base_prompts = [base_prompts[i] for i in cause_iso_shuffle]
        # source_prompts = [source_prompts[i] for i in cause_iso_shuffle]
        # prompt_labels = torch.tensor([prompt_labels[i] for i in cause_iso_shuffle])
        prompt_labels = torch.tensor(prompt_labels)

        base_encoding, base_final_entity_pos = format_prompt_batch(base_prompts, config["device"])
        base_generated_token_ids = [p.first_generated_token_id for p in base_prompts]
        source_generated_token_ids = [p.first_generated_token_id for p in source_prompts]
        source_act_BD = collect_activations(
            model,
            source_prompts,
            layer=config["intervention_layer"],
            device=config["device"],  
        )

        dbm_dataset[attribute] = {
            "base_prompts": base_prompts,
            "source_prompts": source_prompts,
            "base_labels": base_generated_token_ids,
            "source_labels": source_generated_token_ids,
            "base_encoding": base_encoding,
            "base_final_entity_pos": base_final_entity_pos,
            "source_act_BD": source_act_BD,
            "prompt_labels": prompt_labels,
        }
    
    return dbm_dataset, temperatures

def create_dataloaders(dbm_dataset, temperatures, config):
    """Create dataloaders for DBM training."""
    # Calculate steps per epoch
    steps_per_epoch = (config["num_pairs_per_attribute"] // config["dbm_batch_size"])
    num_total_steps = steps_per_epoch * config["num_epochs"]
    
    # Create temperature schedule with warmup
    warmup_steps = config["temperature_warmup_steps"]
    annealing_steps = num_total_steps - warmup_steps
    
    if annealing_steps > 0:
        # Create warmup temperatures (constant)
        warmup_temperatures = torch.full((warmup_steps,), config["temperature_start"])
        # Create annealing temperatures (linear decrease)
        annealing_temperatures = torch.linspace(
            config["temperature_start"], 
            config["temperature_end"], 
            annealing_steps
        )
        # Concatenate warmup and annealing schedules
        temperatures = torch.cat([warmup_temperatures, annealing_temperatures])
    else:
        # If warmup_steps >= num_total_steps, use constant temperature
        temperatures = torch.full((num_total_steps,), config["temperature_start"])
    
    dbm_dataloaders = {}
    
    for attribute in config["attributes"]:
        dbm_dataloaders[attribute] = []
        
        for step_idx in range(num_total_steps):
            batch_start = (step_idx % steps_per_epoch) * config["dbm_batch_size"]
            batch_end = batch_start + config["dbm_batch_size"]
            
            batch_data = {
                k: dbm_dataset[attribute][k][batch_start:batch_end]
                for k in dbm_dataset[attribute].keys()
            }
            batch_data["temperature"] = temperatures[step_idx]
            batch_data["epoch"] = step_idx // steps_per_epoch
            batch_data["step_in_epoch"] = step_idx % steps_per_epoch
            
            dbm_dataloaders[attribute].append(batch_data)
    
    return dbm_dataloaders

# %% 
# Get mask gradient

def get_logit_diff(logits_BLV, source_ids, base_ids, batch_size, prompt_labels):
    logits_BV = logits_BLV[:, -1, :]
    source_logits_B = logits_BV[range(batch_size), source_ids]
    base_logits_B = logits_BV[range(batch_size), base_ids]
    logit_diff_B = source_logits_B - base_logits_B
    logit_diff_B = logit_diff_B * prompt_labels # 1 for cause, -1 for iso

    loss = -logit_diff_B.mean()
    return loss

def get_ce_loss(logits_BLV, source_ids, base_ids, batch_size, prompt_labels):
    logits_BV = logits_BLV[:, -1, :]
    log_preds_BV = torch.log_softmax(logits_BV, dim=-1)
    log_preds_B = log_preds_BV[range(batch_size), source_ids]
    loss = -log_preds_B.mean()
    return loss


#%%
# Add experiment parameters at the top after imports
EXPERIMENT_CONFIG = {
    "num_pairs_per_attribute": 1500,
    "dbm_batch_size": 50,
    "num_epochs": 10,
    "temperature_start": 10,
    "temperature_end": 10,
    "temperature_warmup_steps": 400,  # Number of steps to maintain initial temperature
    "learning_rate": 1e-2,
    "attributes": ["Country", "Continent"],
    "device": "cuda:0",
    "llm_dtype": torch.bfloat16,
    "intervention_layer": 12,
}


intervention_submodule = model.model.layers[EXPERIMENT_CONFIG["intervention_layer"]]

def train_mask(model, sae, dbm_dataloader, temperatures, num_epochs, dbm_batch_size):
    """Train the DBM mask."""
    with model.session() as session:
        # Initialize mask
        loss_history = nnsight.list().save()
        mask_history = nnsight.list().save()
        mask = torch.nn.Parameter(torch.zeros(sae.cfg.d_sae, device=device, dtype=llm_dtype), requires_grad=True).save()
        optimizer = torch.optim.AdamW([mask], lr=EXPERIMENT_CONFIG["learning_rate"])

        with session.iter(dbm_dataloader) as batch:
            batch_encoding = batch["base_encoding"]
            batch_final_entity_pos = batch["base_final_entity_pos"]
            batch_source_act_BD = batch["source_act_BD"]
            batch_source_generated_token_ids = batch["source_labels"]
            batch_base_generated_token_ids = batch["base_labels"]
            batch_temperature = batch["temperature"]
            batch_prompt_labels = batch["prompt_labels"]
            epoch = batch["epoch"]
            step = batch["step_in_epoch"]

            with model.trace(batch_encoding):

                # Get base and source SAE activations
                base_act_BLD = intervention_submodule.output[0]
                base_act_BD = base_act_BLD[range(dbm_batch_size), batch_final_entity_pos, :]
                base_act_BS = sae.encode(base_act_BD.detach())
                source_act_BS = sae.encode(batch_source_act_BD.detach())

                # Subtract base and add source as determined by mask values
                mask_values = torch.sigmoid(mask / batch_temperature)
                masked_diff_BS = mask_values * (source_act_BS - base_act_BS)
                masked_diff_BD = sae.decode(masked_diff_BS)
                base_act_BLD[range(dbm_batch_size), batch_final_entity_pos, :] += masked_diff_BD

                logits_BLV = model.lm_head.output

                mask_history.append(mask.detach().to(torch.float32).save())
                loss = get_ce_loss(logits_BLV, batch_source_generated_token_ids, batch_base_generated_token_ids, dbm_batch_size, batch_prompt_labels)
                loss_history.append(loss.item().save())
                loss.backward()
                optimizer.step()

                session.log(f'Epoch {epoch}, Step {step}: temp={batch_temperature}, loss={loss.item()}')

                nnsight.apply(model.zero_grad)
                nnsight.apply(sae.zero_grad)
                mask.grad.zero_()
            

        final_mask_values = torch.sigmoid(mask / temperatures[-1])
        binary_mask_values = (final_mask_values > 0.5).float().save()

    return binary_mask_values, loss_history, mask_history

# Train mask

def plot_training_history(loss_histories, mask_histories):
    num_attributes = len(loss_histories)
    fig, axs = plt.subplots(num_attributes, 2, figsize=(20, num_attributes * 5))
    
    if num_attributes == 1:
        axs = axs.reshape(1, -1)  # Make axs 2D when there's only one row
    
    for i, (attr, loss_history) in enumerate(loss_histories.items()):
        # Plot loss history
        axs[i, 0].plot(loss_history)
        axs[i, 0].set_title(f'{attr} Loss History')
        axs[i, 0].set_xlabel('Training Step')
        axs[i, 0].set_ylabel('Loss')
        
        # Plot mask history
        mask_values = torch.stack(mask_histories[attr]).cpu().numpy()
        total_steps = len(mask_histories[attr])
        steps = np.arange(total_steps)
        
        # Calculate mean and standard error
        mean_values = np.mean(mask_values, axis=1)
        std_error = np.std(mask_values, axis=1) / np.sqrt(mask_values.shape[1])
        
        # Plot mean line for all steps
        axs[i, 1].plot(steps, mean_values, color='red', label='Mean')
        
        # Add standard error filled area
        axs[i, 1].fill_between(steps, 
                             mean_values - std_error,
                             mean_values + std_error,
                             alpha=0.3, color='red')
        
        # Select 20 equally spaced indices for scatter points
        scatter_indices = np.linspace(0, total_steps-1, 20, dtype=int)
        
        # Create scatter plot for selected steps only
        axs[i, 1].scatter(
            np.repeat(scatter_indices[:, np.newaxis], mask_values.shape[1], axis=1).flatten(),
            mask_values[scatter_indices].flatten(), 
            alpha=0.3, color='blue', label='Individual Features'
        )
        
        axs[i, 1].set_title(f'{attr} Mask Values Over Training')
        axs[i, 1].set_xlabel('Training Step')
        axs[i, 1].set_ylabel('Mask Value')
        axs[i, 1].legend(loc = 'upper left')
    
    plt.tight_layout()
    plt.savefig(f'/share/u/can/SAEBench/sae_bench/evals/ravel/training_history.png')

masks = {}
loss_histories = {}
mask_histories = {}
for attr in EXPERIMENT_CONFIG["attributes"]:
    print(f'Attribute: {attr}')
    torch.cuda.empty_cache()
    gc.collect()

    # Prepare dataset
    dbm_dataset, temperatures = prepare_dbm_dataset(dataset, EXPERIMENT_CONFIG, model)
    
    # Create dataloaders for this attribute
    attr_dataloader = create_dataloaders(dbm_dataset, temperatures, EXPERIMENT_CONFIG)[attr]
    
    # Train mask for this attribute
    masks[attr], loss_histories[attr], mask_histories[attr] = train_mask(
        model,
        sae,
        attr_dataloader,
        temperatures,
        EXPERIMENT_CONFIG["num_epochs"],
        EXPERIMENT_CONFIG["dbm_batch_size"]
    )

    print(f'Finished training mask for attribute: {attr}')
    print(f'Mask mean: {masks[attr].mean()}')

#%%
plot_training_history(loss_histories, mask_histories)
# %%
