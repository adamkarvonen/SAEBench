# Setup Python environment
import torch
from nnsight import LanguageModel


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

intervention_layer = 12
intervention_submodule = model.model.layers[intervention_layer]


# Load dataset

source_prompts = [
    "The capital of France is Paris.",
    "A city in France is Lyon.",
]

base_prompts = [
    "The capital of Germany is Berlin.",
    "A city in Germany is Hamburg.",
]

source_encoding = model.tokenizer.batch_encode_plus(source_prompts, return_tensors="pt", padding=True, padding_side="left", truncation=True, max_length=64)
base_encoding = model.tokenizer.batch_encode_plus(base_prompts, return_tensors="pt", padding=True, padding_side="left", truncation=True, max_length=64)
source_final_entity_pos = [-1, -1]
base_final_entity_pos = [-1, -1]


# Pre-cache source activations
with torch.no_grad():

    with model.trace(source_encoding):
        source_act_BLD = intervention_submodule.output[0].save()

batch_arange = torch.arange(base_encoding.input_ids.shape[0])
source_act_BD = source_act_BLD[batch_arange, source_final_entity_pos, :]

# Dummy generated token ids
base_generated_token_ids = [50, 1234]
source_generated_token_ids = [2345, 45]


# Define mask
temperature = 1.0


# Define loss function
def get_logit_diff(logits_BLV, source_generated_token_ids, base_generated_token_ids):
    logits_BV = logits_BLV[:, -1, :]
    source_logits_B = logits_BV[batch_arange, source_generated_token_ids]
    base_logits_B = logits_BV[batch_arange, base_generated_token_ids]
    logit_diff_B = source_logits_B - base_logits_B
    return logit_diff_B.sum()



# Train mask

num_epochs = 10

# Naively doing model.trace() in a for loop
# for e in range(num_epochs):

#     with model.trace(base_encoding):
#         # Get base and source SAE activations
#         base_act_BLD = intervention_submodule.output[0].save()
#         base_act_BD = base_act_BLD[batch_arange, base_final_entity_pos, :]
#         base_act_BS = sae.encode(base_act_BD)
#         source_act_BS = sae.encode(source_act_BD)

#         # Subtract base and add source as determined by mask values
#         masked_diff_BS = mask_values * (source_act_BS - base_act_BS)
#         base_act_BLD[batch_arange, base_final_entity_pos] += sae.decode(masked_diff_BS)

#         logits_BLV = model.lm_head.output.save()

#     loss = get_logit_diff(logits_BLV, source_generated_token_ids, base_generated_token_ids)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     model.zero_grad()


# Using nnsight session
with model.session() as session:
    mask = torch.nn.Parameter(torch.zeros(model.config.hidden_size, device=device, dtype=llm_dtype), requires_grad=True).save()

    optimizer = torch.optim.AdamW([mask], lr=1e-3)

    with session.iter(range(num_epochs)) as batch:

        with model.trace(base_encoding):
            # Get base and source SAE activations
            base_act_BLD = intervention_submodule.output[0].save()
            base_act_BD = base_act_BLD[batch_arange, base_final_entity_pos, :]

            # Subtract base and add source as determined by mask values
            mask_values = torch.sigmoid(mask / temperature)

            masked_diff_BD = mask_values * (source_act_BD - base_act_BD)
            base_act_BLD[batch_arange, base_final_entity_pos] += masked_diff_BD

            logits_BLV = model.lm_head.output

            # Outside of trace
            loss = get_logit_diff(logits_BLV, source_generated_token_ids, base_generated_token_ids)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

            session.log(loss.item().save())



print(f'Mask mean: {mask.mean()}, std: {mask.std()}')
