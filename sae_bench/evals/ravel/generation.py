from tqdm import tqdm
from jaxtyping import Int
from typing import Optional, Union, List
import torch
from transformers import AutoTokenizer, BatchEncoding, AutoModelForCausalLM

import sae_bench.sae_bench_utils.activation_collection as activation_collection
import sae_bench.evals.ravel.mdbm as mdbm


def custom_left_padding(
    tokenizer: AutoTokenizer,
    input_ids: list[list[int]],
    pad_to_length: Optional[int] = None,
) -> tuple[
    Int[torch.Tensor, "batch_size seq_len"], Int[torch.Tensor, "batch_size seq_len"]
]:
    """
    Left pad the input ids with the pad token.
    """
    max_length = max(len(ids) for ids in input_ids)
    if pad_to_length is not None:
        max_length = pad_to_length

    if hasattr(tokenizer, "pad_token_id"):
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = tokenizer.eos_token_id
    padded_input_ids = [
        [pad_token_id] * (max_length - len(ids)) + ids for ids in input_ids
    ]
    padded_input_ids = torch.tensor(padded_input_ids)
    attention_mask = (padded_input_ids != pad_token_id).long()
    return padded_input_ids, attention_mask


def generate_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids_BL: Union[Int[torch.Tensor, "batch_size seq_len"], List[List[int]]],
    attention_mask_BL: Optional[Int[torch.Tensor, "batch_size seq_len"]] = None,
    max_new_tokens: int = 8,
    llm_batch_size: int = 32,
    return_first_generated_token: bool = False,
):
    """
    Generate completions for a batch of prompts.
    You can either pass
    1. a tokenized and padded input ids + attention masks as torch tensors
    2. a list of lists of tokenized input ids without padding
    """
    num_total_prompts = len(input_ids_BL)

    generations = []
    for batch_begin in tqdm(
        range(0, num_total_prompts, llm_batch_size),
        desc="Generate completions to test model knowledge",
    ):
        # Draw batch from input_ids_BL
        if isinstance(input_ids_BL, torch.Tensor):
            assert attention_mask_BL is not None, (
                "If input_ids_BL is a torch tensor, attention_mask_BL must also be provided."
            )
            input_ids = input_ids_BL[batch_begin : batch_begin + llm_batch_size].to(
                model.device
            )
            attention_mask = attention_mask_BL[
                batch_begin : batch_begin + llm_batch_size
            ].to(model.device)
        else:
            input_ids, attention_mask = custom_left_padding(
                tokenizer, input_ids_BL[batch_begin : batch_begin + llm_batch_size]
            )
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

        # Generate using huggingface model
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding for reproducibility
        )
        generated_ids = output_ids[:, -max_new_tokens:]
        generations.append(generated_ids)

    generations = torch.cat(generations, dim=0)
    generated_strings = tokenizer.batch_decode(generations)
    if return_first_generated_token:
        return generated_strings, generations[:, 0].tolist()
    return generated_strings


if __name__ == "__main__":
    # Test the generation
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device("cuda:0")
    model = LanguageModel(
        "eleutherAI/pythia-70m-deduped", device_map=device, dispatch=True
    )
    tokenizer = AutoTokenizer.from_pretrained("eleutherAI/pythia-70m-deduped")
    tokenizer.pad_token = tokenizer.eos_token

    encoded = model.tokenizer.batch_encode_plus(
        ["Hello, world!", "Moin "],
        return_tensors="pt",
        padding="max_length",
        max_length=20,
    ).to(device)
    input_ids_BL = encoded["input_ids"]
    attention_mask_BL = encoded["attention_mask"]

    generated_strings = generate_batched(
        model, tokenizer, input_ids_BL, attention_mask_BL, max_new_tokens=10
    )
    print(generated_strings)
