import gc
import os
import torch
import math
import yaml

from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer


class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    def __init__(
        self,
        cfg,
        model: HookedTransformer,
        create_dataloader=True,
    ):
        self.cfg = cfg
        self.model = model

        if isinstance(cfg.dataset, str):
            self.dataset = load_dataset(
                cfg.dataset, split="train", streaming=True, trust_remote_code=True
            )
        elif not cfg.dataset:
            raise ValueError("Dataset must be specified.")
        else:
            self.dataset = cfg.dataset

        self.iterable_dataset = iter(self.dataset)

        # check if it's tokenized
        if "tokens" in next(self.iterable_dataset).keys():
            self.cfg.is_dataset_tokenized = True
        elif "text" in next(self.iterable_dataset).keys():
            self.cfg.is_dataset_tokenized = False

        if self.cfg.use_cached_activations:
            # Sanity check: does the cache directory exist?
            assert os.path.exists(
                self.cfg.cached_activations_path
            ), f"Cache directory {self.cfg.cached_activations_path} does not exist."

            self.next_cache_idx = 0  # which file to open next
            self.next_idx_within_buffer = 0  # where to start reading from in that file

        if create_dataloader:
            print("creating data loader")
            print("buffer")
            self.storage_buffer = self.generate_buffer(self.cfg.n_batches_in_store_buffer // 2)
            print("dataloader")
            self.dataloader = self.get_data_loader()

    def get_next_tokenized_data(self):
        """
        Returns next row of tokenized data from dataset
        can deal with text or tokens datasets
        """
        if not self.cfg.is_dataset_tokenized:
            s = next(self.iterable_dataset)["text"]  # what happens when this runs out of dataset?
            tokens = self.model.to_tokens(
                s,
                truncate=True,
                move_to_device=True,
            ).squeeze(0)
            assert len(tokens.shape) == 1, f"tokens.shape should be 1D but was {tokens.shape}"
        else:
            tokens = torch.tensor(
                next(self.iterable_dataset)["tokens"],
                dtype=torch.long,
                device=self.cfg.device,
                requires_grad=False,
            )
        return tokens

    def get_batch_tokenized_data(self):
        """
        Returns a batch of tokenized data
        """
        # Defines useful variables
        store_batch_size = self.cfg.store_batch_size
        context_size = self.cfg.context_size
        device = self.cfg.device

        bos_token = torch.tensor(
            [self.model.tokenizer.bos_token_id],
            dtype=torch.long,
            device=self.cfg.device,
        )

        # Intialise empty batch_tokens tensor
        batch_tokens = torch.zeros(
            size=(store_batch_size, self.cfg.context_size),
            dtype=torch.long,
            device=self.cfg.device,
            requires_grad=False,
        )
        batches_completed = 0
        context_tokens = []
        context_tokens.append(bos_token)
        space_left_in_context = context_size - 1

        # Repeat until all batches are filled
        while batches_completed < store_batch_size:
            try:
                tokenized_data = self.get_next_tokenized_data()[1:]
            except StopIteration:
                if self.cfg.loop_dataset:
                    self.iterable_dataset = iter(self.dataset)
                    tokenized_data = self.get_next_tokenized_data()[1:]
                else:
                    raise

            len_tokenized_data_left = tokenized_data.shape[0]

            # Repeat until we use up all the tokens generated by iterable dataset
            while len_tokenized_data_left > 0 and batches_completed < store_batch_size:
                # If the tokenized data is too short, add it all to the context and get more
                if len_tokenized_data_left < space_left_in_context:
                    context_tokens.append(tokenized_data[-len_tokenized_data_left:])
                    space_left_in_context = space_left_in_context - len_tokenized_data_left
                    len_tokenized_data_left = 0

                # If the tokenized data is too large, just take what you need to fill the context
                elif len_tokenized_data_left >= space_left_in_context:
                    # Makes sure it works if you end exactly at the end of the current context
                    if len_tokenized_data_left > space_left_in_context:
                        end_index = -len_tokenized_data_left + space_left_in_context
                    elif len_tokenized_data_left == space_left_in_context:
                        end_index = None

                    context_tokens.append(tokenized_data[-len_tokenized_data_left:end_index])
                    context_tokens_cat = torch.cat(context_tokens, dim=0)

                    batch_tokens[batches_completed] = context_tokens_cat.unsqueeze(0)

                    if bos_token[0].item() in batch_tokens[batches_completed][1:]:
                        print(
                            bos_token[0].item(), batch_tokens[batches_completed][1:].argmax().item()
                        )

                    batches_completed += 1
                    len_tokenized_data_left = len_tokenized_data_left - space_left_in_context

                    # Create new context and add BOS only if we have not run out of the tokenized data
                    # (otherwise it will be added when we get new tokenized data)
                    context_tokens = []
                    if len_tokenized_data_left > 0:
                        context_tokens.append(bos_token)
                        space_left_in_context = context_size - 1
                    else:
                        space_left_in_context = context_size

        return batch_tokens

    def get_batch_activations(self, batch_tokens):
        """
        Computes activations for a batch of tokens
        Accounts for one head, flattened attn_z, or normal mlp_out
        """
        activations = self.model.run_with_cache(
            batch_tokens,
            names_filter=self.cfg.hook_point,
            stop_at_layer=self.cfg.hook_point_layer + 1,
        )[1][self.cfg.hook_point]

        if self.cfg.remove_bos_tokens:
            activations = activations[:, 1:]

        if self.cfg.flatten_activations_over_layer:
            return activations.view(activations.shape[0], activations.shape[1], -1)
        elif self.cfg.hook_point.endswith(("attn_pattern", "attn_scores")):
            print(activations.shape)
            n_ctx = activations.shape[-1]
            head_acts = activations[:, self.cfg.hook_point_head_index, :, :]
            return head_acts.view(head_acts.shape[0], head_acts.shape[1], -1)
        elif self.cfg.hook_point_head_index is not None:
            return activations[:, :, self.cfg.hook_point_head_index]
        else:
            return activations

    def generate_buffer(self, n_batches_in_buffer):
        """
        Generates buffer of shape (store_batch_size * n_batches_in_buffer * context_size, d_in)
        containing activations
        """
        total_size = self.cfg.store_batch_size * n_batches_in_buffer

        if self.cfg.use_cached_activations:
            buffer = self.load_saved_buffer(n_batches_in_buffer)
            return buffer

        if self.cfg.remove_bos_tokens:
            ctx_size = self.cfg.context_size - 1
        else:
            ctx_size = self.cfg.context_size

        # Initialize empty tensor buffer of the maximum required size
        self.cfg.dtype = torch.float32
        buffer = torch.zeros(
            (total_size, ctx_size, self.cfg.d_in),
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )

        # Iteratively add each batch to buffer
        for buffer_index in range(0, total_size, self.cfg.store_batch_size):
            batch_tokens = self.get_batch_tokenized_data()
            batch_activations = self.get_batch_activations(batch_tokens)
            buffer[buffer_index : buffer_index + self.cfg.store_batch_size] = batch_activations

        # Reshape to (store_batch_size * n_batches_in_buffer * context_size, d_in)
        buffer = buffer.reshape(-1, self.cfg.d_in)

        # Randomise order of buffer
        buffer = buffer[torch.randperm(buffer.shape[0])]

        return buffer

    def load_saved_buffer(self, n_batches_in_buffer):
        # Load the activations from disk
        buffer_total_tokens = (
            self.cfg.store_batch_size * n_batches_in_buffer * self.cfg.context_size
        )

        # Initialize an empty tensor (flattened along all dims except d_in)
        new_buffer = torch.zeros(
            (buffer_total_tokens, self.cfg.d_in), dtype=self.cfg.dtype, device=self.cfg.device
        )
        n_tokens_filled = 0

        # The activations may be split across multiple files,
        # Or we might only want a subset of one file (depending on the sizes)
        while n_tokens_filled < buffer_total_tokens:
            # Load the next file
            # Make sure it exists
            if not os.path.exists(f"{self.cfg.cached_activations_path}/a{self.next_cache_idx}.pt"):
                if self.cfg.loop_dataset:
                    self.next_cache_idx = 0
                    self.next_idx_within_buffer = 0
                    continue
                else:
                    print("\n\nWarning: Ran out of cached activation files earlier than expected.")
                    print(
                        f"Expected to have {buffer_total_tokens} activations, but only found {n_tokens_filled}., self.next_cache_idx, {self.next_cache_idx}"
                    )
                    if (
                        buffer_total_tokens
                        % (self.cfg.total_training_steps * self.cfg.train_batch_size)
                        != 0
                    ):
                        print(
                            "This might just be a rounding error",
                            "your store_batch_size * n_batches_in_buffer * context_size is not divisible by your total_training_tokens",
                        )
                    print(f"Returning a buffer of size {n_tokens_filled} instead.")
                    print("\n\n")
                    new_buffer = new_buffer[:n_tokens_filled]

                    raise IndexError("Not enough activations saved")

            activations = torch.load(
                self.cfg.cached_activations_path + "/a" + str(self.next_cache_idx) + ".pt"
            )
            # log("loaded file")

            # If we only want a subset of the file, take it
            taking_subset_of_file = False
            if n_tokens_filled + activations.shape[0] > buffer_total_tokens:
                activations = activations[: buffer_total_tokens - n_tokens_filled]
                taking_subset_of_file = True

            # Add it to the buffer
            new_buffer[n_tokens_filled : n_tokens_filled + activations.shape[0]] = activations

            # Update counters
            n_tokens_filled += activations.shape[0]
            if taking_subset_of_file:
                self.next_idx_within_buffer = activations.shape[0]
            else:
                self.next_cache_idx += 1
                self.next_idx_within_buffer = 0

        return new_buffer

    def get_data_loader(
        self,
    ) -> DataLoader:
        """
        Return a torch.utils.dataloader which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).

        """
        # Create new buffer by mixing stored and new buffer
        mixing_buffer = torch.cat(
            [self.generate_buffer(self.cfg.n_batches_in_store_buffer // 2), self.storage_buffer]
        )

        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        # Put 50 % in storage & other 50 % in a dataloader
        self.storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        dataloader = iter(
            DataLoader(
                mixing_buffer[mixing_buffer.shape[0] // 2 :],
                batch_size=self.cfg.train_batch_size,
                shuffle=True,
            )
        )

        return dataloader

    def next_batch(self):
        """
        Get the next batch from the current DataLoader.
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            # Try to get the next batch
            return next(self.dataloader)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self.dataloader = self.get_data_loader()
            return next(self.dataloader)
