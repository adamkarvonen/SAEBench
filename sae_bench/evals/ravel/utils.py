def inspect_dataloader(dataloader, tokenizer, n=2, view_strs: bool = False):
    """
    Debug function to inspect the first n elements of the dataloader.

    Args:
        dataloader: The dataloader to inspect
        tokenizer: The tokenizer to use for detokenizing
        n: Number of batches to inspect (default: 2)
    """
    print(f"\n{'=' * 50}\nDEBUGGING DATALOADER CONTENTS\n{'=' * 50}")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n:
            break

        print(f"\n{'-' * 50}\nBATCH {batch_idx + 1}/{n}\n{'-' * 50}")

        (
            base_encoding_BL,
            source_encoding_BL,
            base_pos_B,
            source_pos_B,
            base_pred_B,
            source_pred_B,
        ) = batch

        # Print shapes
        print("\nSHAPES:")
        print(f"  base_encoding_BL['input_ids']: {base_encoding_BL['input_ids'].shape}")
        print(
            f"  base_encoding_BL['attention_mask']: {base_encoding_BL['attention_mask'].shape}"
        )
        print(
            f"  source_encoding_BL['input_ids']: {source_encoding_BL['input_ids'].shape}"
        )
        print(
            f"  source_encoding_BL['attention_mask']: {source_encoding_BL['attention_mask'].shape}"
        )
        print(f"  base_pos_B: {base_pos_B.shape}")
        print(f"  source_pos_B: {source_pos_B.shape}")
        print(f"  base_pred_B: {base_pred_B.shape}")
        print(f"  source_pred_B: {source_pred_B.shape}")

        if view_strs:
            # Inspect individual examples in the batch
            batch_size = base_encoding_BL["input_ids"].shape[0]
            examples_to_show = min(3, batch_size)  # Show at most 3 examples per batch

            for i in range(examples_to_show):
                print(f"\nEXAMPLE {i + 1}/{examples_to_show}:")

                # Get base sequence
                base_ids = base_encoding_BL["input_ids"][i].tolist()
                base_text = tokenizer.decode(base_ids)

                # Get source sequence
                source_ids = source_encoding_BL["input_ids"][i].tolist()
                source_text = tokenizer.decode(source_ids)

                # Get positions and predictions
                base_position = base_pos_B[i].item()
                source_position = source_pos_B[i].item()
                base_prediction = base_pred_B[i].item()
                source_prediction = source_pred_B[i].item()

                # Get tokens at the positions of interest
                base_token_at_pos = tokenizer.decode([base_ids[base_position]])
                source_token_at_pos = tokenizer.decode([source_ids[source_position]])
                base_pred_token = tokenizer.decode([base_prediction])
                source_pred_token = tokenizer.decode([source_prediction])

                # Print everything
                print(f"  BASE TEXT: {base_text}")
                print(f"  SOURCE TEXT: {source_text}")
                print(f"  BASE POSITION: {base_position}")
                print(f"  SOURCE POSITION: {source_position}")
                print(f"  TOKEN AT BASE POSITION: '{base_token_at_pos}'")
                print(f"  TOKEN AT SOURCE POSITION: '{source_token_at_pos}'")
                print(f"  BASE PREDICTION: {base_prediction} ('{base_pred_token}')")
                print(
                    f"  SOURCE PREDICTION: {source_prediction} ('{source_pred_token}')"
                )

    print(f"\n{'=' * 50}\nDEBUGGING COMPLETE\n{'=' * 50}")
