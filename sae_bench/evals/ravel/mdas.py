import sae_lens
import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

import sae_bench.sae_bench_utils.activation_collection as activation_collection
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig


class MDAS(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: RAVELEvalConfig,
        sae: sae_lens.SAE,  # Kept for API compatibility
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae  # Kept for API compatibility
        self.layer_intervened = sae.cfg.hook_layer

        # Get the hidden dimension from the model
        hidden_dim = model.config.hidden_size

        # Initialize a square transformation matrix
        # Using Xavier/Glorot initialization for better training dynamics
        # self.transform_matrix = torch.nn.Parameter(
        #     torch.nn.init.xavier_uniform_(
        #         torch.zeros(
        #             hidden_dim, hidden_dim, device=model.device, dtype=model.dtype
        #         )
        #     ),
        #     requires_grad=True,
        # )

        # Add identity initialization option (can be uncommented if needed)
        self.transform_matrix = torch.nn.Parameter(
            torch.eye(hidden_dim, device=model.device, dtype=torch.float32),
            requires_grad=True,
        )
        self.binary_mask = torch.nn.Parameter(
            torch.zeros(hidden_dim, device=model.device, dtype=torch.float32),
            requires_grad=True,
        )

        self.batch_size = config.llm_batch_size
        self.device = model.device
        self.temperature = 1e-2  # Kept for API compatibility

    def create_intervention_hook(
        self,
        source_rep_BD: torch.Tensor,
        base_pos_B: torch.Tensor,
        training_mode: bool = False,
    ):
        def intervention_hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                resid_BLD = outputs[0]
                rest = outputs[1:]
            else:
                raise ValueError("Unexpected output shape")

            if resid_BLD.shape[1] == 1:
                # This means we are generating with the KV cache and the intervention has already been applied
                return outputs

            # Get the base activations at the target position
            resid_BD = resid_BLD[list(range(resid_BLD.shape[0])), base_pos_B, :]

            # Apply the transformation matrix directly to the source representation
            # This gives us much more flexibility than the binary mask
            rotated_source_BD = torch.matmul(
                source_rep_BD.to(dtype=torch.float32), self.transform_matrix
            )
            rotated_resid_BD = torch.matmul(
                resid_BD.to(dtype=torch.float32), self.transform_matrix
            )

            # Use true binary mask in eval mode, sigmoid in training mode
            if not training_mode:
                mask_values_D = (self.binary_mask > 0).to(dtype=self.binary_mask.dtype)
            else:
                mask_values_D = torch.sigmoid(self.binary_mask / self.temperature)

            # use this to hardcode the mask
            # mask_values_D = torch.zeros_like(binary_mask_D)
            # mask_values_D[:50] = 1

            modified_resid_BD = (
                1 - mask_values_D
            ) * rotated_resid_BD + mask_values_D * rotated_source_BD

            modified_resid_BD = torch.matmul(modified_resid_BD, self.transform_matrix.T)

            # Replace the base activations with the transformed source activations
            resid_BLD[list(range(resid_BLD.shape[0])), base_pos_B, :] = (
                modified_resid_BD.to(dtype=resid_BLD.dtype)
            )

            return (resid_BLD, *rest)

        return intervention_hook

    def forward(
        self,
        base_encoding_BL,
        source_encoding_BL,
        base_pos_B,
        source_pos_B,
        training_mode: bool = False,  # Kept for API compatibility
    ):
        with torch.no_grad():
            # Get source representation
            source_rep = activation_collection.get_layer_activations(
                self.model, self.layer_intervened, source_encoding_BL, source_pos_B
            )

        intervention_hook = self.create_intervention_hook(
            source_rep,
            base_pos_B,
            training_mode,
        )

        handle = activation_collection.get_module(
            self.model, self.layer_intervened
        ).register_forward_hook(intervention_hook)

        logits_BV = self.model(
            input_ids=base_encoding_BL["input_ids"].to(self.model.device),
            attention_mask=base_encoding_BL.get("attention_mask", None),
        ).logits[:, -1, :]

        handle.remove()

        predicted_B = logits_BV.argmax(dim=-1)

        # Format outputs
        predicted_text = []
        for i in range(logits_BV.shape[0]):
            predicted_text.append(self.tokenizer.decode(predicted_B[i]))

        return logits_BV, predicted_text
