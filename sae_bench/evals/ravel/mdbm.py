import gc

import sae_lens
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

import sae_bench.sae_bench_utils.activation_collection as activation_collection
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
from sae_bench.evals.ravel.mdas import MDAS


class MDBM(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: RAVELEvalConfig,
        sae: sae_lens.SAE,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.layer_intervened = sae.cfg.hook_layer
        self.binary_mask = torch.nn.Parameter(
            torch.zeros(sae.cfg.d_sae, device=model.device, dtype=torch.float32),
            requires_grad=True,
        )
        self.batch_size = config.llm_batch_size
        self.device = model.device
        self.temperature: float = 1

    def create_intervention_hook(
        self,
        source_rep_BD: torch.Tensor,
        base_pos_B: torch.Tensor,
        training_mode: bool = False,
        add_error: bool = False,
    ):
        """
        Creates and returns an intervention hook function that applies a binary mask
        to modify activations.

        Args:
            source_rep_BD: Source representation tensor
            base_pos_B: Base positions tensor
            training_mode: Whether to use sigmoid (training) or hard threshold (eval)
            add_error: Whether to add error to the modified activations - we default to False, as it typically degrades performance

        Returns:
            A hook function that can be registered with a PyTorch module
        """

        def intervention_hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                resid_BLD = outputs[0]
                rest = outputs[1:]
            else:
                raise ValueError("Unexpected output shape")

            if resid_BLD.shape[1] == 1:
                # This means we are generating with the KV cache and the intervention has already been applied
                return outputs

            with torch.no_grad():
                source_act_BF = self.sae.encode(source_rep_BD)
                resid_BD = resid_BLD[list(range(resid_BLD.shape[0])), base_pos_B, :]
                base_act_BF = self.sae.encode(resid_BD)

            # Use true binary mask in eval mode, sigmoid in training mode
            if not training_mode:
                mask_values_F = (self.binary_mask > 0).to(dtype=self.binary_mask.dtype)
            else:
                mask_values_F = torch.sigmoid(self.binary_mask / self.temperature)

            modified_act_BF = (
                1 - mask_values_F
            ) * base_act_BF + mask_values_F * source_act_BF

            modified_resid_BD = self.sae.decode(
                modified_act_BF.to(dtype=source_rep_BD.dtype)
            )

            if add_error:
                error_BD = resid_BD - self.sae.decode(base_act_BF)
                modified_resid_BD = modified_resid_BD + error_BD

            resid_BLD[list(range(resid_BLD.shape[0])), base_pos_B, :] = (
                modified_resid_BD
            )

            return (resid_BLD, *rest)

        return intervention_hook

    def forward(
        self,
        base_encoding_BL,
        source_encoding_BL,
        base_pos_B,
        source_pos_B,
        training_mode: bool = False,
    ):
        with torch.no_grad():
            # Get source representation
            source_rep_BD = activation_collection.get_layer_activations(
                self.model, self.layer_intervened, source_encoding_BL, source_pos_B
            )

        intervention_hook = self.create_intervention_hook(
            source_rep_BD,
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


def compute_loss(intervened_logits_BV, target_attr_B):
    """
    Compute multi-task loss combining:
    - Cause loss: Target attribute should match source
    - Iso loss: Other attributes should match base

    NOTE: For cause loss, target_attr_B is the source attribute value.
    For iso loss, target_attr_B is the base attribute value.
    This is set during dataset creation, so we can just use cross entropy loss with target_attr_B
    for both cause and iso loss.

    Returns:
        Tuple of (loss, accuracy) where accuracy is the raw prediction accuracy
        for the final token
    """
    loss = F.cross_entropy(intervened_logits_BV, target_attr_B)

    # Calculate accuracy
    predictions_B = intervened_logits_BV.argmax(dim=-1)
    accuracy = (predictions_B == target_attr_B).float().mean()

    return loss, accuracy


def get_cause_isolation_score_estimates(
    intervened_logits_BV, source_pred_B, base_pred_B
):
    """
    Calculate cause and isolation scores based on predictions.

    Args:
        intervened_logits_BV: Logits from the intervened model
        source_pred_B: Target predictions from source examples
        base_pred_B: Target predictions from base examples

    Returns:
        Tuple of (cause_score, isolation_score, cause_count, isolation_count)
    """
    predictions_B = intervened_logits_BV.argmax(dim=-1)

    # Identify cause and isolation examples
    is_isolation = base_pred_B == source_pred_B
    is_cause = ~is_isolation

    # Count examples in each category
    cause_count = is_cause.sum().item()
    isolation_count = is_isolation.sum().item()

    # Calculate accuracy for each category
    cause_correct = ((predictions_B == source_pred_B) & is_cause).sum().item()
    isolation_correct = ((predictions_B == base_pred_B) & is_isolation).sum().item()

    # Calculate scores (handle division by zero)
    cause_score = cause_correct / cause_count if cause_count > 0 else 0.0
    isolation_score = (
        isolation_correct / isolation_count if isolation_count > 0 else 0.0
    )

    return cause_score, isolation_score, cause_count, isolation_count


@torch.no_grad()
def get_validation_loss(mdbm: MDBM, val_loader: torch.utils.data.DataLoader):
    """Compute validation loss across the validation dataset"""
    mdbm.eval()
    val_loss = 0
    val_accuracy = 0
    val_batch_count = 0
    val_cause_score = 0
    val_isolation_score = 0
    total_cause_count = 0
    total_isolation_count = 0

    with torch.no_grad():
        for batch in val_loader:
            (
                base_encodings_BL,
                source_encodings_BL,
                base_pos_B,
                source_pos_B,
                base_pred_B,
                source_pred_B,
                base_text_str,
                base_label_str,
            ) = batch
            intervened_logits_BV, _ = mdbm(
                base_encodings_BL, source_encodings_BL, base_pos_B, source_pos_B
            )
            loss, accuracy = compute_loss(intervened_logits_BV, source_pred_B)

            # Calculate cause and isolation scores
            cause_score, isolation_score, cause_count, isolation_count = (
                get_cause_isolation_score_estimates(
                    intervened_logits_BV, source_pred_B, base_pred_B
                )
            )

            val_loss += loss.item()
            val_accuracy += accuracy.item()
            val_cause_score += cause_score * cause_count
            val_isolation_score += isolation_score * isolation_count
            total_cause_count += cause_count
            total_isolation_count += isolation_count
            val_batch_count += 1

    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
    avg_val_accuracy = val_accuracy / val_batch_count if val_batch_count > 0 else 0
    avg_val_cause_score = (
        val_cause_score / total_cause_count if total_cause_count > 0 else 0
    )
    avg_val_isolation_score = (
        val_isolation_score / total_isolation_count if total_isolation_count > 0 else 0
    )

    return avg_val_loss, avg_val_accuracy, avg_val_cause_score, avg_val_isolation_score


def train_mdbm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: RAVELEvalConfig,
    sae: sae_lens.SAE,
    train_loader,
    val_loader,
    verbose: bool = False,
    train_mdas: bool = False,
) -> MDBM:
    initial_temperature = 1
    final_temperature = 1e-4
    temperature_schedule = torch.logspace(
        torch.log10(torch.tensor(initial_temperature)),
        torch.log10(torch.tensor(final_temperature)),
        config.num_epochs * len(train_loader),
        device=model.device,
        dtype=model.dtype,
    )

    if train_mdas:
        mdbm = MDAS(
            model,
            tokenizer,
            config,
            sae,
        ).to(model.device)
        optimizer = torch.optim.Adam(
            [mdbm.binary_mask, mdbm.transform_matrix], lr=config.learning_rate
        )
        orthonormal = True
    else:
        mdbm = MDBM(
            model,
            tokenizer,
            config,
            sae,
        ).to(model.device)
        optimizer = torch.optim.Adam([mdbm.binary_mask], lr=config.learning_rate)
        orthonormal = False

    if verbose:
        # Get initial validation loss
        (
            initial_val_loss,
            initial_val_accuracy,
            initial_val_cause_score,
            initial_val_isolation_score,
        ) = get_validation_loss(mdbm, val_loader)  # type: ignore
        print(
            f"Initial validation loss: {initial_val_loss:.4f}, accuracy: {initial_val_accuracy:.4f}, cause score: {initial_val_cause_score:.4f}, isolation score: {initial_val_isolation_score:.4f}"
        )

    for epoch in range(config.num_epochs):
        mdbm.train()
        train_loss = 0
        train_accuracy = 0
        batch_count = 0
        log_count = 0

        gc.collect()
        torch.cuda.empty_cache()

        for batch in train_loader:
            mdbm.temperature = temperature_schedule[
                epoch * len(train_loader) + batch_count
            ].item()
            (
                base_encodings_BL,
                source_encodings_BL,
                base_pos_B,
                source_pos_B,
                base_pred_B,
                source_pred_B,
                base_text_str,
                base_label_str,
            ) = batch

            optimizer.zero_grad()

            intervened_logits_BV, _ = mdbm(
                base_encodings_BL,
                source_encodings_BL,
                base_pos_B,
                source_pos_B,
                training_mode=True,
            )
            loss, accuracy = compute_loss(intervened_logits_BV, source_pred_B)

            loss.backward()
            optimizer.step()
            if train_mdas and orthonormal:
                with torch.no_grad():
                    Q, R = torch.linalg.qr(mdbm.transform_matrix, mode="reduced")
                    # Correct sign to enforce det=+1 for rotation matrices
                    det = torch.det(Q)
                    if det < 0:
                        # Flip the sign of one column to make det=+1
                        Q[:, 0] = -Q[:, 0]
                    mdbm.transform_matrix[...] = Q  # type: ignore

            train_loss += loss.item()
            train_accuracy += accuracy.item()
            batch_count += 1
            log_count += 1

            if log_count % 20 == 0 and verbose:
                print(
                    f"Epoch {epoch + 1}/{config.num_epochs} - "
                    f"Train Loss: {train_loss / log_count:.4f}, "
                    f"Train Accuracy: {train_accuracy / log_count:.4f}"
                )
                train_loss = 0
                train_accuracy = 0
                log_count = 0

        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        avg_train_accuracy = train_accuracy / batch_count if batch_count > 0 else 0
        # Validation

        # Print losses if verbose
        if verbose:
            (
                avg_val_loss,
                avg_val_accuracy,
                avg_val_cause_score,
                avg_val_isolation_score,
            ) = get_validation_loss(mdbm, val_loader)  # type: ignore
            percent_above_zero = (mdbm.binary_mask > 0).float().mean().item()
            print(
                f"Epoch {epoch + 1}/{config.num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Accuracy: {avg_train_accuracy:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Val Accuracy: {avg_val_accuracy:.4f}, "
                f"Percent above zero: {percent_above_zero:.4f}, "
                f"Val Cause Score: {avg_val_cause_score:.4f}, "
                f"Val Isolation Score: {avg_val_isolation_score:.4f}"
            )

        # Early stopping
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     patience_counter = 0
        #     if verbose:
        #         print(f"  New best validation loss: {best_val_loss:.4f}")
        # else:
        #     patience_counter += 1
        #     if verbose:
        #         print(f"  No improvement for {patience_counter} epochs")

        # if patience_counter >= config.early_stop_patience:
        #     print(f"Early stopping at epoch {epoch + 1}")
        #     break

    # if verbose:
    #     print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

    return mdbm  # type: ignore
