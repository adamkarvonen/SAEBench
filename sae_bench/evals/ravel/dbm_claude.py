import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

@dataclass
class TrainingMetrics:
    """Store training metrics for visualization"""
    loss_history: List[float] = None
    sparsity_history: List[float] = None
    kl_div_history: List[float] = None
    mask_evolution: List[torch.Tensor] = None
    temperature_history: List[float] = None
    intervention_accuracy: List[float] = None
    
    def __post_init__(self):
        self.loss_history = []
        self.sparsity_history = []
        self.kl_div_history = []
        self.mask_evolution = []
        self.temperature_history = []
        self.intervention_accuracy = []

class MaskTrainingMonitor:
    """Monitor and visualize mask training progress"""
    def __init__(self, save_dir: str):
        self.metrics = TrainingMetrics()
        self.save_dir = save_dir
        
    def update(self, 
              loss: float, 
              sparsity: float, 
              kl_div: float,
              mask_values: torch.Tensor,
              temperature: float,
              accuracy: float):
        """Update metrics after each training step"""
        self.metrics.loss_history.append(loss)
        self.metrics.sparsity_history.append(sparsity)
        self.metrics.kl_div_history.append(kl_div)
        self.metrics.mask_evolution.append(mask_values.detach().cpu())
        self.metrics.temperature_history.append(temperature)
        self.metrics.intervention_accuracy.append(accuracy)
        
    def plot_training_progress(self, epoch: int):
        """Create comprehensive training visualization"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 20))
        fig.suptitle(f'Training Progress - Epoch {epoch}')
        
        # Plot loss
        axes[0,0].plot(self.metrics.loss_history)
        axes[0,0].set_title('Loss History')
        axes[0,0].set_xlabel('Step')
        axes[0,0].set_ylabel('Loss')
        
        # Plot sparsity
        axes[0,1].plot(self.metrics.sparsity_history)
        axes[0,1].set_title('Mask Sparsity')
        axes[0,1].set_xlabel('Step')
        axes[0,1].set_ylabel('Sparsity Ratio')
        
        # Plot KL divergence
        axes[1,0].plot(self.metrics.kl_div_history)
        axes[1,0].set_title('KL Divergence')
        axes[1,0].set_xlabel('Step')
        axes[1,0].set_ylabel('KL Div')
        
        # Plot temperature
        axes[1,1].plot(self.metrics.temperature_history)
        axes[1,1].set_title('Temperature Annealing')
        axes[1,1].set_xlabel('Step')
        axes[1,1].set_ylabel('Temperature')
        
        # Plot mask evolution heatmap
        mask_history = torch.stack(self.metrics.mask_evolution[-100:])  # Last 100 steps
        sns.heatmap(mask_history.numpy(), ax=axes[2,0], cmap='viridis')
        axes[2,0].set_title('Mask Evolution (Recent Steps)')
        axes[2,0].set_xlabel('Neuron Index')
        axes[2,0].set_ylabel('Step')
        
        # Plot intervention accuracy
        axes[2,1].plot(self.metrics.intervention_accuracy)
        axes[2,1].set_title('Intervention Accuracy')
        axes[2,1].set_xlabel('Step')
        axes[2,1].set_ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_progress_epoch_{epoch}.png')
        plt.close()
        
    def save_metrics(self):
        """Save metrics to CSV for later analysis"""
        metrics_df = pd.DataFrame({
            'step': range(len(self.metrics.loss_history)),
            'loss': self.metrics.loss_history,
            'sparsity': self.metrics.sparsity_history,
            'kl_div': self.metrics.kl_div_history,
            'temperature': self.metrics.temperature_history,
            'accuracy': self.metrics.intervention_accuracy
        })
        metrics_df.to_csv(f'{self.save_dir}/training_metrics.csv', index=False)

def train_mask(model, sae, dbm_dataloader, config):
    """Enhanced training function with monitoring"""
    monitor = MaskTrainingMonitor(save_dir=config['save_dir'])
    
    with model.session() as session:
        # Initialize mask
        mask = torch.nn.Parameter(
            torch.zeros(sae.cfg.d_sae, device=config['device'], 
                       dtype=config['dtype']), 
            requires_grad=True
        ).save()
        
        optimizer = torch.optim.AdamW([mask], lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=len(dbm_dataloader) * config['num_epochs']
        )
        
        for epoch in range(config['num_epochs']):
            for step, batch in enumerate(tqdm(dbm_dataloader)):
                # Get base and source activations
                batch_encoding = batch["base_encoding"]
                batch_final_entity_pos = batch["base_final_entity_pos"]
                batch_source_act_BD = batch["source_act_BD"]
                
                with model.trace(batch_encoding):
                    # Forward pass
                    base_act_BLD = model.model.layers[config['intervention_layer']].output[0]
                    base_act_BD = base_act_BLD[range(config['batch_size']), 
                                             batch_final_entity_pos, :]
                    base_act_BS = sae.encode(base_act_BD.detach())
                    source_act_BS = sae.encode(batch_source_act_BD.detach())

                    # Get temperature for current step
                    current_temp = get_temperature(
                        step, 
                        epoch, 
                        config['num_epochs'], 
                        len(dbm_dataloader),
                        config['temperature_start'],
                        config['temperature_end']
                    )
                    
                    # Apply mask with straight-through estimator
                    mask_values = torch.sigmoid(mask / current_temp)
                    mask_binary = (mask_values > 0.5).float()
                    mask_final = mask_binary + (mask_values - mask_values.detach())
                    
                    # Compute modified activations
                    masked_diff_BS = mask_final * (source_act_BS - base_act_BS)
                    masked_diff_BD = sae.decode(masked_diff_BS)
                    
                    # Update model activations
                    base_act_BLD[range(config['batch_size']), 
                               batch_final_entity_pos, :] += masked_diff_BD

                    # Get model predictions
                    logits_BLV = model.lm_head.output
                    
                    # Compute losses
                    main_loss = compute_intervention_loss(
                        logits_BLV, 
                        batch["source_labels"], 
                        batch["base_labels"],
                        config['batch_size']
                    )
                    
                    kl_div = compute_kl_divergence(
                        logits_BLV,
                        model.lm_head.output_no_intervention
                    )
                    
                    # Compute sparsity
                    sparsity = torch.mean((mask_values > 0.5).float())
                    
                    # Total loss
                    loss = (main_loss + 
                           config['kl_weight'] * kl_div + 
                           config['sparsity_weight'] * sparsity)
                    
                    # Compute accuracy
                    accuracy = compute_intervention_accuracy(
                        logits_BLV,
                        batch["source_labels"]
                    )
                    
                    # Update metrics
                    monitor.update(
                        loss=loss.item(),
                        sparsity=sparsity.item(),
                        kl_div=kl_div.item(),
                        mask_values=mask_values,
                        temperature=current_temp,
                        accuracy=accuracy
                    )
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_([mask], config['grad_clip'])
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
            # Plot progress after each epoch
            monitor.plot_training_progress(epoch)
            
        # Save final metrics
        monitor.save_metrics()
        
        # Get final binary mask
        final_mask = (torch.sigmoid(mask / config['temperature_end']) > 0.5).float()
        
        return final_mask, monitor.metrics

def get_temperature(step: int, 
                   epoch: int, 
                   num_epochs: int,
                   steps_per_epoch: int,
                   temp_start: float,
                   temp_end: float) -> float:
    """Compute temperature with cosine annealing schedule"""
    progress = (epoch * steps_per_epoch + step) / (num_epochs * steps_per_epoch)
    return temp_end + 0.5 * (temp_start - temp_end) * (1 + np.cos(np.pi * progress))

def compute_intervention_loss(logits_BLV, source_ids, base_ids, batch_size):
    """Compute cross entropy loss for intervention"""
    logits_BV = logits_BLV[:, -1, :]
    log_probs = torch.log_softmax(logits_BV, dim=-1)
    target_probs = torch.zeros_like(logits_BV).scatter_(
        1, source_ids.unsqueeze(1), 1
    )
    return -torch.sum(target_probs * log_probs) / batch_size

def compute_kl_divergence(logits_intervention, logits_original):
    """Compute KL divergence between original and intervened distributions"""
    p = torch.softmax(logits_original, dim=-1)
    log_q = torch.log_softmax(logits_intervention, dim=-1)
    return torch.sum(p * (torch.log(p) - log_q)) / p.size(0)

def compute_intervention_accuracy(logits_BLV, source_ids):
    """Compute accuracy of intervention"""
    predictions = torch.argmax(logits_BLV[:, -1, :], dim=-1)
    return torch.mean((predictions == source_ids).float()).item()