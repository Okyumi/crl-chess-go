"""
Training infrastructure for CRL experiments.

Components:
1. TrajectoryBuffer: efficient storage and sampling from game trajectories
2. InfoNCE losses: forward, backward, symmetric, outcome-conditioned
3. Baseline losses: MC classification, value regression
4. CRLTrainer: unified training loop with logging, checkpointing, multi-seed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Callable


# ============================================================
# Trajectory Buffer
# ============================================================

class TrajectoryBuffer:
    """
    Efficient buffer for storing and sampling from game trajectories.
    Pre-flattens states into a contiguous array for fast GPU transfer.
    """
    
    def __init__(self, trajectories: List[Dict], gamma: float = 0.99, device: str = 'cpu'):
        self.gamma = gamma
        self.device = device
        self.trajectories = trajectories
        
        # Pre-flatten everything into arrays
        states_list = []
        traj_ids = []
        time_steps = []
        outcomes = []
        
        for tid, traj in enumerate(trajectories):
            for t, s in enumerate(traj['states']):
                states_list.append(s)
                traj_ids.append(tid)
                time_steps.append(t)
                player_sign = 1 if t % 2 == 0 else -1
                outcomes.append(traj['outcome'] * player_sign)
        
        self.states = np.array(states_list, dtype=np.float32)
        self.traj_ids = np.array(traj_ids, dtype=np.int64)
        self.time_steps = np.array(time_steps, dtype=np.int64)
        self.outcomes = np.array(outcomes, dtype=np.float32)
        self.n_states = len(self.states)
        
        # Build per-trajectory index
        self.traj_starts = {}
        self.traj_lengths = {}
        for tid, traj in enumerate(trajectories):
            mask = self.traj_ids == tid
            indices = np.where(mask)[0]
            if len(indices) > 0:
                self.traj_starts[tid] = int(indices[0])
                self.traj_lengths[tid] = len(traj['states'])
        self.n_trajs = len(trajectories)
        
        # Pre-compute trajectory outcome arrays
        self.traj_outcomes = np.array([t['outcome'] for t in trajectories])
        
        # Move to GPU if requested
        self._states_tensor = None
    
    def _get_states_tensor(self):
        if self._states_tensor is None:
            self._states_tensor = torch.from_numpy(self.states).to(self.device)
        return self._states_tensor
    
    def _geometric_sample(self, max_val: int) -> int:
        """Sample from truncated geometric distribution."""
        p = 1 - self.gamma
        sample = np.random.geometric(p) - 1
        return min(sample, max_val)
    
    def sample_forward_contrastive(self, batch_size: int, n_neg: int = 63) -> Dict[str, torch.Tensor]:
        """
        Forward InfoNCE sampling.
        Positive: (s_t, s_{t+delta}) from same trajectory, delta ~ Geom(1-gamma).
        Negatives: random states from marginal distribution.
        """
        anchors_idx = []
        positives_idx = []
        
        attempts = 0
        while len(anchors_idx) < batch_size and attempts < batch_size * 3:
            attempts += 1
            tid = np.random.randint(self.n_trajs)
            T = self.traj_lengths.get(tid, 0)
            if T < 2:
                continue
            start = self.traj_starts[tid]
            
            t = np.random.randint(0, T - 1)
            delta = max(1, self._geometric_sample(T - t - 1))
            future_t = min(t + delta, T - 1)
            
            anchors_idx.append(start + t)
            positives_idx.append(start + future_t)
        
        B = len(anchors_idx)
        if B == 0:
            return None
        
        neg_idx = np.random.randint(0, self.n_states, size=(B, n_neg))
        
        states_t = self._get_states_tensor()
        return {
            'anchors': states_t[anchors_idx],          # [B, D]
            'positives': states_t[positives_idx],       # [B, D]
            'negatives': states_t[neg_idx.flatten()].view(B, n_neg, -1),  # [B, N, D]
        }
    
    def sample_backward_contrastive(self, batch_size: int, n_neg: int = 63) -> Dict[str, torch.Tensor]:
        """
        Backward InfoNCE sampling (Eysenbach suggestion).
        Pick a state, sample a *previous* state offset by geometric distribution.
        """
        anchors_idx = []
        pasts_idx = []
        
        attempts = 0
        while len(anchors_idx) < batch_size and attempts < batch_size * 3:
            attempts += 1
            tid = np.random.randint(self.n_trajs)
            T = self.traj_lengths.get(tid, 0)
            if T < 2:
                continue
            start = self.traj_starts[tid]
            
            t = np.random.randint(1, T)
            delta = max(1, self._geometric_sample(t))
            past_t = max(0, t - delta)
            
            anchors_idx.append(start + t)
            pasts_idx.append(start + past_t)
        
        B = len(anchors_idx)
        if B == 0:
            return None
        
        neg_idx = np.random.randint(0, self.n_states, size=(B, n_neg))
        
        states_t = self._get_states_tensor()
        return {
            'anchors': states_t[anchors_idx],
            'positives': states_t[pasts_idx],
            'negatives': states_t[neg_idx.flatten()].view(B, n_neg, -1),
        }
    
    def sample_symmetric_contrastive(self, batch_size: int, n_neg: int = 63) -> Dict[str, torch.Tensor]:
        """
        Symmetric sampling: sample pairs in both temporal directions.
        For each pair, we get both forward and backward supervision.
        """
        fwd = self.sample_forward_contrastive(batch_size // 2, n_neg)
        bwd = self.sample_backward_contrastive(batch_size // 2, n_neg)
        if fwd is None or bwd is None:
            return fwd or bwd
        
        return {
            'anchors': torch.cat([fwd['anchors'], bwd['anchors']], dim=0),
            'positives': torch.cat([fwd['positives'], bwd['positives']], dim=0),
            'negatives': torch.cat([fwd['negatives'], bwd['negatives']], dim=0),
        }
    
    def sample_outcome_labeled(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample (state, outcome) pairs for MC classification."""
        idx = np.random.randint(0, self.n_states, size=batch_size)
        states_t = self._get_states_tensor()
        outcomes = self.outcomes[idx]
        
        # Map {-1, 0, 1} -> {2, 1, 0} (class indices)
        labels = np.zeros(batch_size, dtype=np.int64)
        labels[outcomes == 1] = 0   # win
        labels[outcomes == 0] = 1   # draw
        labels[outcomes == -1] = 2  # loss
        
        return {
            'states': states_t[idx],
            'labels': torch.from_numpy(labels).to(self.device),
            'outcomes': torch.from_numpy(outcomes.astype(np.float32)).to(self.device),
        }
    
    def sample_value_regression(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample (state, discounted_outcome) for value regression."""
        states_out = []
        values = []
        
        for _ in range(batch_size):
            tid = np.random.randint(self.n_trajs)
            T = self.traj_lengths.get(tid, 0)
            if T == 0:
                continue
            start = self.traj_starts[tid]
            t = np.random.randint(0, T)
            
            states_out.append(start + t)
            player_sign = 1 if t % 2 == 0 else -1
            discount = self.gamma ** (T - 1 - t)
            values.append(self.trajectories[tid]['outcome'] * player_sign * discount)
        
        states_t = self._get_states_tensor()
        return {
            'states': states_t[states_out],
            'values': torch.FloatTensor(values).to(self.device),
        }
    
    def sample_outcome_contrastive(self, batch_size: int, n_neg: int = 63) -> Dict[str, torch.Tensor]:
        """
        Outcome-conditioned contrastive learning.
        Positive: two states from trajectories with the same outcome.
        Negative: states from trajectories with different outcomes.
        
        Tests whether CRL captures outcome structure when goals are binary.
        """
        # Group states by outcome
        win_idx = np.where(self.outcomes == 1)[0]
        lose_idx = np.where(self.outcomes == -1)[0]
        draw_idx = np.where(self.outcomes == 0)[0]
        
        anchors_idx = []
        pos_idx = []
        neg_indices = []
        
        for _ in range(batch_size):
            # Pick a random outcome group
            groups = [(win_idx, lose_idx, draw_idx), 
                     (lose_idx, win_idx, draw_idx),
                     (draw_idx, win_idx, lose_idx)]
            same, diff1, diff2 = groups[np.random.randint(len(groups))]
            
            if len(same) < 2:
                continue
            
            # Anchor and positive from same outcome
            pair = np.random.choice(len(same), size=2, replace=False)
            anchors_idx.append(same[pair[0]])
            pos_idx.append(same[pair[1]])
            
            # Negatives from different outcomes
            diff_pool = np.concatenate([diff1, diff2]) if len(diff1) > 0 or len(diff2) > 0 else same
            neg = np.random.choice(diff_pool, size=n_neg, replace=True)
            neg_indices.append(neg)
        
        B = len(anchors_idx)
        if B == 0:
            return None
        
        states_t = self._get_states_tensor()
        neg_flat = np.array(neg_indices)
        
        return {
            'anchors': states_t[anchors_idx],
            'positives': states_t[pos_idx],
            'negatives': states_t[neg_flat.flatten()].view(B, n_neg, -1),
        }


# ============================================================
# Loss Functions
# ============================================================

def infonce_loss(critic, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
    """
    InfoNCE loss for contrastive RL.
    
    L = -log[ exp(f(s, s+)) / (exp(f(s, s+)) + sum_i exp(f(s, s_i-))) ]
    """
    anchors = batch['anchors']      # [B, D]
    positives = batch['positives']  # [B, D]
    negatives = batch['negatives']  # [B, N, D]
    
    B, N, D = negatives.shape
    
    # Encode
    anchor_emb = critic.encode_state(anchors)          # [B, E]
    pos_emb = critic.encode_goal(positives)             # [B, E]
    neg_emb = critic.encode_goal(negatives.reshape(B*N, D)).reshape(B, N, -1)  # [B, N, E]
    
    tau = critic.temperature
    
    # Positive logits: [B]
    pos_logits = (anchor_emb * pos_emb).sum(-1) / tau
    
    # Negative logits: [B, N]
    neg_logits = torch.bmm(neg_emb, anchor_emb.unsqueeze(-1)).squeeze(-1) / tau
    
    # InfoNCE
    all_logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  # [B, 1+N]
    labels = torch.zeros(B, dtype=torch.long, device=anchors.device)
    loss = F.cross_entropy(all_logits, labels)
    
    with torch.no_grad():
        acc = (all_logits.argmax(1) == 0).float().mean().item()
        pos_mean = pos_logits.mean().item()
        neg_mean = neg_logits.mean().item()
    
    metrics = {
        'infonce_loss': loss.item(),
        'infonce_acc': acc,
        'pos_logit_mean': pos_mean,
        'neg_logit_mean': neg_mean,
        'temperature': tau.item() if isinstance(tau, torch.Tensor) else tau,
    }
    
    return loss, metrics


def infonce_loss_efficient(critic, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
    """
    Efficient InfoNCE using the full batch as negatives (in-batch negatives).
    All non-diagonal entries in the similarity matrix are negatives.
    This is more sample-efficient: effectively N_neg = batch_size - 1.
    """
    anchors = batch['anchors']      # [B, D]
    positives = batch['positives']  # [B, D]
    
    B = anchors.shape[0]
    
    # Full similarity matrix [B, B]
    logits = critic.compute_logits_matrix(anchors, positives)
    
    # Labels: diagonal is the positive
    labels = torch.arange(B, device=anchors.device)
    loss = F.cross_entropy(logits, labels)
    
    with torch.no_grad():
        acc = (logits.argmax(1) == labels).float().mean().item()
        diag = logits.diag().mean().item()
        offdiag = (logits.sum() - logits.diag().sum()) / (B * (B - 1))
    
    metrics = {
        'infonce_loss': loss.item(),
        'infonce_acc': acc,
        'pos_logit_mean': diag,
        'neg_logit_mean': offdiag.item(),
        'temperature': critic.temperature.item() if isinstance(critic.temperature, torch.Tensor) else critic.temperature,
    }
    
    return loss, metrics


def mc_classification_loss(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
    """Monte Carlo win/draw/loss classification loss."""
    states = batch['states']
    labels = batch['labels']
    
    logits = model(states)
    loss = F.cross_entropy(logits, labels)
    
    with torch.no_grad():
        acc = (logits.argmax(1) == labels).float().mean().item()
        probs = F.softmax(logits, dim=-1)
        # Per-class accuracy
        win_mask = labels == 0
        draw_mask = labels == 1
        loss_mask = labels == 2
        win_acc = (logits.argmax(1)[win_mask] == 0).float().mean().item() if win_mask.any() else 0.0
        draw_acc = (logits.argmax(1)[draw_mask] == 1).float().mean().item() if draw_mask.any() else 0.0
        lose_acc = (logits.argmax(1)[loss_mask] == 2).float().mean().item() if loss_mask.any() else 0.0
    
    metrics = {
        'mc_loss': loss.item(),
        'mc_acc': acc,
        'mc_win_acc': win_acc,
        'mc_draw_acc': draw_acc,
        'mc_lose_acc': lose_acc,
    }
    
    return loss, metrics


def value_regression_loss(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
    """Standard MSE value regression loss."""
    states = batch['states']
    values = batch['values']
    
    predicted = model(states)
    loss = F.mse_loss(predicted, values)
    
    with torch.no_grad():
        mae = (predicted - values).abs().mean().item()
        corr = torch.corrcoef(torch.stack([predicted, values]))[0, 1].item()
        if np.isnan(corr):
            corr = 0.0
    
    metrics = {
        'value_loss': loss.item(),
        'value_mae': mae,
        'value_corr': corr,
    }
    
    return loss, metrics


# ============================================================
# Trainer
# ============================================================

class CRLTrainer:
    """
    Unified trainer with logging, checkpointing, and evaluation.
    """
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cpu', checkpoint_dir: str = 'checkpoints',
                 grad_clip: float = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = grad_clip
        self.history = defaultdict(list)
        self.step_count = 0
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_step(self, loss_fn: Callable, batch: Dict[str, torch.Tensor], 
                   **kwargs) -> Dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, metrics = loss_fn(self.model, batch, **kwargs)
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip
        ).item()
        metrics['grad_norm'] = grad_norm
        
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
            metrics['lr'] = self.scheduler.get_last_lr()[0]
        
        self.step_count += 1
        
        # Log
        for k, v in metrics.items():
            self.history[k].append(v)
        
        return metrics
    
    def save_checkpoint(self, name: str = 'latest'):
        path = os.path.join(self.checkpoint_dir, f'{name}.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'step_count': self.step_count,
            'history': dict(self.history),
        }, path)
    
    def load_checkpoint(self, name: str = 'latest'):
        path = os.path.join(self.checkpoint_dir, f'{name}.pt')
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if self.scheduler and ckpt['scheduler_state_dict']:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.step_count = ckpt['step_count']
            self.history = defaultdict(list, ckpt['history'])
            return True
        return False
    
    def get_summary(self, last_n: int = 100) -> Dict:
        """Get summary of recent metrics."""
        summary = {}
        for k, v in self.history.items():
            recent = v[-last_n:] if len(v) >= last_n else v
            summary[f'{k}_mean'] = float(np.mean(recent))
            summary[f'{k}_std'] = float(np.std(recent))
        return summary
