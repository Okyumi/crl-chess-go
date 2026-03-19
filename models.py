"""
Neural network models for CRL experiments.

Architectures:
1. ResNet-based state encoder (configurable depth/width)
2. Contrastive critic: f(s,g) = phi(s)^T psi(g) / tau  and  f(s,a,g) = phi(s,a)^T psi(g) / tau
3. MC Win classifier: state -> P(win, draw, loss)
4. Standard scalar value network: state -> V(s)
5. Policy network for self-play evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, List


# ============================================================
# Building Blocks
# ============================================================

class ResBlock(nn.Module):
    """Residual block with LayerNorm."""
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    
    def forward(self, x):
        return F.gelu(x + self.net(x))


class ResEncoder(nn.Module):
    """
    ResNet-style MLP encoder.
    Projects input to hidden_dim, then applies n_blocks residual blocks.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256,
                 n_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_proj(x)
        return x


class ConvEncoder(nn.Module):
    """
    Convolutional encoder for board games.
    Input: (batch, n_planes, board_size, board_size)
    Output: (batch, output_dim)
    """
    def __init__(self, n_planes: int, board_size: int, output_dim: int = 256,
                 channels: int = 128, n_blocks: int = 4):
        super().__init__()
        self.board_size = board_size
        self.n_planes = n_planes
        
        # Initial convolution
        layers = [
            nn.Conv2d(n_planes, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        ]
        
        # Residual conv blocks
        for _ in range(n_blocks):
            layers.append(ConvResBlock(channels))
        
        self.conv = nn.Sequential(*layers)
        
        # Global average pool + projection
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, output_dim),
        )
    
    def forward(self, x):
        # x can be flat (batch, state_dim) or already shaped
        if x.dim() == 2:
            batch = x.shape[0]
            x = x.view(batch, self.n_planes, self.board_size, self.board_size)
        x = self.conv(x)
        x = self.head(x)
        return x


class ConvResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
    
    def forward(self, x):
        return F.gelu(x + self.net(x))


# ============================================================
# Contrastive RL Critics
# ============================================================

class ContrastiveCritic(nn.Module):
    """
    CRL Critic: f(s, g) = phi(s)^T psi(g) / tau
    
    The inner product of the learned representations estimates the 
    discounted state occupancy measure, which corresponds to the 
    goal-conditioned value function.
    
    Both encoders are parameterized as ResNet MLPs or ConvNets.
    """
    def __init__(self, state_dim: int, goal_dim: int, embed_dim: int = 256,
                 hidden_dim: int = 512, n_blocks: int = 4, dropout: float = 0.1,
                 encoder_type: str = 'mlp', board_size: int = None, n_planes: int = None,
                 conv_channels: int = 128, learnable_temp: bool = True):
        super().__init__()
        
        if encoder_type == 'conv' and board_size is not None and n_planes is not None:
            self.phi = ConvEncoder(n_planes, board_size, embed_dim, conv_channels, n_blocks)
            self.psi = ConvEncoder(n_planes, board_size, embed_dim, conv_channels, n_blocks)
        else:
            self.phi = ResEncoder(state_dim, hidden_dim, embed_dim, n_blocks, dropout)
            self.psi = ResEncoder(goal_dim, hidden_dim, embed_dim, n_blocks, dropout)
        
        self.embed_dim = embed_dim
        
        if learnable_temp:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0)))
        else:
            self.log_temperature = None
    
    @property
    def temperature(self):
        if self.log_temperature is not None:
            return self.log_temperature.exp().clamp(min=0.01, max=100.0)
        return 1.0
    
    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state and L2-normalize."""
        return F.normalize(self.phi(state), dim=-1)
    
    def encode_goal(self, goal: torch.Tensor) -> torch.Tensor:
        """Encode goal and L2-normalize."""
        return F.normalize(self.psi(goal), dim=-1)
    
    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Compute f(s, g) = phi(s)^T psi(g) / tau."""
        state_emb = self.encode_state(state)
        goal_emb = self.encode_goal(goal)
        return torch.sum(state_emb * goal_emb, dim=-1) / self.temperature
    
    def compute_logits_matrix(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Compute full logits matrix for efficient InfoNCE. Shape: [B_state, B_goal]."""
        state_emb = self.encode_state(state)
        goal_emb = self.encode_goal(goal)
        return (state_emb @ goal_emb.T) / self.temperature


class ContrastiveCriticWithAction(nn.Module):
    """
    CRL Critic with action conditioning: f(s, a, g) = phi(s, a)^T psi(g) / tau
    """
    def __init__(self, state_dim: int, action_dim: int, goal_dim: int, 
                 embed_dim: int = 256, hidden_dim: int = 512, n_blocks: int = 4,
                 dropout: float = 0.1, learnable_temp: bool = True):
        super().__init__()
        self.phi = ResEncoder(state_dim + action_dim, hidden_dim, embed_dim, n_blocks, dropout)
        self.psi = ResEncoder(goal_dim, hidden_dim, embed_dim, n_blocks, dropout)
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        
        if learnable_temp:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0)))
        else:
            self.log_temperature = None
    
    @property
    def temperature(self):
        if self.log_temperature is not None:
            return self.log_temperature.exp().clamp(min=0.01, max=100.0)
        return 1.0
    
    def encode_state_action(self, state: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action_onehot], dim=-1)
        return F.normalize(self.phi(x), dim=-1)
    
    def encode_goal(self, goal: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.psi(goal), dim=-1)
    
    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor, 
                goal: torch.Tensor) -> torch.Tensor:
        sa_emb = self.encode_state_action(state, action_onehot)
        g_emb = self.encode_goal(goal)
        return torch.sum(sa_emb * g_emb, dim=-1) / self.temperature


# ============================================================
# Baseline Models
# ============================================================

class WinPredictorMC(nn.Module):
    """
    Monte Carlo win predictor (from the email discussion).
    
    Maps state -> P(win), P(draw), P(loss) via cross-entropy.
    Equivalent to "learning a value function with Monte Carlo regression, 
    using a cross entropy loss" (Eysenbach's response).
    """
    def __init__(self, state_dim: int, hidden_dim: int = 512, n_blocks: int = 4,
                 dropout: float = 0.1, encoder_type: str = 'mlp',
                 board_size: int = None, n_planes: int = None, conv_channels: int = 128):
        super().__init__()
        
        if encoder_type == 'conv' and board_size is not None:
            self.encoder = ConvEncoder(n_planes, board_size, hidden_dim, conv_channels, n_blocks)
        else:
            self.encoder = ResEncoder(state_dim, hidden_dim, hidden_dim, n_blocks, dropout)
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3),  # win, draw, loss
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return logits for [win, draw, loss]."""
        features = self.encoder(state)
        return self.head(features)
    
    def predict_value(self, state: torch.Tensor) -> torch.Tensor:
        """Return expected value in [-1, 1]."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 0] - probs[:, 2]  # P(win) - P(loss)
    
    def get_features(self, state: torch.Tensor) -> torch.Tensor:
        """Get encoder features (for representation analysis)."""
        return self.encoder(state)


class StandardValueNet(nn.Module):
    """Standard scalar value network: V(s) in [-1, 1]."""
    def __init__(self, state_dim: int, hidden_dim: int = 512, n_blocks: int = 4,
                 dropout: float = 0.1, encoder_type: str = 'mlp',
                 board_size: int = None, n_planes: int = None, conv_channels: int = 128):
        super().__init__()
        
        if encoder_type == 'conv' and board_size is not None:
            self.encoder = ConvEncoder(n_planes, board_size, hidden_dim, conv_channels, n_blocks)
        else:
            self.encoder = ResEncoder(state_dim, hidden_dim, hidden_dim, n_blocks, dropout)
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.encoder(state)
        return self.head(features).squeeze(-1)
    
    def get_features(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)


class PolicyNet(nn.Module):
    """Policy network for self-play evaluation."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512,
                 n_blocks: int = 4, dropout: float = 0.1, encoder_type: str = 'mlp',
                 board_size: int = None, n_planes: int = None, conv_channels: int = 128):
        super().__init__()
        
        if encoder_type == 'conv' and board_size is not None:
            self.encoder = ConvEncoder(n_planes, board_size, hidden_dim, conv_channels, n_blocks)
        else:
            self.encoder = ResEncoder(state_dim, hidden_dim, hidden_dim, n_blocks, dropout)
        
        self.head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.encoder(state)
        logits = self.head(features)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask.bool(), float('-inf'))
        return logits
    
    def get_action(self, state: torch.Tensor, legal_actions: List[int], 
                   temperature: float = 1.0) -> int:
        with torch.no_grad():
            logits = self.forward(state.unsqueeze(0)).squeeze(0)
            mask = torch.full_like(logits, float('-inf'))
            for a in legal_actions:
                if a < len(mask):
                    mask[a] = 0.0
            logits = logits + mask
            if temperature <= 0:
                return logits.argmax().item()
            probs = F.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, 1).item()
