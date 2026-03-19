"""
Experiment configuration.

Each config is a dictionary that fully specifies an experiment run.
Configs can be overridden from the command line or loaded from YAML.
"""

import argparse
import yaml
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    
    # --- Experiment ---
    experiment_name: str = "crl_chess_go"
    seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    n_seeds: int = 5  # How many seeds to run
    device: str = "auto"  # "auto", "cuda", "cpu"
    output_dir: str = "results"
    
    # --- Environment ---
    env_name: str = "connect4"  # "connect4", "chess", "go9", "go13", "go19"
    env_kwargs: dict = field(default_factory=dict)
    
    # --- Data Collection ---
    n_train_games: int = 10000
    n_eval_games: int = 500
    data_collection: str = "random"  # "random" or "self_play"
    
    # --- Model Architecture ---
    encoder_type: str = "mlp"  # "mlp" or "conv"
    embed_dim: int = 256
    hidden_dim: int = 512
    n_blocks: int = 4
    conv_channels: int = 128
    dropout: float = 0.1
    learnable_temp: bool = True
    
    # --- Training ---
    n_epochs: int = 1000  # training steps (batches)
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    grad_clip: float = 1.0
    
    # --- Contrastive ---
    n_negatives: int = 63
    gamma: float = 0.99
    contrastive_mode: str = "forward"  # "forward", "backward", "symmetric", "outcome", "efficient"
    
    # --- Evaluation ---
    eval_every: int = 100
    save_every: int = 500
    n_policy_eval_games: int = 200
    policy_temperature: float = 0.5
    
    # --- Ablations ---
    run_ablations: bool = True
    ablation_embed_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    ablation_n_negatives: List[int] = field(default_factory=lambda: [7, 15, 31, 63, 127, 255])
    ablation_gammas: List[float] = field(default_factory=lambda: [0.9, 0.95, 0.99, 0.995, 0.999])
    ablation_n_blocks: List[int] = field(default_factory=lambda: [1, 2, 4, 6, 8])
    ablation_epochs: int = 500  # Steps per ablation run
    
    def to_dict(self):
        return asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ============================================================
# Preset Configs
# ============================================================

PRESETS = {
    'connect4': ExperimentConfig(
        env_name='connect4',
        n_train_games=10000,
        n_epochs=2000,
        batch_size=512,
        embed_dim=256,
        hidden_dim=512,
        n_blocks=4,
        encoder_type='mlp',
    ),
    'connect4_conv': ExperimentConfig(
        env_name='connect4',
        n_train_games=10000,
        n_epochs=2000,
        batch_size=512,
        embed_dim=256,
        hidden_dim=512,
        n_blocks=4,
        encoder_type='conv',
        conv_channels=64,
    ),
    'chess': ExperimentConfig(
        env_name='chess',
        n_train_games=20000,
        n_epochs=5000,
        batch_size=512,
        embed_dim=256,
        hidden_dim=512,
        n_blocks=6,
        encoder_type='mlp',
        gamma=0.995,
    ),
    'chess_conv': ExperimentConfig(
        env_name='chess',
        n_train_games=20000,
        n_epochs=5000,
        batch_size=512,
        embed_dim=256,
        n_blocks=6,
        encoder_type='conv',
        conv_channels=128,
        gamma=0.995,
    ),
    'go9': ExperimentConfig(
        env_name='go9',
        n_train_games=15000,
        n_epochs=3000,
        batch_size=512,
        embed_dim=256,
        hidden_dim=512,
        n_blocks=6,
        encoder_type='conv',
        conv_channels=128,
        gamma=0.99,
    ),
    'go13': ExperimentConfig(
        env_name='go13',
        n_train_games=20000,
        n_epochs=5000,
        batch_size=512,
        embed_dim=256,
        hidden_dim=512,
        n_blocks=8,
        encoder_type='conv',
        conv_channels=128,
        gamma=0.995,
    ),
    # Quick debug preset
    'debug': ExperimentConfig(
        env_name='connect4',
        n_train_games=500,
        n_epochs=50,
        batch_size=128,
        embed_dim=64,
        hidden_dim=128,
        n_blocks=2,
        n_seeds=1,
        run_ablations=False,
        n_policy_eval_games=20,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description='CRL Board Game Experiments')
    parser.add_argument('--preset', type=str, default=None, choices=list(PRESETS.keys()),
                       help='Use a preset config')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config')
    parser.add_argument('--env', type=str, default=None, help='Override environment')
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--n-epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--n-train-games', type=int, default=None)
    parser.add_argument('--embed-dim', type=int, default=None)
    parser.add_argument('--hidden-dim', type=int, default=None)
    parser.add_argument('--n-blocks', type=int, default=None)
    parser.add_argument('--encoder-type', type=str, default=None, choices=['mlp', 'conv'])
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--n-negatives', type=int, default=None)
    parser.add_argument('--contrastive-mode', type=str, default=None,
                       choices=['forward', 'backward', 'symmetric', 'outcome', 'efficient'])
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--no-ablations', action='store_true')
    parser.add_argument('--experiment-name', type=str, default=None)
    
    args = parser.parse_args()
    
    # Start from preset or default
    if args.preset:
        config = PRESETS[args.preset]
    elif args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = ExperimentConfig()
    
    # Override with CLI args
    if args.env is not None:
        config.env_name = args.env
    if args.seeds is not None:
        config.seeds = args.seeds
        config.n_seeds = len(args.seeds)
    if args.n_epochs is not None:
        config.n_epochs = args.n_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.n_train_games is not None:
        config.n_train_games = args.n_train_games
    if args.embed_dim is not None:
        config.embed_dim = args.embed_dim
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    if args.n_blocks is not None:
        config.n_blocks = args.n_blocks
    if args.encoder_type is not None:
        config.encoder_type = args.encoder_type
    if args.lr is not None:
        config.lr = args.lr
    if args.gamma is not None:
        config.gamma = args.gamma
    if args.n_negatives is not None:
        config.n_negatives = args.n_negatives
    if args.contrastive_mode is not None:
        config.contrastive_mode = args.contrastive_mode
    if args.device is not None:
        config.device = args.device
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.no_ablations:
        config.run_ablations = False
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    
    return config
