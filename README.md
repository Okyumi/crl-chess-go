# Contrastive RL for Board Games

Experimental codebase investigating **Contrastive Reinforcement Learning (CRL)** in discrete board game settings (Chess, Go, Connect-4). Tests whether the InfoNCE-based goal-conditioned RL framework can learn useful state representations and policies in environments where the "goal" is a degenerate binary signal.

## Motivation

In standard GCRL, agents are conditioned on future states as goals. Board games pose a unique challenge: the objective is to win, but winning can arise from enormously many board configurations. This project investigates:

1. **Can CRL learn environment structure** - the "map" of how moves lead to future positions — even when goals are binary?
2. **Forward vs. backward contrastive sampling** — does sampling backward in time improve representations?
3. **CRL vs. MC value classification** — the P(win|s) classifier is equivalent to MC regression with cross-entropy. Does the contrastive objective learn richer representations?
4. **When does CRL collapse toward standard RL** in these settings, and when does it retain structural advantages?

## Environments

| Environment | State Encoding | State Dim | Action Dim | Notes |
|------------|---------------|-----------|-----------|-------|
| **Connect-4** (6×7) | 4 planes (own/opp/valid/turn) | 168 | 7 | Simpler baseline, fast |
| **Chess** (8×8) | 20 planes (AlphaZero-style) | 1280 | 4288 | Full rules via `python-chess` |
| **Go 9×9** | 8 planes (stones/libs/history/valid) | 648 | 82 | Full rules, captures, ko |
| **Go 13×13** | 8 planes | 1352 | 170 | Larger board |

## Setup

### NYUAD HPC (conda env on scratch — avoids home disk quota)

```bash
cd /scratch/$USER/crl-chess-go   # keep repo on scratch
bash scripts/setup_env.sh
conda activate /scratch/$USER/.conda/envs/ccrl
```

SLURM jobs use the same path via `scripts/submit_slurm.sh`.

### Local / other machines

```bash
git clone https://github.com/Okyumi/crl-chess-go.git
cd crl-chess-go

# Create environment
conda create -n ccrl python=3.10 -y
conda activate ccrl

# Install PyTorch (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Or CPU only:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Debug run (fast, ~2 min)
python run_experiments.py --preset debug

# Connect-4 (full, ~30 min on GPU)
python run_experiments.py --preset connect4

# Chess with conv encoder (~2 hrs on GPU)
python run_experiments.py --preset chess_conv

# Go 9x9 (~1 hr on GPU)
python run_experiments.py --preset go9

# All environments
python run_all.py --device cuda
```

## HPC / SLURM

```bash
# Single environment
sbatch scripts/submit_slurm.sh connect4
sbatch scripts/submit_slurm.sh chess
sbatch scripts/submit_slurm.sh go9

# All in parallel
bash scripts/submit_all.sh
```

Edit `scripts/submit_slurm.sh` to match your HPC partition, module loads, and conda environment.

## CLI Options

```
python run_experiments.py --help

Key options:
  --preset {connect4,chess,go9,go13,debug}   Use preset config
  --env ENV              Override environment
  --n-epochs N           Training steps
  --batch-size N         Batch size
  --n-train-games N      Number of games to collect
  --embed-dim N          Embedding dimension
  --hidden-dim N         Hidden dimension
  --n-blocks N           ResNet blocks
  --encoder-type {mlp,conv}  Encoder architecture
  --lr LR                Learning rate
  --gamma G              Discount factor for geometric sampling
  --n-negatives N        Number of negative samples
  --contrastive-mode {forward,backward,symmetric,efficient}
  --device {auto,cuda,cpu}
  --no-ablations         Skip ablation studies
  --seeds S [S ...]      Random seeds
```

## Experiments

### Exp 1: Method Comparison
Trains 6 methods on the same data, same compute budget:
- **CRL Forward** — Standard InfoNCE with forward geometric sampling
- **CRL Backward** — InfoNCE with backward temporal sampling (Eysenbach suggestion)
- **CRL Symmetric** — Both forward and backward
- **CRL Efficient** — In-batch negatives (all batch entries as negatives)
- **MC Win Classifier** — P(win, draw, loss | s) via cross-entropy (email proposal)
- **Value Regression** — Standard V(s) with MSE loss

### Exp 2: Representation Analysis
- Linear probing: 5-fold CV accuracy of logistic regression on frozen embeddings
- Fisher's discriminant ratio: win/loss cluster separation
- t-SNE visualizations colored by game outcome

### Exp 3: Ablations
- Embedding dimension: {32, 64, 128, 256, 512}
- Number of negatives: {7, 15, 31, 63, 127, 255}
- Discount factor γ: {0.9, 0.95, 0.99, 0.995, 0.999}
- Network depth: {1, 2, 4, 6, 8} ResNet blocks
- Contrastive mode: forward / backward / symmetric / efficient

### Exp 4: Multi-Seed Variance
Runs shorter training (500 steps) across 5 seeds for confidence intervals.

### Exp 5: Policy Evaluation
Each method's learned value function selects moves as player 1 against a random opponent.
- CRL uses a **goal-similarity heuristic**: embed winning terminal states, select moves that maximize cosine similarity to the average winning embedding.

## Output

```
results/<env_name>/
├── config.yaml                 # Full experiment config
├── results.json                # All numerical results
├── training_curves_<env>.png   # Loss, accuracy, temperature, logits
├── representation_<env>.png    # Linear probe, Fisher ratio
├── ablations_<env>.png         # All ablation bar charts
├── policy_<env>.png            # Win rates
├── tsne_<env>.png              # t-SNE embeddings
├── crl_forward.pt              # Model checkpoints
├── crl_backward.pt
├── mc_classifier.pt
└── value_regression.pt
```

## Code Structure

```
├── config.py              # ExperimentConfig dataclass, presets, CLI parser
├── environments.py        # FullChessEnv, GoEnv, ConnectFourEnv
├── models.py              # ContrastiveCritic, WinPredictorMC, ValueNet, PolicyNet
├── training.py            # TrajectoryBuffer, InfoNCE losses, trainer
├── run_experiments.py     # Main experiment runner 
├── run_all.py             # Run all environments
├── requirements.txt
├── README.md
└── scripts/
    ├── submit_slurm.sh    # SLURM job script
    └── submit_all.sh      # Submit all envs in parallel
```

## Key References

- [Contrastive Learning as Goal-Conditioned RL](https://arxiv.org/abs/2206.07568) — Eysenbach et al., NeurIPS 2022
- [A Single Goal is All You Need](https://arxiv.org/abs/2408.05804) — Liu, Tang, Eysenbach, 2024
- [Demystifying Emergent Exploration in GCRL](https://mahsa-bastankhah.github.io/demystifying-single-goal-exploration/) — Bastankhah et al., 2025
- [Contrastive Difference Predictive Coding](https://arxiv.org/abs/2310.20141) — TD InfoNCE
