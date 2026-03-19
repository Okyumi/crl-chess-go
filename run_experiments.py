"""
Main experiment runner for CRL in Chess/Go.

Usage:
  # Run full Connect-4 experiments
  python run_experiments.py --preset connect4

  # Run full Chess experiments with conv encoder
  python run_experiments.py --preset chess_conv

  # Run Go 9x9
  python run_experiments.py --preset go9

  # Quick debug run
  python run_experiments.py --preset debug

  # Custom config
  python run_experiments.py --env chess --n-epochs 3000 --encoder-type conv --device cuda

  # Run all environments sequentially
  python run_all.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import time
import os
import sys
from collections import defaultdict
from copy import deepcopy

from environments import FullChessEnv, GoEnv, ConnectFourEnv, collect_random_games
from models import (ContrastiveCritic, ContrastiveCriticWithAction,
                    WinPredictorMC, StandardValueNet, PolicyNet)
from training import (TrajectoryBuffer, infonce_loss, infonce_loss_efficient,
                      mc_classification_loss, value_regression_loss, CRLTrainer)
from config import ExperimentConfig, parse_args


# ============================================================
# Helpers
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)


def get_device(config: ExperimentConfig) -> str:
    if config.device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return config.device


def get_env_info(env_name: str):
    """Return (env_class, env_kwargs, state_dim, action_dim, board_size, n_planes)."""
    if env_name == 'connect4':
        env = ConnectFourEnv()
        return ConnectFourEnv, {}, env.state_dim, env.action_dim, 7, 4  # cols as "board_size" for conv
    elif env_name == 'chess':
        env = FullChessEnv()
        return FullChessEnv, {}, env.state_dim, env.action_dim, 8, 20
    elif env_name.startswith('go'):
        size = int(env_name[2:])
        env = GoEnv(board_size=size)
        return GoEnv, {'board_size': size}, env.state_dim, env.action_dim, size, 8
    else:
        raise ValueError(f"Unknown env: {env_name}")


def smooth(data, window=20):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


# ============================================================
# Core Training Functions
# ============================================================

def train_contrastive(config: ExperimentConfig, buffer: TrajectoryBuffer,
                      state_dim: int, device: str, seed: int,
                      mode: str = 'forward', embed_dim: int = None,
                      n_neg: int = None, gamma: float = None,
                      n_blocks: int = None, n_epochs: int = None,
                      board_size: int = None, n_planes: int = None) -> dict:
    """Train a contrastive critic. Returns history dict."""
    set_seed(seed)
    
    _embed_dim = embed_dim or config.embed_dim
    _n_neg = n_neg or config.n_negatives
    _gamma = gamma or config.gamma
    _n_blocks = n_blocks or config.n_blocks
    _n_epochs = n_epochs or config.n_epochs
    
    # Override buffer gamma if different
    if _gamma != buffer.gamma:
        buffer_copy = TrajectoryBuffer(buffer.trajectories, gamma=_gamma, device=device)
    else:
        buffer_copy = buffer
    
    model = ContrastiveCritic(
        state_dim=state_dim, goal_dim=state_dim, embed_dim=_embed_dim,
        hidden_dim=config.hidden_dim, n_blocks=_n_blocks, dropout=config.dropout,
        encoder_type=config.encoder_type, board_size=board_size, n_planes=n_planes,
        conv_channels=config.conv_channels, learnable_temp=config.learnable_temp,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_n_epochs, eta_min=1e-6)
    
    history = defaultdict(list)
    
    for step in range(_n_epochs):
        # Sample batch
        if mode == 'forward':
            batch = buffer_copy.sample_forward_contrastive(config.batch_size, _n_neg)
        elif mode == 'backward':
            batch = buffer_copy.sample_backward_contrastive(config.batch_size, _n_neg)
        elif mode == 'symmetric':
            batch = buffer_copy.sample_symmetric_contrastive(config.batch_size, _n_neg)
        elif mode == 'outcome':
            batch = buffer_copy.sample_outcome_contrastive(config.batch_size, _n_neg)
        elif mode == 'efficient':
            batch = buffer_copy.sample_forward_contrastive(config.batch_size, _n_neg)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if batch is None:
            continue
        
        # Train step
        model.train()
        optimizer.zero_grad()
        
        if mode == 'efficient':
            loss, metrics = infonce_loss_efficient(model, batch)
        else:
            loss, metrics = infonce_loss(model, batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        
        for k, v in metrics.items():
            history[k].append(v)
        
        if (step + 1) % max(1, _n_epochs // 10) == 0:
            recent_loss = np.mean(history['infonce_loss'][-50:])
            recent_acc = np.mean(history['infonce_acc'][-50:])
            print(f"    [{mode}] step {step+1}/{_n_epochs}: loss={recent_loss:.4f} acc={recent_acc:.3f}")
    
    return {'model': model, 'history': dict(history)}


def train_mc_classifier(config: ExperimentConfig, buffer: TrajectoryBuffer,
                        state_dim: int, device: str, seed: int,
                        board_size: int = None, n_planes: int = None,
                        n_epochs: int = None) -> dict:
    """Train MC win/draw/loss classifier."""
    set_seed(seed)
    _n_epochs = n_epochs or config.n_epochs
    
    model = WinPredictorMC(
        state_dim=state_dim, hidden_dim=config.hidden_dim, n_blocks=config.n_blocks,
        dropout=config.dropout, encoder_type=config.encoder_type,
        board_size=board_size, n_planes=n_planes, conv_channels=config.conv_channels,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_n_epochs, eta_min=1e-6)
    
    history = defaultdict(list)
    
    for step in range(_n_epochs):
        batch = buffer.sample_outcome_labeled(config.batch_size)
        
        model.train()
        optimizer.zero_grad()
        loss, metrics = mc_classification_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        
        for k, v in metrics.items():
            history[k].append(v)
        
        if (step + 1) % max(1, _n_epochs // 10) == 0:
            recent_loss = np.mean(history['mc_loss'][-50:])
            recent_acc = np.mean(history['mc_acc'][-50:])
            print(f"    [MC] step {step+1}/{_n_epochs}: loss={recent_loss:.4f} acc={recent_acc:.3f}")
    
    return {'model': model, 'history': dict(history)}


def train_value_net(config: ExperimentConfig, buffer: TrajectoryBuffer,
                    state_dim: int, device: str, seed: int,
                    board_size: int = None, n_planes: int = None,
                    n_epochs: int = None) -> dict:
    """Train standard value network."""
    set_seed(seed)
    _n_epochs = n_epochs or config.n_epochs
    
    model = StandardValueNet(
        state_dim=state_dim, hidden_dim=config.hidden_dim, n_blocks=config.n_blocks,
        dropout=config.dropout, encoder_type=config.encoder_type,
        board_size=board_size, n_planes=n_planes, conv_channels=config.conv_channels,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_n_epochs, eta_min=1e-6)
    
    history = defaultdict(list)
    
    for step in range(_n_epochs):
        batch = buffer.sample_value_regression(config.batch_size)
        
        model.train()
        optimizer.zero_grad()
        loss, metrics = value_regression_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        
        for k, v in metrics.items():
            history[k].append(v)
        
        if (step + 1) % max(1, _n_epochs // 10) == 0:
            recent_loss = np.mean(history['value_loss'][-50:])
            recent_mae = np.mean(history['value_mae'][-50:])
            print(f"    [Value] step {step+1}/{_n_epochs}: loss={recent_loss:.4f} MAE={recent_mae:.3f}")
    
    return {'model': model, 'history': dict(history)}


# ============================================================
# Evaluation
# ============================================================

def evaluate_representations(models: dict, buffer: TrajectoryBuffer, device: str,
                            n_samples: int = 10000) -> dict:
    """
    Evaluate representation quality:
    1. Linear probing (binary win prediction, 3-class)
    2. Fisher cluster separation
    3. t-SNE visualization data
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    n_samples = min(n_samples, buffer.n_states)
    idx = np.random.choice(buffer.n_states, n_samples, replace=False)
    sample_states = buffer._get_states_tensor()[idx]
    sample_outcomes = buffer.outcomes[idx]
    
    results = {}
    embeddings_for_tsne = {}
    
    for name, model_data in models.items():
        model = model_data['model']
        model.eval()
        
        with torch.no_grad():
            if hasattr(model, 'encode_state'):
                emb = model.encode_state(sample_states).cpu().numpy()
            elif hasattr(model, 'get_features'):
                emb = model.get_features(sample_states).cpu().numpy()
            elif hasattr(model, 'encoder'):
                emb = model.encoder(sample_states).cpu().numpy()
            else:
                continue
        
        embeddings_for_tsne[name] = emb
        
        # Standardize for linear probing
        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(emb)
        
        # Binary probe: win vs not-win
        binary_labels = (sample_outcomes == 1).astype(int)
        try:
            scores_binary = cross_val_score(
                LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs'),
                emb_scaled, binary_labels, cv=5, scoring='accuracy'
            )
            results[f'probe_binary_{name}'] = {
                'mean': float(scores_binary.mean()),
                'std': float(scores_binary.std()),
            }
        except Exception as e:
            results[f'probe_binary_{name}'] = {'mean': 0.0, 'std': 0.0, 'error': str(e)}
        
        # 3-class probe: win/draw/loss
        three_labels = (sample_outcomes + 1).astype(int)  # {0,1,2}
        try:
            scores_3class = cross_val_score(
                LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', multi_class='multinomial'),
                emb_scaled, three_labels, cv=5, scoring='accuracy'
            )
            results[f'probe_3class_{name}'] = {
                'mean': float(scores_3class.mean()),
                'std': float(scores_3class.std()),
            }
        except Exception as e:
            results[f'probe_3class_{name}'] = {'mean': 0.0, 'std': 0.0, 'error': str(e)}
        
        # Fisher cluster separation
        win_emb = emb[sample_outcomes == 1]
        lose_emb = emb[sample_outcomes == -1]
        if len(win_emb) > 10 and len(lose_emb) > 10:
            win_c = win_emb.mean(0)
            lose_c = lose_emb.mean(0)
            inter = np.linalg.norm(win_c - lose_c)
            win_sp = np.mean(np.linalg.norm(win_emb - win_c, axis=1))
            lose_sp = np.mean(np.linalg.norm(lose_emb - lose_c, axis=1))
            fisher = inter / (win_sp + lose_sp + 1e-8)
            results[f'fisher_{name}'] = {
                'ratio': float(fisher),
                'inter_dist': float(inter),
                'win_spread': float(win_sp),
                'lose_spread': float(lose_sp),
            }
        
        print(f"  {name}: binary_probe={results.get(f'probe_binary_{name}', {}).get('mean', 0):.3f}, "
              f"3class_probe={results.get(f'probe_3class_{name}', {}).get('mean', 0):.3f}, "
              f"fisher={results.get(f'fisher_{name}', {}).get('ratio', 0):.3f}")
    
    return results, embeddings_for_tsne, sample_outcomes


def evaluate_policy(env_class, env_kwargs, models: dict, buffer, device: str,
                    n_games: int = 200, temperature: float = 0.5) -> dict:
    """Evaluate policies by playing against random opponent."""
    results = {}
    
    # Random baseline
    wins = 0
    for _ in range(n_games):
        env = env_class(**env_kwargs)
        state = env.reset()
        while not env.done:
            legal = env.get_legal_actions()
            if not legal:
                break
            env.step(np.random.choice(legal))
        wins += (env.winner == 1)
    results['random'] = float(wins / n_games)
    print(f"  random: {results['random']:.3f}")
    
    # Model-based policies
    for name, model_data in models.items():
        model = model_data['model']
        model.eval()
        
        wins = 0
        draws = 0
        for _ in range(n_games):
            env = env_class(**env_kwargs)
            state = env.reset()
            
            while not env.done:
                legal = env.get_legal_actions()
                if not legal:
                    break
                
                if env.turn == 1:  # Our agent
                    best_action = None
                    best_value = -float('inf')
                    
                    for action in legal:
                        env_copy = env.clone()
                        ns, _, _, _ = env_copy.step(action)
                        ns_t = torch.FloatTensor(ns).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            if hasattr(model, 'predict_value'):
                                v = model.predict_value(ns_t).item()
                            elif hasattr(model, 'encode_state') and hasattr(model_data, 'goal_emb'):
                                # CRL: use similarity to winning states
                                v = _crl_value(model, ns_t, model_data.get('goal_emb'))
                            elif hasattr(model, 'forward') and not hasattr(model, 'encode_state'):
                                v = model(ns_t).item()
                            else:
                                v = 0.0
                        
                        if v > best_value:
                            best_value = v
                            best_action = action
                    
                    env.step(best_action if best_action is not None else np.random.choice(legal))
                else:
                    env.step(np.random.choice(legal))
            
            wins += (env.winner == 1)
            draws += (env.winner == 0)
        
        results[name] = float(wins / n_games)
        print(f"  {name}: win_rate={results[name]:.3f} (draws={draws/n_games:.3f})")
    
    # CRL with goal-similarity heuristic
    for crl_name in ['crl_forward', 'crl_backward', 'crl_symmetric', 'crl_efficient']:
        if crl_name not in models:
            continue
        
        crl_model = models[crl_name]['model']
        crl_model.eval()
        
        # Compute goal embedding from winning terminal states
        win_terminals = []
        for traj in buffer.trajectories:
            if traj['outcome'] == 1 and len(traj['states']) > 0:
                win_terminals.append(traj['states'][-1])
        
        if not win_terminals:
            continue
        
        win_t = torch.FloatTensor(np.array(win_terminals[:200])).to(device)
        with torch.no_grad():
            goal_emb = crl_model.encode_goal(win_t).mean(0, keepdim=True)
        
        wins = 0
        draws = 0
        for _ in range(n_games):
            env = env_class(**env_kwargs)
            state = env.reset()
            
            while not env.done:
                legal = env.get_legal_actions()
                if not legal:
                    break
                
                if env.turn == 1:
                    best_action = None
                    best_value = -float('inf')
                    
                    for action in legal:
                        env_copy = env.clone()
                        ns, _, _, _ = env_copy.step(action)
                        ns_t = torch.FloatTensor(ns).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            state_emb = crl_model.encode_state(ns_t)
                            v = (state_emb * goal_emb).sum(-1).item() / crl_model.temperature.item()
                        
                        if v > best_value:
                            best_value = v
                            best_action = action
                    
                    env.step(best_action if best_action is not None else np.random.choice(legal))
                else:
                    env.step(np.random.choice(legal))
            
            wins += (env.winner == 1)
            draws += (env.winner == 0)
        
        results[f'{crl_name}_goalsim'] = float(wins / n_games)
        print(f"  {crl_name}_goalsim: win_rate={results[f'{crl_name}_goalsim']:.3f} (draws={draws/n_games:.3f})")
    
    return results


# ============================================================
# Ablation Runner
# ============================================================

def run_ablations(config: ExperimentConfig, buffer: TrajectoryBuffer,
                  state_dim: int, device: str, seed: int,
                  board_size: int = None, n_planes: int = None) -> dict:
    """Run all ablation studies."""
    results = {}
    abl_epochs = config.ablation_epochs
    
    print("\n  --- Ablation: Embedding Dimension ---")
    for ed in config.ablation_embed_dims:
        r = train_contrastive(config, buffer, state_dim, device, seed,
                             mode='forward', embed_dim=ed, n_epochs=abl_epochs,
                             board_size=board_size, n_planes=n_planes)
        final_acc = float(np.mean(r['history']['infonce_acc'][-50:]))
        final_loss = float(np.mean(r['history']['infonce_loss'][-50:]))
        results[f'embed_dim_{ed}'] = {'acc': final_acc, 'loss': final_loss}
        print(f"    embed_dim={ed}: acc={final_acc:.3f} loss={final_loss:.4f}")
    
    print("\n  --- Ablation: Number of Negatives ---")
    for nn_val in config.ablation_n_negatives:
        r = train_contrastive(config, buffer, state_dim, device, seed,
                             mode='forward', n_neg=nn_val, n_epochs=abl_epochs,
                             board_size=board_size, n_planes=n_planes)
        final_acc = float(np.mean(r['history']['infonce_acc'][-50:]))
        final_loss = float(np.mean(r['history']['infonce_loss'][-50:]))
        results[f'n_neg_{nn_val}'] = {'acc': final_acc, 'loss': final_loss}
        print(f"    n_neg={nn_val}: acc={final_acc:.3f} loss={final_loss:.4f}")
    
    print("\n  --- Ablation: Discount Factor (gamma) ---")
    for g in config.ablation_gammas:
        r = train_contrastive(config, buffer, state_dim, device, seed,
                             mode='forward', gamma=g, n_epochs=abl_epochs,
                             board_size=board_size, n_planes=n_planes)
        final_acc = float(np.mean(r['history']['infonce_acc'][-50:]))
        final_loss = float(np.mean(r['history']['infonce_loss'][-50:]))
        results[f'gamma_{g}'] = {'acc': final_acc, 'loss': final_loss}
        print(f"    gamma={g}: acc={final_acc:.3f} loss={final_loss:.4f}")
    
    print("\n  --- Ablation: Network Depth ---")
    for nb in config.ablation_n_blocks:
        r = train_contrastive(config, buffer, state_dim, device, seed,
                             mode='forward', n_blocks=nb, n_epochs=abl_epochs,
                             board_size=board_size, n_planes=n_planes)
        final_acc = float(np.mean(r['history']['infonce_acc'][-50:]))
        final_loss = float(np.mean(r['history']['infonce_loss'][-50:]))
        results[f'n_blocks_{nb}'] = {'acc': final_acc, 'loss': final_loss}
        print(f"    n_blocks={nb}: acc={final_acc:.3f} loss={final_loss:.4f}")
    
    print("\n  --- Ablation: Contrastive Mode ---")
    for mode in ['forward', 'backward', 'symmetric', 'efficient']:
        r = train_contrastive(config, buffer, state_dim, device, seed,
                             mode=mode, n_epochs=abl_epochs,
                             board_size=board_size, n_planes=n_planes)
        final_acc = float(np.mean(r['history']['infonce_acc'][-50:]))
        final_loss = float(np.mean(r['history']['infonce_loss'][-50:]))
        results[f'mode_{mode}'] = {'acc': final_acc, 'loss': final_loss}
        print(f"    mode={mode}: acc={final_acc:.3f} loss={final_loss:.4f}")
    
    return results


# ============================================================
# Plotting
# ============================================================

def plot_all(results: dict, config: ExperimentConfig, output_dir: str):
    """Generate all plots."""
    os.makedirs(output_dir, exist_ok=True)
    env_name = config.env_name
    
    # 1. Training curves
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {'crl_forward': '#2196F3', 'crl_backward': '#4CAF50',
              'crl_symmetric': '#00BCD4', 'crl_efficient': '#009688',
              'mc_classifier': '#FF9800', 'value_regression': '#9C27B0'}
    
    for name, data in results.get('training', {}).items():
        color = colors.get(name, '#666666')
        label = name.replace('_', ' ').title()
        hist = data.get('history', {})
        
        if 'infonce_loss' in hist:
            axes[0, 0].plot(smooth(hist['infonce_loss']), label=label, color=color, lw=1.5)
        if 'mc_loss' in hist:
            axes[0, 0].plot(smooth(hist['mc_loss']), label=label, color=color, lw=1.5)
        if 'value_loss' in hist:
            axes[0, 0].plot(smooth(hist['value_loss']), label=label, color=color, lw=1.5)
        
        if 'infonce_acc' in hist:
            axes[0, 1].plot(smooth(hist['infonce_acc']), label=label, color=color, lw=1.5)
        if 'mc_acc' in hist:
            axes[0, 1].plot(smooth(hist['mc_acc']), label=label, color=color, lw=1.5)
        
        if 'temperature' in hist:
            axes[1, 0].plot(smooth(hist['temperature']), label=label, color=color, lw=1.5)
        
        if 'pos_logit_mean' in hist:
            axes[1, 1].plot(smooth(hist['pos_logit_mean']), label=f'{label} (pos)', color=color, lw=1.5)
        if 'neg_logit_mean' in hist:
            axes[1, 1].plot(smooth(hist['neg_logit_mean']), label=f'{label} (neg)', color=color, lw=1.5, ls='--')
    
    axes[0, 0].set_title('Training Loss', fontsize=14); axes[0, 0].legend(fontsize=9); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].set_title('Training Accuracy', fontsize=14); axes[0, 1].legend(fontsize=9); axes[0, 1].grid(alpha=0.3)
    axes[1, 0].set_title('Learned Temperature', fontsize=14); axes[1, 0].legend(fontsize=9); axes[1, 0].grid(alpha=0.3)
    axes[1, 1].set_title('Logit Statistics', fontsize=14); axes[1, 1].legend(fontsize=9); axes[1, 1].grid(alpha=0.3)
    for ax in axes.flat:
        ax.set_xlabel('Training Step')
    plt.suptitle(f'Training Curves — {env_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves_{env_name}.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Representation analysis
    rep = results.get('representation', {})
    if rep:
        methods = ['crl_forward', 'crl_backward', 'crl_symmetric', 'crl_efficient',
                   'mc_classifier', 'value_regression']
        method_colors = [colors.get(m, '#666') for m in methods]
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Binary probe
        vals = [rep.get(f'probe_binary_{m}', {}).get('mean', 0) for m in methods]
        stds = [rep.get(f'probe_binary_{m}', {}).get('std', 0) for m in methods]
        present = [(i, m) for i, m in enumerate(methods) if vals[i] > 0]
        if present:
            x = range(len(present))
            axes[0].bar(x, [vals[i] for i, _ in present],
                       yerr=[stds[i] for i, _ in present],
                       color=[method_colors[i] for i, _ in present], alpha=0.8, capsize=4)
            axes[0].set_xticks(list(x))
            axes[0].set_xticklabels([m.replace('_', '\n') for _, m in present], fontsize=9)
            axes[0].set_title('Linear Probe: Win Prediction', fontsize=13)
            axes[0].set_ylabel('Accuracy')
            axes[0].grid(alpha=0.3, axis='y')
        
        # 3-class probe
        vals3 = [rep.get(f'probe_3class_{m}', {}).get('mean', 0) for m in methods]
        stds3 = [rep.get(f'probe_3class_{m}', {}).get('std', 0) for m in methods]
        present3 = [(i, m) for i, m in enumerate(methods) if vals3[i] > 0]
        if present3:
            x = range(len(present3))
            axes[1].bar(x, [vals3[i] for i, _ in present3],
                       yerr=[stds3[i] for i, _ in present3],
                       color=[method_colors[i] for i, _ in present3], alpha=0.8, capsize=4)
            axes[1].set_xticks(list(x))
            axes[1].set_xticklabels([m.replace('_', '\n') for _, m in present3], fontsize=9)
            axes[1].set_title('3-Class Probe: W/D/L', fontsize=13)
            axes[1].set_ylabel('Accuracy')
            axes[1].grid(alpha=0.3, axis='y')
        
        # Fisher ratio
        fisher_vals = [rep.get(f'fisher_{m}', {}).get('ratio', 0) for m in methods]
        present_f = [(i, m) for i, m in enumerate(methods) if fisher_vals[i] > 0]
        if present_f:
            x = range(len(present_f))
            axes[2].bar(x, [fisher_vals[i] for i, _ in present_f],
                       color=[method_colors[i] for i, _ in present_f], alpha=0.8)
            axes[2].set_xticks(list(x))
            axes[2].set_xticklabels([m.replace('_', '\n') for _, m in present_f], fontsize=9)
            axes[2].set_title("Fisher's Cluster Separation", fontsize=13)
            axes[2].set_ylabel("Fisher's Ratio")
            axes[2].grid(alpha=0.3, axis='y')
        
        plt.suptitle(f'Representation Quality — {env_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/representation_{env_name}.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    # 3. Ablations
    abl = results.get('ablations', {})
    if abl:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        def plot_ablation(ax, prefix, values, xlabel, title, color):
            keys = [f'{prefix}_{v}' for v in values]
            accs = [abl.get(k, {}).get('acc', 0) for k in keys]
            present = [(i, v) for i, (v, a) in enumerate(zip(values, accs)) if a > 0]
            if present:
                x = range(len(present))
                ax.bar(x, [accs[i] for i, _ in present], color=color, alpha=0.8)
                ax.set_xticks(list(x))
                ax.set_xticklabels([str(v) for _, v in present])
                for xi, (i, _) in enumerate(present):
                    ax.text(xi, accs[i] + 0.005, f'{accs[i]:.3f}', ha='center', fontsize=8)
            ax.set_xlabel(xlabel); ax.set_ylabel('Accuracy'); ax.set_title(title)
            ax.grid(alpha=0.3, axis='y')
        
        plot_ablation(axes[0,0], 'embed_dim', config.ablation_embed_dims,
                     'Embedding Dim', 'Embedding Dimension', '#2196F3')
        plot_ablation(axes[0,1], 'n_neg', config.ablation_n_negatives,
                     'N Negatives', 'Number of Negatives', '#4CAF50')
        plot_ablation(axes[0,2], 'gamma', config.ablation_gammas,
                     'γ', 'Discount Factor', '#FF9800')
        plot_ablation(axes[1,0], 'n_blocks', config.ablation_n_blocks,
                     'N Blocks', 'Network Depth', '#9C27B0')
        
        # Mode comparison
        modes = ['forward', 'backward', 'symmetric', 'efficient']
        mode_accs = [abl.get(f'mode_{m}', {}).get('acc', 0) for m in modes]
        mode_colors = ['#2196F3', '#4CAF50', '#00BCD4', '#009688']
        present_m = [(i, m) for i, (m, a) in enumerate(zip(modes, mode_accs)) if a > 0]
        if present_m:
            x = range(len(present_m))
            axes[1,1].bar(x, [mode_accs[i] for i, _ in present_m],
                         color=[mode_colors[i] for i, _ in present_m], alpha=0.8)
            axes[1,1].set_xticks(list(x))
            axes[1,1].set_xticklabels([m for _, m in present_m])
            for xi, (i, _) in enumerate(present_m):
                axes[1,1].text(xi, mode_accs[i]+0.005, f'{mode_accs[i]:.3f}', ha='center', fontsize=8)
        axes[1,1].set_xlabel('Mode'); axes[1,1].set_ylabel('Accuracy')
        axes[1,1].set_title('Contrastive Mode'); axes[1,1].grid(alpha=0.3, axis='y')
        
        axes[1,2].axis('off')  # Empty
        
        plt.suptitle(f'Ablation Studies — {env_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ablations_{env_name}.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    # 4. Policy evaluation
    pol = results.get('policy', {})
    if pol:
        fig, ax = plt.subplots(figsize=(10, 6))
        method_names = list(pol.keys())
        win_rates = [pol[m] for m in method_names]
        bar_colors = [colors.get(m.replace('_goalsim', ''), '#666666') for m in method_names]
        bars = ax.bar(range(len(method_names)), win_rates, color=bar_colors, alpha=0.8)
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels([m.replace('_', '\n') for m in method_names], fontsize=9)
        ax.set_ylabel('Win Rate vs Random', fontsize=12)
        ax.set_title(f'Policy Evaluation — {env_name}', fontsize=14)
        ax.axhline(0.5, color='red', ls='--', alpha=0.5, label='50% baseline')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        for b, v in zip(bars, win_rates):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                   f'{v:.3f}', ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/policy_{env_name}.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    # 5. t-SNE
    tsne_data = results.get('tsne_embeddings', {})
    tsne_outcomes = results.get('tsne_outcomes', None)
    if tsne_data and tsne_outcomes is not None:
        try:
            from sklearn.manifold import TSNE
            n_methods = len(tsne_data)
            fig, axes = plt.subplots(1, min(n_methods, 4), figsize=(6*min(n_methods, 4), 5))
            if n_methods == 1:
                axes = [axes]
            
            for ax, (name, emb) in zip(axes, list(tsne_data.items())[:4]):
                # Subsample for speed
                n_sub = min(3000, len(emb))
                idx = np.random.choice(len(emb), n_sub, replace=False)
                emb_sub = emb[idx]
                out_sub = tsne_outcomes[idx]
                
                tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
                coords = tsne.fit_transform(emb_sub)
                
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=out_sub,
                                    cmap='RdYlGn', s=3, alpha=0.5)
                ax.set_title(name.replace('_', ' ').title(), fontsize=12)
                ax.set_xticks([]); ax.set_yticks([])
            
            plt.suptitle(f't-SNE Embeddings (colored by outcome) — {env_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/tsne_{env_name}.png', dpi=200, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  t-SNE plot failed: {e}")


# ============================================================
# Main Runner
# ============================================================

def run_experiment(config: ExperimentConfig):
    """Run full experiment pipeline for a single environment."""
    device = get_device(config)
    env_class, env_kwargs, state_dim, action_dim, board_size, n_planes = get_env_info(config.env_name)
    
    output_dir = os.path.join(config.output_dir, config.env_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config.save(os.path.join(output_dir, 'config.yaml'))
    
    print(f"\n{'='*70}")
    print(f" CRL Experiment: {config.env_name}")
    print(f" Device: {device}")
    print(f" State dim: {state_dim}, Action dim: {action_dim}")
    print(f" Encoder: {config.encoder_type}, Embed dim: {config.embed_dim}")
    print(f" Training: {config.n_epochs} steps, batch {config.batch_size}")
    print(f" Data: {config.n_train_games} games, gamma={config.gamma}")
    print(f"{'='*70}")
    
    # Collect data
    print(f"\n[1/6] Collecting {config.n_train_games} games...")
    t0 = time.time()
    trajs = collect_random_games(env_class, config.n_train_games, verbose=True, **env_kwargs)
    
    outcomes = [t['outcome'] for t in trajs]
    lengths = [t['length'] for t in trajs]
    total_states = sum(len(t['states']) for t in trajs)
    print(f"  Done in {time.time()-t0:.1f}s: {total_states} states, "
          f"avg_len={np.mean(lengths):.1f}, "
          f"W/D/L={outcomes.count(1)}/{outcomes.count(0)}/{outcomes.count(-1)}")
    
    buffer = TrajectoryBuffer(trajs, gamma=config.gamma, device=device)
    
    all_results = {
        'config': config.to_dict(),
        'data': {
            'n_games': len(trajs), 'total_states': total_states,
            'avg_length': float(np.mean(lengths)),
            'win_rate': outcomes.count(1)/len(outcomes),
            'draw_rate': outcomes.count(0)/len(outcomes),
            'loss_rate': outcomes.count(-1)/len(outcomes),
        },
        'training': {},
        'representation': {},
        'ablations': {},
        'policy': {},
    }
    
    # Multi-seed training
    seed = config.seeds[0]  # Use first seed for main experiments
    
    # Train all methods
    print(f"\n[2/6] Training methods (seed={seed})...")
    
    print("  CRL Forward...")
    crl_fwd = train_contrastive(config, buffer, state_dim, device, seed,
                                mode='forward', board_size=board_size, n_planes=n_planes)
    all_results['training']['crl_forward'] = {'history': crl_fwd['history']}
    
    print("  CRL Backward...")
    crl_bwd = train_contrastive(config, buffer, state_dim, device, seed,
                                mode='backward', board_size=board_size, n_planes=n_planes)
    all_results['training']['crl_backward'] = {'history': crl_bwd['history']}
    
    print("  CRL Symmetric...")
    crl_sym = train_contrastive(config, buffer, state_dim, device, seed,
                                mode='symmetric', board_size=board_size, n_planes=n_planes)
    all_results['training']['crl_symmetric'] = {'history': crl_sym['history']}
    
    print("  CRL Efficient (in-batch negatives)...")
    crl_eff = train_contrastive(config, buffer, state_dim, device, seed,
                                mode='efficient', board_size=board_size, n_planes=n_planes)
    all_results['training']['crl_efficient'] = {'history': crl_eff['history']}
    
    print("  MC Win Classifier...")
    mc_res = train_mc_classifier(config, buffer, state_dim, device, seed,
                                 board_size=board_size, n_planes=n_planes)
    all_results['training']['mc_classifier'] = {'history': mc_res['history']}
    
    print("  Value Regression...")
    val_res = train_value_net(config, buffer, state_dim, device, seed,
                              board_size=board_size, n_planes=n_planes)
    all_results['training']['value_regression'] = {'history': val_res['history']}
    
    models = {
        'crl_forward': crl_fwd, 'crl_backward': crl_bwd,
        'crl_symmetric': crl_sym, 'crl_efficient': crl_eff,
        'mc_classifier': mc_res, 'value_regression': val_res,
    }
    
    # Representation analysis
    print(f"\n[3/6] Representation analysis...")
    rep_results, tsne_embs, tsne_outcomes = evaluate_representations(models, buffer, device)
    all_results['representation'] = rep_results
    all_results['tsne_embeddings'] = tsne_embs
    all_results['tsne_outcomes'] = tsne_outcomes
    
    # Ablations
    if config.run_ablations:
        print(f"\n[4/6] Ablation studies...")
        abl_results = run_ablations(config, buffer, state_dim, device, seed,
                                    board_size=board_size, n_planes=n_planes)
        all_results['ablations'] = abl_results
    else:
        print(f"\n[4/6] Ablations skipped.")
    
    # Multi-seed runs for variance estimation
    print(f"\n[5/6] Multi-seed runs ({config.n_seeds} seeds)...")
    multi_seed_results = defaultdict(lambda: defaultdict(list))
    for i, s in enumerate(config.seeds[:config.n_seeds]):
        print(f"  Seed {s} ({i+1}/{config.n_seeds})...")
        for mode in ['forward', 'backward']:
            r = train_contrastive(config, buffer, state_dim, device, s,
                                 mode=mode, n_epochs=min(config.n_epochs, 500),
                                 board_size=board_size, n_planes=n_planes)
            final_acc = float(np.mean(r['history']['infonce_acc'][-50:]))
            multi_seed_results[f'crl_{mode}']['accs'].append(final_acc)
        
        r = train_mc_classifier(config, buffer, state_dim, device, s,
                               board_size=board_size, n_planes=n_planes,
                               n_epochs=min(config.n_epochs, 500))
        final_acc = float(np.mean(r['history']['mc_acc'][-50:]))
        multi_seed_results['mc_classifier']['accs'].append(final_acc)
    
    # Summarize multi-seed
    for method, data in multi_seed_results.items():
        accs = data['accs']
        all_results[f'multiseed_{method}'] = {
            'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
            'values': accs,
        }
        print(f"  {method}: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    
    # Policy evaluation
    print(f"\n[6/6] Policy evaluation ({config.n_policy_eval_games} games)...")
    policy_results = evaluate_policy(
        env_class, env_kwargs, models, buffer, device,
        n_games=config.n_policy_eval_games, temperature=config.policy_temperature
    )
    all_results['policy'] = policy_results
    
    # Plotting
    print(f"\nGenerating plots...")
    plot_all(all_results, config, output_dir)
    
    # Save results (remove non-serializable objects)
    save_results = {k: v for k, v in all_results.items()
                   if k not in ['tsne_embeddings', 'tsne_outcomes']}
    
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(convert(save_results), f, indent=2)
    
    # Save model checkpoints
    for name, mdata in models.items():
        torch.save(mdata['model'].state_dict(),
                  os.path.join(output_dir, f'{name}.pt'))
    
    print(f"\nResults saved to {output_dir}/")
    return all_results


# ============================================================
# Entry Point
# ============================================================

if __name__ == '__main__':
    config = parse_args()
    results = run_experiment(config)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f" SUMMARY: {config.env_name}")
    print(f"{'='*70}")
    
    print("\nTraining (final metrics):")
    for name, data in results.get('training', {}).items():
        hist = data.get('history', {})
        if 'infonce_acc' in hist:
            print(f"  {name}: acc={np.mean(hist['infonce_acc'][-50:]):.3f}")
        if 'mc_acc' in hist:
            print(f"  {name}: acc={np.mean(hist['mc_acc'][-50:]):.3f}")
        if 'value_mae' in hist:
            print(f"  {name}: MAE={np.mean(hist['value_mae'][-50:]):.3f}")
    
    print("\nRepresentation (linear probe binary):")
    for k, v in sorted(results.get('representation', {}).items()):
        if 'probe_binary' in k:
            print(f"  {k}: {v.get('mean', 0):.3f} ± {v.get('std', 0):.3f}")
    
    print("\nPolicy (win rate vs random):")
    for k, v in sorted(results.get('policy', {}).items()):
        print(f"  {k}: {v:.3f}")
