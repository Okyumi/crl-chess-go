"""
Run experiments for all environments sequentially.

Usage:
  python run_all.py                    # All envs with defaults
  python run_all.py --envs connect4 chess go9  # Specific envs
  python run_all.py --device cuda      # Force GPU
"""

import argparse
import time
from config import PRESETS, ExperimentConfig
from run_experiments import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', nargs='+', default=['connect4', 'chess', 'go9'],
                       help='Environments to run')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--no-ablations', action='store_true')
    args = parser.parse_args()
    
    total_start = time.time()
    all_summaries = {}
    
    for env_name in args.envs:
        print(f"\n{'#'*70}")
        print(f"# Running: {env_name}")
        print(f"{'#'*70}")
        
        if env_name in PRESETS:
            config = PRESETS[env_name]
        else:
            config = ExperimentConfig(env_name=env_name)
        
        config.device = args.device
        config.output_dir = args.output_dir
        if args.no_ablations:
            config.run_ablations = False
        
        t0 = time.time()
        results = run_experiment(config)
        elapsed = time.time() - t0
        
        all_summaries[env_name] = {
            'elapsed_minutes': elapsed / 60,
            'policy': results.get('policy', {}),
            'data': results.get('data', {}),
        }
        
        print(f"\n{env_name} completed in {elapsed/60:.1f} min")
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f" ALL DONE — Total: {total_elapsed/60:.1f} min")
    print(f"{'='*70}")
    
    for env_name, summary in all_summaries.items():
        print(f"\n{env_name} ({summary['elapsed_minutes']:.1f} min):")
        print(f"  Data: {summary['data'].get('total_states', '?')} states")
        for method, wr in sorted(summary.get('policy', {}).items()):
            print(f"  Policy {method}: {wr:.3f}")


if __name__ == '__main__':
    main()
