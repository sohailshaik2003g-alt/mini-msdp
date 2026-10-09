"""
Plotting script: learning curves with 95% CI across seeds.
Run after all RL runs finish: python src/plot_results.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import glob
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

METHODS = {
    'msdp_r':        ('MSDP-R (ours)',     '#2196F3'),
    'concat':        ('Concat baseline',   '#FF9800'),
    'ablation_pool': ('Ablation: Pool',    '#9C27B0'),
    'ablation_cls':  ('Ablation: CLS',     '#4CAF50'),
    'state_sanity':  ('State SAC (sanity)','#F44336'),
}

def load_method_logs(method: str, log_dir: str = 'logs') -> dict[int, dict]:
    """Load all seed logs for a method. Returns {seed: {steps, success_rate, mean_return}}."""
    pattern = os.path.join(log_dir, f'{method}_seed*.csv')
    files = glob.glob(pattern)
    seed_data = {}
    for f in files:
        # Ignore test seeds (like seed99)
        basename = os.path.basename(f)
        try:
            seed = int(basename.split('seed')[1].split('.')[0])
        except ValueError:
            continue
        if seed >= 10:  # test seeds
            continue
        steps, sr, mr = [], [], []
        with open(f) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                steps.append(int(row['step']))
                sr.append(float(row['success_rate']))
                mr.append(float(row['mean_return']))
        if len(steps) >= 2:
            seed_data[seed] = {'steps': steps, 'success_rate': sr, 'mean_return': mr}
    return seed_data


def interpolate_to_common_steps(seed_data: dict, n_points: int = 20) -> tuple:
    """Interpolate all seeds to common step axis."""
    valid_data = {k: v for k, v in seed_data.items() if len(v['steps']) >= 2}
    if not valid_data:
        return np.array([]), np.array([]), np.array([])

    all_steps = []
    for d in valid_data.values():
        all_steps.extend(d['steps'])
    step_min = min(all_steps)
    step_max = max(all_steps)
    common_steps = np.linspace(step_min, step_max, n_points)

    sr_interp = []
    mr_interp = []
    for d in valid_data.values():
        s = np.array(d['steps'])
        sr = np.interp(common_steps, s, d['success_rate'])
        mr = np.interp(common_steps, s, d['mean_return'])
        sr_interp.append(sr)
        mr_interp.append(mr)

    sr_arr = np.array(sr_interp)  # (n_seeds, n_points)
    mr_arr = np.array(mr_interp)
    return common_steps, sr_arr, mr_arr


def compute_ci_95(arr: np.ndarray) -> tuple:
    """Compute mean and 95% CI across seeds (axis 0)."""
    n = arr.shape[0]
    mean = arr.mean(0)
    if n == 1:
        return mean, np.zeros_like(mean)
    se = arr.std(0, ddof=1) / np.sqrt(n)
    # t-interval for small n
    t_val = stats.t.ppf(0.975, df=n - 1)
    ci = t_val * se
    return mean, ci


def plot_learning_curves(log_dir: str = 'logs', out_dir: str = 'results/figures'):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('MSDP Mini Reproduction — Push Cube (mock env)', fontsize=13, fontweight='bold')

    for method, (label, color) in METHODS.items():
        seed_data = load_method_logs(method, log_dir)
        if not seed_data:
            continue

        n_seeds = len(seed_data)
        common_steps, sr_arr, mr_arr = interpolate_to_common_steps(seed_data)
        sr_mean, sr_ci = compute_ci_95(sr_arr)
        mr_mean, mr_ci = compute_ci_95(mr_arr)

        for ax, (mean, ci, ylabel) in zip(axes, [
            (sr_mean, sr_ci, 'Success Rate'),
            (mr_mean, mr_ci, 'Mean Return'),
        ]):
            ax.plot(common_steps, mean, label=f'{label} (n={n_seeds})',
                    color=color, linewidth=2)
            ax.fill_between(common_steps, mean - ci, mean + ci,
                            alpha=0.2, color=color)

    for ax, ylabel in zip(axes, ['Success Rate', 'Mean Return']):
        ax.set_xlabel('RL Steps', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylim(-0.05, 1.05)
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'learning_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def print_results_table(log_dir: str = 'logs'):
    """Print final performance table."""
    print("\n" + "=" * 70)
    print("RESULTS TABLE (final eval, mean +/- 95% CI across seeds)")
    print("=" * 70)
    print(f"{'Method':<20} {'Seeds':<6} {'Success Rate':>14} {'Mean Return':>14}")
    print("-" * 70)

    for method, (label, _) in METHODS.items():
        seed_data = load_method_logs(method, log_dir)
        if not seed_data:
            continue

        final_sr = []
        final_mr = []
        for d in seed_data.values():
            if d['success_rate']:
                final_sr.append(d['success_rate'][-1])
                final_mr.append(d['mean_return'][-1])

        n = len(final_sr)
        if n == 0:
            continue

        sr_arr = np.array(final_sr)
        mr_arr = np.array(final_mr)
        sr_mean, sr_ci = float(sr_arr.mean()), (stats.t.ppf(0.975, df=max(n-1,1)) * sr_arr.std(ddof=min(1,n-1)) / np.sqrt(n) if n > 1 else 0)
        mr_mean, mr_ci = float(mr_arr.mean()), (stats.t.ppf(0.975, df=max(n-1,1)) * mr_arr.std(ddof=min(1,n-1)) / np.sqrt(n) if n > 1 else 0)

        print(f"{label:<20} {n:<6} {sr_mean:>7.3f} +/-{sr_ci:>5.3f}  {mr_mean:>7.2f} +/-{mr_ci:>5.2f}")

    print("=" * 70)
    print("\nNote: 3 seeds (paper: 12), 100k steps (paper: 500k), mock env (paper: PyBullet/panda-gym)")
    print("95% CI computed via t-distribution (df=n_seeds-1).")


if __name__ == '__main__':
    plot_learning_curves()
    print_results_table()
