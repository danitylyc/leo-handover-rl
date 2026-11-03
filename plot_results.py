"""
plot_results.py - produce the project figures.

  results/training_curves.png : DQN learning curves across 3 seeds (eval reward,
                                eval handovers/episode, eval quality, eval outage)
  results/comparison.png      : head-to-head bar comparison of all policies
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")


def load_eval_log(seed):
    """Parse runs/seed{N}/eval_log.csv without pandas.
    Eval columns are empty on non-eval rows."""
    p = os.path.join(BASE, "runs", f"seed{seed}", "eval_log.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        header = f.readline().strip().split(",")
        rows = []
        for line in f:
            parts = line.strip().split(",")
            rows.append(dict(zip(header, parts)))
    return rows


def to_col(rows, name):
    return np.array([float(r[name]) for r in rows if r[name] != ""], dtype=float)


def plot_training():
    logs = [load_eval_log(s) for s in (1, 2, 3)]
    logs = [l for l in logs if l is not None]
    if not logs:
        print("no eval_log.csv found")
        return
    df = logs[0]
    xs = np.array([float(r["episode"]) for r in df if r["eval_reward"] != ""])
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    for i, (col, title) in enumerate([
            ("eval_reward", "eval reward (higher better)"),
            ("eval_handovers", "handovers / episode (lower better)"),
            ("eval_quality", "mean link quality (higher better)"),
            ("eval_outage", "outage ratio (lower better)")]):
        ys = np.stack([to_col(l, col) for l in logs])
        mean, std = ys.mean(0), ys.std(0)
        axs[i].plot(xs, mean, color="#1f77b4", lw=1.6)
        axs[i].fill_between(xs, mean - std, mean + std, alpha=0.25, color="#1f77b4")
        axs[i].set_title(title, fontsize=9)
        axs[i].set_xlabel("training episodes", fontsize=8)
        axs[i].grid(alpha=0.3)
    fig.suptitle("DQN handover policy - learning curves (mean +/- std, 3 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "training_curves.png"), dpi=140)
    print("saved results/training_curves.png")


def plot_comparison():
    p = os.path.join(RESULTS, "comparison.csv")
    with open(p) as f:
        header = f.readline().strip().split(",")
        rows = [dict(zip(header, line.strip().split(","))) for line in f]
    names = [r["policy"] for r in rows]
    metrics = [
        ("mean_quality", "mean link quality", True),
        ("mean_sinr", "mean SINR (dB)", True),
        ("handovers", "handovers / episode", False),
        ("outage", "outage ratio", False),
        ("fairness", "load fairness (Jain)", True),
        ("reward", "mean reward", True),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    colors = ["#d62728" if n == "dqn (ensemble)" else "#1f77b4" for n in names]
    for ax, (col, title, higher) in zip(axs.flat, metrics):
        vals = np.array([float(r[col]) for r in rows])
        ax.bar(names, vals, color=colors, alpha=0.85)
        ax.set_title(f"{title} ({'higher' if higher else 'lower'} is better)", fontsize=9)
        ax.tick_params(axis="x", rotation=18, labelsize=8)
        ax.grid(alpha=0.3, axis="y")
        for x, v in zip(range(len(vals)), vals):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Head-to-head: learned DQN vs classical heuristics "
                 "(50 held-out tracks)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "comparison.png"), dpi=140)
    print("saved results/comparison.png")


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    plot_training()
    plot_comparison()
