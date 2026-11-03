"""
evaluate.py - final head-to-head comparison on fixed held-out tracks.

Policies compared:
  * dqn           - learned policy (greedy)
  * nearest       - always the highest-elevation (geometrically nearest) satellite
  * max-sinr      - always the strongest instantaneous SINR
  * longest-los   - always the satellite with the longest remaining visibility
  * keep-or-nearest - stay until visibility is lost, then switch

Metrics (averaged over N held-out tracks, deterministic):
  mean_sinr_db, mean_quality, handovers/episode, outage ratio, Jain load fairness,
  mean_reward.
"""
import argparse
import os

import numpy as np
import torch

from sim.leo_sim import HandoverEnv
from agents.dqn import DQN
from agents.heuristics import KeepOrNearest, NearestSatellite, MaxSINR, LongestLOS

N_TRACKS = 50
TRACK_SEEDS = list(range(1000, 1000 + N_TRACKS))


def jain_fairness(counts):
    """counts: per-satellite cumulative service time."""
    c = np.asarray(counts, dtype=float)
    s = c.sum()
    return float(s * s / (len(c) * (c * c).sum())) if s > 0 else 0.0


def roll(env, policy, agent=None):
    """Run one episode; policy is a callable(state, m, serving) -> action."""
    s = env.reset()
    service = np.zeros(env.n_sats)
    done = False
    total_r = 0.0
    sinr_sum, q_sum = 0.0, 0.0
    while not done:
        m = env._sat_metrics(env.t)
        a = policy(s, m, env.serving)
        s, r, done, info = env.step(a)
        total_r += r
        if info["serving"] is not None:
            service[info["serving"]] += 1
        if info["sinr_db"] != float("-inf"):
            sinr_sum += info["sinr_db"]
            q_sum += info["quality"]
    steps = env.cfg["episode_len"]
    return dict(mean_sinr=sinr_sum / steps, mean_quality=q_sum / steps,
                handovers=env.handovers, outage=env.outage_steps / steps,
                fairness=jain_fairness(service), reward=total_r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/seed1/dqn_final.pt")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    env_kwargs = dict()
    env0 = HandoverEnv(rng=0, **env_kwargs)
    # ensemble: average Q across trained seeds
    base = os.path.dirname(os.path.dirname(args.model))
    agents = []
    for sd in args.seeds.split(","):
        p = os.path.join(base, f"seed{sd}", "dqn_final.pt")
        a = DQN(env0.state_dim, env0.action_dim, seed=int(sd))
        a.load(p)
        agents.append(a)

    def dqn_policy(s, m, serving):
        with torch.no_grad():
            x = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
            q = torch.stack([ag.online(x) for ag in agents]).mean(dim=0)
        return int(q.argmax(dim=1).item())

    policies = {
        "dqn (ensemble)": dqn_policy,
        "nearest": lambda s, m, sv: NearestSatellite().act(s, m, sv),
        "max-sinr": lambda s, m, sv: MaxSINR().act(s, m, sv),
        "longest-los": lambda s, m, sv: LongestLOS().act(s, m, sv),
        "keep-or-nearest": lambda s, m, sv: KeepOrNearest().act(s, m, sv),
    }

    rows = []
    for name, pol in policies.items():
        acc = {k: [] for k in ["mean_sinr", "mean_quality", "handovers", "outage", "fairness", "reward"]}
        for ts in TRACK_SEEDS:
            env = HandoverEnv(rng=ts, **env_kwargs)
            res = roll(env, pol)
            for k in acc:
                acc[k].append(res[k])
        rows.append({"policy": name, **{k: float(np.mean(v)) for k, v in acc.items()}})
        print(f"{name:18s} sinr={np.mean(acc['mean_sinr']):.2f} "
              f"q={np.mean(acc['mean_quality']):.3f} "
              f"ho={np.mean(acc['handovers']):.1f} "
              f"outage={np.mean(acc['outage']):.3f} "
              f"fair={np.mean(acc['fairness']):.3f} r={np.mean(acc['reward']):.1f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "comparison.csv"), "w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["policy", "mean_sinr", "mean_quality",
                                          "handovers", "outage", "fairness", "reward"])
        w.writeheader()
        w.writerows(rows)
    print("saved results/comparison.csv")


if __name__ == "__main__":
    main()
