"""
train_dqn.py - train the DQN handover policy (multiple seeds, checkpoint + resume).

Usage:
  python train_dqn.py --seed 1 --episodes 1200 --eval-interval 100 --out runs
Resume from the latest checkpoint of the same seed:
  python train_dqn.py --seed 1 --resume --out runs
"""
import argparse
import json
import os
import time

import numpy as np

from sim.leo_sim import HandoverEnv
from agents.dqn import DQN

EVAL_TRACKS = [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008]


def run_eval(agent, env_kwargs, n=8):
    total_r, total_ho, total_q, total_out = 0.0, 0, 0.0, 0
    ep_len = None
    for seed in EVAL_TRACKS[:n]:
        env = HandoverEnv(rng=seed, **env_kwargs)
        ep_len = env.cfg["episode_len"]
        s = env.reset()
        done = False
        while not done:
            a = agent.act(s, greedy=True)
            s, r, done, info = env.step(a)
            total_r += r
            total_q += info["quality"]
        total_ho += env.handovers          # cumulative per-episode counters
        total_out += env.outage_steps
    steps = n * ep_len
    return {"eval_reward": total_r / n, "eval_handovers": total_ho / n,
            "eval_quality": total_q / steps, "eval_outage": total_out / steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    env_kwargs = dict()
    out_dir = os.path.join(args.out, f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    env = HandoverEnv(rng=args.seed, **env_kwargs)
    obs_dim = env.state_dim
    n_actions = env.action_dim

    agent = DQN(obs_dim, n_actions, seed=args.seed)
    start_ep = 0
    if args.resume:
        ckpts = [f for f in os.listdir(out_dir) if f.startswith("dqn_ckpt_") and f.endswith(".pt")]
        if ckpts:
            latest = max(ckpts, key=lambda f: int(f.split("_")[2].split(".")[0]))
            agent.load(os.path.join(out_dir, latest))
            start_ep = int(latest.split("_")[2].split(".")[0])
            print(f"resumed from {latest} at episode {start_ep}", flush=True)

    # start epsilon at 0.5: unlike Atari, this MDP is small and the agent already
    # benefits from exploiting early; full random exploration mostly wastes steps
    agent.set_exploration(0.5, 0.05, args.episodes * env.cfg["episode_len"])
    log_rows = []

    t0 = time.time()
    for ep in range(start_ep, args.episodes):
        s = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            agent.update(s, a, r, s2, done)
            s = s2
            ep_r += r
        log_rows.append({"episode": ep + 1, "reward": ep_r,
                         "handovers": info["handovers"]})
        if (ep + 1) % 50 == 0:
            print(f"[seed {args.seed}] ep {ep + 1}/{args.episodes} "
                  f"reward={ep_r:.1f} ho={info['handovers']} "
                  f"eps={agent.eps:.3f} elapsed={time.time() - t0:.0f}s", flush=True)
        if (ep + 1) % args.eval_interval == 0:
            ev = run_eval(agent, env_kwargs)
            log_rows[-1].update(ev)
            print(f"   eval: r={ev['eval_reward']:.2f} ho={ev['eval_handovers']:.1f} "
                  f"q={ev['eval_quality']:.3f} outage={ev['eval_outage']:.3f}", flush=True)
        if (ep + 1) % 400 == 0:
            agent.save(os.path.join(out_dir, f"dqn_ckpt_{ep + 1}.pt"))
            with open(os.path.join(out_dir, f"meta_{ep + 1}.json"), "w") as f:
                json.dump({"episodes": ep + 1, "seed": args.seed}, f)

    agent.save(os.path.join(out_dir, "dqn_final.pt"))
    with open(os.path.join(out_dir, "eval_log.csv"), "w") as f:
        f.write("episode,reward,handovers,eval_reward,eval_handovers,eval_quality,eval_outage\n")
        for row in log_rows:
            f.write(",".join(str(row.get(k, "")) for k in
                             ["episode", "reward", "handovers", "eval_reward",
                              "eval_handovers", "eval_quality", "eval_outage"]) + "\n")
    print(f"[seed {args.seed}] done in {time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
