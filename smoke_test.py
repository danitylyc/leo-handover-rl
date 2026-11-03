"""Quick smoke test for the simulator + agents (dev only)."""
import numpy as np
from sim.leo_sim import HandoverEnv
from agents.dqn import DQN
from agents.heuristics import NearestSatellite, MaxSINR, LongestLOS, KeepOrNearest

env = HandoverEnv(rng=42)
s = env.reset()
print("state_dim", env.state_dim, "action_dim", env.action_dim, "n_sats", env.n_sats)
assert s.shape[0] == env.state_dim, s.shape

# random-policy episode
rng = np.random.default_rng(0)
r, ho, out = 0.0, 0, 0
done = False
while not done:
    s2, rr, done, info = env.step(rng.integers(0, env.action_dim))
    r += rr; ho = info["handovers"]; out = info["outage"]
print(f"random: reward={r:.1f} handovers={ho} outage_steps={out}")

for pol in (NearestSatellite(), MaxSINR(), LongestLOS(), KeepOrNearest()):
    env = HandoverEnv(rng=7)
    s = env.reset(); r = 0.0; done = False
    while not done:
        m = env._sat_metrics(env.t)
        a = pol.act(s, m, env.serving)
        s, rr, done, info = env.step(a)
        r += rr
    print(f"{pol.name:16s} reward={r:.1f} handovers={info['handovers']}")

# DQN 60-episode smoke
agent = DQN(env.state_dim, env.action_dim, seed=0)
agent.set_exploration(1.0, 0.05, 60 * env.cfg["episode_len"])
env = HandoverEnv(rng=0)
losses = []
for ep in range(60):
    s = env.reset(); done = False
    while not done:
        a = agent.act(s)
        s2, rr, done, info = env.step(a)
        l = agent.update(s, a, rr, s2, done)
        s = s2
        if l > 0:
            losses.append(l)
print("dqn smoke: final_eps=%.3f loss_mean=%.4f last50=%.4f" %
      (agent.eps, float(np.mean(losses)), float(np.mean(losses[-50:]))))
print("SMOKE_OK")
