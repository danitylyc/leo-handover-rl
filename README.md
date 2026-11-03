# LEO Handover RL

**Learning satellite-handover decisions for dense low-Earth-orbit (LEO) constellations.**

A from-scratch, self-contained project that frames *which satellite should serve a
moving ground terminal* as a reinforcement-learning problem. It ships a lightweight
physics-based LEO simulator (orbital geometry, Ku-band link budget, per-satellite
load dynamics), a hand-written double-DQN agent, four classical heuristic baselines,
and a reproducible training/evaluation/plotting pipeline. **No RL library, no
satellite-toolbox dependency** — everything is numpy + torch.

---

## Why this problem

LEO mega-constellations (Starlink, OneWeb, …) keep a terminal in view of **several
satellites at the same time**, and the serving satellite rises and sets within
minutes. Each *handover* decision trades off:

- **link quality** — SINR depends on elevation (slant range) and per-satellite gain,
- **handover cost** — signalling, re-acquisition, beam-switching overhead,
- **load balance** — hot satellites degrade everyone they serve.

Classical heuristics optimize one of these at a time (always the highest satellite,
always the strongest signal, always the longest-visible one). None of them sees the
full horizon. A learned policy can in principle trade all three off against each
other — that is the question this project explores end-to-end.

**Simulator-level finding worth knowing:** a 550 km shell sees only ~17° of central
angle (≈ 1,900 km ground footprint), so a sparse constellation leaves *structural
coverage gaps* that no policy can fix. Handover decisions only become meaningful once
the constellation is dense enough to guarantee visible candidates — the config used
here (24 planes × 8 sats) keeps visible candidates available throughout an episode
(measured outage ≈2.5–3%, most of it at handover boundaries).

## What is in this repo

| module | what it does |
|---|---|
| `sim/leo_sim.py` | numpy-only LEO handover MDP: Walker-style constellation, ECEF geometry, elevation/visibility, Ku link budget, load dynamics |
| `agents/dqn.py` | hand-written double DQN (128×2 MLP, replay, soft target update) |
| `agents/heuristics.py` | 4 classical baselines: `nearest`, `max-sinr`, `longest-los`, `keep-or-nearest` |
| `train_dqn.py` | multi-seed training, checkpointing, fixed-track evaluation, CSV logs |
| `evaluate.py` | 50 held-out-track head-to-head: DQN ensemble vs. all baselines |
| `plot_results.py` | learning curves + comparison figures |

### Simulator model (summarized)

- **Orbit:** circular 550 km, inclination 53°, Walker-style phasing across 24 planes;
  orbit planes are inertially fixed and the Earth rotates underneath (RAAN drifts
  westward in ECEF).
- **Link budget:** Ku 12 GHz, spot-beam EIRP 50 dBW, receiver noise floor −125 dBW,
  free-space path loss; per-satellite gain offset (±6 dB) models payload variance.
- **Terminal:** low-latitude mobile user (10–40 m/s), elevation mask 10°.
- **MDP:** state = [terminal geodetic position (2)] + [serving sat: elevation, SINR,
  load (3)] + top-K candidate descriptors: elevation, SINR, remaining visibility,
  load (4K) — candidates are ordered by descending elevation, padding when fewer
  visible; action = {keep, candidate slot 1..K}; reward = link quality − 0.4·load
  − 1.0·handover − 3.0·outage.

## Results (mean over 50 held-out tracks, deterministic seeds)

> Numbers below are produced by `evaluate.py` from the checked-in checkpoints.
> See `results/comparison.png` and `results/training_curves.png`.

| policy | mean quality | SINR (dB) | handovers/ep | outage ratio | fairness | reward |
|---|---|---|---|---|---|---|
| **dqn (3-seed ensemble)** | **0.579** | 1.60 | **4.8** | 0.027 | 0.017 | **84.3** |
| max-sinr | 0.603 | **2.30** | 10.1 | 0.026 | 0.014 | 79.0 |
| longest-los | 0.550 | 1.24 | 10.5 | 0.025 | 0.015 | 66.9 |
| nearest | 0.550 | 1.24 | 10.5 | 0.025 | 0.015 | 66.9 |
| keep-or-nearest | 0.504 | 0.33 | 9.3 | 0.031 | 0.013 | 38.7 |

**What the learned policy actually does:** max-sinr chases the strongest signal on
every step (best quality, most handovers). The DQN policy keeps the link while it is
good and switches only when the serving satellite's quality is about to drop — it
**halves the handover count (4.8 vs 10.1) at only a 4% quality cost, and ends up with
the highest total reward (+6.7% over the best heuristic)**. That is the whole point:
a value function with γ=0.99 implicitly looks ahead, which a greedy single-objective
heuristic cannot.

## Reproduce

```bash
pip install numpy torch matplotlib      # CPU torch is enough

# train 3 seeds (checkpoints + eval log under runs/)
python train_dqn.py --seed 1 --episodes 1200 --out runs
python train_dqn.py --seed 2 --episodes 1200 --out runs
python train_dqn.py --seed 3 --episodes 1200 --out runs

# head-to-head evaluation on 50 held-out tracks
python evaluate.py --model runs/seed1/dqn_final.pt --seeds 1,2,3

# figures
python plot_results.py
```

## Notes / honest limitations

- **Coverage guarantee:** the dense 24×8 single-shell config is a simplification —
  real constellations achieve continuity by stacking several shells. The geometry in
  `_sat_ecef` is exactly the part to swap for a TLE/propagator if you need realism.
- **No link-layer detail:** no modulation, fading or beam pointing; SINR is
  path-loss-dominated.
- **Single-terminal episodes:** load balancing enters through per-satellite load in
  the state/reward; true multi-terminal contention is a natural next step (MARL).
- The learning curves plot what actually happened, including early instability.

## Project layout

```
leo-handover-rl/
├── sim/            # simulator (MDP)
├── agents/         # DQN + heuristics
├── runs/seed{N}/   # checkpoints + eval_log.csv
├── results/        # comparison.csv + figures
├── train_dqn.py    # training entry
├── evaluate.py     # final evaluation
└── plot_results.py # figures
```
