"""
leo_sim.py - lightweight LEO constellation handover simulator (self-contained, numpy-only).

Problem it solves
-----------------
In a low-Earth-orbit (LEO) constellation a ground terminal is served by one satellite
at a time. Satellites cross the sky in minutes, so the terminal must repeatedly decide
whether to KEEP its current serving satellite or HAND OVER to a visible candidate.
Naive policies (always the strongest signal) cause excessive switching, brief outages
and load imbalance across satellites.

This environment turns that decision into a discrete-time MDP:
  state    -> normalized vector (terminal position, serving-sat metrics, per-candidate
              elevation / remaining visibility / load)
  action   -> 0 = keep serving, 1..K = hand over to candidate #k (by descending elevation)
  reward   -> signal quality + load-balancing term - switching cost - outage penalty

The model is deliberately teaching-grade but physically consistent:
  * circular orbits, Walker-delta-like geometry (RAAN + in-plane phase offsets)
  * spherical geometry: central angle -> elevation -> visibility & slant range
  * simplified Ku-band link budget: free-space path loss + receiver noise -> SINR,
    with a per-satellite gain offset (beam-pointing / hardware variance)
  * per-satellite load dynamics -> load-balancing signal inside the reward

numpy-only; no external simulation engine.
"""
import numpy as np

MU = 3.986004418e14        # Earth gravitational parameter [m^3/s^2]
RE = 6.371e6               # Earth radius [m]
C_LIGHT = 2.99792458e8
OMEGA_EARTH = 7.2921159e-5     # Earth rotation rate [rad/s] (WGS84)

DEFAULTS = dict(
    altitude=550e3,        # orbital altitude [m]
    inclination=np.deg2rad(53.0),
    n_planes=24,           # number of orbital planes
    sats_per_plane=8,      # satellites per plane (192 sats total). A 550 km shell only
                           # sees ~17 deg of central angle, so a single sparse layer
                           # leaves structural gaps; this 24x8 Walker config keeps >=2
                           # visible candidates at all times so handover is a real
                           # decision (Starlink achieves continuity by stacking many
                           # shells; we model one dense shell instead)
    freq_hz=12e9,          # Ku-band carrier [Hz]
    eirp_dB=50.0,          # Ku spot-beam EIRP [dBW]
    rx_noise_dB=-125.0,    # receiver noise floor incl. G/T [dBW]
    gain_off_db=6.0,       # per-satellite gain offset [dB] (payload variance: makes
                           # the best-signal choice differ from the best-geometry one)
    el_min=np.deg2rad(10.0),  # minimum usable elevation [rad]
    dt=2.0,                # decision interval [s]
    episode_len=300,       # steps per episode
    K=4,                   # candidate slots in the action space
    load_step=0.02,        # load increase per served step
    load_decay=0.01,       # load decay per unserved step
)


def orbital_period(alt):
    return 2.0 * np.pi * np.sqrt((RE + alt) ** 3 / MU)


def _geo_to_ecef(lat, lon):
    """lat/lon in radians -> unit ECEF vector."""
    cl, sl = np.cos(lat), np.sin(lat)
    cg, sg = np.cos(lon), np.sin(lon)
    return np.array([cl * cg, cl * sg, sl])


def _central_angle(sat_ecef, term_ecef):
    d = np.dot(sat_ecef, term_ecef) / (np.linalg.norm(sat_ecef) * np.linalg.norm(term_ecef))
    return np.arccos(np.clip(d, -1.0, 1.0))


def _elevation(sat_ecef, term_ecef):
    r = np.linalg.norm(sat_ecef)
    phi = _central_angle(sat_ecef, term_ecef)
    slant = np.linalg.norm(sat_ecef - term_ecef)
    sin_el = (r * np.cos(phi) - RE) / slant
    return np.arcsin(np.clip(sin_el, -1.0, 1.0))


def _max_central_angle(alt, el_min):
    """Central angle at which a satellite is exactly at el_min (bisection)."""
    r = RE + alt
    lo, hi = 0.0, np.pi
    target = np.sin(el_min)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        slant = np.sqrt(r * r + RE * RE - 2 * r * RE * np.cos(mid))
        v = (r * np.cos(mid) - RE) / slant
        if v > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class HandoverEnv:
    """Discrete-time LEO handover MDP for one mobile terminal."""

    def __init__(self, rng=None, **kw):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(kw)
        self.rng = np.random.default_rng(rng)
        self.T = orbital_period(self.cfg["altitude"])
        self.omega = 2.0 * np.pi / self.T
        self._init_constellation()
        self.phi_max = _max_central_angle(self.cfg["altitude"], self.cfg["el_min"])
        self.state_dim = 2 + 3 + self.cfg["K"] * 4   # terminal(2) + serving(3) + K candidates(4)
        self.action_dim = self.cfg["K"] + 1  # 0 = keep, 1..K = candidate
        self._reset_history()

    # ------------------------------------------------------------------
    def _init_constellation(self):
        c = self.cfg
        n = c["n_planes"] * c["sats_per_plane"]
        self.n_sats = n
        self.sat_raan = np.empty(n)
        self.sat_phase0 = np.empty(n)
        idx = 0
        for p in range(c["n_planes"]):
            raan = 2 * np.pi * p / c["n_planes"]
            for m in range(c["sats_per_plane"]):
                phase0 = 2 * np.pi * m / c["sats_per_plane"]
                phase0 += 2 * np.pi * p / (c["n_planes"] * c["sats_per_plane"]) * 0.5
                self.sat_raan[idx] = raan
                self.sat_phase0[idx] = phase0
                idx += 1
        self.gain_off = np.zeros(n)  # set per episode (hardware variance)

    def _sat_ecef(self, t):
        """ECEF positions at time t.

        Orbit planes are inertially fixed; the Earth rotates underneath, so in
        ECEF coordinates the RAAN drifts westward at the Earth's rotation rate.
        This matters: without it the orbital belt stays frozen over the ground and
        leaves permanent coverage gaps."""
        r = RE + self.cfg["altitude"]
        inc = self.cfg["inclination"]
        theta = self.sat_phase0 + self.omega * t
        raan = self.sat_raan - OMEGA_EARTH * t          # westward drift in ECEF
        ct, st = np.cos(theta), np.sin(theta)
        cg, sg = np.cos(raan), np.sin(raan)
        ci, si = np.cos(inc), np.sin(inc)
        x = r * (ct * cg - st * sg * ci)
        y = r * (ct * sg + st * cg * ci)
        z = r * (st * si)
        return np.stack([x, y, z], axis=1)  # (n_sats, 3)

    # ------------------------------------------------------------------
    def _sample_terminal(self):
        """Sample a terminal start; reset() re-samples until the first state is visible."""
        lat = self.rng.uniform(-0.35, 0.35)          # low-latitude belt (rad)
        lon = self.rng.uniform(-2.6, 2.6)
        speed = self.rng.uniform(10.0, 40.0)         # m/s (vehicle-scale)
        bearing = self.rng.uniform(0.0, 2 * np.pi)
        return lat, lon, speed, bearing

    def _advance_terminal(self, dt):
        """Move the terminal along a great-circle-ish local bearing (small-step planar approx)."""
        d = self.term_speed * dt
        # planar displacement in local ENU, converted back to lat/lon (fine for ~km scale)
        dlon = d * np.sin(self.term_bearing) / (RE * np.cos(self.term_lat))
        dlat = d * np.cos(self.term_bearing) / RE
        self.term_lat = np.clip(self.term_lat + dlat, -1.4, 1.4)
        self.term_lon += dlon

    # ------------------------------------------------------------------
    def _sat_metrics(self, t):
        """Per satellite (vectorized): elevation, slant range, sinr_db, visibility, LOS."""
        c = self.cfg
        sats = self._sat_ecef(t)                     # (n,3)
        term = RE * _geo_to_ecef(self.term_lat, self.term_lon)   # (3,)
        rn = np.linalg.norm(sats, axis=1)            # (n,)
        d = (sats @ term) / (rn * RE)
        phis = np.arccos(np.clip(d, -1.0, 1.0))
        slants = np.linalg.norm(sats - term, axis=1)
        sin_el = (rn * np.cos(phis) - RE) / slants
        els = np.arcsin(np.clip(sin_el, -1.0, 1.0))
        lam = C_LIGHT / c["freq_hz"]
        fspl = 20 * np.log10(4 * np.pi * slants / lam)
        sinr = c["eirp_dB"] + self.gain_off - fspl - c["rx_noise_dB"]
        visible = els >= c["el_min"]
        los = np.where(visible, (self.phi_max - phis) / (self.omega * 0.5 + 1e-9), 0.0)
        los = np.clip(los, 0.0, 600.0)
        return dict(els=els, slants=slants, sinr=sinr, visible=visible, los=los)

    # ------------------------------------------------------------------
    def _build_state(self, m):
        c = self.cfg
        els, sinr, los, visible = m["els"], m["sinr"], m["los"], m["visible"]
        K = c["K"]
        # SINR normalization: sim range ~[-17, +10] dB -> [0, 1]
        def snr_norm(v):
            return float(np.clip((v + 17.0) / 27.0, 0.0, 1.0))
        # serving info
        if self.serving is not None:
            cur_el = els[self.serving] / (np.pi / 2)
            cur_snr = snr_norm(sinr[self.serving])
            cur_load = self.loads[self.serving]
        else:
            cur_el, cur_snr, cur_load = 0.0, 0.0, 0.0
        s = [self.term_lat / 0.7, self.term_lon / 3.0, cur_el, cur_snr, cur_load]
        # candidates: visible sats sorted by descending elevation, top K
        cand = np.where(visible)[0]
        order = cand[np.argsort(-els[cand])][:K]
        for i in range(K):
            if i < len(order):
                si = order[i]
                s += [els[si] / (np.pi / 2), snr_norm(sinr[si]),
                      los[si] / 300.0, self.loads[si]]
            else:
                s += [-1.0, -1.0, -1.0, -1.0]
        return np.array(s, dtype=np.float32)

    def _reward(self, m, handover, outage):
        c = self.cfg
        if self.serving is not None:
            # map SINR ~[-10, +10] dB into [0, 1] link quality
            q = np.clip((m["sinr"][self.serving] + 10.0) / 20.0, 0.0, 1.0)
            load = self.loads[self.serving]
        else:
            q, load = 0.0, 0.0
        return q - 0.4 * load - 1.0 * handover - 3.0 * outage

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.loads = self.rng.uniform(0.10, 0.40, size=self.n_sats)
        self.t0 = self.rng.uniform(0.0, self.T)          # constellation phase offset
        self.t = self.t0
        self.step_no = 0
        # re-sample the terminal start until at least one satellite is visible,
        # so the episode begins with a valid link (dense shell: almost always true)
        for _ in range(200):
            self.term_lat, self.term_lon, self.term_speed, self.term_bearing = self._sample_terminal()
            m = self._sat_metrics(self.t)
            if m["visible"].any():
                break
        self.gain_off = self.rng.uniform(-self.cfg["gain_off_db"], self.cfg["gain_off_db"],
                                         size=self.n_sats)
        m = self._sat_metrics(self.t)
        # initial serving: best visible by SINR (fallback: any visible)
        vis = np.where(m["visible"])[0]
        if len(vis) > 0:
            self.serving = int(vis[np.argmax(m["sinr"][vis])])
        else:
            self.serving = None
        self.handovers = 0
        self.outage_steps = 0
        self._reset_history()
        return self._build_state(m)

    def step(self, action):
        c = self.cfg
        a = int(action)
        m = self._sat_metrics(self.t)
        vis = np.where(m["visible"])[0]
        cand = vis[np.argsort(-m["els"][vis])][:c["K"]] if len(vis) > 0 else np.array([], dtype=int)

        # decide target satellite
        handover = False
        if self.serving is None:
            target = int(cand[0]) if len(cand) > 0 else None
            if target is not None and target != self.serving:
                handover = True
        elif a == 0:
            target = self.serving if self.serving in vis else None
        else:
            k = a - 1
            if k < len(cand):
                target = int(cand[k])
            else:
                target = self.serving
        if target is None:
            # keep is impossible (serving gone) -> force best visible
            target = int(cand[0]) if len(cand) > 0 else None
            handover = True
        if target is not None and target != self.serving:
            handover = True

        # apply handover
        if handover:
            self.handovers += 1
        self.serving = target

        # advance world
        self.t += c["dt"]
        self._advance_terminal(c["dt"])
        self.step_no += 1

        # load dynamics
        self.loads = np.maximum(self.loads - c["load_decay"], 0.02)
        if self.serving is not None:
            self.loads[self.serving] = np.minimum(self.loads[self.serving] + c["load_step"], 1.0)

        # outage: no serving or serving below el_min
        m2 = self._sat_metrics(self.t)
        outage = 0
        if self.serving is None or m2["els"][self.serving] < c["el_min"]:
            outage = 1
            self.outage_steps += 1

        r = self._reward(m2, handover, outage)
        done = self.step_no >= c["episode_len"]
        return self._build_state(m2), r, done, dict(
            sinr_db=float(m2["sinr"][self.serving]) if self.serving is not None else float("-inf"),
            quality=float(np.clip((m2["sinr"][self.serving] + 10.0) / 20.0, 0.0, 1.0)) if self.serving is not None else 0.0,
            handovers=self.handovers, outage=self.outage_steps,
            serving=self.serving, loads=self.loads.copy())

    # ------------------------------------------------------------------
    def _reset_history(self):
        self._last_phi = None
