"""
dqn.py - self-implemented Deep Q-Network agent (double DQN, soft target update).

Implementation notes (kept deliberately small and dependency-light):
  * MLP Q-network: state -> 128 -> 128 -> num_actions
  * experience replay buffer (circular, numpy-backed)
  * soft target-network update (tau) instead of periodic hard copy
  * double-DQN action selection: online net picks the action, target net values it
  * epsilon-greedy exploration with linear decay
Everything is written from scratch on numpy + torch (CPU); no RL library is used.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.cap = capacity
        self.s = np.empty((capacity, 0), dtype=np.float32)  # resized lazily
        self._n = 0
        self._pos = 0

    def _ensure(self, obs_dim):
        if self.s.shape[1] != obs_dim:
            self.s = np.empty((self.cap, obs_dim), dtype=np.float32)
            self.a = np.empty(self.cap, dtype=np.int64)
            self.r = np.empty(self.cap, dtype=np.float32)
            self.s2 = np.empty((self.cap, obs_dim), dtype=np.float32)
            self.d = np.empty(self.cap, dtype=bool)
            self._n = 0
            self._pos = 0

    def push(self, s, a, r, s2, done):
        self._ensure(s.shape[0])
        i = self._pos
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = done
        self._pos = (i + 1) % self.cap
        self._n = min(self._n + 1, self.cap)

    def sample(self, batch):
        idx = np.random.randint(0, self._n, size=batch)
        return (torch.as_tensor(self.s[idx]), torch.as_tensor(self.a[idx]),
                torch.as_tensor(self.r[idx]), torch.as_tensor(self.s2[idx]),
                torch.as_tensor(self.d[idx]))


class DQN:
    def __init__(self, obs_dim, n_actions, seed=0, lr=3e-4, gamma=0.99,
                 tau=0.005, batch=64, hidden=128):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.batch = batch
        self.online = QNet(obs_dim, n_actions, hidden)
        self.target = QNet(obs_dim, n_actions, hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.0  # set by caller: (eps - eps_min) / total_steps
        self.steps = 0

    def set_exploration(self, eps, eps_min, total_steps):
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = (eps - eps_min) / max(1, total_steps)

    def act(self, s, greedy=False):
        if not greedy and np.random.rand() < self.eps:
            return int(np.random.randint(0, self.n_actions))
        with torch.no_grad():
            q = self.online(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def update(self, s, a, r, s2, done):
        self.steps += 1
        self.eps = max(self.eps - self.eps_decay, self.eps_min)
        self.buffer.push(np.asarray(s, np.float32), a, r, np.asarray(s2, np.float32), done)
        if self.buffer._n < self.batch:
            return 0.0
        s, a, r, s2, d = self.buffer.sample(self.batch)
        with torch.no_grad():
            # double DQN: online selects, target evaluates
            next_acts = self.online(s2).argmax(dim=1, keepdim=True)
            q_next = self.target(s2).gather(1, next_acts).squeeze(1)
            y = r + self.gamma * q_next * (1.0 - d.float())
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        # Huber loss: robust to the sparse -3 outage reward (vs MSE spikes)
        loss = torch.nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # soft update target
        with torch.no_grad():
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * op.data)
        return float(loss.item())

    def save(self, path):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "opt": self.opt.state_dict(), "steps": self.steps, "eps": self.eps,
                    "n_actions": self.n_actions}, path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.online.load_state_dict(ck["online"])
        self.target.load_state_dict(ck["target"])
        self.opt.load_state_dict(ck["opt"])
        self.steps = ck["steps"]
        self.eps = ck["eps"]
