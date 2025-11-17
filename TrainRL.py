#!/usr/bin/env python3
"""
Lightweight RL training scaffold that replaces the PATH-based solver with a
learned policy. The goal is the same: two agents maintain a distance band
while moving to paired targets and avoiding circular obstacles. Only standard
Python/NumPy/PyTorch are used; no network calls or heavy dependencies.
"""

import glob
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Reuse the game solver to produce expert demonstrations
import CollaborativeGameSolver as GameSolver

# -----------------------------
# Scenario and env definitions
# -----------------------------

@dataclass
class Scenario:
    name: str
    x1_init: np.ndarray
    x2_init: np.ndarray
    x1_des: np.ndarray
    theta_des: float
    obstacles: List[Dict[str, np.ndarray]]


SCENARIOS: List[Scenario] = [
    Scenario(
        name="Scenario_6_WithObs",
        x1_init=np.array([[3.0, 0.0]]),
        x2_init=np.array([[0.0, 0.0]]),
        x1_des=np.array([[7.0, 0.0]]),
        theta_des=np.deg2rad(60.0),
        obstacles=[{"Pos": np.array([[6.0, 2.0]]), "diam": 0.5}],
    ),
    Scenario(
        name="Scenario_6_Switch_WithObs",
        x1_init=np.array([[0.0, 0.0]]),
        x2_init=np.array([[3.0, 0.0]]),
        x1_des=np.array([[7.0, 4.0]]),
        theta_des=np.deg2rad(240.0),
        obstacles=[{"Pos": np.array([[4.5, 2.0]]), "diam": 0.5}],
    ),
]


def random_scenarios(
    num: int = 4,
    seed: int = None,
    workspace: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 8.0), (-2.0, 6.0)),
    min_separation: float = 1.0,
    max_separation: float = 4.0,
    obstacle_prob: float = 0.7,
    max_obstacles: int = 1,
) -> List[Scenario]:
    """
    Generate a list of randomized scenarios with reasonable spacing/obstacle placement.
    """
    rng = np.random.default_rng(seed)
    scenarios: List[Scenario] = []
    for i in range(num):
        # Sample a base point and a separation vector
        base = np.array(
            [
                [rng.uniform(*workspace[0]), rng.uniform(*workspace[1])]
            ]
        )
        sep_dist = rng.uniform(min_separation, max_separation)
        sep_angle = rng.uniform(0, 2 * math.pi)
        sep_vec = sep_dist * np.array([[math.cos(sep_angle), math.sin(sep_angle)]])

        x1_init = base
        x2_init = base + sep_vec

        # Clamp into workspace
        x2_init[0, 0] = np.clip(x2_init[0, 0], *workspace[0])
        x2_init[0, 1] = np.clip(x2_init[0, 1], *workspace[1])

        # Sample goal near top/right-ish to keep motion meaningful
        x1_des = base + rng.uniform(2.0, 4.0, size=(1, 2)) * rng.choice([-1, 1], size=(1, 2))
        x1_des[0, 0] = np.clip(x1_des[0, 0], *workspace[0])
        x1_des[0, 1] = np.clip(x1_des[0, 1], *workspace[1])

        theta_des = rng.uniform(0, 2 * math.pi)

        obstacles: List[Dict[str, np.ndarray]] = []
        if rng.random() < obstacle_prob:
            n_obs = rng.integers(1, max_obstacles + 1)
            for _ in range(n_obs):
                pos = np.array(
                    [
                        [
                            rng.uniform(*workspace[0]),
                            rng.uniform(*workspace[1]),
                        ]
                    ]
                )
                diam = float(rng.uniform(0.3, 0.7))
                obstacles.append({"Pos": pos, "diam": diam})

        scenarios.append(
            Scenario(
                name=f"Random_{i}",
                x1_init=x1_init,
                x2_init=x2_init,
                x1_des=x1_des,
                theta_des=theta_des,
                obstacles=obstacles,
            )
        )
    return scenarios


@dataclass
class EnvConfig:
    dt: float = 0.1
    horizon: int = 10
    tf: float = 10.0
    d_delta: float = 0.05
    v1_max: float = 1.5
    a1_max: float = 5.0
    v2_max: float = 3.0
    a2_max: float = 25.0
    reward_distance_w: float = 1.0
    reward_band_penalty: float = 5.0
    reward_obs_penalty: float = 10.0
    reward_action_penalty: float = 0.01
    success_bonus: float = 10.0
    max_obstacles: int = 3  # pad observation if fewer obstacles are present


def limited_cmd(a: np.ndarray, a_max: float) -> np.ndarray:
    norm = np.linalg.norm(a)
    if norm > a_max:
        return a / norm * a_max
    return a


class CollaborativeEnv:
    """
    Minimal Gym-like environment.
    Observation: [x1,y1,v1x,v1y, x2,y2,v2x,v2y, x1_des,y1_des, x2_des,y2_des,
                  (obs1_x, obs1_y, obs1_relx, obs1_rely), ... padded to max_obstacles]
    Action: 2D acceleration for the robot (agent controls only the robot).
    Human uses a simple PD-like tracking toward its goal.
    """

    def __init__(self, config: EnvConfig, scenarios: List[Scenario]):
        self.cfg = config
        self.scenarios = scenarios
        self.current: Scenario = None
        self.reset()

    def reset(self) -> np.ndarray:
        self.current = random.choice(self.scenarios)
        self.t = 0.0
        self.step_count = 0

        self.x1 = self.current.x1_init.copy()
        self.x2 = self.current.x2_init.copy()
        self.v1 = np.zeros((1, 2))
        self.v2 = np.zeros((1, 2))

        d_init = np.linalg.norm(self.x1 - self.x2)
        self.d_min, self.d_max = d_init - self.cfg.d_delta, d_init + self.cfg.d_delta
        self.x2_des = self.current.x1_des + d_init * np.array(
            [[math.cos(self.current.theta_des), math.sin(self.current.theta_des)]]
        )

        return self._obs()

    def _obs(self) -> np.ndarray:
        obs_parts = [
            self.x1.flatten(),
            self.v1.flatten(),
            self.x2.flatten(),
            self.v2.flatten(),
            self.current.x1_des.flatten(),
            self.x2_des.flatten(),
        ]
        # Add obstacle absolute positions and relative vectors to robot; pad to max_obstacles
        for i_obs in range(self.cfg.max_obstacles):
            if i_obs < len(self.current.obstacles):
                center = self.current.obstacles[i_obs]["Pos"].flatten()
                rel = center - self.x2.flatten()
            else:
                center = np.zeros(2)
                rel = np.zeros(2)
            obs_parts.extend([center, rel])
        return np.concatenate(obs_parts)

    def _human_policy(self) -> np.ndarray:
        """Greedy accel toward its goal with velocity saturation."""
        to_goal = self.current.x1_des - self.x1
        desired_v = limited_cmd(to_goal / max(np.linalg.norm(to_goal), 1e-6), self.cfg.v1_max)
        a = 2.0 * (desired_v - self.v1)
        return limited_cmd(a, self.cfg.a1_max)

    def _apply_dynamics(self, a1: np.ndarray, a2: np.ndarray) -> None:
        dt = self.cfg.dt
        self.x1 = self.x1 + dt * self.v1 + 0.5 * dt**2 * a1
        self.v1 = self.v1 + dt * a1
        self.x2 = self.x2 + dt * self.v2 + 0.5 * dt**2 * a2
        self.v2 = self.v2 + dt * a2

    def _band_penalty(self) -> float:
        dist = np.linalg.norm(self.x1 - self.x2)
        if dist < self.d_min or dist > self.d_max:
            return self.cfg.reward_band_penalty * abs(dist - np.clip(dist, self.d_min, self.d_max))
        return 0.0

    def _obs_penalty(self) -> float:
        penalty = 0.0
        for obs in self.current.obstacles:
            center, diam = obs["Pos"], obs["diam"]
            if np.linalg.norm(self.x2 - center) < diam / 2.0:
                penalty += self.cfg.reward_obs_penalty
        return penalty

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        a_robot = limited_cmd(action.reshape(1, 2), self.cfg.a2_max)
        a_human = self._human_policy()

        self._apply_dynamics(a_human, a_robot)
        self.t += self.cfg.dt
        self.step_count += 1

        dist_cost = np.linalg.norm(self.x2 - self.x2_des)
        band_pen = self._band_penalty()
        obs_pen = self._obs_penalty()
        act_pen = self.cfg.reward_action_penalty * float(np.linalg.norm(a_robot))
        reward = -(
            self.cfg.reward_distance_w * dist_cost
            + band_pen
            + obs_pen
            + act_pen
        )

        done = False
        success = False
        if (np.linalg.norm(self.x1 - self.current.x1_des) < 0.2) and (
            np.linalg.norm(self.x2 - self.x2_des) < 0.2
        ):
            reward += self.cfg.success_bonus
            done, success = True, True
        if self.t >= self.cfg.tf:
            done = True

        info = {"success": success, "band_penalty": band_pen, "obs_penalty": obs_pen}
        return self._obs(), reward, done, info


# -----------------------------
# Simple policy + trainer
# -----------------------------


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collect_episode(env: CollaborativeEnv, policy: PolicyNet, device: torch.device):
    obs_list, act_list, rew_list = [], [], []
    obs = env.reset()
    done = False
    while not done:
        obs_t = torch.from_numpy(obs).float().to(device)
        # Deterministic policy for now; add noise for exploration.
        action = policy(obs_t).cpu().detach().numpy()
        next_obs, reward, done, info = env.step(action)

        obs_list.append(obs)
        act_list.append(action)
        rew_list.append(reward)
        obs = next_obs
    return obs_list, act_list, rew_list


def finish_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    g = 0.0
    returns = []
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    return list(reversed(returns))


def train(num_episodes: int = 100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CollaborativeEnv(EnvConfig(), SCENARIOS)
    obs_dim = env._obs().shape[0]
    policy = PolicyNet(obs_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    for ep in range(num_episodes):
        obs_list, act_list, rew_list = collect_episode(env, policy, device)
        returns = torch.tensor(finish_returns(rew_list), dtype=torch.float32, device=device)
        obs_batch = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
        act_batch = torch.tensor(np.array(act_list), dtype=torch.float32, device=device)

        pred_actions = policy(obs_batch)
        loss = ((pred_actions - act_batch) ** 2).mean() - returns.mean() * 1e-3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"Episode {ep:03d} | Len {len(rew_list):03d} | "
            f"Return {sum(rew_list):6.2f} | Loss {loss.item():.4f}"
        )


# -----------------------------
# PATH solver demonstrations
# -----------------------------

def rollout_with_solver(
    scenario: Scenario,
    alpha: float = 0.5,
    cfg: EnvConfig = EnvConfig(),
) -> Dict[str, np.ndarray]:
    """
    Use the PATH-based solver as an expert to produce one trajectory of
    (state, action) pairs. We mirror the stepping logic from CollaborativeGame.py
    but without plotting or multiprocessing.
    """
    x1_init, x2_init = scenario.x1_init, scenario.x2_init
    x1_des, theta_des = scenario.x1_des, scenario.theta_des
    obs_list: List[np.ndarray] = []
    act_robot_list: List[np.ndarray] = []
    act_human_list: List[np.ndarray] = []
    time_list: List[float] = []

    d_init = np.linalg.norm(x1_init - x2_init)
    x2_des = x1_des + d_init * np.array(
        [[math.cos(theta_des), math.sin(theta_des)]]
    )

    GameSol = GameSolver.CollaborativeGame(
        N=cfg.horizon, dt=cfg.dt, d=d_init, Obstcles=scenario.obstacles
    )

    t = 0.0
    Tf = cfg.tf
    x1_state, x2_state = x1_init.copy(), x2_init.copy()
    v1_state, v2_state = np.zeros((1, 2)), np.zeros((1, 2))
    EndSimulation = False
    i_acc = 0
    GameSol.success = False
    reverse_init = False
    avoid_Obs = 0.0

    while not EndSimulation:
        if GameSol.success:
            z0 = GameSol.z0
            z0[GameSol.indx_x1:GameSol.indx_x1 + cfg.horizon] = GameSol.sol.x1_sol[1:, 0]
            z0[GameSol.indx_y1:GameSol.indx_y1 + cfg.horizon] = GameSol.sol.x1_sol[1:, 1]
            z0[GameSol.indx_vx1:GameSol.indx_vx1 + cfg.horizon] = GameSol.sol.v1_sol[1:, 0]
            z0[GameSol.indx_vy1:GameSol.indx_vy1 + cfg.horizon] = GameSol.sol.v1_sol[1:, 1]
            z0[GameSol.indx_ax1:GameSol.indx_ax1 + cfg.horizon - 1] = GameSol.sol.a1_sol[1:, 0]
            z0[GameSol.indx_ay1:GameSol.indx_ay1 + cfg.horizon - 1] = GameSol.sol.a1_sol[1:, 1]
            z0[GameSol.indx_x2:GameSol.indx_x2 + cfg.horizon] = GameSol.sol.x2_sol[1:, 0]
            z0[GameSol.indx_y2:GameSol.indx_y2 + cfg.horizon] = GameSol.sol.x2_sol[1:, 1]
            z0[GameSol.indx_vx2:GameSol.indx_vx2 + cfg.horizon] = GameSol.sol.v2_sol[1:, 0]
            z0[GameSol.indx_vy2:GameSol.indx_vy2 + cfg.horizon] = GameSol.sol.v2_sol[1:, 1]
            z0[GameSol.indx_ax2:GameSol.indx_ax2 + cfg.horizon - 1] = GameSol.sol.a2_sol[1:, 0]
            z0[GameSol.indx_ay2:GameSol.indx_ay2 + cfg.horizon - 1] = GameSol.sol.a2_sol[1:, 1]

            GameSol.Solve(
                t, x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, alpha, z0=z0, avoid_Obs=avoid_Obs
            )
        if not GameSol.success:
            x1_guess, v1_guess, a1_guess = GameSol.MPC_guess_human_calc(x1_state, v1_state, x1_des)
            x2_guess, v2_guess, a2_guess = GameSol.MPC_guess_robot_calc(x2_state, v2_state, x2_des, x_partner=x1_guess, reverse_init=reverse_init)
            reverse_init = False

            z0 = np.zeros_like(GameSol.z0)
            H = cfg.horizon
            z0[GameSol.indx_x1:GameSol.indx_x1 + H + 1] = x1_guess[:, 0]
            z0[GameSol.indx_y1:GameSol.indx_y1 + H + 1] = x1_guess[:, 1]
            z0[GameSol.indx_vx1:GameSol.indx_vx1 + H + 1] = v1_guess[:, 0]
            z0[GameSol.indx_vy1:GameSol.indx_vy1 + H + 1] = v1_guess[:, 1]
            z0[GameSol.indx_ax1:GameSol.indx_ax1 + H] = a1_guess[:, 0]
            z0[GameSol.indx_ay1:GameSol.indx_ay1 + H] = a1_guess[:, 1]
            z0[GameSol.indx_x2:GameSol.indx_x2 + H + 1] = x2_guess[:, 0]
            z0[GameSol.indx_y2:GameSol.indx_y2 + H + 1] = x2_guess[:, 1]
            z0[GameSol.indx_vx2:GameSol.indx_vx2 + H + 1] = v2_guess[:, 0]
            z0[GameSol.indx_vy2:GameSol.indx_vy2 + H + 1] = v2_guess[:, 1]
            z0[GameSol.indx_ax2:GameSol.indx_ax2 + H] = a2_guess[:, 0]
            z0[GameSol.indx_ay2:GameSol.indx_ay2 + H] = a2_guess[:, 1]

            GameSol.Solve(
                t, x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, alpha, z0=z0, avoid_Obs=avoid_Obs
            )

        if GameSol.success:
            i_acc = 0
        else:
            i_acc += 1

        # Expert action comes from solver's robot acceleration at i_acc
        a1_cmd = limited_cmd(GameSol.sol.a1_sol[i_acc, :] , GameSol.a1_max)
        a2_cmd = limited_cmd(GameSol.sol.a2_sol[i_acc, :], GameSol.a2_max)

        # Build observation before stepping
        # Observation with obstacle context
        obs_parts = [
            x1_state.flatten(),
            v1_state.flatten(),
            x2_state.flatten(),
            v2_state.flatten(),
            x1_des.flatten(),
            x2_des.flatten(),
        ]
        for i_obs in range(cfg.max_obstacles):
            if i_obs < len(scenario.obstacles):
                center = scenario.obstacles[i_obs]["Pos"].flatten()
                rel = center - x2_state.flatten()
            else:
                center = np.zeros(2)
                rel = np.zeros(2)
            obs_parts.extend([center, rel])
        obs = np.concatenate(obs_parts)
        obs_list.append(obs)
        act_robot_list.append(a2_cmd.copy())
        act_human_list.append(a1_cmd.copy())
        time_list.append(t)

        # Step dynamics
        x1_state = x1_state + cfg.dt * v1_state + 0.5 * cfg.dt ** 2 * a1_cmd
        v1_state = v1_state + cfg.dt * a1_cmd
        x2_state = x2_state + cfg.dt * v2_state + 0.5 * cfg.dt ** 2 * a2_cmd
        v2_state = v2_state + cfg.dt * a2_cmd
        t += cfg.dt

        if np.linalg.norm(GameSol.sol.v2_sol[-1, :]) < 0.5 and np.linalg.norm(GameSol.sol.x2_sol[-1, :] - x2_des) > 0.5:
            avoid_Obs = 1.0 - avoid_Obs
            GameSol.success = False
        if t >= Tf or (
            np.linalg.norm(x1_state - x1_des) < 0.2 and np.linalg.norm(x2_state - x2_des) < 0.2
        ):
            EndSimulation = True

    return {
        "obs": np.array(obs_list),
        "act_robot": np.array(act_robot_list),
        "act_human": np.array(act_human_list),
        "time": np.array(time_list),
        "scenario": scenario.name,
        "alpha": alpha,
    }


def _rollout_task(args):
    scenario, alpha = args
    return rollout_with_solver(scenario, alpha)


def generate_solver_action_dataset(
    out_path: str = "solver_actions_dataset.npz",
    alphas: List[float] = (0.01, 0.5, 0.99),
    episodes_per_scenario: int = 2,
    max_workers: int = None,
    num_random_scenarios: int = 4,
):
    """
    Run the original PATH-solver-driven simulation for multiple scenarios/alphas
    and save state -> (human, robot) action pairs.
    """
    base_scenarios = SCENARIOS[:]  # copy defaults
    if num_random_scenarios > 0:
        base_scenarios = base_scenarios + random_scenarios(num=num_random_scenarios)

    tasks = []
    for scenario in base_scenarios:
        for alpha in alphas:
            for _ in range(episodes_per_scenario):
                tasks.append((scenario, alpha))

    if max_workers is None:
        import os
        max_workers = int(max(1, (os.cpu_count() or 2) / 2))

    demos = []
    # Use processes to match prior PATH solver usage pattern
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for result in ex.map(_rollout_task, tasks):
            demos.append(result)

    obs = np.concatenate([d["obs"] for d in demos], axis=0)
    act_robot = np.concatenate([d["act_robot"] for d in demos], axis=0)
    act_human = np.concatenate([d["act_human"] for d in demos], axis=0)
    time = np.concatenate([d["time"] for d in demos], axis=0)
    meta = {
        "scenario": [d["scenario"] for d in demos],
        "alpha": [d["alpha"] for d in demos],
        "lengths": [len(d["obs"]) for d in demos],
    }
    np.savez(out_path, obs=obs, act_robot=act_robot, act_human=act_human, time=time, meta=meta)
    print(f"Saved {obs.shape[0]} samples to {out_path}")


def load_solver_action_datasets(pattern: str = "solver_actions_dataset*.npz") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one or more solver demonstration datasets matched by pattern.
    Returns concatenated (obs, act_robot) arrays.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No datasets found for pattern: {pattern}")

    obs_list, act_list = [], []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        if "obs" not in data or "act_robot" not in data:
            raise KeyError(f"{path} is missing expected keys 'obs'/'act_robot'")
        obs = data["obs"]
        acts = data["act_robot"]
        if obs.shape[0] != acts.shape[0]:
            raise ValueError(f"{path} obs/actions length mismatch: {obs.shape[0]} vs {acts.shape[0]}")
        obs_list.append(obs)
        act_list.append(acts)
        print(f"Loaded {obs.shape[0]} samples from {path}")

    obs_all = np.concatenate(obs_list, axis=0)
    act_all = np.concatenate(act_list, axis=0)
    print(f"Total samples: {obs_all.shape[0]} from {len(paths)} files")
    return obs_all, act_all


def train_from_dataset(
    pattern: str = "solver_actions_dataset*.npz",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    checkpoint_path: str = "policy_from_solver.pt",
):
    """
    Supervised training of PolicyNet on solver demonstration datasets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_np, acts_np = load_solver_action_datasets(pattern)

    obs_tensor = torch.tensor(obs_np, dtype=torch.float32)
    acts_tensor = torch.tensor(acts_np, dtype=torch.float32)

    dataset_size = obs_tensor.shape[0]
    n_val = int(max(1, val_split * dataset_size)) if dataset_size > 10 else 0
    indices = list(range(dataset_size))
    random.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_ds = torch.utils.data.TensorDataset(obs_tensor[train_idx], acts_tensor[train_idx])
    val_ds = torch.utils.data.TensorDataset(obs_tensor[val_idx], acts_tensor[val_idx]) if n_val > 0 else None

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size) if val_ds else None

    obs_dim = obs_tensor.shape[1]
    policy = PolicyNet(obs_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        policy.train()
        total_loss = 0.0
        for obs_b, act_b in train_loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            pred = policy(obs_b)
            loss = ((pred - act_b) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * obs_b.size(0)
        train_loss = total_loss / len(train_ds)

        val_loss = None
        if val_loader:
            policy.eval()
            with torch.no_grad():
                vloss = 0.0
                for obs_b, act_b in val_loader:
                    obs_b = obs_b.to(device)
                    act_b = act_b.to(device)
                    pred = policy(obs_b)
                    loss = ((pred - act_b) ** 2).mean()
                    vloss += loss.item() * obs_b.size(0)
                val_loss = vloss / len(val_ds)

        if val_loss is None:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}")
        else:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    torch.save(policy.state_dict(), checkpoint_path)
    print(f"Saved trained policy to {checkpoint_path}")
    return policy


if __name__ == "__main__":
    # To create a supervised dataset from the PATH solver, uncomment:
    # generate_solver_action_dataset(out_path="solver_actions_dataset.npz", alphas=[0.05])
    # To run the quick RL training stub, comment the line above and uncomment below:
    # train(num_episodes=5)
    # To train from previously saved solver datasets:
    train_from_dataset(pattern="solver_actions_dataset*.npz", epochs=100, batch_size=64)
