from __future__ import annotations
import math
import itertools
from typing import Iterable, List, Sequence, Tuple, Any, Optional
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def _all_subsets_of_size_k(n: int, k: int) -> Iterable[Tuple[int, ...]]:
    return itertools.combinations(range(n), k)

class SpatialMABEnvironment:
    def __init__(self, n: int, k: int, x_true: npt.ArrayLike, mu_true: npt.ArrayLike, sigma_true: npt.ArrayLike, *, lambda_true: float = 1.0, beta_true: float = 1.0, seed: int | np.random.Generator | None = None) -> None:
        self.n, self.k = int(n), int(k)
        self.x_true = np.asarray(x_true, dtype=float)
        self.mu_true = np.asarray(mu_true, dtype=float)
        self.sigma_true = np.asarray(sigma_true, dtype=float)
        if self.x_true.shape != (self.n,) or self.mu_true.shape != (self.n,) or self.sigma_true.shape != (self.n,): raise ValueError("shape mismatch")
        if (self.sigma_true < 0).any(): raise ValueError("sigma must be non-negative")
        self.lambda_true = float(lambda_true)
        self.beta_true = float(beta_true)
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    def play_subset(self, subset: Sequence[int]) -> Tuple[np.ndarray, float]:
        subset = np.asarray(subset, dtype=int)
        if subset.size != self.k or len({*subset}) != self.k: raise ValueError("subset must contain K distinct indices.")
        if (subset < 0).any() or (subset >= self.n).any(): raise IndexError("subset contains out-of-range indices.")
        x_subset = self.x_true[subset]
        x_bar = x_subset.mean()
        noise = self._rng.normal(0.0, self.sigma_true[subset])
        r = self.mu_true[subset] + noise
        penalty = self.lambda_true * np.abs(x_subset - x_bar) ** self.beta_true
        R_vec = r - penalty
        return R_vec, float(R_vec.sum())

def init_agent_estimates(agent_type: str, true_mu: npt.ArrayLike, true_x: npt.ArrayLike, params: dict[str, Any] | None, *, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None: rng = np.random.default_rng()
    params = {} if params is None else params
    true_mu = np.asarray(true_mu, dtype=float)
    true_x = np.asarray(true_x, dtype=float)
    N = true_mu.size
    if agent_type == "naive": return np.zeros(N), np.zeros(N)
    if agent_type == "single_noisy":
        sigma = float(params.get("sigma", 0.1))
        if sigma < 0: raise ValueError("sigma must be non-negative.")
        est_mu = rng.normal(true_mu, sigma)
        est_x = rng.normal(true_x, sigma)
        return est_mu, est_x
    if agent_type == "sep_noisy":
        sigma_mu = float(params.get("sigma_mu", 0.2))
        sigma_x = float(params.get("sigma_x", 0.4))
        if sigma_mu < 0 or sigma_x < 0: raise ValueError("sigmas must be non-negative.")
        est_mu = rng.normal(true_mu, sigma_mu)
        est_x = rng.normal(true_x, sigma_x)
        return est_mu, est_x
    if agent_type == "dist_based":
        agent_loc = float(params.get("agent_loc", 0.0))
        noise_min = float(params.get("noise_min", 0.1))
        noise_slope = float(params.get("noise_slope", 0.2))
        if noise_min < 0 or noise_slope < 0: raise ValueError("noise parameters must be non-negative.")
        dist = np.abs(true_x - agent_loc)
        sigmas = noise_min + noise_slope * dist
        est_mu = rng.normal(true_mu, sigmas)
        est_x = rng.normal(true_x, sigmas)
        return est_mu, est_x
    raise ValueError(f"unknown agent_type = {agent_type!r}")

class _SubsetStrategy:
    def __init__(self, hat_mu: np.ndarray, hat_x: np.ndarray, lambda_hat: float, *, N: int, K: int, gamma: float, beta: float, max_cache: int = 10_000, sample_exploit: int = 5_000) -> None:
        self.mu_hat, self.x_hat = hat_mu, hat_x
        self.lambda_hat = lambda_hat
        self.N, self.K = int(N), int(K)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self._t = 0
        total_subsets = math.comb(self.N, self.K)
        self._subset_space = list(_all_subsets_of_size_k(self.N, self.K)) if total_subsets <= max_cache else None
        self._total_subsets = total_subsets
        self._sample_exploit = sample_exploit

    def _predicted_reward(self, subset: Tuple[int, ...]) -> float:
        idx = np.fromiter(subset, dtype=int, count=self.K)
        mu_i = self.mu_hat[idx]
        x_i = self.x_hat[idx]
        x_bar = x_i.mean()
        dist_b = np.abs(x_i - x_bar) ** self.beta
        return float((mu_i - self.lambda_hat * dist_b).sum())

    def _best_subset_cached(self) -> Tuple[int, ...]:
        best_S = None
        best_val = -np.inf
        for S in self._subset_space:
            val = self._predicted_reward(S)
            if val > best_val:
                best_S, best_val = S, val
        return best_S

    def _best_subset_sampled(self, rng: np.random.Generator) -> Tuple[int, ...]:
        best_S = None
        best_val = -np.inf
        for _ in range(self._sample_exploit):
            S = tuple(rng.choice(self.N, self.K, replace=False))
            val = self._predicted_reward(S)
            if val > best_val:
                best_S, best_val = S, val
        return best_S

    def choose_subset(self, rng: np.random.Generator) -> Tuple[int, ...]:
        self._t += 1
        eps_t = min(self.gamma / self._t, 1.0)
        if rng.random() < eps_t:
            return tuple(rng.choice(self.N, self.K, replace=False))
        if self._subset_space is not None:
            return self._best_subset_cached()
        return self._best_subset_sampled(rng)

def _safe_power(arr: np.ndarray, exponent: float, eps: float = 1e-12) -> np.ndarray:
    return np.abs(arr) + eps if exponent < 1.0 else np.abs(arr)

def _backprop_update(subset: Sequence[int], R_obs: np.ndarray, mu_hat: np.ndarray, x_hat: np.ndarray, lambda_hat: float, *, beta: float, lr_mu: float, lr_x: float, lr_lambda: float) -> float:
    idx = np.asarray(subset, dtype=int)
    mu_s = mu_hat[idx]
    x_s = x_hat[idx]
    x_bar = float(x_s.mean())
    dist = x_s - x_bar
    dist_b = _safe_power(dist, beta) ** beta
    R_pred = mu_s - lambda_hat * dist_b
    err = R_pred - R_obs
    grad_mu = err
    grad_lambda = -(err * dist_b).sum()
    sign_term = beta * lambda_hat * np.sign(dist) * _safe_power(dist, beta - 1)
    grad_x = -err * sign_term
    coupling = sign_term.mean()
    grad_x += err.sum() * coupling
    mu_hat[idx] -= lr_mu * grad_mu
    x_hat[idx] -= lr_x * grad_x
    lambda_hat_new = lambda_hat - lr_lambda * grad_lambda
    return max(0.0, lambda_hat_new)

class Agent:
    def __init__(self, n: int, k: int, agent_type: str, true_mu: npt.ArrayLike, true_x: npt.ArrayLike, *, agent_params: dict[str, Any] | None = None, gamma: float = 0.3, beta: float = 1.0, lambda_init: float = 1.0, lr_mu: float = 0.05, lr_x: float = 0.05, lr_lambda: float = 0.05, seed: int | np.random.Generator | None = None) -> None:
        self.n, self.k = int(n), int(k)
        self.beta = float(beta)
        self.lr_mu, self.lr_x, self.lr_lambda = lr_mu, lr_x, lr_lambda
        self._rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.mu_hat, self.x_hat = init_agent_estimates(agent_type, true_mu, true_x, agent_params, rng=self._rng)
        self.lambda_hat = float(lambda_init)
        self._strategy = _SubsetStrategy(self.mu_hat, self.x_hat, self.lambda_hat, N=self.n, K=self.k, gamma=gamma, beta=self.beta)

    def act(self) -> Tuple[int, ...]:
        return self._strategy.choose_subset(self._rng)

    def learn(self, subset: Sequence[int], R_vec: np.ndarray) -> None:
        self.lambda_hat = _backprop_update(subset=subset, R_obs=R_vec, mu_hat=self.mu_hat, x_hat=self.x_hat, lambda_hat=self.lambda_hat, beta=self.beta, lr_mu=self.lr_mu, lr_x=self.lr_x, lr_lambda=self.lr_lambda)
        self._sync_strategy()

    def _sync_strategy(self) -> None:
        self._strategy.lambda_hat = self.lambda_hat

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500    , help="number of time steps")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--monitor", action="store_true")
    args = parser.parse_args()

    N, K = 6, 3
    x_true = np.array([0.0, 1.5, 3.0, 6.0, 7.0, 9.0])
    mu_true = np.array([1.0, 2.0, 1.5, 3.5, 4.0, 2.5])
    sigma = np.full(N, 0.2)

    env = SpatialMABEnvironment(N, K, x_true, mu_true, sigma, lambda_true=0.8, beta_true=1.0, seed=1)
    agent = Agent(n=N, k=K, agent_type="single_noisy", true_mu=mu_true, true_x=x_true, agent_params={"sigma": 0.1}, gamma=0.55, beta=1.0, lambda_init=1.0, seed=2)

    rewards = []
    monitor_data = []
    for t in range(1, args.steps + 1):
        S = agent.act()
        R_vec, R_sum = env.play_subset(S)
        agent.learn(S, R_vec)
        rewards.append(R_sum)
        monitor_data.append({"t": t, "subset": S, "R_vec": R_vec.tolist(), "R_sum": R_sum, "lambda_hat": agent.lambda_hat})
        if R_sum < 5.0:
            print(f"[Low reward] t={t}, R_sum={R_sum:.2f}, subset={S}, R_vec={R_vec}")
        if args.verbose or t in {1, args.steps}:
            print(f"t={t:4d}  S={S}  R_sum={R_sum:6.2f}  lambda_hat={agent.lambda_hat:6.3f}")
        if t % 200 == 0 or t in {1, args.steps}:
            mu_err = np.abs(agent.mu_hat - mu_true)
            x_err = np.abs(agent.x_hat - x_true)
            mu_corr = np.corrcoef(agent.mu_hat, mu_true)[0, 1]
            x_corr = np.corrcoef(agent.x_hat, x_true)[0, 1]
            print(f"\n[Monitor @ t={t}]")
            print(f"  Mean abs error in mu_hat: {mu_err.mean():.4f}")
            print(f"  Mean abs error in x_hat:  {x_err.mean():.4f}")
            print(f"  Corr(mu_hat, mu_true):    {mu_corr:.4f}")
            print(f"  Corr(x_hat, x_true):      {x_corr:.4f}")
            print(f"  lambda_hat:               {agent.lambda_hat:.4f}")

    monitor_data.sort(key=lambda x: x["R_sum"])
    print("\n--- Lowest reward episodes ---")
    for entry in monitor_data[:5]:
        print(f"t={entry['t']}, R_sum={entry['R_sum']:.2f}, subset={entry['subset']}, R_vec={entry['R_vec']}")

    def moving_average(data, window_size=10):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.steps + 1), rewards, marker="o", linestyle='-', label="Raw rewards", alpha=0.4)
    plt.plot(range(10, args.steps + 1), moving_average(rewards, 10), color="red", label="Smoothed (MA window=10)")
    plt.xlabel("Time step")
    plt.ylabel("Reward sum")
    plt.title("Reward sum over time (with smoothing)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()