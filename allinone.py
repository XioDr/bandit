# -*- coding: utf-8 -*-
r"""
Spatial Combinatorial MAB — fast implementation with **UCB + ε‑greedy**.
----------------------------------------------------------------------------
* Vectorised reward computation                                      (fast)
* Subset cache or Monte‑Carlo fallback                               (scales)
* Upper‑Confidence‑Bound exploration                                 (smart)
* Extra decaying‑ε factor (γ / t)                                    (option)

Set ``gamma = 0``  → pure‑UCB  
Set ``gamma > 0``  → ε‑greedy‑with‑UCB hybrid
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence, Tuple, Optional, Dict

import numpy as np
import numpy.typing as npt

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def all_subsets_of_size(n: int, k: int) -> Iterable[Tuple[int, ...]]:
    """Yield every unordered k‑subset of ``range(n)``."""
    if not (0 <= k <= n):
        raise ValueError("Need 0 ≤ k ≤ n.")
    return combinations(range(n), k)


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
class SpatialMABEnvironment:
    """1‑D spatial combinatorial MAB simulator."""

    def __init__(
        self,
        *,
        n: int,
        k: int,
        x_true: npt.NDArray[np.floating],
        mu_true: npt.NDArray[np.floating],
        sigma_true: npt.NDArray[np.floating],
        lambda_true: float = 1.0,
        beta_true: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if (
            x_true.shape != (n,)
            or mu_true.shape != (n,)
            or sigma_true.shape != (n,)
        ):
            raise ValueError("x_true, mu_true and sigma_true must be shape (n,).")

        self.n, self.k = int(n), int(k)
        self.x_true = x_true.astype(float)
        self.mu_true = mu_true.astype(float)
        self.sigma_true = sigma_true.astype(float)
        self.lambda_true = float(lambda_true)
        self.beta_true = float(beta_true)

        self.rng = np.random.default_rng(seed)

    # --------------------------------------------------------------------- #
    def play_subset(self, subset: Sequence[int]) -> Tuple[np.ndarray, float]:
        subset = np.asarray(subset, dtype=int)
        if subset.size != self.k or np.unique(subset).size != self.k:
            raise ValueError(f"Provide exactly {self.k} distinct indices.")

        x_bar = self.x_true[subset].mean()
        base = self.rng.normal(self.mu_true[subset], self.sigma_true[subset])
        penalties = self.lambda_true * np.abs(self.x_true[subset] - x_bar) ** self.beta_true
        rewards = base - penalties
        return rewards, float(rewards.sum())


# --------------------------------------------------------------------------- #
# agent‑side helpers
# --------------------------------------------------------------------------- #
def init_agent_estimates(
    *,
    agent_type: str,
    true_mu: npt.NDArray[np.floating],
    true_x: npt.NDArray[np.floating],
    params: dict | None = None,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct noisy priors (μ̂, x̂) for common agent archetypes."""
    rng = np.random.default_rng() if rng is None else rng
    params = params or {}
    n = true_mu.size

    if agent_type == "naive":
        return np.zeros(n), np.zeros(n)

    if agent_type == "single_noisy":
        sigma = params.get("sigma", 0.3)
        return rng.normal(true_mu, sigma), rng.normal(true_x, sigma)

    if agent_type == "sep_noisy":
        return (
            rng.normal(true_mu, params.get("sigma_mu", 0.2)),
            rng.normal(true_x, params.get("sigma_x", 0.4)),
        )

    if agent_type == "dist_based":
        agent_loc = params.get("agent_loc", 0.0)
        dist = np.abs(true_x - agent_loc)
        sigmas = params.get("noise_min", 0.1) + params.get("noise_slope", 0.2) * dist
        return rng.normal(true_mu, sigmas), rng.normal(true_x, sigmas)

    raise ValueError(f"Unknown agent_type={agent_type!r}.")


# --------------------------------------------------------------------------- #
# subset‑selection strategy:  UCB  +  ε‑greedy
# --------------------------------------------------------------------------- #
@dataclass
class UCBSubsetStrategy:
    """
    Upper‑Confidence‑Bound policy on the *subset* space with optional ε‑greedy
    exploration.

    Parameters
    ----------
    ucb_c : float
        Exploration constant used in the UCB bonus term
        ``c * sqrt(log t / N_t(S))``.
    gamma : float
        Coefficient for the decaying ε schedule ``ε_t = min(gamma / t, 1)``.
        *Set to 0 for pure UCB*.
    """

    est_mu: npt.NDArray[np.floating]
    est_x: npt.NDArray[np.floating]
    est_lambda: float
    k: int
    ucb_c: float = 1.0
    gamma: float = 0.0
    beta: float = 1.0
    rng: np.random.Generator | None = None
    max_cached_subsets: int = 100_000
    sample_budget: int = 5_000

    # internal
    subsets_array: Optional[np.ndarray] = None
    subset_means: Optional[np.ndarray] = None
    subset_counts: Optional[np.ndarray] = None
    _subset_to_idx: Optional[Dict[Tuple[int, ...], int]] = None
    n: int = 0
    t: int = 0

    # --------------------------------------------------------------------- #
    def __post_init__(self) -> None:
        self.n = self.est_mu.size
        if self.est_x.shape != (self.n,):
            raise ValueError("est_mu and est_x must share shape.")

        self.rng = np.random.default_rng() if self.rng is None else self.rng

        num_subsets = math.comb(self.n, self.k)
        if num_subsets <= self.max_cached_subsets:
            # Cache every k‑subset of range(n) in an int32 array (num_subsets, k)
            self.subsets_array = np.fromiter(
                (i for S in all_subsets_of_size(self.n, self.k) for i in S),
                dtype=np.int32,
                count=num_subsets * self.k,
            ).reshape(num_subsets, self.k)
            self.subset_means = np.zeros(num_subsets, dtype=float)
            self.subset_counts = np.zeros(num_subsets, dtype=int)
            # build dictionary for O(1) lookup when updating stats later
            self._subset_to_idx = {tuple(S.tolist()): idx
                                   for idx, S in enumerate(self.subsets_array)}

    # --------------------------------------------------------------------- #
    # vectorised reward prediction
    def _predicted_rewards_vec(self, subsets: np.ndarray) -> np.ndarray:
        mu_vals = self.est_mu[subsets]                       # (S, k)
        x_vals = self.est_x[subsets]                         # (S, k)
        x_bar = x_vals.mean(axis=1, keepdims=True)           # (S, 1)
        penalties = self.est_lambda * np.abs(x_vals - x_bar) ** self.beta
        return (mu_vals - penalties).sum(axis=1)             # (S,)

    # --------------------------------------------------------------------- #
    def _select_ucb_idx_cached(self) -> int:
        """Return index of subset that maximises UCB (cache mode)."""
        counts = self.subset_counts
        means = self.subset_means
        with np.errstate(divide="ignore"):
            bonus = self.ucb_c * np.sqrt(np.log(self.t + 1) / np.maximum(counts, 1))
        ucb = means + bonus
        return int(ucb.argmax())

    # --------------------------------------------------------------------- #
    def choose_subset(self) -> Tuple[int, ...]:
        """Return a k‑subset according to ε‑greedy‑with‑UCB policy."""
        self.t += 1

        # ---------------------------------------------------------------- #
        # 1. ε‑greedy exploration (uniform random subset)
        eps_t = min(self.gamma / self.t, 1.0)
        if self.rng.random() < eps_t:
            return tuple(self.rng.choice(self.n, self.k, replace=False))

        # ---------------------------------------------------------------- #
        # 2. UCB exploitation / directed exploration
        if self.subsets_array is not None:
            idx = self._select_ucb_idx_cached()
            return tuple(int(i) for i in self.subsets_array[idx])

        # Too many subsets → Monte‑Carlo UCB on a sampled batch
        sampled = self.rng.choice(self.n, (self.sample_budget, self.k), replace=False)
        rewards = self._predicted_rewards_vec(sampled)
        bonus = self.ucb_c * np.sqrt(np.log(self.t + 1))     # treat N=0
        ucb = rewards + bonus
        best_idx = int(ucb.argmax())
        return tuple(int(i) for i in sampled[best_idx])

    # --------------------------------------------------------------------- #
    def update_statistics(self, subset: Sequence[int], R_sum: float) -> None:
        """Update empirical mean / count for the chosen subset (cache mode)."""
        if self.subsets_array is None:
            return  # MC mode — we don't keep per‑subset stats

        key = tuple(sorted(map(int, subset)))
        idx = self._subset_to_idx[key]                       # type: ignore[index]
        cnt = self.subset_counts[idx] + 1
        self.subset_counts[idx] = cnt
        # incremental mean
        self.subset_means[idx] += (R_sum - self.subset_means[idx]) / cnt


# --------------------------------------------------------------------------- #
# SGD parameter update  (unchanged)
# --------------------------------------------------------------------------- #
def backprop_update(
    subset: Sequence[int],
    R_obs: np.ndarray,
    mu_hat: np.ndarray,
    x_hat: np.ndarray,
    lambda_hat: float,
    *,
    beta: float = 1.0,
    lr_mu: float = 0.05,
    lr_x: float = 0.05,
    lr_lambda: float = 0.05,
) -> float:
    subset = np.asarray(subset, dtype=int)
    mu_s, x_s = mu_hat[subset], x_hat[subset]
    k = subset.size

    x_bar = x_s.mean()
    delta = np.abs(x_s - x_bar) ** beta
    R_pred = mu_s - lambda_hat * delta
    err = R_pred - R_obs

    grad_mu = err
    grad_lambda = -np.dot(err, delta)
    sign = np.sign(x_s - x_bar) * beta * lambda_hat * np.abs(x_s - x_bar) ** (beta - 1)
    grad_x = -err * sign + err.sum() * (sign / k).sum()

    mu_hat[subset] -= lr_mu * grad_mu
    x_hat[subset]  -= lr_x * grad_x
    lambda_hat    -= lr_lambda * grad_lambda
    return float(lambda_hat)


# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #
def train(
    env: SpatialMABEnvironment,
    strategy: UCBSubsetStrategy,
    *,
    steps: int = 1_000,
    beta: float | None = None,
    lr_mu: float = 0.05,
    lr_x: float = 0.05,
    lr_lambda: float = 0.05,
    profile: bool = False,
) -> np.ndarray:
    import time

    beta = strategy.beta if beta is None else beta
    rewards = np.empty(steps, dtype=float)

    t0 = time.perf_counter()
    for t in range(steps):
        subset = strategy.choose_subset()
        R_vec, R_sum = env.play_subset(subset)
        rewards[t] = R_sum

        strategy.est_lambda = backprop_update(
            subset,
            R_vec,
            strategy.est_mu,
            strategy.est_x,
            strategy.est_lambda,
            beta=beta,
            lr_mu=lr_mu,
            lr_x=lr_x,
            lr_lambda=lr_lambda,
        )

        strategy.update_statistics(subset, R_sum)

    if profile:
        dt = time.perf_counter() - t0
        print(f"[train] {steps} rounds in {dt:.3f}s — {steps/dt:.1f} rounds/s")

    return rewards


def plot_rewards(rewards: np.ndarray, *, window: int = 50) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(rewards, label="instantaneous", alpha=0.3)
    if window > 1 and rewards.size >= window:
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, rewards.size), ma, label=f"{window}-step MA")
    ax.set_xlabel("Round")
    ax.set_ylabel("Total reward")
    ax.set_title("Training performance")
    ax.legend()
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# sanity test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, k = 20, 5
    x_true = np.sort(rng.uniform(0.0, 1.0, n))
    mu_true = rng.uniform(0.0, 1.0, n)
    sigma_true = np.full(n, 0.1)

    env = SpatialMABEnvironment(
        n=n,
        k=k,
        x_true=x_true,
        mu_true=mu_true,
        sigma_true=sigma_true,
        lambda_true=1.0,
        beta_true=1.0,
        seed=42,
    )

    est_mu, est_x = init_agent_estimates(
        agent_type="sep_noisy",
        true_mu=mu_true,
        true_x=x_true,
        params={"sigma_mu": 0.2, "sigma_x": 0.4},
        rng=rng,
    )

    strategy = UCBSubsetStrategy(
        est_mu=est_mu,
        est_x=est_x,
        est_lambda=1.0,
        k=k,
        ucb_c=1.0,      # UCB exploration constant
        gamma=0.0,      # ε‑greedy coefficient  ← you asked for gamma = 0.1
        rng=rng,
        max_cached_subsets=200_000,
        sample_budget=5_000,
    )

    rewards = train(env, strategy, steps=100_000, profile=True)
    plot_rewards(rewards, window=50)
