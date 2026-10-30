"""
Pricing of a European down-and-out call option.

- Monte Carlo simulation (vectorised with NumPy), with a 95% confidence interval
- Validation against the closed-form price (continuous barrier monitoring)
- Broadie-Glasserman-Kou correction to approximate discrete (daily) monitoring
- Sensitivity of the price to volatility and to the barrier level

Author: Hamza Mennani
"""

import argparse
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

BGK_BETA = 0.5826  # -zeta(1/2) / sqrt(2*pi), Broadie-Glasserman-Kou constant


@dataclass(frozen=True)
class OptionParams:
    S0: float = 100.0     # spot price
    K: float = 100.0      # strike
    B: float = 90.0       # down-and-out barrier (B < S0)
    T: float = 1.0        # maturity in years
    r: float = 0.05       # risk-free rate (continuous)
    sigma: float = 0.20   # volatility
    n_steps: int = 252    # monitoring dates (daily)


# ---------------------------------------------------------------------------
# Closed-form prices
# ---------------------------------------------------------------------------
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def down_and_out_call_continuous(S, K, B, T, r, sigma):
    """Closed-form down-and-out call, continuous monitoring, case B <= K, no dividend."""
    if S <= B:
        return 0.0
    if B > K:
        raise NotImplementedError("Closed form implemented for B <= K only.")
    lam = (r + 0.5 * sigma**2) / sigma**2
    y = np.log(B**2 / (S * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
    down_and_in = (S * (B / S) ** (2 * lam) * norm.cdf(y)
                   - K * np.exp(-r * T) * (B / S) ** (2 * lam - 2) * norm.cdf(y - sigma * np.sqrt(T)))
    return black_scholes_call(S, K, T, r, sigma) - down_and_in


def down_and_out_call_discrete_bgk(p: OptionParams):
    """Approximation of discrete monitoring: shift the barrier down by exp(-beta * sigma * sqrt(dt))."""
    dt = p.T / p.n_steps
    B_adj = p.B * np.exp(-BGK_BETA * p.sigma * np.sqrt(dt))
    return down_and_out_call_continuous(p.S0, p.K, B_adj, p.T, p.r, p.sigma)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def simulate_paths(p: OptionParams, n_paths: int, rng: np.random.Generator):
    """Geometric Brownian motion paths, shape (n_paths, n_steps + 1)."""
    dt = p.T / p.n_steps
    z = rng.standard_normal((n_paths, p.n_steps))
    log_increments = (p.r - 0.5 * p.sigma**2) * dt + p.sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = p.S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    return paths


def monte_carlo_down_and_out_call(p: OptionParams, n_paths: int = 100_000, seed: int = 42,
                                  batch_size: int = 20_000):
    """Monte Carlo price with daily barrier monitoring. Returns (price, standard error)."""
    rng = np.random.default_rng(seed)
    discounted_payoffs = []
    remaining = n_paths
    while remaining > 0:
        n = min(batch_size, remaining)
        paths = simulate_paths(p, n, rng)
        alive = paths.min(axis=1) > p.B
        payoff = np.where(alive, np.maximum(paths[:, -1] - p.K, 0.0), 0.0)
        discounted_payoffs.append(np.exp(-p.r * p.T) * payoff)
        remaining -= n
    x = np.concatenate(discounted_payoffs)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_sample_paths(p: OptionParams, n_sample: int = 20, seed: int = 7):
    paths = simulate_paths(p, n_sample, np.random.default_rng(seed))
    knocked = paths.min(axis=1) <= p.B
    fig, ax = plt.subplots(figsize=(10, 5))
    for path, ko in zip(paths, knocked):
        ax.plot(path, color="grey" if ko else "tab:blue", alpha=0.6, lw=1)
    ax.axhline(p.B, color="red", ls="--", label=f"Barrier B = {p.B}")
    ax.axhline(p.K, color="black", ls=":", label=f"Strike K = {p.K}")
    ax.set_title("Sample paths (grey = knocked out)")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Underlying price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sensitivities(p: OptionParams):
    vols = np.linspace(0.05, 0.50, 46)
    barriers = np.linspace(60, 99, 40)
    price_vs_vol = [down_and_out_call_discrete_bgk(replace(p, sigma=v)) for v in vols]
    vanilla_vs_vol = [black_scholes_call(p.S0, p.K, p.T, p.r, v) for v in vols]
    price_vs_barrier = [down_and_out_call_discrete_bgk(replace(p, B=b)) for b in barriers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(vols * 100, price_vs_vol, label="Down-and-out call")
    ax1.plot(vols * 100, vanilla_vs_vol, ls="--", label="Vanilla call")
    ax1.set_title("Price vs volatility")
    ax1.set_xlabel("Volatility (%)")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(barriers, price_vs_barrier)
    ax2.set_title("Price vs barrier level")
    ax2.set_xlabel("Barrier B")
    ax2.set_ylabel("Price")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Price a European down-and-out call option.")
    parser.add_argument("--paths", type=int, default=100_000, help="number of Monte Carlo paths")
    parser.add_argument("--show", action="store_true", help="display charts instead of only saving them")
    args = parser.parse_args()

    p = OptionParams()
    mc_price, se = monte_carlo_down_and_out_call(p, n_paths=args.paths)
    continuous = down_and_out_call_continuous(p.S0, p.K, p.B, p.T, p.r, p.sigma)
    discrete = down_and_out_call_discrete_bgk(p)
    vanilla = black_scholes_call(p.S0, p.K, p.T, p.r, p.sigma)

    print("European down-and-out call")
    print(f"  S0={p.S0}, K={p.K}, B={p.B}, T={p.T}, r={p.r}, sigma={p.sigma}, daily monitoring")
    print(f"  Monte Carlo ({args.paths:,} paths) : {mc_price:.4f}  "
          f"[95% CI {mc_price - 1.96 * se:.4f} ; {mc_price + 1.96 * se:.4f}]")
    print(f"  Closed form, discrete (BGK)     : {discrete:.4f}")
    print(f"  Closed form, continuous barrier : {continuous:.4f}")
    print(f"  Vanilla Black-Scholes call      : {vanilla:.4f}")

    plot_sample_paths(p).savefig("sample_paths.png", dpi=120)
    plot_sensitivities(p).savefig("sensitivities.png", dpi=120)
    print("Charts saved: sample_paths.png, sensitivities.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
