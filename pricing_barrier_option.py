# Hamza M. \
# pricing_barrier_option\
# Monte Carlo simulation to price a European Down-and-Out Call Option\
\
\
\
import numpy as np\
import matplotlib.pyplot as plt\
\
\
\
\
# === Parameters ===\
S0 = 100        # Initial stock price\
K = 100         # Strike price\
B = 90          # Barrier level (down-and-out)\
T = 1.0         # Time to maturity (1 year)\
r = 0.05        # Risk-free rate\
sigma = 0.2     # Volatility\
n_paths = 10000\
n_steps = 252   # Daily steps\
\
dt = T / n_steps\
discount_factor = np.exp(-r * T)\
\
np.random.seed(42)\
payoffs = []\
\
# === Simulation ===\
for _ in range(n_paths):\
    S = S0\
    knocked_out = False\
    for _ in range(n_steps):\
        z = np.random.normal()\
        S *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)\
        if S <= B:\
            knocked_out = True\
            break\
    payoff = max(S - K, 0) if not knocked_out else 0\
    payoffs.append(payoff)\
\
price = discount_factor * np.mean(payoffs)\
\
# === Plot sample paths ===\
n_sample = 20\
sample_paths = np.zeros((n_sample, n_steps + 1))\
for i in range(n_sample):\
    S = [S0]\
    for _ in range(n_steps):\
        z = np.random.normal()\
        S.append(S[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z))\
    sample_paths[i] = S\
\
plt.figure(figsize=(10, 5))\
for i in range(n_sample):\
    plt.plot(sample_paths[i], alpha=0.6)\
plt.axhline(B, color='red', linestyle='--', label='Barrier')\
plt.title('Sample Monte Carlo Paths (Down-and-Out Barrier Option)')\
plt.xlabel('Steps')\
plt.ylabel('Price')\
plt.legend()\
plt.grid(True)\
plt.tight_layout()\
plt.show()\
\
# === Output the price ===\
print(f"Estimated price of the down-and-out call option: \{price:.2f\}")\
}
