{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww29200\viewh16140\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Hamza M. \
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