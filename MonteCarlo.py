import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#random seed for reproducibility
np.random.seed(2026)

S0 = 100  #price
mu = 0.1 # expected return (drift)
sigma = 0.30 # volatility
T = 2  # random shock (Brownian motion)
steps = 252
n_simulations = 1000
dt = T / steps


Z = np.random.standard_normal((steps, n_simulations))
prices = np.zeros((steps + 1, n_simulations)) #price matrix - a column represents one simulation
prices[0] = S0

for t in range(1, steps + 1):
    prices[t] = prices[t - 1] * np.exp(
        (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t - 1]
    ) # Drift and diffusion term


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, steps)
ax.set_ylim(prices.min() * 0.9, prices.max() * 1.1)

ax.set_title("Monte Carlo Stock Price Simulation")
ax.set_xlabel("Time (Days)")
ax.set_ylabel("Stock Price")


# Create a line object for each simulation path
lines = [
    ax.plot([], [], linewidth=1, alpha=0.6)[0]
    for _ in range(n_simulations)
]

mean_line, = ax.plot([], [], linestyle="--", linewidth=2, label="Mean Path")
ax.legend()

def init():
    for line in lines:
        line.set_data([], [])
    mean_line.set_data([], [])
    return lines + [mean_line]

def update(frame):
    x = np.arange(frame)
    for i, line in enumerate(lines): # Update each simulation path
        line.set_data(x, prices[:frame, i])
    mean_line.set_data(x, prices[:frame].mean(axis=1)) #Update mean path across all simulations
    return lines + [mean_line]


ani = FuncAnimation(
    fig,
    update,
    frames=steps,
    init_func=init,
    interval=40,   # ms between frames
    blit=True
)

plt.show()
