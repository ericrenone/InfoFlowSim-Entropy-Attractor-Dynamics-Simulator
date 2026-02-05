import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import entropy as scipy_entropy
from dataclasses import dataclass
import warnings

# ===================== Production Setup =====================
warnings.filterwarnings('ignore')
plt.style.use('dark_background')

# ===================== Configuration =====================
@dataclass
class InfoFlowConfig:
    n_states: int = 15
    n_steps: int = 1000
    dt: float = 0.12
    alpha_init: float = 3.5
    alpha_final: float = 0.05
    beta_init: float = 0.1
    beta_final: float = 18.0
    temperature: float = 0.008
    random_seed: int = 42

# ===================== Simulator =====================
class InfoFlowSimulator:
    def __init__(self, config: InfoFlowConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.target_dist = self._create_attractor()
        self.state = np.ones(config.n_states) / config.n_states

    def _create_attractor(self) -> np.ndarray:
        x = np.arange(self.config.n_states)
        dist = np.exp(-x / 2.5)
        return dist / dist.sum()

    def step(self, t: int):
        # α and β schedule (sigmoid transition)
        prog = t / self.config.n_steps
        trans = 1 / (1 + np.exp(-12 * (prog - 0.5)))
        alpha = self.config.alpha_init * (1 - trans) + self.config.alpha_final * trans
        beta = self.config.beta_init * (1 - trans) + self.config.beta_final * trans

        # Metrics
        entropy_val = scipy_entropy(self.state, base=2)
        kl_div = np.sum(self.state * np.log2(np.maximum(self.state, 1e-12) / self.target_dist))

        # Gradient (canonical constants)
        ln2_inv = 1 / np.log(2)
        grad = beta * (np.log2(self.state / self.target_dist) + ln2_inv) \
               - alpha * (-np.log2(np.maximum(self.state, 1e-12)) - ln2_inv)

        # Project gradient onto simplex
        grad -= np.dot(self.state, grad)

        # Euler-Maruyama update with multiplicative noise
        dstate = -self.config.dt * (self.state * grad)
        noise = np.sqrt(2 * self.config.temperature * self.config.dt) * np.sqrt(self.state) * self.rng.standard_normal(self.config.n_states)
        self.state = np.maximum(self.state + dstate + noise, 1e-12)
        self.state /= self.state.sum()

        return self.state.copy(), entropy_val, kl_div, alpha, beta

    def animate(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # --- State Simplex ---
        bars = ax1.bar(range(self.config.n_states), self.state, color='#00d4ff', alpha=0.7)
        ax1.step(range(self.config.n_states), self.target_dist, where='mid', color='#ff0055', ls='--', label='Target')
        ax1.set_ylim(0, 0.7)
        ax1.set_title("State Simplex")
        ax1.legend(frameon=False)

        # --- Information Flow Topology ---
        entropy_hist, kl_hist = [], []
        phase_trail, = ax2.plot([], [], color='#00ff88', lw=1.5, alpha=0.9)
        phase_head = ax2.scatter([], [], color='white', s=60, edgecolors='#00ff88', zorder=5)
        ax2.set_xlim(0, 4); ax2.set_ylim(0, 4)
        ax2.set_title("Information Flow Topology")
        ax2.set_xlabel("Entropy (H)"); ax2.set_ylabel("KL Divergence")

        # --- Metrics Timeline ---
        times, h_vals, kl_vals = [], [], []
        line_h, = ax3.plot([], [], color='#00d4ff', label='Entropy (H)')
        line_kl, = ax3.plot([], [], color='#ff0055', label='KL Divergence')
        ax3.set_xlim(0, self.config.n_steps)
        ax3.set_ylim(0, max(4, 1.2*np.max(self.target_dist)))  # dynamic upper bound
        ax3.set_title("Metrics Timeline")
        ax3.legend(loc='upper right', frameon=False)

        # --- Update function ---
        def update(frame):
            state, entropy_val, kl_div, alpha, beta = self.step(frame)

            # Update bars
            for bar, val in zip(bars, state):
                bar.set_height(val)

            # Update phase trajectory
            entropy_hist.append(entropy_val)
            kl_hist.append(kl_div)
            phase_trail.set_data(entropy_hist, kl_hist)
            phase_head.set_offsets([[entropy_val, kl_div]])

            # Update metrics timeline
            times.append(frame)
            h_vals.append(entropy_val)
            kl_vals.append(kl_div)
            line_h.set_data(times, h_vals)
            line_kl.set_data(times, kl_vals)

            fig.suptitle(
                f"InfoFlowSim | Step {frame+1}/{self.config.n_steps} | "
                f"α: {alpha:.2f}, β: {beta:.1f} | H: {entropy_val:.2f}, KL: {kl_div:.2f}",
                color='white', fontsize=14, fontweight='bold'
            )
            return list(bars) + [phase_trail, phase_head, line_h, line_kl]

        ani = FuncAnimation(fig, update, frames=range(self.config.n_steps), interval=20, blit=False, repeat=False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


# ===================== Run Example =====================
if __name__ == "__main__":
    config = InfoFlowConfig()
    sim = InfoFlowSimulator(config)
    sim.animate()
