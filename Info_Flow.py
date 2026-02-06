import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import entropy as scipy_entropy
from dataclasses import dataclass
import warnings

# ===================== Production Setup =====================
warnings.filterwarnings('ignore')
plt.style.use('dark_background')

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
    random_seed: int = 1

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
        prog = t / self.config.n_steps
        trans = 1 / (1 + np.exp(-12 * (prog - 0.5)))
        alpha = self.config.alpha_init * (1 - trans) + self.config.alpha_final * trans
        beta = self.config.beta_init * (1 - trans) + self.config.beta_final * trans

        h_val = scipy_entropy(self.state, base=2)
        kl_val = np.sum(self.state * np.log2(np.maximum(self.state, 1e-12) / self.target_dist))

        ln2_inv = 1 / np.log(2)
        grad = beta * (np.log2(self.state / self.target_dist) + ln2_inv) \
               - alpha * (-np.log2(np.maximum(self.state, 1e-12)) - ln2_inv)

        grad -= np.dot(self.state, grad)
        dstate = -self.config.dt * (self.state * grad)
        noise = np.sqrt(2 * self.config.temperature * self.config.dt) * np.sqrt(self.state) * self.rng.standard_normal(self.config.n_states)
        self.state = np.maximum(self.state + dstate + noise, 1e-12)
        self.state /= self.state.sum()

        return self.state.copy(), h_val, kl_val, alpha, beta

    def animate(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
        
        # Consistent Color Palette
        c_h = '#00d4ff'   # Cyan for Entropy
        c_kl = '#ff0055'  # Red/Pink for KL
        c_flow = '#00ff88' # Green for the Path

        # --- 1. State Simplex (L) ---
        bars = ax1.bar(range(self.config.n_states), self.state, color=c_h, alpha=0.5, label='Current State')
        ax1.step(range(self.config.n_states), self.target_dist, where='mid', color=c_kl, ls='--', lw=2, label='Target (KL Target)')
        ax1.set_ylim(0, 0.7)
        ax1.set_title("I. Probability Simplex", fontsize=12, pad=15)
        ax1.legend(loc='upper right', frameon=False)

        # --- 2. Information Flow Topology (M) ---
        h_hist, kl_hist = [], []
        phase_trail, = ax2.plot([], [], color=c_flow, lw=1.5, alpha=0.6)
        phase_head = ax2.scatter([], [], color='white', s=60, edgecolors=c_flow, zorder=5)
        ax2.set_title("II. Topology: H vs KL", fontsize=12, pad=15)
        ax2.set_xlabel("KL Divergence (Accuracy)", color=c_kl)
        ax2.set_ylabel("Entropy (Diversity)", color=c_h)
        ax2.grid(True, alpha=0.1)

        # --- 3. Metrics Timeline (R) ---
        times, h_vals, kl_vals = [], [], []
        line_h, = ax3.plot([], [], color=c_h, lw=2, label='Entropy (H)')
        line_kl, = ax3.plot([], [], color=c_kl, lw=2, label='KL Divergence')
        ax3.set_xlim(0, self.config.n_steps)
        ax3.set_title("III. Metrics Timeline", fontsize=12, pad=15)
        ax3.set_xlabel("Steps")
        ax3.legend(loc='upper right', frameon=False)

        def update(frame):
            state, h, kl, alpha, beta = self.step(frame)

            # Update Bars
            for bar, val in zip(bars, state): bar.set_height(val)

            # Update Topology
            h_hist.append(h)
            kl_hist.append(kl)
            phase_trail.set_data(kl_hist, h_hist)
            phase_head.set_offsets([[kl, h]])
            
            # Dynamic Rectangle Scaling for Topology
            ax2.set_xlim(min(kl_hist)*0.95, max(kl_hist)*1.05)
            ax2.set_ylim(min(h_hist)*0.95, max(h_hist)*1.05)

            # Update Timeline
            times.append(frame)
            h_vals.append(h)
            kl_vals.append(kl)
            line_h.set_data(times, h_vals)
            line_kl.set_data(times, kl_vals)
            ax3.set_ylim(0, max(max(h_vals), max(kl_vals)) * 1.1)

            fig.suptitle(
                f"InfoFlow Dynamics | Step {frame} | α (Exploration): {alpha:.2f} | β (Exploitation): {beta:.1f}",
                color='white', fontsize=14, y=0.98
            )
            return list(bars) + [phase_trail, phase_head, line_h, line_kl]

        ani = FuncAnimation(fig, update, frames=range(self.config.n_steps), interval=10, blit=False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.show()

# ===================== Run =====================
if __name__ == "__main__":
    sim = InfoFlowSimulator(InfoFlowConfig())
    sim.animate()
