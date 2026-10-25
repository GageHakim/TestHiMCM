"""
HiMCM Part 1 -- two-compartment honeybee colony model.

State variables (number of adult bees):
    H = P_hive  : hive (nurse/house) bees
    F = P_for   : foragers
    P = H + F   : total colony

Model:
    dH/dt = K1 * sqrt(F)  -  K2 * H / F  -  g1 * H
    dF/dt = K2 * H / F  -  K3 * F**(3/2)  -  g2 * F
    dP/dt = dH/dt + dF/dt

Term meanings:
    K1 * sqrt(F)   : births, proportional to food income. Foragers need a foraging area
                     proportional to F, so the radius (and trip length) grows like sqrt(F);
                     each forager brings in food at a rate ~ 1/sqrt(F), so total food
                     ~ F / sqrt(F) = sqrt(F). Each new bee costs a fixed amount of food.
    K2 * H / F     : hive bees recruited to foraging (more recruitment when foragers are scarce)
    K3 * F**(3/2)  : foraging-distance deaths. Each forager's risk grows with trip length
                     ~ sqrt(F), so total deaths ~ F * sqrt(F).
    g1, g2         : background per-capita death rates

Also included: a "revised" model with a queen-limited birth rate and
social-inhibition recruitment (Khoury, Myerscough & Barron 2011, PLoS ONE 6(4):e18491)
for comparison.

Run:  .venv/bin/python bee_model.py
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

EPS = 1e-9  # keeps H/F and sqrt finite when a compartment hits zero


# --------------------------------------------------------------------------
# Main model
# --------------------------------------------------------------------------
@dataclass
class Params:
    K1: float = 18.0       # births per sqrt(forager)      [bees^(1/2) / day]
    K2: float = 800.0      # recruitment coefficient       [bees / day]
    K3: float = 1e-4       # distance-death coefficient    [bees^(-1/2) / day]
    g1: float = 0.01       # hive-bee death rate           [1/day]
    g2: float = 0.15       # forager death rate            [1/day]


def calibrate(H_star, F_star, g1=0.01, g2=0.15, K3=1e-4):
    """Pick K1, K2 so that (H_star, F_star) is an equilibrium of the model."""
    K2 = F_star / H_star * (K3 * F_star**1.5 + g2 * F_star)
    K1 = (K2 * H_star / F_star + g1 * H_star) / np.sqrt(F_star)
    return Params(K1=K1, K2=K2, K3=K3, g1=g1, g2=g2)


def rhs(t, y, p: Params):
    H, F = np.maximum(y, 0.0)
    births = p.K1 * np.sqrt(F)
    recruit = p.K2 * H / (F + EPS)
    dH = births - recruit - p.g1 * H
    dF = recruit - p.K3 * F**1.5 - p.g2 * F
    return [dH, dF]


def jacobian(H, F, p: Params, h=1e-3):
    """Numerical Jacobian of rhs at (H, F) -- used for equilibrium stability."""
    J = np.zeros((2, 2))
    y0 = np.array([H, F], dtype=float)
    for j in range(2):
        dy = np.zeros(2)
        dy[j] = h * max(1.0, abs(y0[j]))
        J[:, j] = (np.array(rhs(0, y0 + dy, p)) - np.array(rhs(0, y0 - dy, p))) / (2 * dy[j])
    return J


# --------------------------------------------------------------------------
# Revised model (Khoury et al. 2011 style) for comparison
# --------------------------------------------------------------------------
@dataclass
class RevisedParams:
    L: float = 2000.0      # max eggs/day the queen can lay
    w: float = 27000.0     # colony size at which births are half of L
    alpha: float = 0.25    # max recruitment rate when there are no foragers [1/day]
    sigma: float = 0.75    # strength of social inhibition by foragers
    g1: float = 0.0        # hive-bee death rate [1/day]
    g2: float = 0.24       # forager death rate  [1/day]


def rhs_revised(t, y, p: RevisedParams):
    H, F = np.maximum(y, 0.0)
    N = H + F + EPS
    births = p.L * N / (p.w + N)                            # saturates at L: no blow-up
    recruit = H * max(p.alpha - p.sigma * F / N, 0.0)       # finite when F = 0
    dH = births - recruit - p.g1 * H
    dF = recruit - p.g2 * F
    return [dH, dF]


# --------------------------------------------------------------------------
# Simulation helpers
# --------------------------------------------------------------------------
def blowup(t, y, *args):
    return 1e6 - (y[0] + y[1])        # stop if colony exceeds 1,000,000 bees
blowup.terminal = True


def simulate(f, y0, params, t_end=365.0):
    return solve_ivp(f, (0, t_end), y0, args=(params,), method="LSODA",
                     events=blowup, max_step=0.5, rtol=1e-8, atol=1e-6)


def main():
    # Calibrate so a 30,000-bee colony (20k hive / 10k foragers) is an equilibrium.
    H_star, F_star = 20000.0, 10000.0
    p = calibrate(H_star, F_star)
    print("Model parameters:", p)

    eig = np.linalg.eigvals(jacobian(H_star, F_star, p))
    print(f"Equilibrium (H, F) = ({H_star:.0f}, {F_star:.0f}); Jacobian eigenvalues = {eig}")
    print("  -> stable" if np.all(eig.real < 0) else "  -> UNSTABLE (saddle/source)")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    for scale in [0.01, 0.5, 1.0, 2.0]:
        sol = simulate(rhs, [scale * H_star, scale * F_star], p)
        H, F = sol.y
        end = f"blows up at t={sol.t[-1]:.0f} d" if sol.status == 1 else f"t={sol.t[-1]:.0f} d"
        print(f"  start at {scale:.0%} of equilibrium: final H = {max(H[-1], 0):,.0f}, "
              f"F = {max(F[-1], 0):,.0f} ({end})")
        line, = axes[0].plot(sol.t, H + F, label=f"start {scale:.0%} of eq.")
        axes[1].plot(sol.t, H, color=line.get_color())
        axes[1].plot(sol.t, F, "--", color=line.get_color())
    axes[0].set_title("Model: total colony P(t)")
    axes[0].set_xlabel("day"); axes[0].set_ylabel("bees"); axes[0].legend(fontsize=8)
    axes[1].set_title("Model: hive bees H (solid) and foragers F (dashed)")
    axes[1].set_xlabel("day"); axes[1].set_ylabel("bees")

    ax = axes[2]
    rp = RevisedParams()
    for y0 in [[5000, 1000], [16000, 8000], [40000, 20000]]:
        sol = simulate(rhs_revised, y0, rp)
        ax.plot(sol.t, sol.y[0] + sol.y[1], label=f"start H={y0[0]}, F={y0[1]}")
    ax.set_title("Revised model (queen-limited births): P(t)")
    ax.set_xlabel("day"); ax.set_ylabel("bees"); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("bee_model.png", dpi=150)
    print("Saved bee_model.png")


if __name__ == "__main__":
    main()
