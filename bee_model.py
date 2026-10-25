"""
HiMCM Part 1 -- two-compartment honeybee colony model.

State variables (number of adult bees):
    H = P_hive  : hive (nurse/house) bees
    F = P_for   : foragers
    P = H + F   : total colony

Model as written on the whiteboard:
    dH/dt = K1 * (F * H)**b / sqrt(F + H)  -  K2 * H / F  -  g1 * H   (b = 1 as written)
    dF/dt = K2 * H / F  -  K3 * F**a  -  g2 * F          (a = 1/2 as written)
    dP/dt = dH/dt + dF/dt

Term meanings (our reading):
    K1 * (F*H)**b/sqrt(F+H) : new adults emerging (needs foragers for food, hive bees for brood care)
    K2 * H/F           : hive bees recruited to foraging (more recruitment when foragers are scarce)
    K3 * F**a          : extra forager loss (e.g. lost while foraging / predation / pesticide)
    g1, g2             : background per-capita death rates

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
# Whiteboard model
# --------------------------------------------------------------------------
@dataclass
class Params:
    K1: float = 1.559e-3   # birth coefficient            [bees^(-1/2) / day]
    K2: float = 800.0      # recruitment coefficient      [bees / day]
    K3: float = 1.0        # extra forager loss coeff.    [bees^(1-a) / day]
    a: float = 0.5         # exponent on F in the K3 term (whiteboard: 1/2)
    b: float = 1.0         # exponent on F*H in births (whiteboard: 1; sqrt variant: 1/2)
    g1: float = 0.01       # hive-bee death rate          [1/day]
    g2: float = 0.15       # forager death rate           [1/day]


def calibrate(H_star, F_star, g1=0.01, g2=0.15, K3=1.0, a=0.5, b=1.0):
    """Pick K1, K2 so that (H_star, F_star) is an equilibrium of the whiteboard model."""
    K2 = F_star / H_star * (K3 * F_star**a + g2 * F_star)
    K1 = (K2 * H_star / F_star + g1 * H_star) * np.sqrt(H_star + F_star) / (F_star * H_star)**b
    return Params(K1=K1, K2=K2, K3=K3, a=a, b=b, g1=g1, g2=g2)


def rhs(t, y, p: Params):
    H, F = np.maximum(y, 0.0)
    births = p.K1 * (F * H)**p.b / np.sqrt(F + H + EPS)
    recruit = p.K2 * H / (F + EPS)
    dH = births - recruit - p.g1 * H
    dF = recruit - p.K3 * F**p.a - p.g2 * F
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
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    variants = [(1.0, "Whiteboard model (F*H)"), (0.5, "Sqrt variant (sqrt(F*H))")]
    for ax, (b, name) in zip(axes, variants):
        p = calibrate(H_star, F_star, b=b)
        print(f"{name} parameters:", p)

        eig = np.linalg.eigvals(jacobian(H_star, F_star, p))
        print(f"Equilibrium (H, F) = ({H_star:.0f}, {F_star:.0f}); Jacobian eigenvalues = {eig}")
        print("  -> stable" if np.all(eig.real < 0) else "  -> UNSTABLE (saddle/source)")

        for scale, ls in [(0.01, "-."), (0.9, "--"), (1.0, "-"), (1.1, ":"), (2.0, "-")]:
            sol = simulate(rhs, [scale * H_star, scale * F_star], p)
            H, F = sol.y
            end = f"blows up at t={sol.t[-1]:.0f} d" if sol.status == 1 else f"t={sol.t[-1]:.0f} d"
            print(f"  start at {scale:.0%} of equilibrium: final N = {max(H[-1] + F[-1], 0):,.0f} ({end})")
            ax.plot(sol.t, H + F, ls, label=f"start {scale:.0%} of eq. ({end})")
        ax.set_title(f"{name}: total colony P(t)")
        ax.set_xlabel("day"); ax.set_ylabel("bees"); ax.set_yscale("symlog", linthresh=100)
        ax.legend(fontsize=8)

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
