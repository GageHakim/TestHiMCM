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

Parameters are rough literature estimates (see Params) except K3, which has no good
data and is treated as a free parameter. The equilibrium colony size is an OUTPUT.

Also included: a "revised" model with a queen-limited birth rate and
social-inhibition recruitment (Khoury, Myerscough & Barron 2011, PLoS ONE 6(4):e18491)
for comparison.

Run:  .venv/bin/python bee_model.py
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

EPS = 1e-9  # keeps H/F and sqrt finite when a compartment hits zero

# K1 and K2 multiply terms that are not scale-free (sqrt(F) and H/F), so field
# measurements of per-bee rates have to be tied to the colony they were measured in.
F_REF = 10000.0  # foragers in a typical field colony where per-bee rates are measured


# --------------------------------------------------------------------------
# Main model
# --------------------------------------------------------------------------
# K1: pollen budget. ~20% of foragers collect pollen, ~7 trips/day, ~15 mg per trip
#     => ~21 mg pollen per forager per day; raising one worker takes ~140 mg pollen
#     (Brodschneider & Crailsheim 2010). So ~0.15 new bees per forager per day at F_REF,
#     and K1 = 0.15 * sqrt(F_REF). Sanity check: 1,500 births/day at F_REF, matching
#     a queen's typical 1,500-2,000 eggs/day (Winston 1987).
BIRTHS_PER_FORAGER = 0.2 * 7 * 15 / 140

# K2: workers start foraging at ~21 days old (Winston 1987), so a hive bee leaves the
#     hive compartment at total rate ~1/21 per day = recruitment K2/F + death g1.
AGE_AT_FORAGING = 21.0


@dataclass
class Params:
    K1: float = BIRTHS_PER_FORAGER * np.sqrt(F_REF)   # ~15   [bees^(1/2) / day]
    K2: float = (1 / AGE_AT_FORAGING - 0.01) * F_REF  # ~380  [bees / day]
    K3: float = 1e-4   # distance deaths: NO DATA, free parameter  [bees^(-1/2) / day]
    g1: float = 0.01   # hive bees: low mortality, ~1%/day                 [1/day]
    g2: float = 0.154  # foragers live ~1 week after they start foraging   [1/day]


def equilibrium(p: Params):
    """Unique positive equilibrium (H*, F*).

    dH/dt = 0  ->  H = K1 F^(3/2) / (K2 + g1 F)
    dF/dt = 0  ->  K1 K2 / (K2 + g1 F) = K3 F + g2 sqrt(F)
    The left side decreases in F and the right side increases, so there is one root.
    """
    F = brentq(lambda F: p.K1 * p.K2 / (p.K2 + p.g1 * F) - p.K3 * F - p.g2 * np.sqrt(F),
               1e-9, 1e9)
    H = p.K1 * F**1.5 / (p.K2 + p.g1 * F)
    return H, F


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
    p = Params()
    print("Model parameters:", p)

    H_star, F_star = equilibrium(p)
    eig = np.linalg.eigvals(jacobian(H_star, F_star, p))
    print(f"Equilibrium: H = {H_star:,.0f}, F = {F_star:,.0f}, P = {H_star + F_star:,.0f}")
    print(f"  births = {p.K1 * np.sqrt(F_star):,.0f}/day, "
          f"distance deaths = {p.K3 * F_star**1.5:,.0f}/day")
    print(f"  Jacobian eigenvalues = {eig}",
          "-> stable" if np.all(eig.real < 0) else "-> UNSTABLE (saddle/source)")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for start in [(200, 100), (8000, 4000), (40000, 20000)]:
        sol = simulate(rhs, start, p)
        H, F = sol.y
        print(f"  start H={start[0]:,}, F={start[1]:,}: after {sol.t[-1]:.0f} d "
              f"H = {max(H[-1], 0):,.0f}, F = {max(F[-1], 0):,.0f}")
        line, = axes[0, 0].plot(sol.t, H + F, label=f"start H={start[0]:,}, F={start[1]:,}")
        axes[0, 1].plot(sol.t, H, color=line.get_color())
        axes[0, 1].plot(sol.t, F, "--", color=line.get_color())
    axes[0, 0].axhline(H_star + F_star, color="gray", lw=0.8, ls=":")
    axes[0, 0].set_title("Model: total colony P(t)")
    axes[0, 0].set_xlabel("day"); axes[0, 0].set_ylabel("bees"); axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Model: hive bees H (solid) and foragers F (dashed)")
    axes[0, 1].set_xlabel("day"); axes[0, 1].set_ylabel("bees")

    # K3 has no data: show how the equilibrium depends on it.
    print("Equilibrium vs K3:")
    K3s = np.logspace(-7, -2, 60)
    eq = np.array([equilibrium(Params(K3=k)) for k in K3s])
    for k in [0.0, 1e-5, 1e-4, 1e-3, 1e-2]:
        H, F = equilibrium(Params(K3=k))
        print(f"  K3 = {k:g}: P = {H + F:,.0f} (H = {H:,.0f}, F = {F:,.0f})")
    ax = axes[1, 0]
    ax.semilogx(K3s, eq.sum(axis=1), label="total P*")
    ax.semilogx(K3s, eq[:, 0], label="hive H*")
    ax.semilogx(K3s, eq[:, 1], "--", label="foragers F*")
    ax.axhspan(20000, 60000, color="gray", alpha=0.15, label="typical colony 20k-60k")
    ax.set_title("Equilibrium colony size vs distance-death K3")
    ax.set_xlabel("K3"); ax.set_ylabel("bees"); ax.legend(fontsize=8)

    ax = axes[1, 1]
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
