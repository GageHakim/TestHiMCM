# HiMCM Part 1 — Honeybee Colony Population Model

A two-compartment ODE model of a honeybee colony: **hive (house) bees** and **foragers**.

```
.venv/bin/python bee_model.py      # prints equilibrium + stability, saves bee_model.png
```

(Setup: `python3 -m venv .venv && .venv/bin/pip install numpy scipy matplotlib`)

---

## Current model

Let H = P_hive (hive bees), F = P_for (foragers), P = H + F.

```
dH/dt = K1·√F  −  K2·H/F  −  γ1·H
dF/dt = K2·H/F  −  K3·F^(3/2)  −  γ2·F
dP/dt = dH/dt + dF/dt
```

| Term | Meaning |
|---|---|
| K1·√F | **Births = food income.** Foragers need a foraging area ∝ F, so the foraging radius (trip length) ∝ √F. Each forager's food rate ∝ 1/√F, so total food ∝ F/√F = √F. Each new bee costs a fixed amount of food. |
| K2·H/F | **Recruitment** of hive bees to foraging. Faster when foragers are scarce (foragers inhibit recruitment). |
| K3·F^(3/2) | **Distance deaths.** Each forager's risk ∝ trip length ∝ √F, so total deaths ∝ F·√F. |
| γ1·H, γ2·F | Background death rates. |

---

## Changes from the original whiteboard model

Original:

```
dH/dt = K1·F·H/√(F+H)  −  K2·H/F  −  γ1·H
dF/dt = K2·H/F  −  K3·F^(1/2)  −  γ2·F
```

### 1. Birth term: `K1·F·H/√(F+H)` → `K1·√F`

**Problem:** The original birth term scales like (colony size)^1.5, while deaths scale linearly. Births always eventually outrun deaths, so the population reaches infinity in finite time. Its equilibrium is a saddle point: with constants calibrated to a 30k-bee equilibrium (eigenvalues +0.040, −0.385), a colony starting 10% below collapsed to 0 in ~100 days, and one starting 10% above passed 1,000,000 bees by day 110.

**Things we tried:**
- `K1·√(F·H)/√(F+H)` (square root on the product): stable, but has no physical justification beyond "the smaller of F and H limits births."
- `K1·√F / H` (food per hive bee): wrong direction. More nurses means fewer births, and a 300-bee colony would produce ~18,000 births/day, ~10× what a queen can lay.

**Fix:** Births are proportional to food income, which is ∝ √F by the foraging-radius argument above. This scales like (colony size)^0.5, so deaths eventually catch up and the colony levels off.

### 2. Forager death term: `K3·F^(1/2)` → `K3·F^(3/2)`

The foraging-radius argument says each forager flies farther (∝ √F) as the colony grows. If risk is proportional to distance, each forager's death rate is ∝ √F, so **total** deaths are F·√F = F^(3/2). The original √F was the *total food* scaling, not the death scaling.

### 3. Constants from biology, not from a chosen colony size

The first version picked a 30,000-bee equilibrium and solved for K1 and K2 backwards, which makes the colony size an *input*. Now every constant except K3 comes from bee biology, so the equilibrium colony size is a *prediction* we can compare to real colonies.

### Unchanged

- The recruitment term `K2·H/F`. Known issue: infinite when F → 0 (the code adds a tiny ε). Also not scale-free (see `F_REF` below).
- The two-compartment structure (newborns enter as hive bees, then become foragers).

---

## How the constants were determined

`K1·√F` and `K2·H/F` depend on colony size in a way that isn't scale-free. A per-bee field measurement only pins down K1 or K2 at the colony size where it was measured, so we tie both to a **reference colony of F_REF = 10,000 foragers** (a typical field colony).

| Constant | Value | Derivation |
|---|---|---|
| **γ2** forager death | 0.154 /day | Foragers survive ~1 week after they start foraging [3, 4]. 1/6.5 days ≈ 0.154. |
| **γ1** hive-bee death | 0.01 /day | Hive bees work inside and have low mortality [1]. ~1%/day is a placeholder; Khoury et al. [4] use 0. |
| **K2** recruitment | ≈ 376 bees/day | Workers begin foraging at ~21 days of age [1, 5]. A hive bee leaves the hive compartment at total rate 1/21 per day = K2/F + γ1, so K2 = (1/21 − 0.01)·F_REF ≈ 376. |
| **K1** births | 15 bees^½/day | Pollen budget: ~20% of foragers collect pollen [1], ~7 trips/day [1], ~15 mg per trip (both pollen loads) [1] → ~21 mg pollen per forager per day. Raising one worker takes ~140 mg pollen [2]. So ~0.15 new bees per forager per day at F_REF, and K1 = 0.15·√F_REF = 15. **Check:** 1,500 births/day at F_REF, consistent with a queen's typical 1,500–2,000 eggs/day [1]. |
| **K3** distance death | 10⁻⁴ (free) | No data. Treated as a free parameter and swept (see below). |

### Result

With the constants above and K3 = 10⁻⁴:

- **Equilibrium:** 17,065 hive bees + 6,296 foragers = **23,361 bees**, inside the typical 20,000–60,000 range for a colony
- Births ≈ 1,190/day, distance deaths ≈ 50/day
- **Stable:** Jacobian eigenvalues −0.020, −0.378. Colonies starting at 300 bees and at 60,000 bees both converge to it within about 200 days.
- There is exactly **one** positive equilibrium for any constants: setting both equations to zero gives K1·K2/(K2 + γ1·F) = K3·F + γ2·√F, where the left side decreases in F and the right increases.

### Sensitivity to K3

| K3 | Equilibrium P | H | F |
|---|---|---|---|
| 0 | 25,753 | 18,950 | 6,804 |
| 10⁻⁵ | 25,491 | 18,743 | 6,748 |
| 10⁻⁴ | 23,361 | 17,065 | 6,296 |
| 10⁻³ | 12,820 | 8,891 | 3,929 |
| 10⁻² | 2,172 | 1,192 | 980 |

K3 barely matters below ~10⁻⁴. The colony stays in the realistic range (above 20,000 bees) for K3 up to about 2.8×10⁻⁴.

---

## Open issues

- **No collapse:** every colony recovers, however small. Real colonies below a critical size fail (see Khoury et al. [4]).
- **No queen cap:** births grow like √F without limit instead of saturating at ~2,000/day.
- **Recruitment `K2·H/F`** is infinite at F = 0 and depends on the reference colony size.
- **No brood delay** (~21 days egg → adult), **food stores**, or **seasons**.
- `bee_model.py` also contains a comparison model (Khoury et al. [4] style, queen-limited births), plotted in the bottom-right panel.

---

## Sources

> ⚠️ These references and figures were written from memory and **have not been checked against the originals**. Verify each number in the source before citing it in the paper. The forager death rate and the pollen-trip figures (trips/day, mg/trip, % pollen foragers) are the least certain and vary a lot between studies.

1. Winston, M. L. (1987). *The Biology of the Honey Bee*. Harvard University Press. — queen egg-laying rate, age at first foraging, pollen foraging behavior, hive-bee mortality.
2. Brodschneider, R., & Crailsheim, K. (2010). Nutrition and health in honey bees. *Apidologie*, 41(3), 278–294. — pollen needed to rear one worker (~125–145 mg).
3. Visscher, P. K., & Dukas, R. (1997). Survivorship of foraging honey bees. *Insectes Sociaux*, 44, 1–5. — forager life expectancy after the start of foraging.
4. Khoury, D. S., Myerscough, M. R., & Barron, A. B. (2011). A quantitative model of honey bee colony population dynamics. *PLoS ONE*, 6(4), e18491. — hive/forager compartment model, social inhibition of recruitment, colony collapse threshold.
5. Robinson, G. E. (1992). Regulation of division of labor in insect societies. *Annual Review of Entomology*, 37, 637–665. — age polyethism, age at onset of foraging.
