from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

ROOT = Path.home() / "battery/paper_C"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD POLICY-LEVEL RESULTS FROM SCRIPT 08
# ============================================================

f = RESULTS / "MIT_policy_cyclelife_summary.csv"
df = pd.read_csv(f)

xcol = "C_peak"
ycol = "cycle_life_median"

d = df[
    ["charge_policy", xcol, ycol]
].dropna().copy()

print("Loaded:", f)
print("Number of policies:", len(d))

print("\nPOLICIES")
print(d.to_string(index=False))


# ============================================================
# 1. BASE SPEARMAN ASSOCIATION
# ============================================================

rho, p_asymptotic = spearmanr(
    d[xcol],
    d[ycol]
)

print("\n" + "=" * 70)
print("BASE ASSOCIATION")
print("=" * 70)
print(f"Spearman rho: {rho:.4f}")
print(f"Scipy p-value: {p_asymptotic:.6f}")


# ============================================================
# 2. EXACT PERMUTATION TEST
#
# With only 8 policies:
# 8! = 40,320 permutations
# so an exact enumeration is practical.
# ============================================================

import itertools

x = d[xcol].to_numpy(dtype=float)
y = d[ycol].to_numpy(dtype=float)

observed_rho = spearmanr(x, y).statistic

perm_rhos = []

for perm in itertools.permutations(y):
    r = spearmanr(x, perm).statistic
    perm_rhos.append(r)

perm_rhos = np.asarray(perm_rhos)

# Two-sided exact permutation probability
p_exact = np.mean(
    np.abs(perm_rhos) >= abs(observed_rho) - 1e-12
)

print("\n" + "=" * 70)
print("EXACT PERMUTATION TEST")
print("=" * 70)
print("Permutations:", len(perm_rhos))
print(f"Observed rho: {observed_rho:.4f}")
print(f"Exact two-sided p: {p_exact:.6f}")


# ============================================================
# 3. LEAVE-ONE-POLICY-OUT SENSITIVITY
# ============================================================

loo_rows = []

for i in range(len(d)):

    reduced = d.drop(d.index[i])

    r, p = spearmanr(
        reduced[xcol],
        reduced[ycol]
    )

    loo_rows.append({
        "removed_policy": d.iloc[i]["charge_policy"],
        "n_remaining": len(reduced),
        "spearman_rho": r,
        "p_value": p
    })

loo = pd.DataFrame(loo_rows)

loo_file = RESULTS / "MIT_Cpeak_leave_one_policy_out.csv"
loo.to_csv(loo_file, index=False)

print("\n" + "=" * 70)
print("LEAVE-ONE-POLICY-OUT")
print("=" * 70)
print(
    loo.round(4).to_string(index=False)
)

print("\nLOO rho range:")
print(
    f"{loo['spearman_rho'].min():.4f} "
    f"to {loo['spearman_rho'].max():.4f}"
)


# ============================================================
# 4. BOOTSTRAP ACROSS POLICIES
#
# Because resampling may occasionally produce constant x,
# those replicates are discarded.
#
# Important:
# This is a descriptive nonparametric bootstrap over only
# eight experimental policies. Interpret the CI cautiously.
# ============================================================

rng = np.random.default_rng(42)

B = 20000
boot_rhos = []

for _ in range(B):

    idx = rng.integers(
        0,
        len(d),
        size=len(d)
    )

    xb = x[idx]
    yb = y[idx]

    # Spearman undefined if all sampled x or y are identical
    if (
        len(np.unique(xb)) < 2 or
        len(np.unique(yb)) < 2
    ):
        continue

    r = spearmanr(xb, yb).statistic

    if np.isfinite(r):
        boot_rhos.append(r)

boot_rhos = np.asarray(boot_rhos)

ci_low, ci_high = np.percentile(
    boot_rhos,
    [2.5, 97.5]
)

print("\n" + "=" * 70)
print("POLICY-LEVEL BOOTSTRAP")
print("=" * 70)
print("Requested replicates:", B)
print("Valid replicates:", len(boot_rhos))
print(f"Observed rho: {observed_rho:.4f}")
print(
    f"Bootstrap 95% percentile interval: "
    f"[{ci_low:.4f}, {ci_high:.4f}]"
)


# ============================================================
# 5. EXPLICIT EXTREME-POLICY REMOVAL
# ============================================================

lowest_life_idx = d[ycol].idxmin()
highest_life_idx = d[ycol].idxmax()

extreme_rows = []

for label, idx in [
    ("remove_lowest_cycle_life", lowest_life_idx),
    ("remove_highest_cycle_life", highest_life_idx)
]:

    reduced = d.drop(idx)

    r, p = spearmanr(
        reduced[xcol],
        reduced[ycol]
    )

    extreme_rows.append({
        "test": label,
        "removed_policy": d.loc[idx, "charge_policy"],
        "spearman_rho": r,
        "p_value": p
    })

extreme = pd.DataFrame(extreme_rows)

print("\n" + "=" * 70)
print("EXTREME-POLICY REMOVAL")
print("=" * 70)
print(
    extreme.round(4).to_string(index=False)
)


# ============================================================
# 6. REPEAT ROBUSTNESS FOR WEIGHTED C-RATE
# ============================================================

x2col = "C_weighted_0_80"

d2 = df[
    ["charge_policy", x2col, ycol]
].dropna().copy()

rho2, p2 = spearmanr(
    d2[x2col],
    d2[ycol]
)

x2 = d2[x2col].to_numpy(dtype=float)
y2 = d2[ycol].to_numpy(dtype=float)

perm2 = []

for perm in itertools.permutations(y2):
    perm2.append(
        spearmanr(x2, perm).statistic
    )

perm2 = np.asarray(perm2)

p2_exact = np.mean(
    np.abs(perm2) >= abs(rho2) - 1e-12
)

print("\n" + "=" * 70)
print("WEIGHTED C-RATE CHECK")
print("=" * 70)
print(f"Spearman rho: {rho2:.4f}")
print(f"Scipy p-value: {p2:.6f}")
print(f"Exact permutation p: {p2_exact:.6f}")


# ============================================================
# 7. SUMMARY TABLE
# ============================================================

summary = pd.DataFrame([
    {
        "metric": "Peak C-rate",
        "n_policies": len(d),
        "spearman_rho": observed_rho,
        "exact_permutation_p": p_exact,
        "bootstrap_CI_low": ci_low,
        "bootstrap_CI_high": ci_high,
        "LOO_rho_min": loo["spearman_rho"].min(),
        "LOO_rho_max": loo["spearman_rho"].max()
    },
    {
        "metric": "SOC-weighted C-rate",
        "n_policies": len(d2),
        "spearman_rho": rho2,
        "exact_permutation_p": p2_exact,
        "bootstrap_CI_low": np.nan,
        "bootstrap_CI_high": np.nan,
        "LOO_rho_min": np.nan,
        "LOO_rho_max": np.nan
    }
])

summary_file = RESULTS / "MIT_policy_robustness_summary.csv"
summary.to_csv(summary_file, index=False)


# ============================================================
# 8. ROBUSTNESS FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.scatter(
    np.arange(len(loo)),
    loo["spearman_rho"],
    s=70
)

ax.axhline(
    observed_rho,
    linestyle="--",
    linewidth=1.5,
    label=f"All policies: rho={observed_rho:.2f}"
)

ax.axhline(
    0,
    linewidth=1
)

ax.set_xticks(
    np.arange(len(loo))
)

ax.set_xticklabels(
    [
        p.split("-newstructure")[0]
        for p in loo["removed_policy"]
    ],
    rotation=45,
    ha="right",
    fontsize=8
)

ax.set_ylabel(
    "Spearman correlation after policy removal"
)

ax.set_xlabel(
    "Removed charging policy"
)

ax.set_title(
    "Leave-One-Policy-Out Robustness of Peak C-rate–Cycle-Life Association"
)

ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

figfile = FIGS / "Fig_MIT_policy_robustness.png"
fig.savefig(figfile, dpi=300)
plt.close(fig)


print("\n" + "=" * 70)
print("FINAL ROBUSTNESS SUMMARY")
print("=" * 70)
print(summary.round(4).to_string(index=False))

print("\nSaved:")
print(loo_file)
print(summary_file)
print(figfile)
