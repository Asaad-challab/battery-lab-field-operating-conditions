from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.home() / "battery/paper_C"
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
RESULTS = ROOT / "results"

FIGS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD MIT SUMMARY
# ============================================================

f = TABLES / "MIT_LFP_lab_domain.csv"
df = pd.read_csv(f)

print("Loaded:", f)
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Only cells with valid cycle-life labels
d = df.dropna(
    subset=[
        "cycle_life",
        "C1",
        "C2",
        "Q1_pct",
        "Tavg_mean_C",
        "Tmax_global_C"
    ]
).copy()

print("\nCells with valid cycle life:", len(d))


# ============================================================
# 1. DEFINE DESCRIPTIVE CHARGING-POLICY METRICS
# ============================================================

# SOC-weighted mean fast-charge C-rate over 0-80% SOC.
#
# Policy:
# C1 from 0 -> Q1
# C2 from Q1 -> 80%
#
# This is NOT an electrochemical stress index.
# It is only a descriptive charging-rate metric.

d["C_weighted_0_80"] = (
    d["C1"] * d["Q1_pct"] +
    d["C2"] * (80.0 - d["Q1_pct"])
) / 80.0

# Maximum commanded fast-charge rate
d["C_peak"] = d[["C1", "C2"]].max(axis=1)

# Temperature rise relative to nominal 30 C chamber condition
d["Tavg_rise_from_30"] = d["Tavg_mean_C"] - 30.0
d["Tmax_rise_from_30"] = d["Tmax_global_C"] - 30.0


# ============================================================
# 2. CELL-LEVEL CORRELATIONS
# ============================================================

variables = [
    "C1",
    "C2",
    "Q1_pct",
    "C_weighted_0_80",
    "C_peak",
    "Tavg_mean_C",
    "Tmax_global_C",
    "Tavg_rise_from_30",
    "Tmax_rise_from_30",
]

corr_rows = []

for x in variables:

    tmp = d[[x, "cycle_life"]].dropna()

    pearson = tmp[x].corr(
        tmp["cycle_life"],
        method="pearson"
    )

    spearman = tmp[x].corr(
        tmp["cycle_life"],
        method="spearman"
    )

    corr_rows.append({
        "variable": x,
        "n": len(tmp),
        "pearson_r": pearson,
        "spearman_rho": spearman
    })

corr = pd.DataFrame(corr_rows)

corr_file = RESULTS / "MIT_cell_level_stress_cyclelife_correlations.csv"
corr.to_csv(corr_file, index=False)

print("\n" + "="*70)
print("CELL-LEVEL ASSOCIATIONS WITH CYCLE LIFE")
print("="*70)
print(corr.round(3).to_string(index=False))


# ============================================================
# 3. POLICY-LEVEL AGGREGATION
# ============================================================

policy = (
    d.groupby(
        ["charge_policy", "C1", "Q1_pct", "C2"],
        as_index=False
    )
    .agg(
        n_cells=("cycle_life", "size"),

        cycle_life_mean=("cycle_life", "mean"),
        cycle_life_median=("cycle_life", "median"),
        cycle_life_std=("cycle_life", "std"),
        cycle_life_min=("cycle_life", "min"),
        cycle_life_max=("cycle_life", "max"),

        Tavg_mean_C=("Tavg_mean_C", "mean"),
        Tmax_mean_C=("Tmax_global_C", "mean"),
        Tmax_max_C=("Tmax_global_C", "max"),

        C_weighted_0_80=("C_weighted_0_80", "first"),
        C_peak=("C_peak", "first")
    )
)

# IQR separately
iqr = (
    d.groupby("charge_policy")["cycle_life"]
    .quantile([0.25, 0.75])
    .unstack()
)

iqr.columns = ["cycle_life_q25", "cycle_life_q75"]
iqr["cycle_life_IQR"] = (
    iqr["cycle_life_q75"] -
    iqr["cycle_life_q25"]
)

policy = policy.merge(
    iqr,
    left_on="charge_policy",
    right_index=True,
    how="left"
)

policy = policy.sort_values(
    "cycle_life_median"
).reset_index(drop=True)

policy_file = RESULTS / "MIT_policy_cyclelife_summary.csv"
policy.to_csv(policy_file, index=False)

print("\n" + "="*70)
print("POLICY-LEVEL SUMMARY")
print("="*70)

cols = [
    "charge_policy",
    "n_cells",
    "C1",
    "Q1_pct",
    "C2",
    "C_weighted_0_80",
    "Tmax_mean_C",
    "cycle_life_median",
    "cycle_life_IQR",
]

print(
    policy[cols]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 4. POLICY-LEVEL CORRELATIONS
# ============================================================

policy_variables = [
    "C1",
    "C2",
    "Q1_pct",
    "C_weighted_0_80",
    "C_peak",
    "Tavg_mean_C",
    "Tmax_mean_C"
]

policy_corr_rows = []

for x in policy_variables:

    tmp = policy[
        [x, "cycle_life_median"]
    ].dropna()

    policy_corr_rows.append({
        "variable": x,
        "n_policies": len(tmp),

        "pearson_r":
            tmp[x].corr(
                tmp["cycle_life_median"],
                method="pearson"
            ),

        "spearman_rho":
            tmp[x].corr(
                tmp["cycle_life_median"],
                method="spearman"
            )
    })

policy_corr = pd.DataFrame(policy_corr_rows)

policy_corr_file = (
    RESULTS /
    "MIT_policy_level_stress_cyclelife_correlations.csv"
)

policy_corr.to_csv(policy_corr_file, index=False)

print("\n" + "="*70)
print("POLICY-LEVEL ASSOCIATIONS WITH MEDIAN CYCLE LIFE")
print("="*70)
print(policy_corr.round(3).to_string(index=False))


# ============================================================
# 5. FIGURE — POLICY LIFE WITH WITHIN-POLICY VARIABILITY
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

plot_df = policy.sort_values(
    "cycle_life_median"
).copy()

yerr_low = (
    plot_df["cycle_life_median"] -
    plot_df["cycle_life_q25"]
)

yerr_high = (
    plot_df["cycle_life_q75"] -
    plot_df["cycle_life_median"]
)

ax.errorbar(
    np.arange(len(plot_df)),
    plot_df["cycle_life_median"],
    yerr=np.vstack([yerr_low, yerr_high]),
    fmt="o",
    capsize=4
)

ax.set_xticks(np.arange(len(plot_df)))

ax.set_xticklabels(
    plot_df["charge_policy"],
    rotation=45,
    ha="right",
    fontsize=8
)

ax.set_ylabel("Cycle life [cycles]")
ax.set_xlabel("Fast-charging policy")

ax.set_title(
    "MIT LFP Cycle Life across Fast-Charging Policies"
)

ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

f1 = FIGS / "Fig_MIT_policy_cyclelife.png"
fig.savefig(f1, dpi=300)
plt.close(fig)


# ============================================================
# 6. FIGURE — WEIGHTED C-RATE VS POLICY MEDIAN LIFE
# ============================================================

fig, ax = plt.subplots(figsize=(7.5, 5.5))

ax.scatter(
    policy["C_weighted_0_80"],
    policy["cycle_life_median"],
    s=70
)

for _, r in policy.iterrows():

    ax.annotate(
        r["charge_policy"].split("-newstructure")[0],
        (
            r["C_weighted_0_80"],
            r["cycle_life_median"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=7
    )

ax.set_xlabel(
    "SOC-weighted fast-charge C-rate over 0–80% SOC [C]"
)

ax.set_ylabel(
    "Median cycle life [cycles]"
)

ax.set_title(
    "Fast-Charging Severity versus Median Cycle Life"
)

ax.grid(alpha=0.25)

fig.tight_layout()

f2 = FIGS / "Fig_MIT_Crate_vs_cyclelife.png"
fig.savefig(f2, dpi=300)
plt.close(fig)


# ============================================================
# 7. FIGURE — TEMPERATURE VS POLICY MEDIAN LIFE
# ============================================================

fig, ax = plt.subplots(figsize=(7.5, 5.5))

ax.scatter(
    policy["Tmax_mean_C"],
    policy["cycle_life_median"],
    s=70
)

for _, r in policy.iterrows():

    ax.annotate(
        r["charge_policy"].split("-newstructure")[0],
        (
            r["Tmax_mean_C"],
            r["cycle_life_median"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=7
    )

ax.set_xlabel(
    "Mean observed maximum cell temperature [°C]"
)

ax.set_ylabel(
    "Median cycle life [cycles]"
)

ax.set_title(
    "Thermal Response versus Median Cycle Life"
)

ax.grid(alpha=0.25)

fig.tight_layout()

f3 = FIGS / "Fig_MIT_temperature_vs_cyclelife.png"
fig.savefig(f3, dpi=300)
plt.close(fig)


print("\nSaved tables:")
print(corr_file)
print(policy_file)
print(policy_corr_file)

print("\nSaved figures:")
print(f1)
print(f2)
print(f3)

print("\nIMPORTANT:")
print(
    "These are observational associations within the MIT "
    "fast-charge experiment. Do not interpret correlation "
    "as an isolated causal effect of C-rate or temperature."
)
