from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.home() / "battery/paper_C"
TABLES = ROOT / "tables"
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

kit = pd.read_csv(TABLES / "KIT_lab_domain.csv")
mit = pd.read_csv(TABLES / "MIT_LFP_lab_domain.csv")
tum = pd.read_csv(TABLES / "TUM_field_profiles.csv")

tum_chg = tum[tum["state"] == "charging"].copy()

print("KIT:", kit.shape)
print("MIT:", mit.shape)
print("TUM charging:", tum_chg.shape)


# ============================================================
# HELPER: weighted quantile
# ============================================================

def weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = (
        np.isfinite(values) &
        np.isfinite(weights) &
        (weights > 0)
    )

    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return np.nan

    idx = np.argsort(values)
    values = values[idx]
    weights = weights[idx]

    cdf = np.cumsum(weights) / np.sum(weights)

    return np.interp(q, cdf, values)


# ============================================================
# 1. FIELD PERCENTILE TABLE
# ============================================================

rows = []

for vehicle in sorted(tum_chg["vehicle"].unique()):

    v = tum_chg[tum_chg["vehicle"] == vehicle]

    row = {"vehicle": vehicle}

    for variable, prefix in [
        ("C_rate", "C_rate"),
        ("SOC", "SOC"),
        ("T_max", "Tmax"),
    ]:

        x = v[v["variable"] == variable]

        if len(x) == 0:
            continue

        values = x["x_value"].to_numpy()
        weights = x["weight"].to_numpy()

        # Tiny negative C-rate bins around zero are histogram artefacts
        if variable == "C_rate":
            values = np.maximum(values, 0.0)

        row[f"{prefix}_p05"] = weighted_quantile(
            values, weights, 0.05
        )
        row[f"{prefix}_p50"] = weighted_quantile(
            values, weights, 0.50
        )
        row[f"{prefix}_p95"] = weighted_quantile(
            values, weights, 0.95
        )

        row[f"{prefix}_max"] = np.max(values)

    rows.append(row)

field = pd.DataFrame(rows)

field_out = TABLES / "TUM_field_percentiles.csv"
field.to_csv(field_out, index=False)

print("\n==============================================")
print("TUM FIELD PERCENTILES")
print("==============================================")
print(field.round(3).to_string(index=False))


# ============================================================
# 2. LAB–FIELD C-RATE COMPARISON
# ============================================================

field_p95_max = field["C_rate_p95"].max()
field_p95_median = field["C_rate_p95"].median()

kit_min = kit["charge_C_rate"].min()
kit_max = kit["charge_C_rate"].max()

mit_fast_min = min(
    mit["C1"].min(),
    mit["C2"].min()
)

mit_fast_max = max(
    mit["C1"].max(),
    mit["C2"].max()
)

comparison = pd.DataFrame([
    {
        "domain": "TUM field",
        "metric": "Median vehicle P95",
        "C_rate": field_p95_median
    },
    {
        "domain": "TUM field",
        "metric": "Highest vehicle P95",
        "C_rate": field_p95_max
    },
    {
        "domain": "KIT lab",
        "metric": "Minimum protocol rate",
        "C_rate": kit_min
    },
    {
        "domain": "KIT lab",
        "metric": "Maximum protocol rate",
        "C_rate": kit_max
    },
    {
        "domain": "MIT lab",
        "metric": "Minimum fast-charge rate",
        "C_rate": mit_fast_min
    },
    {
        "domain": "MIT lab",
        "metric": "Maximum fast-charge rate",
        "C_rate": mit_fast_max
    },
])

comparison_out = TABLES / "lab_field_Crate_reference.csv"
comparison.to_csv(comparison_out, index=False)

print("\n==============================================")
print("LAB–FIELD C-RATE REFERENCE")
print("==============================================")
print(comparison.round(3).to_string(index=False))


# ============================================================
# FIGURE 1:
# TUM charging C-rate distributions
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

for vehicle in sorted(tum_chg["vehicle"].unique()):

    x = tum_chg[
        (tum_chg["vehicle"] == vehicle) &
        (tum_chg["variable"] == "C_rate")
    ].copy()

    if len(x) == 0:
        continue

    x["x_plot"] = x["x_value"].clip(lower=0)

    # Normalize each vehicle independently
    total = x["weight"].sum()

    if total <= 0:
        continue

    x["probability"] = x["weight"] / total

    ax.plot(
        x["x_plot"],
        x["probability"],
        label=vehicle,
        linewidth=1.5
    )

# Lab reference rates
for rate in sorted(kit["charge_C_rate"].dropna().unique()):
    ax.axvline(
        rate,
        linestyle="--",
        linewidth=0.8,
        alpha=0.5
    )

ax.axvspan(
    mit_fast_min,
    mit_fast_max,
    alpha=0.10,
    label="MIT fast-charge range"
)

ax.set_xlim(left=0)
ax.set_xlabel("Charging C-rate [C]")
ax.set_ylabel("Normalized field exposure")
ax.set_title(
    "Real-EV Charging Exposure versus Laboratory Charging Stress"
)

ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.2)

fig.tight_layout()

f1 = FIGS / "Fig_Crate_field_vs_lab.png"
fig.savefig(f1, dpi=300)
plt.close(fig)


# ============================================================
# FIGURE 2:
# TUM T_max field distributions + lab thermal domains
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

for vehicle in sorted(tum_chg["vehicle"].unique()):

    x = tum_chg[
        (tum_chg["vehicle"] == vehicle) &
        (tum_chg["variable"] == "T_max")
    ].copy()

    if len(x) == 0:
        continue

    total = x["weight"].sum()

    if total <= 0:
        continue

    x["probability"] = x["weight"] / total

    ax.plot(
        x["x_value"],
        x["probability"],
        label=vehicle,
        linewidth=1.5
    )

# KIT controlled temperature envelope
kit_Tmin = kit["temperature_C"].min()
kit_Tmax = kit["temperature_C"].max()

ax.axvspan(
    kit_Tmin,
    kit_Tmax,
    alpha=0.08,
    label="KIT protocol temperature range"
)

# MIT observed surface-temperature envelope
mit_Tmin = mit["Tmin_global_C"].min()
mit_Tmax = mit["Tmax_global_C"].max()

ax.axvspan(
    mit_Tmin,
    mit_Tmax,
    alpha=0.08,
    label="MIT measured cell-temperature range"
)

ax.set_xlabel("Maximum battery temperature [°C]")
ax.set_ylabel("Normalized field exposure")
ax.set_title(
    "Real-EV Thermal Exposure versus Laboratory Thermal Domains"
)

ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.2)

fig.tight_layout()

f2 = FIGS / "Fig_temperature_field_vs_lab.png"
fig.savefig(f2, dpi=300)
plt.close(fig)


# ============================================================
# FIGURE 3:
# TUM charging SOC distributions
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

for vehicle in sorted(tum_chg["vehicle"].unique()):

    x = tum_chg[
        (tum_chg["vehicle"] == vehicle) &
        (tum_chg["variable"] == "SOC")
    ].copy()

    if len(x) == 0:
        continue

    total = x["weight"].sum()

    if total <= 0:
        continue

    x["probability"] = x["weight"] / total

    ax.plot(
        x["x_value"],
        x["probability"],
        label=vehicle,
        linewidth=1.5
    )

# MIT high-rate charging region
ax.axvspan(
    0,
    80,
    alpha=0.07,
    label="MIT high-rate charging region"
)

ax.axvline(
    80,
    linestyle="--",
    linewidth=1.2
)

ax.set_xlim(0, 100)
ax.set_xlabel("State of charge [%]")
ax.set_ylabel("Normalized field exposure")
ax.set_title(
    "Real-EV Charging SOC Exposure and MIT Fast-Charge Region"
)

ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.2)

fig.tight_layout()

f3 = FIGS / "Fig_SOC_field_vs_lab.png"
fig.savefig(f3, dpi=300)
plt.close(fig)


# ============================================================
# STRESS AMPLIFICATION — DESCRIPTIVE ONLY
# ============================================================

stress = pd.DataFrame([
    {
        "reference": "KIT maximum / highest TUM vehicle P95",
        "ratio": kit_max / field_p95_max
    },
    {
        "reference": "MIT maximum / highest TUM vehicle P95",
        "ratio": mit_fast_max / field_p95_max
    },
    {
        "reference": "KIT maximum / median TUM vehicle P95",
        "ratio": kit_max / field_p95_median
    },
    {
        "reference": "MIT maximum / median TUM vehicle P95",
        "ratio": mit_fast_max / field_p95_median
    },
])

stress_out = TABLES / "descriptive_stress_amplification.csv"
stress.to_csv(stress_out, index=False)

print("\n==============================================")
print("DESCRIPTIVE STRESS AMPLIFICATION")
print("==============================================")
print(stress.round(2).to_string(index=False))

print("\nSaved figures:")
print(f1)
print(f2)
print(f3)

print("\nSaved tables:")
print(field_out)
print(comparison_out)
print(stress_out)
