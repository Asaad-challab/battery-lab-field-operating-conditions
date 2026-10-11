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

tum = pd.read_csv(TABLES / "TUM_field_profiles.csv")
kit = pd.read_csv(TABLES / "KIT_lab_domain.csv")

# ------------------------------------------------------------
# IMPORTANT LIMITATION
# ------------------------------------------------------------
# TUM public JSON gives marginal distributions, not joint SOC-T.
# Therefore we cannot create a true joint probability distribution
# by multiplying marginals.
#
# Instead, this script constructs a conservative DOMAIN comparison:
# - field SOC support by vehicle
# - field T_max support by vehicle
# - KIT protocol SOC windows and temperature setpoints
#
# This is a rectangular operating-domain comparison, NOT a measured
# joint density.
# ------------------------------------------------------------

tum = tum[tum["state"] == "charging"].copy()

rows = []

for vehicle in sorted(tum["vehicle"].unique()):

    soc = tum[
        (tum["vehicle"] == vehicle) &
        (tum["variable"] == "SOC")
    ]

    temp = tum[
        (tum["vehicle"] == vehicle) &
        (tum["variable"] == "T_max")
    ]

    rows.append({
        "vehicle": vehicle,
        "SOC_min": soc["x_value"].min(),
        "SOC_max": soc["x_value"].max(),
        "Tmax_min_C": temp["x_value"].min(),
        "Tmax_max_C": temp["x_value"].max(),
    })

field_domains = pd.DataFrame(rows)

out_csv = RESULTS / "TUM_SOC_temperature_domains.csv"
field_domains.to_csv(out_csv, index=False)

print("\nFIELD DOMAINS")
print(field_domains.round(2).to_string(index=False))


# ============================================================
# KIT DESIGN POINTS
# ============================================================

kit_points = (
    kit[
        ["temperature_C", "SOC_min_pct", "SOC_max_pct"]
    ]
    .dropna()
    .drop_duplicates()
    .sort_values(["temperature_C", "SOC_min_pct", "SOC_max_pct"])
)

print("\nKIT SOC-T DESIGN")
print(kit_points.to_string(index=False))


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

# Plot field support rectangles
for _, r in field_domains.iterrows():

    width = r["SOC_max"] - r["SOC_min"]
    height = r["Tmax_max_C"] - r["Tmax_min_C"]

    rect = plt.Rectangle(
        (r["SOC_min"], r["Tmax_min_C"]),
        width,
        height,
        fill=False,
        linewidth=1.5,
        alpha=0.8
    )

    ax.add_patch(rect)

    ax.text(
        r["SOC_min"] + width / 2,
        r["Tmax_min_C"] + height / 2,
        r["vehicle"],
        ha="center",
        va="center",
        fontsize=8
    )

# Overlay KIT SOC windows at each temperature setpoint
for _, r in kit_points.iterrows():

    ax.plot(
        [r["SOC_min_pct"], r["SOC_max_pct"]],
        [r["temperature_C"], r["temperature_C"]],
        linewidth=4,
        alpha=0.7
    )

ax.set_xlim(0, 100)
ax.set_ylim(
    min(0, field_domains["Tmax_min_C"].min() - 2),
    max(45, field_domains["Tmax_max_C"].max() + 2)
)

ax.set_xlabel("State of charge [%]")
ax.set_ylabel("Battery temperature [°C]")

ax.set_title(
    "Field Charging SOC–Temperature Support versus KIT Laboratory Design"
)

ax.grid(alpha=0.2)

fig.tight_layout()

figfile = FIGS / "Fig_joint_SOC_temperature_domain.png"
fig.savefig(figfile, dpi=300)
plt.close(fig)

print("\nSaved:")
print(figfile)
print(out_csv)


# ============================================================
# SIMPLE DOMAIN COVERAGE TEST
# ============================================================

kit_T_min = kit["temperature_C"].min()
kit_T_max = kit["temperature_C"].max()

kit_SOC_min = kit["SOC_min_pct"].min()
kit_SOC_max = kit["SOC_max_pct"].max()

coverage_rows = []

for _, r in field_domains.iterrows():

    soc_inside = (
        r["SOC_min"] >= kit_SOC_min and
        r["SOC_max"] <= kit_SOC_max
    )

    temp_inside = (
        r["Tmax_min_C"] >= kit_T_min and
        r["Tmax_max_C"] <= kit_T_max
    )

    coverage_rows.append({
        "vehicle": r["vehicle"],
        "SOC_support_inside_KIT": soc_inside,
        "temperature_support_inside_KIT": temp_inside,
        "joint_rectangular_support_inside_KIT":
            soc_inside and temp_inside
    })

coverage = pd.DataFrame(coverage_rows)

coverage_file = RESULTS / "KIT_vs_TUM_SOC_temperature_support.csv"
coverage.to_csv(coverage_file, index=False)

print("\nDOMAIN SUPPORT CHECK")
print(coverage.to_string(index=False))

print("\nSaved:")
print(coverage_file)
