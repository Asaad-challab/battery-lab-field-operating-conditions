from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# Paths
# ============================================================

ROOT = Path.home() / "battery/paper_C"

FIELD_FILE = ROOT / "tables/TUM_field_profiles.csv"
KIT_FILE   = ROOT / "tables/KIT_lab_domain.csv"
MIT_FILE   = ROOT / "tables/MIT_LFP_lab_domain.csv"

OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load data
# ============================================================

field = pd.read_csv(FIELD_FILE)
kit   = pd.read_csv(KIT_FILE)
mit   = pd.read_csv(MIT_FILE)


# ============================================================
# TUM FIELD DATA
# Use CHARGING exposure only
# ============================================================

field = field[field["state"] == "charging"].copy()

# Clean only obvious histogram-edge numerical artefacts.
field.loc[
    (field["variable"] == "SOC") &
    (field["x_value"] < 0),
    "x_value"
] = 0.0

field.loc[
    (field["variable"] == "SOC") &
    (field["x_value"] > 100),
    "x_value"
] = 100.0

field.loc[
    (field["variable"] == "C_rate") &
    (field["x_value"] < 0) &
    (field["x_value"] > -0.02),
    "x_value"
] = 0.0


# ============================================================
# DEFINE LAB DOMAINS
# ============================================================

# ------------------------------------------------------------
# KIT
# ------------------------------------------------------------
#
# Use the actual KIT protocol table.
#
# SOC support:
# minimum SOC_min_pct -> maximum SOC_max_pct
#
# Temperature support:
# minimum -> maximum controlled temperature_C
#
# C-rate:
# actual nominal charging-rate values in the KIT table.
#
# IMPORTANT:
# We distinguish:
#   BELOW = field exposure below the laboratory stress range
#   INSIDE = represented by laboratory range
#   ABOVE = field exposure exceeding laboratory range
# ------------------------------------------------------------

KIT_BOUNDS = {
    "SOC": (
        float(kit["SOC_min_pct"].min()),
        float(kit["SOC_max_pct"].max())
    ),

    "C_rate": (
        float(kit["charge_C_rate"].min()),
        float(kit["charge_C_rate"].max())
    ),

    "T_max": (
        float(kit["temperature_C"].min()),
        float(kit["temperature_C"].max())
    ),
}


# ------------------------------------------------------------
# MIT
# ------------------------------------------------------------
#
# MIT fast-charge region:
# 0--80% SOC.
#
# C-rate:
# use all fast-charge policy rates C1 and C2.
#
# Temperature:
# use measured global cell-temperature support.
#
# This should be interpreted as the observed MIT accelerated
# fast-charge laboratory domain, NOT a complete MIT probability
# distribution.
# ------------------------------------------------------------

mit_c_rates = np.concatenate([
    mit["C1"].dropna().to_numpy(dtype=float),
    mit["C2"].dropna().to_numpy(dtype=float),
])

MIT_BOUNDS = {
    "SOC": (
        float(mit["fast_charge_SOC_start_pct"].min()),
        float(mit["fast_charge_SOC_end_pct"].max())
    ),

    "C_rate": (
        float(np.min(mit_c_rates)),
        float(np.max(mit_c_rates))
    ),

    "T_max": (
        float(mit["Tmin_global_C"].min()),
        float(mit["Tmax_global_C"].max())
    ),
}


# ============================================================
# Helper functions
# ============================================================

def weighted_fraction(mask, weights):
    """Weighted percentage for Boolean mask."""
    weights = np.asarray(weights, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    total = np.sum(weights)

    if total <= 0:
        return np.nan

    return 100.0 * np.sum(weights[mask]) / total


def normalized_distance_to_interval(values, low, high):
    """
    Normalized distance from [low, high].

    Inside range -> 0
    Below range  -> (low - x) / range_width
    Above range  -> (x - high) / range_width
    """
    values = np.asarray(values, dtype=float)

    width = high - low

    if width <= 0:
        return np.full_like(values, np.nan, dtype=float)

    d = np.zeros_like(values, dtype=float)

    below = values < low
    above = values > high

    d[below] = (low - values[below]) / width
    d[above] = (values[above] - high) / width

    return d


def analyse_variable(d, low, high):
    values = d["x_value"].to_numpy(dtype=float)
    weights = d["weight"].to_numpy(dtype=float)

    below = values < low
    inside = (values >= low) & (values <= high)
    above = values > high

    dist = normalized_distance_to_interval(
        values,
        low,
        high
    )

    total_weight = weights.sum()

    if total_weight <= 0:
        weighted_distance = np.nan
    else:
        weighted_distance = np.average(
            dist,
            weights=weights
        )

    return {
        "field_min": float(np.min(values)),
        "field_max": float(np.max(values)),

        "below_pct": weighted_fraction(
            below,
            weights
        ),

        "inside_pct": weighted_fraction(
            inside,
            weights
        ),

        "above_pct": weighted_fraction(
            above,
            weights
        ),

        "mean_normalized_distance": weighted_distance,
    }


# ============================================================
# Exposure-weighted analysis
# ============================================================

rows = []

LABS = {
    "KIT": KIT_BOUNDS,
    "MIT": MIT_BOUNDS,
}

vehicles = sorted(field["vehicle"].unique())


for lab_name, bounds in LABS.items():

    for vehicle in vehicles:

        for variable, (low, high) in bounds.items():

            d = field[
                (field["vehicle"] == vehicle) &
                (field["variable"] == variable)
            ].copy()

            if d.empty:
                continue

            stats = analyse_variable(
                d,
                low,
                high
            )

            rows.append({
                "lab": lab_name,
                "vehicle": vehicle,
                "variable": variable,

                "lab_lower": low,
                "lab_upper": high,

                **stats,
            })


res = pd.DataFrame(rows)


# ============================================================
# Weighted aggregate across ALL vehicles
# ============================================================

aggregate_rows = []

for lab_name, bounds in LABS.items():

    for variable, (low, high) in bounds.items():

        d = field[
            field["variable"] == variable
        ].copy()

        if d.empty:
            continue

        stats = analyse_variable(
            d,
            low,
            high
        )

        aggregate_rows.append({
            "lab": lab_name,
            "vehicle": "ALL",
            "variable": variable,

            "lab_lower": low,
            "lab_upper": high,

            **stats,
        })


aggregate = pd.DataFrame(aggregate_rows)

full = pd.concat(
    [res, aggregate],
    ignore_index=True
)


# ============================================================
# Save main table
# ============================================================

out_main = (
    OUT_DIR /
    "exposure_weighted_lab_field_representativeness.csv"
)

full.to_csv(
    out_main,
    index=False
)


# ============================================================
# Summary table
# ============================================================

summary = (
    full[
        full["vehicle"] == "ALL"
    ][
        [
            "lab",
            "variable",
            "lab_lower",
            "lab_upper",
            "field_min",
            "field_max",
            "below_pct",
            "inside_pct",
            "above_pct",
            "mean_normalized_distance",
        ]
    ]
    .copy()
)

out_summary = (
    OUT_DIR /
    "exposure_weighted_summary.csv"
)

summary.to_csv(
    out_summary,
    index=False
)


# ============================================================
# Per-vehicle C-rate stress classification
# ============================================================

crate = full[
    full["variable"] == "C_rate"
].copy()

out_crate = (
    OUT_DIR /
    "exposure_weighted_Crate_by_vehicle.csv"
)

crate.to_csv(
    out_crate,
    index=False
)


# ============================================================
# Print results
# ============================================================

print("\n============================================================")
print("LAB DOMAINS")
print("============================================================")

print("\nKIT:")
for k, v in KIT_BOUNDS.items():
    print(f"{k:10s}: {v[0]:.4f} -> {v[1]:.4f}")

print("\nMIT:")
for k, v in MIT_BOUNDS.items():
    print(f"{k:10s}: {v[0]:.4f} -> {v[1]:.4f}")


print("\n============================================================")
print("EXPOSURE-WEIGHTED LAB-to-FIELD REPRESENTATIVENESS")
print("ALL VEHICLES")
print("============================================================")

print(
    summary.round(3).to_string(index=False)
)


print("\n============================================================")
print("C-RATE: PER-VEHICLE FIELD EXPOSURE")
print("============================================================")

print(
    crate[
        [
            "lab",
            "vehicle",
            "lab_lower",
            "lab_upper",
            "below_pct",
            "inside_pct",
            "above_pct",
            "mean_normalized_distance",
        ]
    ]
    .round(3)
    .to_string(index=False)
)


print("\nSaved:")
print(out_main)
print(out_summary)
print(out_crate)
