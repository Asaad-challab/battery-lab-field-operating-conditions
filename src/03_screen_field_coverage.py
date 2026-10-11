from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.home() / "battery/paper_C"
FIELD_FILE = ROOT / "tables/TUM_field_profiles.csv"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FIELD_FILE)

# ============================================================
# Use CHARGING exposure only
# ============================================================
x = df[df["state"] == "charging"].copy()

# Clean only obvious histogram-edge numerical artefacts.
# Do NOT arbitrarily remove valid field observations.
x.loc[(x["variable"] == "SOC") & (x["x_value"] < 0), "x_value"] = 0.0
x.loc[(x["variable"] == "SOC") & (x["x_value"] > 100), "x_value"] = 100.0

x.loc[
    (x["variable"] == "C_rate") &
    (x["x_value"] < 0) &
    (x["x_value"] > -0.02),
    "x_value"
] = 0.0

# ============================================================
# KIT documented protocol envelope
#
# SOC:
#   KIT cycling windows span 0-100%.
#
# Temperature:
#   controlled operating setpoints span 0-40 C.
#
# C-rate:
#   nominal CC charging extends to 5/3 C.
#   Cyclic/profile CC-CV cutoff is ~0.1 C.
#
# IMPORTANT:
# These are protocol limits, not a measured KIT probability
# distribution. Therefore results are "screening coverage".
# ============================================================

bounds = {
    "SOC":    (0.0, 100.0),
    "C_rate": (0.10, 5/3),
    "T_max":  (0.0, 40.0),
}

rows = []

for vehicle in sorted(x["vehicle"].unique()):
    for variable, (low, high) in bounds.items():

        d = x[
            (x["vehicle"] == vehicle) &
            (x["variable"] == variable)
        ].copy()

        if d.empty:
            continue

        total = d["weight"].sum()

        inside = d[
            (d["x_value"] >= low) &
            (d["x_value"] <= high)
        ]["weight"].sum()

        below = d[
            d["x_value"] < low
        ]["weight"].sum()

        above = d[
            d["x_value"] > high
        ]["weight"].sum()

        rows.append({
            "vehicle": vehicle,
            "variable": variable,
            "KIT_lower": low,
            "KIT_upper": high,
            "field_min": d["x_value"].min(),
            "field_max": d["x_value"].max(),
            "coverage_pct": 100 * inside / total,
            "below_pct": 100 * below / total,
            "above_pct": 100 * above / total,
        })

res = pd.DataFrame(rows)

out = OUT_DIR / "screening_field_coverage.csv"
res.to_csv(out, index=False)

print("Saved:", out)

print("\n============================================================")
print("FIELD EXPOSURE INSIDE KIT DOCUMENTED PROTOCOL ENVELOPE")
print("============================================================")

for variable in bounds:
    print(f"\n--- {variable} ---")

    z = res[res["variable"] == variable].copy()

    print(
        z[
            [
                "vehicle",
                "field_min",
                "field_max",
                "coverage_pct",
                "below_pct",
                "above_pct",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

print("\n============================================================")
print("MEAN ACROSS VEHICLES")
print("============================================================")

summary = (
    res.groupby("variable")
       [["coverage_pct", "below_pct", "above_pct"]]
       .mean()
       .round(2)
)

print(summary)
