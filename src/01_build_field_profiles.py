from pathlib import Path
import json
import pandas as pd
import numpy as np

TUM_JSON = Path.home() / "Data/battery/TUM/electric-vehicle-uds-dataset/data/json"
OUT = Path.home() / "battery/paper_C/tables"
OUT.mkdir(parents=True, exist_ok=True)

variables = {
    900: "SOC",
    1288: "C_rate",
    1208: "T_min",
    1209: "T_max",
    15: "Ambient_T",
}

rows = []

for state in ["charging", "track"]:
    for value_id, variable in variables.items():
        files = sorted(TUM_JSON.glob(f"{state}_*_{value_id}.json"))

        for f in files:
            with open(f, "r") as fh:
                d = json.load(fh)

            x = np.asarray(d["data"]["x_values"], dtype=float)
            y = np.asarray(d["data"]["y_values"], dtype=float)

            valid = np.isfinite(x) & np.isfinite(y) & (y > 0)

            if not valid.any():
                continue

            x_valid = x[valid]
            y_valid = y[valid]

            total = y_valid.sum()

            for xv, yv in zip(x_valid, y_valid):
                rows.append({
                    "vehicle": d["vehicle_id"],
                    "state": state,
                    "value_id": value_id,
                    "variable": variable,
                    "x_value": xv,
                    "weight": yv,
                    "weight_norm": yv / total if total > 0 else np.nan,
                })

df = pd.DataFrame(rows)

out_file = OUT / "TUM_field_profiles.csv"
df.to_csv(out_file, index=False)

print("Saved:", out_file)
print("Shape:", df.shape)

print("\nVehicles:")
print(sorted(df["vehicle"].unique()))

print("\nVariables:")
print(df["variable"].value_counts())

print("\nCharging ranges:")
chg = df[df["state"] == "charging"]

summary = (
    chg.groupby(["vehicle", "variable"])["x_value"]
       .agg(["min", "max", "count"])
       .reset_index()
)

print(summary.to_string(index=False))
