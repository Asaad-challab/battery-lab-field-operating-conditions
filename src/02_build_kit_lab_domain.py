from pathlib import Path
import pandas as pd

KIT_FILE = (
    Path.home()
    / "Data/battery/KIT_2024/result_v2/unpacked/10.35097-1969/data/dataset"
    / "KIT_76_conditions_decoded.csv"
)

OUT = Path.home() / "battery/paper_C/tables"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(KIT_FILE)

# For charging-domain comparison, use cyclic + profile aging.
lab = df[df["aging_type"].isin(["Cyclic", "Profile"])].copy()

# Keep variables relevant to the field comparison.
cols = [
    "parameter_id",
    "aging_type",
    "temperature_C",
    "SOC_min_pct",
    "SOC_max_pct",
    "charge_C_rate",
    "profile",
    "n_cells",
]

lab = lab[cols]

out_file = OUT / "KIT_lab_domain.csv"
lab.to_csv(out_file, index=False)

print("Saved:", out_file)
print("Shape:", lab.shape)

print("\nAGING TYPES:")
print(lab["aging_type"].value_counts())

print("\nTEMPERATURE LEVELS:")
print(sorted(lab["temperature_C"].dropna().unique()))

print("\nSOC WINDOWS:")
print(
    lab[["SOC_min_pct", "SOC_max_pct"]]
    .dropna()
    .drop_duplicates()
    .sort_values(["SOC_min_pct", "SOC_max_pct"])
    .to_string(index=False)
)

print("\nCHARGE C-RATES:")
print(sorted(lab["charge_C_rate"].dropna().unique()))

print("\nLAB ENVELOPE:")
print(
    "Temperature:",
    lab["temperature_C"].min(),
    "to",
    lab["temperature_C"].max(),
    "°C",
)
print(
    "SOC:",
    lab["SOC_min_pct"].min(),
    "to",
    lab["SOC_max_pct"].max(),
    "%"
)
print(
    "Charge C-rate:",
    lab["charge_C_rate"].min(),
    "to",
    lab["charge_C_rate"].max(),
    "C"
)
