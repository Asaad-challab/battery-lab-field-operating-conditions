from pathlib import Path
import pandas as pd
import re

ROOT = Path.home() / "battery/paper_C"
f = ROOT / "tables/MIT_LFP_2018_policy_summary.csv"

df = pd.read_csv(f)

def parse_policy(s):
    """
    Example:
    5.6C(19%)-4.6C-newstructure
       -> C1=5.6, Q1=19, C2=4.6
    """
    m = re.search(
        r"([0-9.]+)C\(([0-9.]+)%\)-([0-9.]+)C",
        str(s)
    )

    if not m:
        return pd.Series([None, None, None])

    return pd.Series([
        float(m.group(1)),
        float(m.group(2)),
        float(m.group(3)),
    ])

df[["C1", "Q1_pct", "C2"]] = df["charge_policy"].apply(parse_policy)

# Experimental protocol:
# C1 from 0 -> Q1 SOC
# C2 from Q1 -> 80% SOC
# then 1C CC-CV above 80% SOC.
df["fast_charge_SOC_start_pct"] = 0.0
df["fast_charge_SOC_end_pct"] = 80.0
df["post80_C_rate"] = 1.0

out = ROOT / "tables/MIT_LFP_lab_domain.csv"
df.to_csv(out, index=False)

print("Saved:", out)
print("Shape:", df.shape)

print("\nUNIQUE MIT POLICIES:")
cols = ["charge_policy", "C1", "Q1_pct", "C2"]

print(
    df[cols]
    .drop_duplicates()
    .sort_values(["C1", "Q1_pct"])
    .to_string(index=False)
)

print("\nMIT FAST-CHARGE DOMAIN")
print("C1:", df["C1"].min(), "to", df["C1"].max(), "C")
print("C2:", df["C2"].min(), "to", df["C2"].max(), "C")
print("Q1:", df["Q1_pct"].min(), "to", df["Q1_pct"].max(), "% SOC")
print(
    "Measured cell Tmax:",
    round(df["Tmax_global_C"].min(),2),
    "to",
    round(df["Tmax_global_C"].max(),2),
    "°C"
)
