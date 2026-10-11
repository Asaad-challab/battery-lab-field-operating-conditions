from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.home() / "battery/paper_C"

kit = pd.read_csv(ROOT / "tables/KIT_lab_domain.csv")
mit = pd.read_csv(ROOT / "tables/MIT_LFP_lab_domain.csv")
tum = pd.read_csv(ROOT / "tables/TUM_field_profiles.csv")

rows = []

# ============================================================
# KIT
# ============================================================

rows.append({
    "dataset": "KIT",
    "source_type": "Lab",
    "chemistry": "NMC/C-SiO",
    "scale": "Cell",
    "temperature_min_C": kit["temperature_C"].min(),
    "temperature_max_C": kit["temperature_C"].max(),
    "SOC_min_pct": kit["SOC_min_pct"].min(),
    "SOC_max_pct": kit["SOC_max_pct"].max(),
    "charge_C_min": kit["charge_C_rate"].min(),
    "charge_C_max": kit["charge_C_rate"].max(),
    "note": "Controlled factorial aging protocol"
})

# ============================================================
# MIT
# ============================================================

rows.append({
    "dataset": "MIT",
    "source_type": "Lab",
    "chemistry": "LFP/graphite",
    "scale": "Cell",
    "temperature_min_C": mit["Tmin_global_C"].min(),
    "temperature_max_C": mit["Tmax_global_C"].max(),
    "SOC_min_pct": 0.0,
    "SOC_max_pct": 100.0,
    "charge_C_min": min(
        mit["C1"].min(),
        mit["C2"].min(),
        mit["post80_C_rate"].min()
    ),
    "charge_C_max": max(
        mit["C1"].max(),
        mit["C2"].max(),
        mit["post80_C_rate"].max()
    ),
    "note": "Fast-charge aging protocol; 0-80% high-rate then 1C CC-CV"
})

# ============================================================
# TUM charging field
# ============================================================

chg = tum[tum["state"] == "charging"].copy()

def weighted_quantile(values, weights, q):
    values = np.asarray(values)
    weights = np.asarray(weights)

    idx = np.argsort(values)
    values = values[idx]
    weights = weights[idx]

    cdf = np.cumsum(weights) / np.sum(weights)
    return np.interp(q, cdf, values)

for vehicle in sorted(chg["vehicle"].unique()):

    v = chg[chg["vehicle"] == vehicle]

    soc = v[v["variable"] == "SOC"]
    cr  = v[v["variable"] == "C_rate"]
    tmp = v[v["variable"] == "T_max"]

    rows.append({
        "dataset": f"TUM_{vehicle}",
        "source_type": "Field",
        "chemistry": "Production EV pack",
        "scale": "Pack",
        "temperature_min_C": tmp["x_value"].min(),
        "temperature_max_C": tmp["x_value"].max(),
        "SOC_min_pct": soc["x_value"].min(),
        "SOC_max_pct": soc["x_value"].max(),
        "charge_C_min": cr["x_value"].min(),
        "charge_C_max": cr["x_value"].max(),

        "SOC_p05": weighted_quantile(
            soc["x_value"], soc["weight"], 0.05
        ),
        "SOC_p50": weighted_quantile(
            soc["x_value"], soc["weight"], 0.50
        ),
        "SOC_p95": weighted_quantile(
            soc["x_value"], soc["weight"], 0.95
        ),

        "C_rate_p05": weighted_quantile(
            cr["x_value"], cr["weight"], 0.05
        ),
        "C_rate_p50": weighted_quantile(
            cr["x_value"], cr["weight"], 0.50
        ),
        "C_rate_p95": weighted_quantile(
            cr["x_value"], cr["weight"], 0.95
        ),

        "Tmax_p05": weighted_quantile(
            tmp["x_value"], tmp["weight"], 0.05
        ),
        "Tmax_p50": weighted_quantile(
            tmp["x_value"], tmp["weight"], 0.50
        ),
        "Tmax_p95": weighted_quantile(
            tmp["x_value"], tmp["weight"], 0.95
        ),

        "note": "Real EV charging exposure"
    })

out = pd.DataFrame(rows)

outfile = ROOT / "tables/cross_dataset_operating_domain.csv"
out.to_csv(outfile, index=False)

print("Saved:", outfile)
print("\n")
print(out.round(2).to_string(index=False))
