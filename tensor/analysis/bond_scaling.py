import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

folder = "results_F1_bondN_theta045_cut1e8_chi512"
files = sorted(glob.glob(f"{folder}/*_steady_summary.csv"))

if not files:
    raise SystemExit(f"No hay CSV en {folder}")

rows = []
for f in files:
    d = pd.read_csv(f)
    if len(d) == 0:
        continue
    r = d.iloc[0].to_dict()
    r["file"] = f
    m = re.search(r"seed(\d+)", f)
    r["seed"] = int(m.group(1)) if m else None
    rows.append(r)

df = pd.DataFrame(rows)
df = df.sort_values(["N", "seed"])
df.to_csv("F1_raw_bond_vs_N_theta045_cut1e8_chi512.csv", index=False)

summary = (
    df.groupby("N")
      .agg(
          n_runs=("BOND_peak_all", "count"),
          chi_mean=("BOND_peak_all", "mean"),
          chi_std=("BOND_peak_all", "std"),
          chi_min=("BOND_peak_all", "min"),
          chi_max=("BOND_peak_all", "max"),
          saturated_any=("saturated", "any"),
          early_stop_any=("early_stopped_any", "any"),
          sec_cycle_mean=("sec_per_traj_cycle", "mean"),
          maxdim=("maxdim", "max"),
          cutoff=("cutoff", "first"),
      )
      .reset_index()
      .sort_values("N")
)
summary["chi_std"] = summary["chi_std"].fillna(0.0)
summary.to_csv("F1_summary_bond_vs_N_theta045_cut1e8_chi512.csv", index=False)

print("\nResumen Figura 1:")
print(summary.to_string(index=False))

plt.figure(figsize=(7.2, 4.6))

ok = summary[summary["saturated_any"] == False]
sat = summary[summary["saturated_any"] == True]

plt.errorbar(
    ok["N"], ok["chi_mean"], yerr=ok["chi_std"],
    marker="o", capsize=3, linestyle="-",
    label=r"$\chi_{\rm peak}$, media de 3 semillas"
)

if len(sat):
    plt.errorbar(
        sat["N"], sat["chi_mean"], yerr=sat["chi_std"],
        marker="x", capsize=3, linestyle="none",
        label="saturado / cota inferior"
    )

plt.axhline(512, linestyle="--", linewidth=1, label=r"$\chi_{\max}=512$")
plt.axhline(0.9*512, linestyle=":", linewidth=1, label=r"$0.9\chi_{\max}$")

plt.xlabel(r"$N$")
plt.ylabel(r"Bond dimension pico $\chi_{\rm peak}$")
plt.title(r"$\theta=0.45$, NCYCLES=4, cutoff=$10^{-8}$")
plt.legend()
plt.tight_layout()
plt.savefig("F1_bond_peak_vs_N_theta045_cut1e8_chi512.png", dpi=300)
plt.savefig("F1_bond_peak_vs_N_theta045_cut1e8_chi512.pdf")

print("\nGuardado:")
print("  F1_raw_bond_vs_N_theta045_cut1e8_chi512.csv")
print("  F1_summary_bond_vs_N_theta045_cut1e8_chi512.csv")
print("  F1_bond_peak_vs_N_theta045_cut1e8_chi512.png")
print("  F1_bond_peak_vs_N_theta045_cut1e8_chi512.pdf")
