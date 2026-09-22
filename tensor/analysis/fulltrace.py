
#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REFS = {
    "E": -1.084500814291,
    "MZ": -0.159938170214,
    "MX": -0.535583127785,
    "ZZ": -0.495448311060,
}
N = 20

def find_results_dir(root: Path) -> Path:
    candidates = [
        root / "results_N20",
        root / "export_N20_fulltrace_theta045" / "results_N20",
        root,
    ]
    for c in candidates:
        if c.exists() and any(c.glob("N20_fulltrace_theta045*_theta_samples_samples.csv")):
            return c
    raise SystemExit(f"No fulltrace sample CSVs found under {root}")

def rep_from_name(name: str) -> int:
    m = re.search(r"_rep(\d+)", name)
    return int(m.group(1)) if m else -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".", help="N20_work or exported folder")
    ap.add_argument("--outdir", default="analysis_repro_outputs/fulltrace_theta045")
    ap.add_argument("--steady-frac", type=float, default=2/3)
    ap.add_argument("--errorbar-stride", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.root)
    results = find_results_dir(root)
    outdir = Path(args.outdir)
    plotdir = outdir / "plots"
    tabledir = outdir / "tables"
    plotdir.mkdir(parents=True, exist_ok=True)
    tabledir.mkdir(parents=True, exist_ok=True)

    files = sorted(results.glob("N20_fulltrace_theta045_rt075_chi512_xchi384_cut1e5_xcut3e5_ncy731_rep*_theta0.45_ncy731_theta_samples_samples.csv"))
    if not files:
        files = sorted(results.glob("N20_fulltrace_theta045*_theta_samples_samples.csv"))

    rows = []
    validation = []
    for f in files:
        df = pd.read_csv(f)
        rep = rep_from_name(f.name)
        required = ["cycle", "E", "MZ", "MX", "ZZ", "BOND"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SystemExit(f"{f.name} is missing columns {missing}. Available columns: {list(df.columns)}")
        df["rep"] = rep
        df["file"] = f.name
        rows.append(df)
        validation.append({
            "file": f.name,
            "rep": rep,
            "cycle_min": int(df["cycle"].min()),
            "cycle_max": int(df["cycle"].max()),
            "nrows": len(df),
            "full_trace_ok": (df["cycle"].min() == 1 and df["cycle"].max() == 731 and len(df) == 731),
        })

    if not rows:
        raise SystemExit("No CSVs loaded.")

    data = pd.concat(rows, ignore_index=True)
    data["E_over_N"] = data["E"] / N
    data.to_csv(tabledir / "N20_fulltrace_theta045_individual_samples.csv", index=False)
    pd.DataFrame(validation).to_csv(tabledir / "N20_fulltrace_validation.csv", index=False)

    # Summary by cycle
    agg = data.groupby("cycle")
    summary = pd.DataFrame({"cycle": sorted(data["cycle"].unique())})
    for col in ["E_over_N", "E", "MZ", "MX", "ZZ", "BOND"]:
        g = agg[col]
        tmp = pd.DataFrame({
            "cycle": g.mean().index,
            f"{col}_mean": g.mean().values,
            f"{col}_std": g.std().values,
            f"{col}_count": g.count().values,
        })
        tmp[f"{col}_sem"] = tmp[f"{col}_std"] / np.sqrt(tmp[f"{col}_count"])
        summary = summary.merge(tmp, on="cycle", how="left")
    summary.to_csv(tabledir / "N20_fulltrace_theta045_cycle_summary.csv", index=False)

    max_cycle = int(summary["cycle"].max())
    steady_start = int(np.floor(args.steady_frac * max_cycle)) + 1

    # Energy main plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for rep, g in data.groupby("rep"):
        ax.plot(g["cycle"], g["E_over_N"], alpha=0.22, linewidth=1, label=f"rep {rep}")
    ax.plot(summary["cycle"], summary["E_over_N_mean"], linewidth=1.8, label="media")
    ax.axhline(REFS["E"] / N, linestyle="--", label="Gibbs MPS")
    ax.axvspan(steady_start, max_cycle, alpha=0.15, label="ventana steady")
    ax.set_xlabel("reset cycle")
    ax.set_ylabel("E/N")
    ax.set_title("N=20: energía vs reset cycle (θ=0.45)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plotdir / "N20_fulltrace_theta045_energy_vs_cycle.png", dpi=180)
    plt.close(fig)

    # Energy sparse errorbar plot
    stride = max(1, args.errorbar_stride)
    sparse = summary.iloc[::stride].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(summary["cycle"], summary["E_over_N_mean"], linewidth=1.5, label="media")
    ax.errorbar(sparse["cycle"], sparse["E_over_N_mean"], yerr=sparse["E_over_N_sem"],
                fmt="o", capsize=3, markersize=4, label=f"media ± SEM cada {stride} ciclos")
    ax.axhline(REFS["E"] / N, linestyle="--", label="Gibbs MPS")
    ax.axvspan(steady_start, max_cycle, alpha=0.15, label="ventana steady")
    ax.set_xlabel("reset cycle")
    ax.set_ylabel("E/N")
    ax.set_title("N=20: energía vs reset cycle (θ=0.45), errorbars espaciadas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plotdir / "N20_fulltrace_theta045_energy_vs_cycle_errorbars_sparse.png", dpi=180)
    plt.close(fig)

    # Multipanel observables
    obs_specs = [
        ("E_over_N", "E/N", REFS["E"] / N),
        ("MZ", "MZ", REFS["MZ"]),
        ("MX", "MX", REFS["MX"]),
        ("ZZ", "ZZ", REFS["ZZ"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, ylabel, ref) in zip(axes.ravel(), obs_specs):
        for rep, g in data.groupby("rep"):
            y = g[col] if col != "E_over_N" else g["E_over_N"]
            ax.plot(g["cycle"], y, alpha=0.18, linewidth=1)
        ax.plot(summary["cycle"], summary[f"{col}_mean"], linewidth=1.8)
        ax.axhline(ref, linestyle="--")
        ax.axvspan(steady_start, max_cycle, alpha=0.12)
        ax.set_title(ylabel)
        ax.set_xlabel("cycle")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    fig.suptitle("N=20 full trace θ=0.45: observables vs reset cycle")
    fig.tight_layout()
    fig.savefig(plotdir / "N20_fulltrace_theta045_observables_multipanel.png", dpi=180)
    plt.close(fig)

    # Bond plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for rep, g in data.groupby("rep"):
        ax.plot(g["cycle"], g["BOND"], alpha=0.22, linewidth=1)
    ax.plot(summary["cycle"], summary["BOND_mean"], linewidth=1.8, label="media")
    ax.fill_between(summary["cycle"], summary["BOND_mean"]-summary["BOND_sem"], summary["BOND_mean"]+summary["BOND_sem"], alpha=0.2, label="±SEM")
    ax.axvspan(steady_start, max_cycle, alpha=0.15, label="ventana steady")
    ax.set_xlabel("reset cycle")
    ax.set_ylabel("bond dimension")
    ax.set_title("N=20: bond dimension vs reset cycle (θ=0.45)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plotdir / "N20_fulltrace_theta045_bond_vs_cycle.png", dpi=180)
    plt.close(fig)

    print("Loaded files:")
    for f in files:
        print(" ", f)
    print(f"Saved outputs to {outdir}")

if __name__ == "__main__":
    main()
