
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
OBS = ["E", "MZ", "MX", "ZZ"]

SPEC = {
    0.35: {"ncycles": 1250, "reps": [801,802,803,804,805,811,812,813]},
    0.45: {"ncycles": 800,  "reps": [801,802,803,804,805,806,807,811]},
    0.55: {"ncycles": 550,  "reps": [801,802,803,804,805,806,807,808]},
    0.65: {"ncycles": 400,  "reps": [801,802,803,804,805,806,807,808]},
}

def sem(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy()
    return np.nan if len(x) <= 1 else x.std(ddof=1)/np.sqrt(len(x))

def std(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy()
    return np.nan if len(x) <= 1 else x.std(ddof=1)

def fit_line(x, y, yerr=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)[mask]
    if yerr is not None and len(x) > 2 and np.all(np.isfinite(yerr)) and np.all(yerr > 0):
        coef, cov = np.polyfit(x, y, 1, w=1/yerr, cov=True)
    elif len(x) > 2:
        coef, cov = np.polyfit(x, y, 1, cov=True)
    else:
        coef = np.polyfit(x, y, 1)
        cov = np.full((2,2), np.nan)
    slope, intercept = coef
    yhat = slope*x + intercept
    ss_res = np.sum((y-yhat)**2)
    ss_tot = np.sum((y-y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, (np.sqrt(cov[1,1]) if np.isfinite(cov[1,1]) else np.nan), r2

def find_results_dir(root: Path) -> Path:
    candidates = [
        root / "results_N20",
        root / "export_N20_for_chatgpt" / "results_N20",
        root,
    ]
    for c in candidates:
        if c.exists() and any(c.glob("N20_squeeze_theta*_theta_scaling.csv")):
            return c
    raise SystemExit(f"No N20 squeeze theta_scaling CSVs found under {root}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".", help="N20_work or exported folder")
    ap.add_argument("--outdir", default="analysis_repro_outputs/theta2_balanced8")
    args = ap.parse_args()

    root = Path(args.root)
    results = find_results_dir(root)
    outdir = Path(args.outdir)
    plotdir = outdir / "plots"
    tabledir = outdir / "tables"
    plotdir.mkdir(parents=True, exist_ok=True)
    tabledir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for theta, info in SPEC.items():
        thtag = f"{int(round(theta*100)):03d}"
        ncy = info["ncycles"]
        for rep in info["reps"]:
            f = results / f"N20_squeeze_theta{thtag}_rt075_chi512_xchi384_cut1e5_xcut3e5_ncy{ncy}_rep{rep}_theta_scaling.csv"
            if not f.exists():
                missing.append(str(f))
                continue
            df = pd.read_csv(f)
            row = df.iloc[0].to_dict()
            row["rep"] = rep
            row["file"] = f.name
            rows.append(row)

    individual = pd.DataFrame(rows)
    individual.to_csv(tabledir / "N20_balanced8_individual.csv", index=False)
    if missing:
        (tabledir / "MISSING.txt").write_text("\n".join(missing), encoding="utf-8")

    if individual.empty:
        raise SystemExit("No rows loaded.")

    summary_rows = []
    for theta, g in individual.groupby("theta"):
        row = {
            "theta": float(theta),
            "theta2": float(g["theta2"].iloc[0]),
            "ncycles": int(g["ncycles"].iloc[0]),
            "n_reps": len(g),
            "any_saturated": bool(g["saturated"].astype(bool).any()),
            "any_early_stopped": bool(g["early_stopped_any"].astype(bool).any()),
        }
        for obs in OBS:
            for suffix in ["mean", "abs_error", "mean_abs_drift"]:
                col = f"{obs}_{suffix}"
                vals = pd.to_numeric(g[col], errors="coerce")
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_std"] = std(vals)
                row[f"{col}_sem"] = sem(vals)
        for col in ["BOND_peak_all", "BOND_mean", "BOND_max", "walltime_sec", "sec_per_traj_cycle"]:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce")
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_std"] = std(vals)
                row[f"{col}_sem"] = sem(vals)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("theta")
    summary.to_csv(tabledir / "N20_balanced8_summary_by_theta.csv", index=False)

    fit_rows = []
    for obs in OBS:
        x = summary["theta2"].to_numpy()
        y = summary[f"{obs}_mean_mean"].to_numpy()
        yerr = summary[f"{obs}_mean_sem"].to_numpy()
        slope, O0, O0err, r2 = fit_line(x, y)
        slopew, O0w, O0werr, r2w = fit_line(x, y, yerr)
        fit_rows.append({
            "obs": obs,
            "O0_OLS": O0,
            "O0_OLS_err": O0err,
            "slope_OLS": slope,
            "r2_OLS": r2,
            "O0_weighted": O0w,
            "O0_weighted_err": O0werr,
            "slope_weighted": slopew,
            "r2_weighted": r2w,
            "Gibbs_ref": REFS[obs],
            "abs_O0_OLS_minus_Gibbs": abs(O0-REFS[obs]),
            "abs_O0_weighted_minus_Gibbs": abs(O0w-REFS[obs]),
        })
    fits = pd.DataFrame(fit_rows)
    fits.to_csv(tabledir / "N20_balanced8_fits.csv", index=False)

    # Main individual plots
    for obs in OBS:
        fig, ax = plt.subplots(figsize=(8,5.5))
        for th, g in individual.groupby("theta"):
            ax.scatter(g["theta2"], g[f"{obs}_mean"], alpha=0.45, s=24)
        ax.errorbar(summary["theta2"], summary[f"{obs}_mean_mean"], yerr=summary[f"{obs}_mean_sem"],
                    fmt="o", capsize=4, label="media ± SEM")
        slope, O0, _, r2 = fit_line(summary["theta2"], summary[f"{obs}_mean_mean"])
        xfit = np.linspace(0, summary["theta2"].max()*1.05, 300)
        ax.plot(xfit, slope*xfit + O0, label=f"fit lineal, R²={r2:.2f}")
        ax.axhline(REFS[obs], linestyle="--", label="Gibbs MPS")
        for _, r in summary.iterrows():
            ax.annotate(f"θ={r['theta']:.2f}", (r["theta2"], r[f"{obs}_mean_mean"]),
                        textcoords="offset points", xytext=(5,5), fontsize=8)
        ax.set_xlabel(r"$\theta^2$")
        ax.set_ylabel(obs)
        ax.set_title(f"N=20 balanced8: {obs} vs $\\theta^2$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plotdir / f"N20_balanced8_{obs}_vs_theta2.png", dpi=180)
        plt.close(fig)

    # Multipanel
    fig, axes = plt.subplots(2,2,figsize=(11,8))
    for ax, obs in zip(axes.ravel(), OBS):
        x = summary["theta2"].to_numpy()
        y = summary[f"{obs}_mean_mean"].to_numpy()
        yerr = summary[f"{obs}_mean_sem"].to_numpy()
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
        slope, O0, _, _ = fit_line(x, y)
        xfit = np.linspace(0, max(x)*1.05, 300)
        ax.plot(xfit, slope*xfit + O0)
        ax.axhline(REFS[obs], linestyle="--")
        ax.set_title(obs)
        ax.set_xlabel(r"$\theta^2$")
        ax.set_ylabel(obs)
    fig.suptitle("N=20 balanced8: observables vs $\\theta^2$")
    fig.tight_layout()
    fig.savefig(plotdir / "N20_balanced8_observables_multipanel.png", dpi=180)
    plt.close(fig)

    # Diagnostics
    for col, ylabel, fname, title in [
        ("E_abs_error_mean", "|E - E_Gibbs|", "N20_balanced8_E_abs_error_vs_theta2.png", "N=20: error energético"),
        ("E_mean_abs_drift_mean", "E drift", "N20_balanced8_E_drift_vs_theta2.png", "N=20: drift energético"),
        ("BOND_peak_all_mean", "BOND_peak_all", "N20_balanced8_BOND_peak_vs_theta2.png", "N=20: bond pico"),
    ]:
        fig, ax = plt.subplots(figsize=(8,5.5))
        sem_col = col.replace("_mean", "_sem")
        yerr = summary[sem_col] if sem_col in summary.columns else None
        ax.errorbar(summary["theta2"], summary[col], yerr=yerr, fmt="o-", capsize=4)
        ax.set_xlabel(r"$\theta^2$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(plotdir / fname, dpi=180)
        plt.close(fig)

    print(f"Missing files: {len(missing)}")
    print(summary[["theta","theta2","ncycles","n_reps","E_mean_mean","E_mean_sem","E_abs_error_mean","E_mean_abs_drift_mean","BOND_peak_all_mean","any_saturated","any_early_stopped"]].to_string(index=False))
    print("\nFits:")
    print(fits[["obs","O0_OLS","Gibbs_ref","abs_O0_OLS_minus_Gibbs","slope_OLS","r2_OLS"]].to_string(index=False))
    print(f"\nSaved outputs to {outdir}")

if __name__ == "__main__":
    main()
