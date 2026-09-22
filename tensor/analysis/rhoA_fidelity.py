#!/usr/bin/env python3
import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def infer_N(path, df):
    if "N" in df.columns:
        vals = df["N"].dropna().unique()
        if len(vals) > 0:
            return int(vals[0])
    m = re.search(r"N(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    raise ValueError(f"No puedo inferir N de {path}")


def infer_rep(path):
    m = re.search(r"rep(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def first_sustained(y, t, threshold, window, below=True):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    for i in range(0, len(y) - window + 1):
        block = y[i:i + window]
        if not np.all(np.isfinite(block)):
            continue
        ok = np.all(block <= threshold) if below else np.all(block >= threshold)
        if ok:
            return float(t[i])
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results_fidelity_T1")
    ap.add_argument("--outdir", default="plots_fidelity_T1_final")
    ap.add_argument("--steady-frac", type=float, default=0.8)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--plateau-tol", type=float, default=0.05)
    ap.add_argument("--eps-infid", type=float, default=0.05)
    ap.add_argument("--eps-D", type=float, default=0.05)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input, "**", "*rhoA_trace*timeseries.csv"), recursive=True))
    if not files:
        raise SystemExit("No encuentro timeseries CSV en results_fidelity_T1/")

    all_dfs = []
    rep_rows = []

    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            continue

        N = infer_N(f, df)
        rep = infer_rep(f)

        df["N"] = N
        df["rep"] = rep
        df["source_file"] = f

        if "cycle" not in df.columns:
            raise ValueError(f"Falta columna cycle en {f}")

        if "time_filter" not in df.columns:
            df["time_filter"] = df["cycle"]

        if "F_rhoA_mean" not in df.columns:
            raise ValueError(f"Falta F_rhoA_mean en {f}")

        if "trace_distance_rhoA_mean" not in df.columns:
            raise ValueError(f"Falta trace_distance_rhoA_mean en {f}")

        df["infidelity"] = 1.0 - df["F_rhoA_mean"]

        cmax = df["cycle"].max()
        c0 = args.steady_frac * cmax
        ss = df[df["cycle"] >= c0]

        rep_rows.append({
            "N": N,
            "rep": rep,
            "file": f,
            "n_points": len(df),
            "cycle_max": float(df["cycle"].max()),
            "F_ss_rep": float(ss["F_rhoA_mean"].mean()),
            "infid_ss_rep": float(ss["infidelity"].mean()),
            "D_ss_rep": float(ss["trace_distance_rhoA_mean"].mean()),
            "F_final_rep": float(df["F_rhoA_mean"].iloc[-1]),
            "infid_final_rep": float(df["infidelity"].iloc[-1]),
            "D_final_rep": float(df["trace_distance_rhoA_mean"].iloc[-1]),
            "bond_peak_rep": float(df["BOND_mean"].max()) if "BOND_mean" in df.columns else np.nan,
        })

        all_dfs.append(df)

    data = pd.concat(all_dfs, ignore_index=True)
    rep_summary = pd.DataFrame(rep_rows).sort_values(["N", "rep"])
    rep_summary.to_csv(outdir / "fidelity_replicate_summary.csv", index=False)

    final_rows = []

    for N, dN in sorted(data.groupby("N")):
        curve = dN.groupby("cycle").agg(
            time=("time_filter", "mean"),
            F_mean=("F_rhoA_mean", "mean"),
            F_sem=("F_rhoA_mean", sem),
            D_mean=("trace_distance_rhoA_mean", "mean"),
            D_sem=("trace_distance_rhoA_mean", sem),
            infid_mean=("infidelity", "mean"),
            infid_sem=("infidelity", sem),
        ).reset_index()

        reps = rep_summary[rep_summary["N"] == N]
        ntraj = len(reps)

        F_ss = float(reps["F_ss_rep"].mean())
        F_ss_sem = sem(reps["F_ss_rep"])
        infid_ss = float(reps["infid_ss_rep"].mean())
        infid_ss_sem = sem(reps["infid_ss_rep"])
        D_ss = float(reps["D_ss_rep"].mean())
        D_ss_sem = sem(reps["D_ss_rep"])

        # Tiempo de termalización estricto por infidelidad absoluta
        t_infid_abs = first_sustained(
            curve["infid_mean"],
            curve["time"],
            args.eps_infid,
            args.window,
            below=True,
        )
        cyc_infid_abs = first_sustained(
            curve["infid_mean"],
            curve["cycle"],
            args.eps_infid,
            args.window,
            below=True,
        )

        # Tiempo por distancia de traza absoluta
        t_D_abs = first_sustained(
            curve["D_mean"],
            curve["time"],
            args.eps_D,
            args.window,
            below=True,
        )
        cyc_D_abs = first_sustained(
            curve["D_mean"],
            curve["cycle"],
            args.eps_D,
            args.window,
            below=True,
        )

        # Tiempo de entrada al plateau estacionario
        infid_plateau_threshold = infid_ss * (1.0 + args.plateau_tol)
        D_plateau_threshold = D_ss * (1.0 + args.plateau_tol)

        t_infid_plateau = first_sustained(
            curve["infid_mean"],
            curve["time"],
            infid_plateau_threshold,
            args.window,
            below=True,
        )
        cyc_infid_plateau = first_sustained(
            curve["infid_mean"],
            curve["cycle"],
            infid_plateau_threshold,
            args.window,
            below=True,
        )

        t_D_plateau = first_sustained(
            curve["D_mean"],
            curve["time"],
            D_plateau_threshold,
            args.window,
            below=True,
        )
        cyc_D_plateau = first_sustained(
            curve["D_mean"],
            curve["cycle"],
            D_plateau_threshold,
            args.window,
            below=True,
        )

        curve.to_csv(outdir / f"N{N}_averaged_fidelity_timeseries.csv", index=False)

        final_rows.append({
            "N": N,
            "ntraj": ntraj,
            "F_ss": F_ss,
            "F_ss_sem": F_ss_sem,
            "infidelity_ss": infid_ss,
            "infidelity_ss_sem": infid_ss_sem,
            "D_ss": D_ss,
            "D_ss_sem": D_ss_sem,
            "eps_infid": args.eps_infid,
            "t_th_infid_abs": t_infid_abs,
            "cycle_th_infid_abs": cyc_infid_abs,
            "eps_D": args.eps_D,
            "t_th_D_abs": t_D_abs,
            "cycle_th_D_abs": cyc_D_abs,
            "plateau_tol": args.plateau_tol,
            "infid_plateau_threshold": infid_plateau_threshold,
            "t_th_infid_plateau": t_infid_plateau,
            "cycle_th_infid_plateau": cyc_infid_plateau,
            "D_plateau_threshold": D_plateau_threshold,
            "t_th_D_plateau": t_D_plateau,
            "cycle_th_D_plateau": cyc_D_plateau,
            "bond_peak_mean": float(reps["bond_peak_rep"].mean()),
            "bond_peak_max": float(reps["bond_peak_rep"].max()),
            "n_points": int(curve.shape[0]),
        })

        plt.figure()
        plt.errorbar(curve["cycle"], curve["infid_mean"], yerr=curve["infid_sem"], linewidth=1, marker=".", markersize=2, capsize=2)
        plt.axhline(args.eps_infid, linestyle="--", label=f"abs eps={args.eps_infid}")
        plt.axhline(infid_plateau_threshold, linestyle=":", label="plateau")
        plt.xlabel("cycle")
        plt.ylabel(r"$1-F_A(t)$")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"N{N}_infidelity_vs_cycle.png", dpi=260)
        plt.close()

        plt.figure()
        plt.errorbar(curve["cycle"], curve["D_mean"], yerr=curve["D_sem"], linewidth=1, marker=".", markersize=2, capsize=2)
        plt.axhline(args.eps_D, linestyle="--", label=f"abs eps={args.eps_D}")
        plt.axhline(D_plateau_threshold, linestyle=":", label="plateau")
        plt.xlabel("cycle")
        plt.ylabel(r"$D_A(t)$")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"N{N}_trace_distance_vs_cycle.png", dpi=260)
        plt.close()

        plt.figure()
        plt.errorbar(curve["cycle"], curve["F_mean"], yerr=curve["F_sem"], linewidth=1, marker=".", markersize=2, capsize=2)
        plt.xlabel("cycle")
        plt.ylabel(r"$F_A(t)$")
        plt.tight_layout()
        plt.savefig(outdir / f"N{N}_fidelity_vs_cycle.png", dpi=260)
        plt.close()

    summary = pd.DataFrame(final_rows).sort_values("N")
    summary.to_csv(outdir / "fidelity_T1_N_scaling_summary.csv", index=False)

    print(summary.to_string(index=False))

    plt.figure()
    plt.errorbar(summary["N"], summary["F_ss"], yerr=summary["F_ss_sem"], marker="o", capsize=3)
    plt.xlabel("N")
    plt.ylabel(r"$F_A^{ss}$")
    plt.tight_layout()
    plt.savefig(outdir / "fidelity_vs_N.png", dpi=300)
    plt.close()

    plt.figure()
    plt.errorbar(summary["N"], summary["infidelity_ss"], yerr=summary["infidelity_ss_sem"], marker="o", capsize=3)
    plt.xlabel("N")
    plt.ylabel(r"$1-F_A^{ss}$")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(outdir / "infidelity_vs_N.png", dpi=300)
    plt.close()

    plt.figure()
    plt.errorbar(summary["N"], summary["D_ss"], yerr=summary["D_ss_sem"], marker="o", capsize=3)
    plt.xlabel("N")
    plt.ylabel(r"$D_A^{ss}$")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(outdir / "trace_distance_ss_vs_N.png", dpi=300)
    plt.close()

    good = summary[np.isfinite(summary["t_th_infid_plateau"])]
    if not good.empty:
        plt.figure()
        plt.plot(good["N"], good["t_th_infid_plateau"], marker="o")
        plt.xlabel("N")
        plt.ylabel(r"$t_{\rm th}$ from fidelity plateau")
        plt.tight_layout()
        plt.savefig(outdir / "thermalization_time_vs_N_from_fidelity_plateau.png", dpi=300)
        plt.close()

    good = summary[np.isfinite(summary["t_th_infid_abs"])]
    if not good.empty:
        plt.figure()
        plt.plot(good["N"], good["t_th_infid_abs"], marker="o")
        plt.xlabel("N")
        plt.ylabel(r"$t_{\rm th}$ from $1-F_A<\epsilon$")
        plt.tight_layout()
        plt.savefig(outdir / "thermalization_time_vs_N_from_fidelity_abs.png", dpi=300)
        plt.close()

    good = summary[np.isfinite(summary["t_th_D_plateau"])]
    if not good.empty:
        plt.figure()
        plt.plot(good["N"], good["t_th_D_plateau"], marker="o")
        plt.xlabel("N")
        plt.ylabel(r"$t_{\rm th}$ from trace-distance plateau")
        plt.tight_layout()
        plt.savefig(outdir / "thermalization_time_vs_N_from_D_plateau.png", dpi=300)
        plt.close()

    good = summary[np.isfinite(summary["t_th_D_abs"])]
    if not good.empty:
        plt.figure()
        plt.plot(good["N"], good["t_th_D_abs"], marker="o")
        plt.xlabel("N")
        plt.ylabel(r"$t_{\rm th}$ from $D_A<\epsilon$")
        plt.tight_layout()
        plt.savefig(outdir / "thermalization_time_vs_N_from_D_abs.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
