from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

N = 20
REF_E = -1.084500814291
REF_E_OVER_N = REF_E / N

FIT_MIN = 1
FIT_MAX_LIST = [60, 80, 100, 120, 150]

OUT_PLOTS = Path("plots_N20")
OUT_RESULTS = Path("results_N20")
OUT_PLOTS.mkdir(exist_ok=True)
OUT_RESULTS.mkdir(exist_ok=True)

files = sorted(Path("results_N20").glob(
    "N20_fulltrace_theta045_rt075_chi512_xchi384_cut1e5_xcut3e5_ncy731_rep*_theta0.45_ncy731_theta_samples_samples.csv"
))

print("Files encontrados:")
for f in files:
    print(" ", f.name)

if not files:
    raise SystemExit("No fulltrace sample files found.")

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["E_over_N"] = df["E"] / N
    dfs.append(df[["cycle", "E_over_N"]])

all_df = pd.concat(dfs, ignore_index=True)

summary = all_df.groupby("cycle")["E_over_N"].agg(["mean", "std", "count"]).reset_index()
summary["sem"] = summary["std"] / np.sqrt(summary["count"])

summary.to_csv(OUT_RESULTS / "N20_theta045_relaxation_cycle_summary.csv", index=False)

def exp_relax(c, Einf, A, tau):
    return Einf + A * np.exp(-c / tau)

def r2_score(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

fit_rows = []

plt.figure(figsize=(9, 6))
plt.plot(summary["cycle"], summary["mean"], label="media fulltrace", linewidth=1.6)
plt.fill_between(
    summary["cycle"],
    summary["mean"] - summary["sem"],
    summary["mean"] + summary["sem"],
    alpha=0.20,
    label="± SEM"
)
plt.axhline(REF_E_OVER_N, linestyle="--", color="black", label="Gibbs MPS")

colors_x = np.linspace(0.15, 0.85, len(FIT_MAX_LIST))

for fit_max, cx in zip(FIT_MAX_LIST, colors_x):
    fit = summary[(summary["cycle"] >= FIT_MIN) & (summary["cycle"] <= fit_max)].copy()

    x = fit["cycle"].to_numpy(float)
    y = fit["mean"].to_numpy(float)

    # Estimaciones iniciales
    Einf0 = summary[summary["cycle"] > 500]["mean"].mean()
    A0 = y[0] - Einf0
    tau0 = 10.0

    try:
        popt, pcov = curve_fit(
            exp_relax,
            x,
            y,
            p0=[Einf0, A0, tau0],
            bounds=([-np.inf, -np.inf, 0.1], [np.inf, np.inf, 500.0]),
            maxfev=50000,
        )

        Einf, A, tau = popt
        perr = np.sqrt(np.diag(pcov))
        yhat = exp_relax(x, *popt)
        r2 = r2_score(y, yhat)
        rmse = np.sqrt(np.mean((y - yhat)**2))

        fit_rows.append({
            "fit_min": FIT_MIN,
            "fit_max": fit_max,
            "Einf": Einf,
            "Einf_err": perr[0],
            "A": A,
            "A_err": perr[1],
            "tau": tau,
            "tau_err": perr[2],
            "R2": r2,
            "RMSE": rmse,
            "Gibbs_E_over_N": REF_E_OVER_N,
            "abs_Einf_minus_Gibbs": abs(Einf - REF_E_OVER_N),
        })

        xfit = np.linspace(FIT_MIN, fit_max, 400)
        plt.plot(
            xfit,
            exp_relax(xfit, *popt),
            "--",
            linewidth=1.4,
            label=f"fit 1-{fit_max}: τ={tau:.2f}"
        )

        # Plot individual de cada ventana
        plt_ind = plt.figure(figsize=(8, 5.5))
        ax = plt_ind.gca()
        ax.plot(summary["cycle"], summary["mean"], label="media", linewidth=1.5)
        ax.fill_between(
            summary["cycle"],
            summary["mean"] - summary["sem"],
            summary["mean"] + summary["sem"],
            alpha=0.20,
            label="± SEM"
        )
        ax.plot(xfit, exp_relax(xfit, *popt), "--", label=f"fit exp, τ={tau:.2f}")
        ax.axhline(REF_E_OVER_N, linestyle="--", color="black", label="Gibbs MPS")
        ax.axvspan(FIT_MIN, fit_max, alpha=0.10, label="fit window")
        ax.set_xlabel("reset cycle")
        ax.set_ylabel("E/N")
        ax.set_title(f"N=20 θ=0.45: fit relajación, ciclos {FIT_MIN}-{fit_max}")
        ax.legend()
        plt_ind.tight_layout()
        plt_ind.savefig(OUT_PLOTS / f"N20_theta045_relaxation_fit_1_{fit_max}.png", dpi=170)
        plt.close(plt_ind)

    except Exception as e:
        print(f"[WARN] fit_max={fit_max} falló: {e}")
        fit_rows.append({
            "fit_min": FIT_MIN,
            "fit_max": fit_max,
            "Einf": np.nan,
            "Einf_err": np.nan,
            "A": np.nan,
            "A_err": np.nan,
            "tau": np.nan,
            "tau_err": np.nan,
            "R2": np.nan,
            "RMSE": np.nan,
            "Gibbs_E_over_N": REF_E_OVER_N,
            "abs_Einf_minus_Gibbs": np.nan,
        })

plt.xlabel("reset cycle")
plt.ylabel("E/N")
plt.title("N=20 θ=0.45: comparación de ventanas de ajuste")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PLOTS / "N20_theta045_relaxation_fit_windows_comparison.png", dpi=170)
plt.close()

fits = pd.DataFrame(fit_rows)
fits.to_csv(OUT_RESULTS / "N20_theta045_relaxation_time_fit_scan.csv", index=False)

print("\n=== Resultados fit exponencial ===")
print(fits.to_string(index=False))

# Plot tau vs fit_max
plt.figure(figsize=(7, 5))
plt.errorbar(fits["fit_max"], fits["tau"], yerr=fits["tau_err"], fmt="o-", capsize=4)
plt.xlabel("fit_max")
plt.ylabel(r"$\tau$ [ciclos]")
plt.title("Robustez del tiempo de relajación efectivo")
plt.tight_layout()
plt.savefig(OUT_PLOTS / "N20_theta045_relaxation_tau_vs_fitmax.png", dpi=170)
plt.close()

# Plot Einf vs fit_max
plt.figure(figsize=(7, 5))
plt.errorbar(fits["fit_max"], fits["Einf"], yerr=fits["Einf_err"], fmt="o-", capsize=4, label=r"$E_\infty/N$")
plt.axhline(REF_E_OVER_N, linestyle="--", color="black", label="Gibbs MPS")
plt.xlabel("fit_max")
plt.ylabel(r"$E_\infty/N$")
plt.title("Meseta efectiva estimada vs ventana de ajuste")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PLOTS / "N20_theta045_relaxation_Einf_vs_fitmax.png", dpi=170)
plt.close()

print("\nGuardado:")
print(" results_N20/N20_theta045_relaxation_time_fit_scan.csv")
print(" results_N20/N20_theta045_relaxation_cycle_summary.csv")
print(" plots_N20/N20_theta045_relaxation_fit_windows_comparison.png")
print(" plots_N20/N20_theta045_relaxation_tau_vs_fitmax.png")
print(" plots_N20/N20_theta045_relaxation_Einf_vs_fitmax.png")
print(" plots_N20/N20_theta045_relaxation_fit_1_60.png etc.")
