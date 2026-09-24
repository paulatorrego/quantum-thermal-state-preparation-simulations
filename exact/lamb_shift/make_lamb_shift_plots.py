from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--root", default="analysis_definitive_N3_mixed_v2_2")
parser.add_argument("--out", default="analysis_clean_focus_v2_2")
args = parser.parse_args()

ROOT = Path(args.root)
OUT = Path(args.out)
PLOTS = OUT / "plots"
TABLES = OUT / "tables"
PLOTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "tables" / "compact_results.csv")

all_variants = [
    "stored_rho_mf",
    "instant_fmax",
    "f2avg_Hmf",
    "f2avg_rho_diagnostic",
]
variants = [v for v in all_variants if v in set(df["variant"])]

vlabel = {
    "stored_rho_mf": r"stored $f=1$",
    "instant_fmax": r"instant $f_{\max}$",
    "f2avg_Hmf": r"$f^2$-avg $H_{\rm mf}$",
    "f2avg_rho_diagnostic": r"$f^2$-avg $\rho_{\rm eff}$",
}
vcolor = {
    "stored_rho_mf": "C0",
    "instant_fmax": "C1",
    "f2avg_Hmf": "C2",
    "f2avg_rho_diagnostic": "C3",
}
lstyle = {
    0.0: dict(linestyle="-", marker=None),
    2.0: dict(linestyle="--", marker="o"),
}

pd.DataFrame({"variant": variants}).to_csv(TABLES / "variants_included.csv", index=False)
print("Variantes incluidas:", variants)

def base_rows(dataset):
    return (
        df[(df["dataset"] == dataset) & (df["variant"] == "stored_rho_mf")]
        .drop_duplicates(["theta2", "lambda"])
        .sort_values(["lambda", "theta2"])
    )

def plot_rhoeff_variants(dataset, ycol, ylabel, fname, xcol="theta2"):
    sub = df[df["dataset"] == dataset].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for v in variants:
        for lam in [0.0, 2.0]:
            s = sub[(sub["variant"] == v) & np.isclose(sub["lambda"], lam)].sort_values(xcol)
            if s.empty:
                continue
            ax.plot(
                s[xcol], s[ycol],
                color=vcolor[v],
                linewidth=2.2,
                label=vlabel[v] + (r", $\lambda=0$" if lam == 0 else r", $\lambda=2$"),
                **lstyle[lam],
            )
    ax.set_title(dataset)
    ax.set_xlabel(r"$\theta$" if xcol == "theta" else r"$\theta^2$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / fname, dpi=240)
    plt.close(fig)

def plot_gibbs_improvement(dataset):
    b = base_rows(dataset)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for lam in [0.0, 2.0]:
        s = b[np.isclose(b["lambda"], lam)]
        ax.plot(
            s["theta2"], s["D_fix_beta"],
            color="C0",
            linewidth=2.4,
            label=(r"$D(\rho_{\rm fix},\rho_\beta)$, $\lambda=0$" if lam == 0
                   else r"$D(\rho_{\rm fix},\rho_\beta)$, $\lambda=2$"),
            **lstyle[lam],
        )
    ax.set_title(dataset + r": mejora por randomización")
    ax.set_xlabel(r"$\theta^2$")
    ax.set_ylabel(r"$D(\rho_{\rm fix},\rho_\beta)$")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{dataset}_clean_Dfixbeta_lambda0_vs_lambda2.png", dpi=240)
    plt.close(fig)

def plot_clean_summary_no_cpaper_formula(dataset, lam):
    sub = df[(df["dataset"] == dataset) & np.isclose(df["lambda"], lam)].copy()
    b = sub[sub["variant"] == "stored_rho_mf"].drop_duplicates("theta2").sort_values("theta2")

    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4))
    ax = axs.ravel()

    # 1. logdiff rhoeff
    for v in variants:
        s = sub[sub["variant"] == v].sort_values("theta2")
        ax[0].plot(s["theta2"], s["norm_logeff_minus_logfix_dev_F"],
                   color=vcolor[v], marker="o", label=vlabel[v])
    ax[0].set_title(r"1. $\|\mathrm{dev}(\log\rho_{\rm eff}-\log\rho_{\rm fix})\|_F$")
    ax[0].set_xlabel(r"$\theta^2$")
    ax[0].set_ylabel("norma")
    ax[0].legend(fontsize=7)

    # 2. state distances
    for v in variants:
        s = sub[sub["variant"] == v].sort_values("theta2")
        ax[1].plot(s["theta2"], s["D_fix_eff"], color=vcolor[v], marker="o", label=vlabel[v])
    ax[1].plot(b["theta2"], b["D_fix_beta"], color="k", linestyle="--", marker="x",
               label=r"$D(\rho_{\rm fix},\rho_\beta)$")
    ax[1].set_title("2. Distancias directas")
    ax[1].set_xlabel(r"$\theta^2$")
    ax[1].set_ylabel("trace distance")
    ax[1].legend(fontsize=7)

    # 3. norms, sin Cpaper_formula duplicado
    ax[2].plot(b["theta2"], b["norm_Hfixcorr_F"], marker="o", label=r"$H_{\rm fix,corr}$")
    ax[2].plot(b["theta2"], b["norm_theta2_Cpaper_F"], marker="o", label=r"$\theta^2 C_{\rm paper}$")
    ax[2].plot(b["theta2"], b["norm_theta2_GLS_F"], marker="o", label=r"$\theta^2 G_{\rm LS}$")
    ax[2].plot(b["theta2"], b["norm_theta2_DeltaG_F"], marker="o", label=r"$\theta^2\Delta G$")
    for v in variants:
        s = sub[sub["variant"] == v].sort_values("theta2")
        ax[2].plot(s["theta2"], s["norm_Hmf_F"], color=vcolor[v], linestyle="--",
                   label=r"$H_{\rm mf}$ " + vlabel[v])
    ax[2].set_title("3. Normas: escala, no estructura")
    ax[2].set_xlabel(r"$\theta^2$")
    ax[2].set_ylabel("Frobenius")
    ax[2].legend(fontsize=6)

    # 4. similarity, sin Cpaper_formula duplicado
    ax[3].plot(b["theta2"], b["abs_cos_Cnum_Cpaper_offE"], marker="o", label=r"$C_{\rm paper}$")
    ax[3].plot(b["theta2"], b["abs_cos_Cnum_GLS_offE"], marker="o", label=r"$G_{\rm LS}$")
    ax[3].plot(b["theta2"], b["abs_cos_Cnum_DeltaG_offE"], marker="o", label=r"$\Delta G$")
    for v in variants:
        s = sub[sub["variant"] == v].sort_values("theta2")
        ax[3].plot(s["theta2"], s["abs_cos_Cnum_Cmf_offE"], color=vcolor[v], linestyle="--",
                   label=r"$C_{\rm mf}$ " + vlabel[v])
    ax[3].set_title(r"4. $|s|$ off-diagonal en base de energía")
    ax[3].set_xlabel(r"$\theta^2$")
    ax[3].set_ylim(-0.05, 1.05)
    ax[3].legend(fontsize=6)

    # 5. residuals, sin Cpaper_formula duplicado
    ax[4].plot(b["theta2"], b["resid_Cnum_vs_Cpaper_offE"], marker="o", label=r"$C_{\rm paper}$")
    ax[4].plot(b["theta2"], b["resid_Cnum_vs_GLS_offE"], marker="o", label=r"$G_{\rm LS}$")
    ax[4].plot(b["theta2"], b["resid_Cnum_vs_DeltaG_offE"], marker="o", label=r"$\Delta G$")
    for v in variants:
        s = sub[sub["variant"] == v].sort_values("theta2")
        ax[4].plot(s["theta2"], s["resid_Cnum_vs_Cmf_offE"], color=vcolor[v], linestyle="--",
                   label=r"$C_{\rm mf}$ " + vlabel[v])
    ax[4].set_title("5. Residual tras escala óptima")
    ax[4].set_xlabel(r"$\theta^2$")
    ax[4].set_ylabel("menor es mejor")
    ax[4].legend(fontsize=6)

    # 6. candidate Gibbs states, sin Cpaper_formula duplicado
    ax[5].plot(b["theta2"], b["D_fix_beta"], color="k", linestyle="--", marker="x", label=r"$\rho_\beta$")
    ax[5].plot(b["theta2"], b["D_fix_Gibbs_Hs_plus_theta2_Cpaper"],
               marker="o", label=r"$H_S+\theta^2 C_{\rm paper}$")
    for v in variants:
        s = sub[sub["variant"] == v].sort_values("theta2")
        ax[5].plot(s["theta2"], s["D_fix_Gibbs_Hs_plus_Hmf"],
                   color=vcolor[v], linestyle="--", label=r"$H_S+H_{\rm mf}$ " + vlabel[v])
    ax[5].set_title(r"6. $D(\rho_{\rm fix},$ Gibbs candidato$)$")
    ax[5].set_xlabel(r"$\theta^2$")
    ax[5].set_ylabel("trace distance")
    ax[5].legend(fontsize=6)

    for a in ax:
        a.grid(True, alpha=0.25)
    fig.suptitle(f"{dataset}, lambda={lam:g} -- sin duplicar Cpaper_formula", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOTS / f"{dataset}_lambda{lam:g}_clean_summary_no_Cpaper_formula.png", dpi=240)
    plt.close(fig)

def plot_cpaper_focus(dataset):
    b = base_rows(dataset)

    # 1. H scale: Hfixcorr vs theta^2 Cpaper
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for lam in [0.0, 2.0]:
        s = b[np.isclose(b["lambda"], lam)]
        ax.plot(s["theta2"], s["norm_Hfixcorr_F"],
                color="C0", linewidth=2.4,
                label=(r"$\|H_{\rm fix,corr}\|$, $\lambda=0$" if lam == 0
                       else r"$\|H_{\rm fix,corr}\|$, $\lambda=2$"),
                **lstyle[lam])
        ax.plot(s["theta2"], s["norm_theta2_Cpaper_F"],
                color="C1", linewidth=2.4,
                label=(r"$\|\theta^2 C_{\rm paper}\|$, $\lambda=0$" if lam == 0
                       else r"$\|\theta^2 C_{\rm paper}\|$, $\lambda=2$"),
                **lstyle[lam])
    ax.set_title(dataset + r": escala Hamiltoniana")
    ax.set_xlabel(r"$\theta^2$")
    ax.set_ylabel("Frobenius norm")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{dataset}_Cnum_Cpaper_Hscale.png", dpi=240)
    plt.close(fig)

    # 2. C scale: Cnum vs Cpaper
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for lam in [0.0, 2.0]:
        s = b[np.isclose(b["lambda"], lam)].copy()
        s["norm_Cnum_F"] = s["norm_Hfixcorr_F"] / s["theta2"]
        s["norm_Cpaper_F"] = s["norm_theta2_Cpaper_F"] / s["theta2"]
        ax.plot(s["theta2"], s["norm_Cnum_F"],
                color="C0", linewidth=2.4,
                label=(r"$\|C_{\rm num}\|$, $\lambda=0$" if lam == 0
                       else r"$\|C_{\rm num}\|$, $\lambda=2$"),
                **lstyle[lam])
        ax.plot(s["theta2"], s["norm_Cpaper_F"],
                color="C1", linewidth=2.4,
                label=(r"$\|C_{\rm paper}\|$, $\lambda=0$" if lam == 0
                       else r"$\|C_{\rm paper}\|$, $\lambda=2$"),
                **lstyle[lam])
    ax.set_title(dataset + r": escala perturbativa")
    ax.set_xlabel(r"$\theta^2$")
    ax.set_ylabel("Frobenius norm")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / f"{dataset}_Cnum_Cpaper_Cscale.png", dpi=240)
    plt.close(fig)

    # 3. structural focus: cos, abs cos, alpha, residual
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    ax = axs.ravel()
    metrics = [
        ("cos_Cnum_Cpaper_offE", r"$s(C_{\rm num},C_{\rm paper})$", None),
        ("abs_cos_Cnum_Cpaper_offE", r"$|s(C_{\rm num},C_{\rm paper})|$", (-0.05, 1.05)),
        ("alpha_Cnum_vs_Cpaper_offE", r"$\alpha^\star$ in $C_{\rm num}\simeq\alpha C_{\rm paper}$", None),
        ("resid_Cnum_vs_Cpaper_offE", r"$r(C_{\rm paper})$", None),
    ]
    for k, (col, ylabel, ylim) in enumerate(metrics):
        for lam in [0.0, 2.0]:
            s = b[np.isclose(b["lambda"], lam)]
            ax[k].plot(s["theta2"], s[col],
                       color="C0", linewidth=2.4,
                       label=(r"$\lambda=0$" if lam == 0 else r"$\lambda=2$"),
                       **lstyle[lam])
        ax[k].set_xlabel(r"$\theta^2$")
        ax[k].set_ylabel(ylabel)
        if ylim:
            ax[k].set_ylim(*ylim)
        ax[k].grid(True, alpha=0.25)
        ax[k].legend(fontsize=8)
    fig.suptitle(dataset + r": comparación aislada $C_{\rm num}$ vs $C_{\rm paper}$", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOTS / f"{dataset}_Cnum_Cpaper_offdiag_alpha_residual.png", dpi=240)
    plt.close(fig)

for dataset in ["mixed_general", "mixed_theta_wide"]:
    plot_gibbs_improvement(dataset)

    plot_rhoeff_variants(
        dataset,
        "D_fix_eff",
        r"$D(\rho_{\rm fix},\rho_{\rm eff}^{(v)})$",
        f"{dataset}_4rhoeff_Dfixeff_vs_theta2.png",
    )

    plot_rhoeff_variants(
        dataset,
        "norm_logeff_minus_logfix_dev_F",
        r"$\|\mathrm{dev}(\log\rho_{\rm eff}^{(v)}-\log\rho_{\rm fix})\|_F$",
        f"{dataset}_4rhoeff_logdiff_vs_theta2.png",
    )

    plot_rhoeff_variants(
        dataset,
        "abs_cos_Cnum_Cmf_offE",
        r"$|s(C_{\rm num},C_{\rm mf}^{(v)})|$",
        f"{dataset}_4rhoeff_Cmf_similarity_vs_theta2.png",
    )

    plot_rhoeff_variants(
        dataset,
        "resid_Cnum_vs_Cmf_offE",
        r"$r(C_{\rm mf}^{(v)})$",
        f"{dataset}_4rhoeff_Cmf_residual_vs_theta2.png",
    )

    for lam in [0.0, 2.0]:
        plot_clean_summary_no_cpaper_formula(dataset, lam)

    plot_cpaper_focus(dataset)

print("Plots limpios escritos en:", PLOTS.resolve())
