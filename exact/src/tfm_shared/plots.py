# tfm_shared/plots.py

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import replace
from types import SimpleNamespace
from IPython.display import display

from .core import (
    expect,
    trace_distance_dm,
    fidelity_dm,
    energy_basis_matrix,
    metrics_vs_target,
    state_error_summary,
    trajectory_error_arrays,
    maximally_mixed,
    all_plus,
    make_op,
    mat,
    project_to_physical_dm,
    total_coherence_vs_resets,
    coherence_norm_in_energy_basis,
    run_scheduled_protocol,
    heat_capacity_from_variance,
    gaussian_filter_values,
)

# ============================================================
# PLOTS
# ============================================================

def plot_energy_metrics(res, Hs, rho_gibbs=None, rho_fp=None, title=""):
    cycles = np.arange(len(res["energies"]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Energía
    axes[0].plot(cycles, res["energies"], marker="o", ms=3, label="E(r)")
    if rho_gibbs is not None:
        Eg = expect(Hs, rho_gibbs)
        axes[0].axhline(Eg, ls="--", label="E_Gibbs")
    if rho_fp is not None:
        Efp = expect(Hs, rho_fp)
        axes[0].axhline(Efp, ls=":", label="E_punto_fijo")
    axes[0].set_xlabel("resets")
    axes[0].set_ylabel("energía")
    axes[0].set_title("Evolución de energía")
    axes[0].legend()

    # Fidelidad / distancia a Gibbs
    if rho_gibbs is not None and res["rhos"] is not None:
        Fg, Dg = metrics_vs_target(res["rhos"], rho_gibbs)
        axes[1].plot(cycles, Fg, marker="o", ms=3)
        axes[1].set_xlabel("resets")
        axes[1].set_ylabel("F(ρ_r, ρ_Gibbs)")
        axes[1].set_title("Fidelidad a Gibbs")

        axes[2].plot(cycles, Dg, marker="o", ms=3)
        axes[2].set_xlabel("resets")
        axes[2].set_ylabel("D_tr(ρ_r, ρ_Gibbs)")
        axes[2].set_title("Distancia traza a Gibbs")
    else:
        axes[1].axis("off")
        axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_bath_readout(res, basis="Z"):
    basis = basis.upper()
    data = res[f"bath_{basis.lower()}"]

    plt.figure(figsize=(6, 4))
    plt.imshow(data.T, aspect="auto")
    plt.colorbar(label=f"<{basis}> del baño antes del reset")
    plt.xlabel("reset")
    plt.ylabel("qubit de baño")
    plt.title(f"Lectura del baño en base {basis}")
    plt.show()



    
def plot_steady_state_matrix_compare(rho_ss, rho_gibbs, Hs, title=""):
    evals, rho_ss_e = energy_basis_matrix(rho_ss, Hs)
    _, rho_g_e = energy_basis_matrix(rho_gibbs, Hs)

    A = np.abs(rho_ss_e)
    B = np.abs(rho_g_e)
    C = np.abs(rho_ss_e - rho_g_e)

    vmax_main = max(A.max(), B.max(), 1e-12)
    vmax_diff = max(C.max(), 1e-12)

    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[1, 1, 1, 0.05, 0.05],
        wspace=0.35
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    cax0 = fig.add_subplot(gs[0, 3])
    cax1 = fig.add_subplot(gs[0, 4])

    im0 = ax0.imshow(A, vmin=0, vmax=vmax_main)
    im1 = ax1.imshow(B, vmin=0, vmax=vmax_main)
    im2 = ax2.imshow(C, vmin=0, vmax=vmax_diff)

    ax0.set_title(r"$|\rho_{ss}|$ en base de energía")
    ax1.set_title(r"$|\rho_{\rm Gibbs}|$ en base de energía")
    ax2.set_title(r"$|\rho_{ss}-\rho_{\rm Gibbs}|$")

    for ax in [ax0, ax1, ax2]:
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        ax.set_xticks(range(A.shape[1]))
        ax.set_yticks(range(A.shape[0]))

    cb0 = fig.colorbar(im1, cax=cax0)
    cb0.set_label(r"$|\rho_{ij}|$")

    cb1 = fig.colorbar(im2, cax=cax1)
    cb1.set_label(r"$|(\rho_{ss}-\rho_{\rm Gibbs})_{ij}|$")

    fig.suptitle(title)
    plt.show()
    
# para ver poblaciones y coherencias en base de energía a lo largo de la trayectoria
def plot_single_spin_energy_basis_observables(res, Hs, title=""):
    pops0, pops1 = [], []
    coh_re, coh_im, coh_abs = [], [], []

    for rho in res["rhos"]:
        _, rho_e = energy_basis_matrix(rho, Hs)
        pops0.append(np.real(rho_e[0, 0]))
        pops1.append(np.real(rho_e[1, 1]))
        coh_re.append(np.real(rho_e[0, 1]))
        coh_im.append(np.imag(rho_e[0, 1]))
        coh_abs.append(np.abs(rho_e[0, 1]))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(pops0, label=r"$\rho_{00}$")
    ax[0].plot(pops1, label=r"$\rho_{11}$")
    ax[0].set_xlabel("reset")
    ax[0].set_ylabel("población")
    ax[0].set_title("Poblaciones en base de energía")
    ax[0].legend()

    ax[1].plot(coh_re, label=r"$\Re(\rho_{01})$")
    ax[1].plot(coh_im, label=r"$\Im(\rho_{01})$")
    ax[1].plot(coh_abs, label=r"$|\rho_{01}|$")
    ax[1].set_xlabel("reset")
    ax[1].set_ylabel("coherencia")
    ax[1].set_title("Coherencias en base de energía")
    ax[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    


# ============================================================
# ESTUDIO THETA^2 TIPO PAPER
# ============================================================

# ============================================================
# ESTUDIO THETA^2 TIPO PAPER — CORREGIDO
# ============================================================

# ============================================================
# THETA SCALING STUDY — VERSIÓN LIMPIA Y CORREGIDA
# ============================================================
# Qué hace distinto:
#   1. Usa el punto fijo EXACTO por superoperador cuando cabe (dS^2 ≤ ~256).
#   2. Si el sistema es mayor, hace late_time con n_burn escalado como θ⁻²
#      (porque el tiempo de mezcla del protocolo ∝ θ⁻²).
#   3. Una sola llamada → un solo plot + una sola tabla.
#   4. No fuerza fp_mode internamente: respeta lo que se le pasa.
# ============================================================
 
def theta_scaling_study(
    cfg_runner,
    base_params,
    theta_grid,
    rho0=None,
    fit_theta_max=0.20,      # rango pequeño-θ para hacer el fit log-log
    eps=1e-12,
    fp_mode="auto",          # "superop", "late_time", o "auto"
    burn_ref_theta=0.25,     # a qué θ corresponden base_params["n_cycles"] ciclos
    min_burn=200,
    max_burn=20000,
    use_final_state=False,   # False = medir en el punto fijo (recomendado)
    title_suffix="",
    show_table=True,
):
    """
    Estudia cómo escalan los errores al estado de Gibbs en función de θ.
    Para el protocolo de Lloyd & Abanin se espera ||σ − σ_β|| ∝ θ².
 
    Parámetros clave:
        cfg_runner     : función tipo run_single_spin_case(params, ...)
        base_params    : dict con los parámetros del protocolo (todo menos theta)
        theta_grid     : array de valores de θ a barrer
        fp_mode        : "superop" (exacto, dS pequeño) o "late_time" (con burn escalado)
        burn_ref_theta : valor de θ para el cual base_params["n_cycles"] es suficiente;
                         el burn-in para otros θ se escala como (burn_ref/θ)².
    """
    rows = []
 
    for th in theta_grid:
        params = base_params.copy()
        params["theta"] = float(th)
 
        # Para late_time: escalar el número de ciclos como θ⁻²
        # (tiempo de mezcla del protocolo ∝ θ⁻²)
        if fp_mode == "late_time" or (fp_mode == "auto"):
            ref_cycles = base_params.get("n_cycles", 200)
            scale = (burn_ref_theta / th) ** 2
            needed = int(np.clip(ref_cycles * scale, min_burn, max_burn))
            params["n_cycles"] = needed
 
        out = cfg_runner(
            params,
            rho0=rho0,
            compute_fp=True,
            fp_mode=fp_mode,
        )
 
        if use_final_state:
            rho_ref = out["res"]["rhos"][-1]
            summary = state_error_summary(rho_ref, out["rho_g"], out["Hs"])
            E_ref = out["Efinal"]
        else:
            rho_ref = out["rho_fp"]
            summary = out["fp_vs_gibbs"]
            E_ref = out["Efp"]
 
        rows.append({
            "theta": th,
            "theta2": th**2,
            "population_error": summary["population_error_l1"],
            "coherence_error":  summary["coherence_norm"],
            "trace_distance":   summary["trace_distance"],
            "fidelity":         summary["fidelity"],
            "Egibbs":           out["Eg"],
            "Eref":             E_ref,
            "E_bias":           abs(E_ref - out["Eg"]),
            "fp_method":        out["fp_method_used"],
            "n_cycles_used":    params.get("n_cycles", None),
        })
 
    df = pd.DataFrame(rows)
 
    # --- Fits robustos solo en la región de θ pequeño ---
    mask_base = df["theta"] <= fit_theta_max
    def safe_slope(xs, ys):
        if (xs > 0).sum() >= 2 and (ys > eps).sum() >= 2:
            m = (xs > 0) & (ys > eps)
            return float(np.polyfit(np.log(xs[m]), np.log(ys[m]), 1)[0])
        return np.nan
 
    slope_pop  = safe_slope(df.loc[mask_base, "theta"].values,
                            df.loc[mask_base, "population_error"].values)
    slope_coh  = safe_slope(df.loc[mask_base, "theta"].values,
                            df.loc[mask_base, "coherence_error"].values)
    slope_dist = safe_slope(df.loc[mask_base, "theta"].values,
                            df.loc[mask_base, "trace_distance"].values)
 
    # --- Plot: 3 paneles ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
 
    # Panel 1: log-log con guía ∝ θ²
    ax = axes[0]
    ax.loglog(df["theta"], df["population_error"], 'o-',
              label=f'population (slope={slope_pop:.2f})')
    if (df["coherence_error"] > eps).any():
        ax.loglog(df["theta"], df["coherence_error"], 's--',
                  label=f'coherence (slope={slope_coh:.2f})')
    ax.loglog(df["theta"], df["trace_distance"], '^:',
              label=f'trace dist (slope={slope_dist:.2f})')
    # Guía ∝ θ² anclada al punto más pequeño
    ref_t = df["theta"].min()
    ref_y = df.loc[df["theta"] == ref_t, "trace_distance"].values[0]
    ax.loglog(df["theta"], ref_y * (df["theta"]/ref_t)**2, 'k--', alpha=0.4,
              label=r'$\propto \theta^2$')
    ax.set_xlabel(r'$\theta$'); ax.set_ylabel('error')
    ax.set_title(r'Errores vs $\theta$ (log-log)')
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
 
    # Panel 2: lineal en θ²
    ax = axes[1]
    ax.plot(df["theta2"], df["population_error"], 'o-',  label='population')
    ax.plot(df["theta2"], df["trace_distance"],   '^:',  label='trace distance')
    if (df["coherence_error"] > eps).any():
        ax.plot(df["theta2"], df["coherence_error"], 's--', label='coherence')
    ax.set_xlabel(r'$\theta^2$'); ax.set_ylabel('error')
    ax.set_title(r'Errores vs $\theta^2$ (lineal)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
 
    # Panel 3: sesgo energético
    ax = axes[2]
    ax.plot(df["theta"], df["E_bias"], 'o-', label=r'$|E_{\rm ref}-E_\beta|$')
    ax.set_xlabel(r'$\theta$'); ax.set_ylabel('energy bias')
    ax.set_title("Sesgo energético"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
 
    fig.suptitle(f"θ-scaling [{df['fp_method'].iloc[0]}] {title_suffix}", y=1.02)
    plt.tight_layout()
    plt.show()
 
    print(f"Pendientes log-log (θ ≤ {fit_theta_max}): "
          f"pop={slope_pop:.2f}, coh={slope_coh:.2f}, dist={slope_dist:.2f}")
    print(f"Método punto fijo: {df['fp_method'].iloc[0]}")
    if df['fp_method'].iloc[0] == "late_time":
        print(f"n_cycles usado: min={df['n_cycles_used'].min()}, "
              f"max={df['n_cycles_used'].max()} (escalado como θ⁻²)")
 
    if show_table:
        display(df.round(8))
 
    return df, {"slope_pop": slope_pop,
                "slope_coh": slope_coh,
                "slope_dist": slope_dist}
    
    
# ============================================================
# FUNCIONES GENÉRICAS PARA CUALQUIER SISTEMA
# ============================================================
# Reemplazan: single_spin_heat_capacity_study, plot_single_spin_extras,
#             compare_single_spin_system_paulis, compare_single_spin_variants

def heat_capacity_study(runner, base_params, n_sys, beta_grid,
                         rho0=None, use_fp=True):
    """
    Capacidad calorífica vs β para cualquier sistema.
    C_v = β² (⟨H²⟩ - ⟨H⟩²)
    """
    if rho0 is None:
        rho0 = maximally_mixed(n_sys)

    rows = []
    for b in beta_grid:
        p = base_params.copy(); p["beta"] = float(b)
        out = runner(p, rho0=rho0, compute_fp=use_fp)
        Hs = out["Hs"]; Hs_mat = mat(Hs)
        rho = out["rho_fp"] if (use_fp and out["rho_fp"] is not None) else out["res"]["rhos"][-1]

        E  = expect(Hs_mat, rho)
        E2 = expect(Hs_mat @ Hs_mat, rho)
        Cv = b**2 * (E2 - E**2)

        # Valores exactos del Gibbs
        rho_g = out["rho_g"]
        Eg  = expect(Hs_mat, rho_g)
        E2g = expect(Hs_mat @ Hs_mat, rho_g)
        Cv_exact = b**2 * (E2g - Eg**2)

        rows.append({"beta": b, "E": E, "E_gibbs": Eg,
                      "Cv": Cv, "Cv_exact": Cv_exact})

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["beta"], df["E"], "o-", label="protocolo", ms=4)
    axes[0].plot(df["beta"], df["E_gibbs"], "k--", label="Gibbs exacto")
    axes[0].set_xlabel(r"$\beta$"); axes[0].set_ylabel("E")
    axes[0].set_title("Energía vs β"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(df["beta"], df["Cv"], "o-", label="protocolo", ms=4)
    axes[1].plot(df["beta"], df["Cv_exact"], "k--", label="Gibbs exacto")
    axes[1].set_xlabel(r"$\beta$"); axes[1].set_ylabel(r"$C_v$")
    axes[1].set_title(r"Capacidad calorífica vs $\beta$")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle(f"n_sys={n_sys}, d={2**n_sys}", y=1.02)
    plt.tight_layout(); plt.show()
    display(df.round(6))
    return df

def heat_capacity_fine_study(
    runner,
    base_params,
    beta_grid,
    rho0=None,
    fp_mode="superop",
    title="",
):
    """
    Estudio fino de la capacidad calorífica: compara tres representaciones 
    de C_V(T_th) y las tres representaciones de la energía del protocolo 
    frente al Gibbs exacto, barriendo la temperatura termodinámica T_th=1/β.

    Panel 1 — Energía vs T_th:
        * E_Gibbs  = ⟨H⟩_ρ_β  (referencia exacta)
        * E_fp     = ⟨H⟩_ρ_fp (punto fijo del canal del protocolo)
        * E_final  = ⟨H⟩ tras n_cycles de protocolo desde rho0
      Si E_final coincide con E_fp → el protocolo ha convergido.
      Si además E_fp coincide con E_Gibbs → el punto fijo es térmico.

    Panel 2 — Capacidad calorífica vs T_th:
        * C_V^Gibbs,var = β² Var_ρ_β(H)             (referencia termodinámica)
        * C_V^fp,var    = β² Var_ρ_fp(H)            (varianza sobre fp)
        * C_V^fp,fd     = dE_fp / dT_th             (derivada numérica de E_fp)

    Panel 3 — Qué tan térmico es ρ_fp:
        * D_tr(ρ_fp, ρ_β)
        * coherencia l1 off-diagonal de ρ_fp en base de energía

    Parámetros
    ----------
    runner : callable
        Ejecutor de un caso del protocolo (run_single_spin_case, etc.).
        Debe devolver un dict con claves: "Hs", "rho_g", "rho_fp", "Eg", 
        "Efp", "Efinal".
    base_params : dict
        Parámetros base del protocolo. La clave "beta" se sobreescribe por
        cada valor de beta_grid.
    beta_grid : array-like
        Valores de β a barrer.
    rho0 : np.ndarray | None
        Estado inicial. Si None, usa maximally_mixed(n_sys) dentro del runner.
    fp_mode : str
        "superop" (exacto hasta ~4 qubits) o "late_time" (sistemas grandes).
    title : str
        Sufijo opcional para los títulos.
    """
    rows = []

    for beta in beta_grid:
        params = dict(base_params)
        params["beta"] = float(beta)

        out = runner(
            params,
            rho0=rho0,
            compute_fp=True,
            fp_mode=fp_mode,
        )

        Hs     = out["Hs"]
        rho_g  = out["rho_g"]
        rho_fp = out["rho_fp"]

        rows.append({
            "beta":   float(beta),
            "T_th":   1.0 / float(beta),

            "Egibbs": float(out["Eg"]),
            "Efp":    float(out["Efp"]),
            "Efinal": float(out["Efinal"]),

            "Cv_gibbs_var": heat_capacity_from_variance(rho_g,  Hs, beta),
            "Cv_fp_var":    heat_capacity_from_variance(rho_fp, Hs, beta),

            "Dtr_fp_gibbs": float(trace_distance_dm(rho_fp, rho_g)),
            "F_fp_gibbs":   float(fidelity_dm(rho_fp, rho_g)),
            "coh_fp_l1":    coherence_norm_in_energy_basis(rho_fp, Hs, norm="l1"),
        })

    df = pd.DataFrame(rows).sort_values("T_th").reset_index(drop=True)

    # Derivadas numéricas respecto a T_th
    df["Cv_gibbs_fd"] = np.gradient(df["Egibbs"].values, df["T_th"].values)
    df["Cv_fp_fd"]    = np.gradient(df["Efp"].values,    df["T_th"].values)

    # Desviaciones útiles
    df["dE_abs"]      = np.abs(df["Efp"] - df["Egibbs"])
    df["dCv_var_abs"] = np.abs(df["Cv_fp_var"] - df["Cv_gibbs_var"])
    df["dCv_fd_abs"]  = np.abs(df["Cv_fp_fd"]  - df["Cv_gibbs_fd"])

    # -------- plots --------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    # Panel 1: Energía con las 3 curvas
    axes[0].plot(df["T_th"], df["Egibbs"], "o-",  lw=2, label=r"$E_{\rm Gibbs}$ (exacto)")
    axes[0].plot(df["T_th"], df["Efp"],    "s--", lw=2, label=r"$E_{\rm fp}$ (punto fijo)")
    axes[0].plot(df["T_th"], df["Efinal"], "^:",  lw=2, alpha=0.8,
                 label=r"$E_{\rm final}$ (tras $n_{\rm cycles}$)")
    axes[0].set_xlabel(r"$T_{\rm th} = 1/\beta$")
    axes[0].set_ylabel("E")
    axes[0].set_title("Energía: exacta vs protocolo")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Panel 2: Capacidad calorífica
    axes[1].plot(df["T_th"], df["Cv_gibbs_var"], "o-",  lw=2,
                 label=r"$C_V^{\rm Gibbs}$ (var)")
    axes[1].plot(df["T_th"], df["Cv_fp_var"],    "s--", lw=2,
                 label=r"$C_V^{\rm fp}$ (var)")
    axes[1].plot(df["T_th"], df["Cv_fp_fd"],     "^-.", lw=2,
                 label=r"$C_V^{\rm fp}$ (dE/dT)")
    axes[1].set_xlabel(r"$T_{\rm th} = 1/\beta$")
    axes[1].set_ylabel(r"$C_V$")
    axes[1].set_title("Capacidad calorífica: exacta vs protocolo")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # Panel 3: Qué tan térmico es ρ_fp
    axes[2].plot(df["T_th"], df["Dtr_fp_gibbs"], "o-",  lw=2,
                 label=r"$D_{\rm tr}(\rho_{fp}, \rho_\beta)$")
    axes[2].plot(df["T_th"], df["coh_fp_l1"],    "s--", lw=2,
                 label=r"$\|\rho_{fp}^{\rm off-diag}\|_1$")
    axes[2].set_xlabel(r"$T_{\rm th} = 1/\beta$")
    axes[2].set_ylabel("magnitud del error")
    axes[2].set_title(r"Qué tan térmico es $\rho_{fp}$")
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

    display(df.round(8))
    return df


def plot_system_extras(out, n_sys, title="", show_bath_heatmap=True):
    """
    Observables extra para cualquier sistema.
    Reemplaza plot_single_spin_extras.
    Muestra: magnetización Z por qubit, poblaciones en base de energía,
    pureza, entropía, observables del baño.
    """
    res = out["res"]; rhos = res["rhos"]
    Hs = out["Hs"]; n_bath = out["cfg"].n_bath

    # Magnetización Z por qubit del sistema
    mag_z = np.array([[expect(make_op(n_sys, [(1.0, {q: "Z"})]), rho)
                        for q in range(n_sys)] for rho in rhos])

    # Poblaciones y coherencia total en base de energía
    pops_gs, coh_total = [], []
    for rho in rhos:
        _, rho_e = energy_basis_matrix(rho, Hs)
        pops_gs.append(np.real(rho_e[0, 0]))
        off = rho_e - np.diag(np.diag(rho_e))
        coh_total.append(np.sum(np.abs(off)))

    # Pureza y entropía
    pur = [float(np.real(np.trace(rho @ rho))) for rho in rhos]
    def vn_entropy(rho):
        vals = np.linalg.eigvalsh(project_to_physical_dm(rho))
        vals = np.clip(np.real(vals), 1e-15, None)
        return float(-np.sum(vals * np.log(vals)))
    SvN = [vn_entropy(rho) for rho in rhos]

    # Baño
    bz = res["bath_z"]
    p_exc_b = (1 - bz) / 2

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # (1) Magnetización Z por qubit
    for q in range(n_sys):
        ax[0, 0].plot(mag_z[:, q], label=f"qubit {q}")
    ax[0, 0].set_title(r"$\langle Z_q \rangle$ por qubit del sistema")
    ax[0, 0].set_xlabel("reset"); ax[0, 0].legend(fontsize=7); ax[0, 0].grid(alpha=0.3)

    # (2) Población del GS y coherencia total
    ax[0, 1].plot(pops_gs, label=r"$\rho_{00}$ (GS)")
    ax[0, 1].plot(coh_total, label=r"$\sum|\rho_{ab}^{\rm off}|$")
    ax[0, 1].set_title("Población GS y coherencia total")
    ax[0, 1].set_xlabel("reset"); ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)

    # (3) Pureza y entropía
    ax[0, 2].plot(pur, label="pureza")
    ax[0, 2].plot(SvN, label="entropía vN")
    ax[0, 2].set_title("Pureza y entropía"); ax[0, 2].set_xlabel("reset")
    ax[0, 2].legend(); ax[0, 2].grid(alpha=0.3)

    # (4) ⟨Z⟩ del baño por qubit
    for mu in range(n_bath):
        ax[1, 0].plot(bz[:, mu], label=f"bath {mu}")
    ax[1, 0].set_title(r"$\langle Z_\mu^{\rm bath}\rangle$ antes del reset")
    ax[1, 0].set_xlabel("reset"); ax[1, 0].legend(fontsize=7); ax[1, 0].grid(alpha=0.3)

    # (5) Excitación del baño
    for mu in range(n_bath):
        ax[1, 1].plot(p_exc_b[:, mu], label=f"bath {mu}")
    ax[1, 1].set_title(r"$p_{\rm exc}$ del baño"); ax[1, 1].set_xlabel("reset")
    ax[1, 1].legend(fontsize=7); ax[1, 1].grid(alpha=0.3)

    # (6) Heatmap del baño
    if show_bath_heatmap:
        im = ax[1, 2].imshow(bz.T, aspect="auto", cmap="RdBu_r")
        ax[1, 2].set_title("Lectura del baño (Z)")
        ax[1, 2].set_xlabel("reset"); ax[1, 2].set_ylabel("qubit baño")
        fig.colorbar(im, ax=ax[1, 2], fraction=0.046, pad=0.04)
    else:
        ax[1, 2].axis("off")

    fig.suptitle(title if title else f"Extras (n_sys={n_sys})")
    plt.tight_layout(); plt.show()


def compare_system_paulis(runner, base_params, n_sys,
                           paulis=("X", "Y", "Z"),
                           rho0_energy=None, rho0_coh=None,
                           normalize_by_Eg=True):
    """
    Compara el operador de acoplamiento A_μ = P para distintos Paulis.
    Reemplaza compare_single_spin_system_paulis.
    Funciona para cualquier sistema.
    """
    if rho0_energy is None:
        rho0_energy = maximally_mixed(n_sys)
    if rho0_coh is None:
        rho0_coh = all_plus(n_sys)

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for pauli in paulis:
        p = base_params.copy(); p["sys_A"] = pauli
        out_E = runner(p, rho0=rho0_energy, compute_fp=True)
        out_C = runner(p, rho0=rho0_coh, compute_fp=False)

        Eg = out_E["Eg"]
        Efp = out_E["Efp"]
        D = trace_distance_dm(out_E["rho_fp"], out_E["rho_g"])
        F = fidelity_dm(out_E["rho_fp"], out_E["rho_g"])

        # Coherencia final
        _, rho_e = energy_basis_matrix(out_C["res"]["rhos"][-1], out_C["Hs"])
        off = rho_e - np.diag(np.diag(rho_e))
        coh = np.sum(np.abs(off))

        rows.append({"A": pauli, "E_fp": Efp, "E_gibbs": Eg,
                      "|E_fp-E_β|": abs(Efp - Eg),
                      "D_tr": D, "F": F, "|coh|_final": coh})

        # Plots
        E_traj = out_E["res"]["energies"]
        if normalize_by_Eg and abs(Eg) > 1e-10:
            E_traj = E_traj / abs(Eg)
        axes[0].plot(E_traj, label=f"A={pauli}", lw=1.5)

        coh_traj, _ = total_coherence_vs_resets(
            rho0_coh, out_C["compiled"], out_C["Hs"], out_C["cfg"].n_cycles)
        axes[1].plot(coh_traj, label=f"A={pauli}", lw=1.5)

    if normalize_by_Eg:
        axes[0].axhline(1.0, color="k", ls="--", label=r"$E_\beta$")
        axes[0].set_ylabel(r"$E / |E_\beta|$")
    else:
        axes[0].axhline(Eg, color="k", ls="--", label=r"$E_\beta$")
        axes[0].set_ylabel("E")
    axes[0].set_xlabel("ciclo"); axes[0].set_title("Energía (desde I/d)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("ciclo"); axes[1].set_ylabel(r"$\sum|\rho_{ab}^{\rm off}|$")
    axes[1].set_title(r"Coherencia total (desde $|+...+\rangle$)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle(f"Comparación de acoplo A (n_sys={n_sys})", y=1.02)
    plt.tight_layout(); plt.show()

    df = pd.DataFrame(rows)
    display(df.round(6))
    return df


def compare_system_variants(runner, base_params, n_sys,
                              rho0=None, normalize_by_Eg=True):
    """
    Compara variantes del protocolo: distintos A, con/sin randomización.
    Reemplaza compare_single_spin_variants.
    """
    if rho0 is None:
        rho0 = maximally_mixed(n_sys)

    variants = {
        "A=Y":           dict(sys_A="Y"),
        "A=X":           dict(sys_A="X"),
        "A=Z":           dict(sys_A="Z"),
        "A=(Z+Y)/√2":   dict(sys_A=[(1/np.sqrt(2), {0: "Z"}),
                                     (1/np.sqrt(2), {0: "Y"})]),
        "A=Y + random":  dict(sys_A="Y", randomize=True, randomization_lambda=1.0),
    }

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for name, extra in variants.items():
        p = base_params.copy(); p.update(extra)
        out = runner(p, rho0=rho0, compute_fp=True)

        Eg  = out["Eg"]
        Efp = out["Efp"]
        D   = trace_distance_dm(out["rho_fp"], out["rho_g"])
        F   = fidelity_dm(out["rho_fp"], out["rho_g"])

        _, rho_e = energy_basis_matrix(out["rho_fp"], out["Hs"])
        off = rho_e - np.diag(np.diag(rho_e))
        coh = np.sum(np.abs(off))

        rows.append({"variante": name, "E_fp": Efp,
                      "|E-E_β|": abs(Efp - Eg), "D_tr": D, "F": F, "|coh|": coh})

        E_traj = out["res"]["energies"]
        if normalize_by_Eg and abs(Eg) > 1e-10:
            E_traj = E_traj / abs(Eg)
        axes[0].plot(E_traj, label=name, lw=1.5)
        axes[1].plot([trace_distance_dm(r, out["rho_g"]) for r in out["res"]["rhos"]],
                      label=name, lw=1.5)

    if normalize_by_Eg:
        axes[0].axhline(1.0, color="k", ls="--"); axes[0].set_ylabel(r"$E/|E_\beta|$")
    axes[0].set_xlabel("ciclo"); axes[0].set_title("Energía")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("ciclo"); axes[1].set_ylabel(r"$D_{\rm tr}$")
    axes[1].set_title("Distancia a Gibbs"); axes[1].set_yscale("log")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, which="both")

    fig.suptitle(f"Comparación de variantes (n_sys={n_sys})", y=1.02)
    plt.tight_layout(); plt.show()

    df = pd.DataFrame(rows)
    display(df.round(6))
    return df

# ============================================================
# TABLAS Y PRINTS ORDENADOS
# ============================================================
def case_summary_dataframe(out, label="case"):
    row = {
        "label": label,
        "n_sys": out["cfg"].n_sys,
        "n_bath": out["cfg"].n_bath,
        "beta": out["cfg"].beta,
        "theta": out["cfg"].theta,
        "delta": out["cfg"].delta,
        "MT": out["cfg"].MT,
        "n_cycles": out["cfg"].n_cycles,
        "E0": out["E0"],
        "Efinal": out["Efinal"],
        "Egibbs": out["Eg"],
        "Efp": out["Efp"],
        "|Efinal-Eg|": abs(out["Efinal"] - out["Eg"]),
        "|Efp-Eg|": abs(out["Efp"] - out["Eg"]) if np.isfinite(out["Efp"]) else np.nan,
        "F(final,Gibbs)": out["final_vs_gibbs"]["fidelity"],
        "Dtr(final,Gibbs)": out["final_vs_gibbs"]["trace_distance"],
    }
    if out["fp_vs_gibbs"] is not None:
        row["F(fp,Gibbs)"] = out["fp_vs_gibbs"]["fidelity"]
        row["Dtr(fp,Gibbs)"] = out["fp_vs_gibbs"]["trace_distance"]
        row["PopErr(fp)"] = out["fp_vs_gibbs"]["population_error_l1"]
        row["CohErr(fp)"] = out["fp_vs_gibbs"]["coherence_norm"]
    else:
        row["F(fp,Gibbs)"] = np.nan
        row["Dtr(fp,Gibbs)"] = np.nan
        row["PopErr(fp)"] = np.nan
        row["CohErr(fp)"] = np.nan

    return pd.DataFrame([row])

def print_case_report(out, label="case", digits=6):
    df = case_summary_dataframe(out, label=label).round(digits)
    print("=" * 80)
    print(f"REPORTE: {label}")
    print("=" * 80)
    print(f"Hs terms     : {out['cfg'].system_terms}")
    print(f"beta         : {out['cfg'].beta}")
    print(f"theta        : {out['cfg'].theta}")
    print(f"delta        : {out['cfg'].delta}")
    print(f"MT           : {out['cfg'].MT}")
    print(f"n_cycles     : {out['cfg'].n_cycles}")
    print(f"fp method    : {out['fp_method_used']}")
    print("-" * 80)
    display(df)


# ============================================================
# DASHBOARD GENÉRICO
# ============================================================

def plot_case_dashboard(out, title=""):
    res = out["res"]
    Hs = out["Hs"]
    rho_g = out["rho_g"]
    rho_fp = out["rho_fp"]

    cycles = np.arange(len(res["energies"]))
    pop_err, coh_err, dtr_g, fid_g = trajectory_error_arrays(res["rhos"], rho_g, Hs)

    if rho_fp is not None:
        fid_fp = np.array([fidelity_dm(r, rho_fp) for r in res["rhos"]], dtype=float)
        dtr_fp = np.array([trace_distance_dm(r, rho_fp) for r in res["rhos"]], dtype=float)
    else:
        fid_fp = None
        dtr_fp = None

    has_bath = ("bath_x" in res) and (len(res["bath_x"]) > 0)

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))

    # (1) Energía
    ax[0, 0].plot(cycles, res["energies"], marker="o", ms=3, label="E(r)")
    ax[0, 0].axhline(out["Eg"], ls="--", label=r"$E_{\rm Gibbs}$")
    if rho_fp is not None:
        ax[0, 0].axhline(out["Efp"], ls=":", label=r"$E_{\rm fp}$")
    ax[0, 0].set_title("Energía")
    ax[0, 0].set_xlabel("reset")
    ax[0, 0].set_ylabel("E")
    ax[0, 0].legend()

    # (2) Fidelidades
    ax[0, 1].plot(cycles, fid_g, marker="o", ms=3, label=r"$F(\rho_r,\rho_\beta)$")
    if fid_fp is not None:
        ax[0, 1].plot(cycles, fid_fp, marker="s", ms=3, label=r"$F(\rho_r,\rho_{\rm fp})$")
    ax[0, 1].set_title("Fidelidades")
    ax[0, 1].set_xlabel("reset")
    ax[0, 1].set_ylabel("F")
    ax[0, 1].legend()

    # (3) Distancias traza
    ax[0, 2].plot(cycles, dtr_g, marker="o", ms=3, label=r"$D(\rho_r,\rho_\beta)$")
    if dtr_fp is not None:
        ax[0, 2].plot(cycles, dtr_fp, marker="s", ms=3, label=r"$D(\rho_r,\rho_{\rm fp})$")
    ax[0, 2].set_title("Distancias traza")
    ax[0, 2].set_xlabel("reset")
    ax[0, 2].set_ylabel("D")
    ax[0, 2].legend()

    # (4) Errores en base de energía
    ax[1, 0].plot(cycles, pop_err, marker="o", ms=3, label="population error")
    ax[1, 0].plot(cycles, coh_err, marker="s", ms=3, label="coherence norm")
    ax[1, 0].set_title("Errores en base de energía")
    ax[1, 0].set_xlabel("reset")
    ax[1, 0].set_ylabel("error")
    ax[1, 0].legend()

    # (5) Observables del baño
    if has_bath:
        bx = np.mean(res["bath_x"], axis=1)
        by = np.mean(res["bath_y"], axis=1)
        bz = np.mean(res["bath_z"], axis=1)
        ax[1, 1].plot(np.arange(1, len(bx) + 1), bx, label=r"$\langle X_B\rangle$")
        ax[1, 1].plot(np.arange(1, len(by) + 1), by, label=r"$\langle Y_B\rangle$")
        ax[1, 1].plot(np.arange(1, len(bz) + 1), bz, label=r"$\langle Z_B\rangle$")
        ax[1, 1].set_title("Baño antes del reset")
        ax[1, 1].set_xlabel("reset")
        ax[1, 1].legend()
    else:
        ax[1, 1].axis("off")

    # (6) energía normalizada
    if abs(out["Eg"]) > 1e-14:
        ax[1, 2].plot(cycles, res["energies"] / out["Eg"], marker="o", ms=3)
        ax[1, 2].axhline(1.0, ls="--")
        ax[1, 2].set_title(r"Energía normalizada $E/E_\beta$")
        ax[1, 2].set_xlabel("reset")
        ax[1, 2].set_ylabel(r"$E/E_\beta$")
    else:
        ax[1, 2].axis("off")

    fig.suptitle(title if title else "Dashboard del caso")
    plt.tight_layout()
    plt.show()


# ============================================================
# MATRICES FINALES CON BARRAS DE COLOR
# ============================================================

def plot_state_matrix_compare_with_colorbars(rho_ss, rho_gibbs, Hs, title=""):
    _, rho_ss_e = energy_basis_matrix(rho_ss, Hs)
    _, rho_g_e = energy_basis_matrix(rho_gibbs, Hs)

    A = np.abs(rho_ss_e)
    B = np.abs(rho_g_e)
    C = np.abs(rho_ss_e - rho_g_e)

    vmax_main = max(A.max(), B.max(), 1e-12)
    vmax_diff = max(C.max(), 1e-12)

    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.05, 0.05], wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    cax0 = fig.add_subplot(gs[0, 3])
    cax1 = fig.add_subplot(gs[0, 4])

    im0 = ax0.imshow(A, vmin=0, vmax=vmax_main)
    im1 = ax1.imshow(B, vmin=0, vmax=vmax_main)
    im2 = ax2.imshow(C, vmin=0, vmax=vmax_diff)

    ax0.set_title(r"$|\rho_{ss}|$ en base de energía")
    ax1.set_title(r"$|\rho_{\rm Gibbs}|$ en base de energía")
    ax2.set_title(r"$|\rho_{ss}-\rho_{\rm Gibbs}|$")

    for a in [ax0, ax1, ax2]:
        a.set_xlabel("j")
        a.set_ylabel("i")

    cb0 = fig.colorbar(im1, cax=cax0)
    cb0.set_label(r"$|\rho_{ij}|$")

    cb1 = fig.colorbar(im2, cax=cax1)
    cb1.set_label(r"$|(\rho_{ss}-\rho_{\rm Gibbs})_{ij}|$")

    fig.suptitle(title)
    plt.show()
    
    
    # ============================================================
# BARRIDOS 1D REUTILIZABLES
# ============================================================

def run_param_sweep(cfg_runner, base_params, sweep_name, values, rho0=None):
    rows = []
    cases = []

    for val in values:
        params = base_params.copy()
        params[sweep_name] = val
        out = cfg_runner(params, rho0=rho0, compute_fp=True)

        row = {
            sweep_name: val,
            "Efinal": out["Efinal"],
            "Egibbs": out["Eg"],
            "Efp": out["Efp"],
            "|Efinal-Eg|": abs(out["Efinal"] - out["Eg"]),
            "|Efp-Eg|": abs(out["Efp"] - out["Eg"]),
            "F(final,Gibbs)": out["final_vs_gibbs"]["fidelity"],
            "Dtr(final,Gibbs)": out["final_vs_gibbs"]["trace_distance"],
        }
        if out["fp_vs_gibbs"] is not None:
            row["F(fp,Gibbs)"] = out["fp_vs_gibbs"]["fidelity"]
            row["Dtr(fp,Gibbs)"] = out["fp_vs_gibbs"]["trace_distance"]
        else:
            row["F(fp,Gibbs)"] = np.nan
            row["Dtr(fp,Gibbs)"] = np.nan

        rows.append(row)
        cases.append(out)

    df = pd.DataFrame(rows)
    return df, cases

def plot_energy_sweep_grid(cfg_runner, base_params, sweep_map, rho0=None, figsize=(18, 10)):
    """
    sweep_map ejemplo:
    {
        "beta":  [0.25, 0.5, 1.0, 1.5, 2.0],
        "theta": [0.05, 0.08, 0.10, 0.12, 0.15],
        ...
    }
    """
    names = list(sweep_map.keys())
    n_panels = len(names)
    nrows = int(np.ceil(n_panels / 3))
    ncols = min(3, n_panels)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    sweep_tables = {}

    for ax, name in zip(axes, names):
        values = sweep_map[name]
        df, cases = run_param_sweep(cfg_runner, base_params, name, values, rho0=rho0)
        sweep_tables[name] = df

        for val, out in zip(values, cases):
            line, = ax.plot(out["res"]["energies"], label=f"{name}={val}")
            c = line.get_color()
            ax.axhline(out["Eg"], color=c, ls="--", alpha=0.85)
            ax.axhline(out["Efp"], color=c, ls=":", alpha=0.85)

        ax.set_title(f"Barrido en {name}")
        ax.set_xlabel("reset")
        ax.set_ylabel("E")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for j in range(len(names), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Barridos de energía vs resets", y=1.02)
    plt.tight_layout()
    plt.show()

    print("=" * 90)
    print("TABLAS RESUMEN DE BARRIDOS")
    print("=" * 90)
    for name in names:
        print(f"\n--- Sweep: {name} ---")
        display(sweep_tables[name].round(6))

    return sweep_tables


# ============================================================
# HEATMAP 2D DE MÉTRICAS (MUY ÚTIL PARA RESTO DE HAMILTONIANOS)
# ============================================================

def plot_metric_heatmap(cfg_runner, base_params, x_name, x_values, y_name, y_values, metric="E_bias", rho0=None):
    """
    metric puede ser:
      - 'E_bias'        -> |Efp - Eg|
      - 'D_fp_gibbs'    -> Dtr(fp, Gibbs)
      - 'F_fp_gibbs'    -> F(fp, Gibbs)
      - 'PopErr_fp'     -> error de poblaciones del fp
      - 'CohErr_fp'     -> norma de coherencias del fp
    """
    Z = np.zeros((len(y_values), len(x_values)), dtype=float)

    for iy, yv in enumerate(y_values):
        for ix, xv in enumerate(x_values):
            params = base_params.copy()
            params[x_name] = xv
            params[y_name] = yv
            out = cfg_runner(params, rho0=rho0, compute_fp=True)

            if metric == "E_bias":
                val = abs(out["Efp"] - out["Eg"])
            elif metric == "D_fp_gibbs":
                val = out["fp_vs_gibbs"]["trace_distance"]
            elif metric == "F_fp_gibbs":
                val = out["fp_vs_gibbs"]["fidelity"]
            elif metric == "PopErr_fp":
                val = out["fp_vs_gibbs"]["population_error_l1"]
            elif metric == "CohErr_fp":
                val = out["fp_vs_gibbs"]["coherence_norm"]
            else:
                raise ValueError("Métrica no reconocida")

            Z[iy, ix] = val

    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[min(x_values), max(x_values), min(y_values), max(y_values)]
    )
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(f"Heatmap: {metric}")
    plt.colorbar(im, label=metric)
    plt.show()

    return Z


def plot_resource_scan(df, param_name, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(df[param_name], df["depth"], "o-", label="depth")
    axes[0].plot(df[param_name], df["size"], "s--", label="size")
    axes[0].set_title("Depth / size")
    axes[0].set_xlabel(param_name)
    axes[0].legend()

    axes[1].plot(df[param_name], df["cx"], "o-", label="cx")
    axes[1].plot(df[param_name], df["measure"], "s--", label="measure")
    axes[1].plot(df[param_name], df["reset"], "d-.", label="reset")
    axes[1].set_title("2-qubit y operaciones no unitarias")
    axes[1].set_xlabel(param_name)
    axes[1].legend()

    axes[2].plot(df[param_name], df["rz"], "o-", label="rz")
    axes[2].plot(df[param_name], df["sx"], "s--", label="sx")
    axes[2].plot(df[param_name], df["x"], "d-.", label="x")
    axes[2].set_title("Puertas nativas 1-qubit")
    axes[2].set_xlabel(param_name)
    axes[2].legend()

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    
def plot_system_extras(out, n_sys, title="", show_bath_heatmap=True):
    """
    Extras genéricos para cualquier sistema:
    - magnetización Z por qubit
    - población del GS y coherencia total en base de energía
    - pureza y entropía
    - observables del baño
    """
    res = out["res"]
    rhos = res["rhos"]
    Hs = out["Hs"]
    n_bath = out["cfg"].n_bath

    # magnetización Z por qubit
    mag_z = np.array([
        [expect(make_op(n_sys, [(1.0, {q: "Z"})]), rho) for q in range(n_sys)]
        for rho in rhos
    ], dtype=float)

    # base de energía
    pops_gs = []
    coh_total = []
    for rho in rhos:
        _, rho_e = energy_basis_matrix(rho, Hs)
        pops_gs.append(np.real(rho_e[0, 0]))
        off = rho_e - np.diag(np.diag(rho_e))
        coh_total.append(np.sum(np.abs(off)))

    # pureza y entropía
    pur = np.array([float(np.real(np.trace(rho @ rho))) for rho in rhos], dtype=float)

    def vn_entropy(rho):
        vals = np.linalg.eigvalsh(project_to_physical_dm(rho))
        vals = np.clip(np.real(vals), 1e-15, None)
        return float(-np.sum(vals * np.log(vals)))

    SvN = np.array([vn_entropy(rho) for rho in rhos], dtype=float)

    # baño
    bz = res["bath_z"]
    p_exc_b = (1 - bz) / 2

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # (1) Z del sistema
    for q in range(n_sys):
        ax[0, 0].plot(mag_z[:, q], label=f"qubit {q}")
    ax[0, 0].set_title(r"$\langle Z_q \rangle$ del sistema")
    ax[0, 0].set_xlabel("reset")
    ax[0, 0].legend(fontsize=7)
    ax[0, 0].grid(alpha=0.3)

    # (2) población GS + coherencia total
    ax[0, 1].plot(pops_gs, label=r"$\rho_{00}^{(E)}$")
    ax[0, 1].plot(coh_total, label=r"$\sum_{a\neq b} |\rho_{ab}^{(E)}|$")
    ax[0, 1].set_title("Base de energía")
    ax[0, 1].set_xlabel("reset")
    ax[0, 1].legend()
    ax[0, 1].grid(alpha=0.3)

    # (3) pureza + entropía
    ax[0, 2].plot(pur, label="pureza")
    ax[0, 2].plot(SvN, label="entropía vN")
    ax[0, 2].set_title("Pureza y entropía")
    ax[0, 2].set_xlabel("reset")
    ax[0, 2].legend()
    ax[0, 2].grid(alpha=0.3)

    # (4) Z del baño
    for mu in range(n_bath):
        ax[1, 0].plot(bz[:, mu], label=f"bath {mu}")
    ax[1, 0].set_title(r"$\langle Z_\mu^{\rm bath}\rangle$")
    ax[1, 0].set_xlabel("reset")
    ax[1, 0].legend(fontsize=7)
    ax[1, 0].grid(alpha=0.3)

    # (5) excitación del baño
    for mu in range(n_bath):
        ax[1, 1].plot(p_exc_b[:, mu], label=f"bath {mu}")
    ax[1, 1].set_title(r"$p_{\rm exc}$ del baño")
    ax[1, 1].set_xlabel("reset")
    ax[1, 1].legend(fontsize=7)
    ax[1, 1].grid(alpha=0.3)

    # (6) heatmap del baño
    if show_bath_heatmap:
        im = ax[1, 2].imshow(bz.T, aspect="auto", cmap="RdBu_r")
        ax[1, 2].set_title("Lectura del baño (Z)")
        ax[1, 2].set_xlabel("reset")
        ax[1, 2].set_ylabel("qubit baño")
        fig.colorbar(im, ax=ax[1, 2], fraction=0.046, pad=0.04)
    else:
        ax[1, 2].axis("off")

    fig.suptitle(title if title else f"Extras (n_sys={n_sys})")
    plt.tight_layout()
    plt.show()
    
    
def compare_parameter_cases(
    runner,
    cases,
    rho0,
    title_prefix="",
    legend_suffix="",
    fp_mode="auto",
):
    """
    Comparación genérica de casos etiquetados.
    cases = {
        "caso 1": {...params...},
        "caso 2": {...params...},
    }
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    rows = []
    outs = {}

    for i, (label, pars) in enumerate(cases.items()):
        out = runner(pars, rho0=rho0, compute_fp=True, fp_mode=fp_mode)
        outs[label] = out

        d = np.array([trace_distance_dm(r, out["rho_g"]) for r in out["res"]["rhos"]], dtype=float)
        color = color_cycle[i % len(color_cycle)]
        full_label = label + legend_suffix

        axes[0].plot(out["res"]["energies"], label=full_label, color=color, lw=2.4, alpha=0.95)
        axes[1].plot(d, label=full_label, color=color, lw=2.2, alpha=0.95)
        axes[2].plot(d, label=full_label, color=color, lw=2.2, alpha=0.95)

        if out["rho_fp"] is not None:
            axes[0].axhline(
                out["Efp"], ls="--", lw=2.0, color=color, alpha=0.9,
                label=fr"$E_{{\mathrm{{fp}}}}$ {label}"
            )

        rows.append({
            "label": label,
            "Egibbs": out["Eg"],
            "Efinal": out["Efinal"],
            "Efp": out["Efp"],
            "|Efinal-Eg|": abs(out["Efinal"] - out["Eg"]),
            "|Efp-Eg|": abs(out["Efp"] - out["Eg"]) if np.isfinite(out["Efp"]) else np.nan,
            "Dtr(final,Gibbs)": out["final_vs_gibbs"]["trace_distance"],
            "F(final,Gibbs)": out["final_vs_gibbs"]["fidelity"],
            "Dtr(fp,Gibbs)": out["fp_vs_gibbs"]["trace_distance"] if out["fp_vs_gibbs"] is not None else np.nan,
            "F(fp,Gibbs)": out["fp_vs_gibbs"]["fidelity"] if out["fp_vs_gibbs"] is not None else np.nan,
        })

    first_out = next(iter(outs.values()))
    axes[0].axhline(
        first_out["Eg"], ls=":", lw=2.6, color="black", alpha=0.9, label=r"$E_{\rm Gibbs}$"
    )

    axes[0].set_title("Energía")
    axes[1].set_title("Distancia a Gibbs (log)")
    axes[2].set_title("Distancia a Gibbs (lineal)")
    axes[1].set_yscale("log")

    for ax in axes:
        ax.set_xlabel("ciclo")
        ax.grid(alpha=0.28)

    axes[0].set_ylabel("E")
    axes[1].set_ylabel(r"$D_{\rm tr}(\rho_r,\rho_\beta)$")
    axes[2].set_ylabel(r"$D_{\rm tr}(\rho_r,\rho_\beta)$")

    axes[0].legend(fontsize=8, loc="best")
    axes[1].legend(fontsize=8, loc="best")
    axes[2].legend(fontsize=8, loc="best")

    plt.suptitle(title_prefix, y=1.03)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(rows)
    display(df.round(8))
    return df, outs

def compare_mt_cases(
    runner,
    cases,
    rho0,
    title_prefix="",
    legend_suffix="",
    fp_mode="auto",
):
    """
    Wrapper fino para casos tipo MT/delta.
    """
    return compare_parameter_cases(
        runner=runner,
        cases=cases,
        rho0=rho0,
        title_prefix=title_prefix,
        legend_suffix=legend_suffix,
        fp_mode=fp_mode,
    )
    
# ============================================================
# OBJETIVO 1.2 — COMPARAR A = X, Y, Z (CORREGIDO)
# ============================================================
    
def compare_system_couplings(
    runner,
    base_params,
    coupling_cases,
    n_sys,
    rho0_energy=None,
    rho0_coh=None,
    normalize_by_Eg=True,
    coherence_mode="total",  # "total" o "abs01"
):
    """
    Compara distintos operadores de acoplo sistema-baño.

    Muestra dos estados iniciales:
      fila 1: rho0 = I/d
      fila 2: rho0 = |+...+><+...+|

    Para cada uno muestra:
      1) energía
      2) coherencia en base de energía
      3) distancia a Gibbs

    coherence_mode:
      - "total": sum_{a!=b} |rho_ab^(E)|
      - "abs01": |rho_01^(E)|, útil para single spin
    """

    if rho0_energy is None:
        rho0_energy = maximally_mixed(n_sys)
    if rho0_coh is None:
        rho0_coh = all_plus(n_sys)

    initial_states = [
        (r"$\rho_0=I/d$", rho0_energy),
        (r"$\rho_0=|+\cdots+\rangle\langle+\cdots+|$", rho0_coh),
    ]

    def coherence_traj(rhos, Hs):
        vals = []
        for rho in rhos:
            if coherence_mode == "total":
                vals.append(coherence_norm_in_energy_basis(rho, Hs, norm="l1"))
            elif coherence_mode == "abs01":
                _, rho_e = energy_basis_matrix(rho, Hs)
                if rho_e.shape[0] < 2:
                    vals.append(0.0)
                else:
                    vals.append(abs(rho_e[0, 1]))
            else:
                raise ValueError("coherence_mode debe ser 'total' o 'abs01'")
        return np.asarray(vals, dtype=float)

    coh_label = (
        r"$\sum_{a\neq b}|\rho_{ab}^{(E)}|$"
        if coherence_mode == "total"
        else r"$|\rho_{01}^{(E)}|$"
    )

    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)

    for row_idx, (rho_label, rho0) in enumerate(initial_states):
        for name, extra in coupling_cases.items():
            p = base_params.copy()
            p.update(extra)

            out = runner(p, rho0=rho0, compute_fp=True)

            Hs = out["Hs"]
            rho_g = out["rho_g"]
            rhos = out["res"]["rhos"]
            cycles = np.arange(len(rhos))

            # Energía
            Eg = out["Eg"]
            E_traj = np.asarray(out["res"]["energies"], dtype=float)
            if normalize_by_Eg and abs(Eg) > 1e-14:
                E_plot = E_traj / Eg
                E_ylabel = r"$E/E_\beta$"
            else:
                E_plot = E_traj
                E_ylabel = r"$E$"

            # Coherencia y distancia
            coh = coherence_traj(rhos, Hs)
            dtr = np.asarray(
                [trace_distance_dm(rho, rho_g) for rho in rhos],
                dtype=float,
            )

            axes[row_idx, 0].plot(cycles, E_plot, lw=2, label=name)
            axes[row_idx, 1].plot(cycles, coh, lw=2, label=name)
            axes[row_idx, 2].plot(cycles, dtr, lw=2, label=name)

            rows.append({
                "rho0": rho_label,
                "case": name,
                "Efinal": out["Efinal"],
                "Egibbs": out["Eg"],
                "Efinal/Eg": out["Efinal"] / out["Eg"] if abs(out["Eg"]) > 1e-14 else np.nan,
                "|Efinal-Eg|": abs(out["Efinal"] - out["Eg"]),
                "Dtr(final,Gibbs)": out["final_vs_gibbs"]["trace_distance"],
                "F(final,Gibbs)": out["final_vs_gibbs"]["fidelity"],
                "Dtr(fp,Gibbs)": (
                    out["fp_vs_gibbs"]["trace_distance"]
                    if out["fp_vs_gibbs"] is not None else np.nan
                ),
                "F(fp,Gibbs)": (
                    out["fp_vs_gibbs"]["fidelity"]
                    if out["fp_vs_gibbs"] is not None else np.nan
                ),
                "coh_max": float(np.max(coh)),
                "coh_final": float(coh[-1]),
            })

        # Formato por fila
        if normalize_by_Eg:
            axes[row_idx, 0].axhline(
                1.0, ls="--", color="k", alpha=0.6,
                label=r"$E/E_\beta=1$" if row_idx == 0 else None,
            )

        axes[row_idx, 0].set_ylabel(rho_label + "\n" + E_ylabel)
        axes[row_idx, 1].set_ylabel(coh_label)
        axes[row_idx, 2].set_ylabel(r"$D_{\rm tr}(\rho_r,\rho_\beta)$")
        axes[row_idx, 2].set_yscale("log")

    axes[0, 0].set_title("Energía")
    axes[0, 1].set_title("Coherencia en base de energía")
    axes[0, 2].set_title("Distancia a Gibbs")

    for ax in axes[-1, :]:
        ax.set_xlabel("ciclo")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)

    axes[0, 2].legend(fontsize=8, loc="best")

    plt.suptitle("Comparación de acoplos relevantes", y=1.02)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(rows)
    display(df.round(6))
    return df

def compare_protocol_variants(
    runner,
    base_params,
    variant_cases,
    rho0,
    fp_mode="auto",
    n_random_seeds=1,
    coherence_mode="total",   # "total" o "single_spin_abs01"
    title="Comparación de variantes del protocolo",
):
    """
    Compara variantes del protocolo (base, random, rewind, ambos, etc.)
    mostrando:
      1) energía E(r)
      2) distancia a Gibbs (log)
      3) coherencia

    Parámetros
    ----------
    runner : callable
        Función tipo run_single_spin_case(params, rho0=..., compute_fp=..., fp_mode=...)
    base_params : dict
        Parámetros base del protocolo.
    variant_cases : dict
        Por ejemplo:
        {
            "base": {},
            "random": {"randomize": True, "randomization_lambda": 1.0},
            "rewind": {"rewind": True},
            "both": {"randomize": True, "randomization_lambda": 1.0, "rewind": True},
        }
    rho0 : np.ndarray
        Estado inicial.
    fp_mode : str
        "auto", "superop" o "late_time".
    n_random_seeds : int
        Número de semillas a promediar para variantes con randomize=True.
    coherence_mode : str
        - "total": suma total de coherencias off-diagonal en base de energía
        - "single_spin_abs01": |rho_01^(E)| (ideal para n_sys=1)
    title : str
        Título de la figura.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    rows = []
    outs = {}

    for name, extra in variant_cases.items():
        local_outs = []

        n_runs = n_random_seeds if extra.get("randomize", False) else 1
        for s in range(n_runs):
            p = base_params.copy()
            p.update(extra)
            p["seed"] = p.get("seed", 1234) + s

            out = runner(p, rho0=rho0, compute_fp=True, fp_mode=fp_mode)
            local_outs.append(out)

        # --- promedio de energías
        energies = np.mean([o["res"]["energies"] for o in local_outs], axis=0)

        # --- promedio de distancia a Gibbs
        dtr = np.mean([
            [trace_distance_dm(r, o["rho_g"]) for r in o["res"]["rhos"]]
            for o in local_outs
        ], axis=0)

        # --- promedio de coherencia
        if coherence_mode == "total":
            coh = np.mean([
                [coherence_norm_in_energy_basis(r, o["Hs"], norm="l1") for r in o["res"]["rhos"]]
                for o in local_outs
            ], axis=0)
            coh_label = r"$\sum_{a\neq b} |\rho_{ab}^{(E)}|$"
            coh_table_key = "coh_total_final_mean"
        elif coherence_mode == "single_spin_abs01":
            coh = np.mean([
                [np.abs(energy_basis_matrix(r, o["Hs"])[1][0, 1]) for r in o["res"]["rhos"]]
                for o in local_outs
            ], axis=0)
            coh_label = r"$|\rho_{01}^{(E)}|$"
            coh_table_key = "abs_rho01_final_mean"
        else:
            raise ValueError("coherence_mode debe ser 'total' o 'single_spin_abs01'")

        out0 = local_outs[0]

        # --- plots
        axes[0].plot(energies, lw=2, label=name)
        axes[1].plot(dtr, lw=2, label=name)
        axes[2].plot(coh, lw=2, label=name)

        # línea de Gibbs en energía
        axes[0].axhline(out0["Eg"], ls="--", color="k", alpha=0.12)

        # --- tabla
        row = {
            "variante": name,
            "n_runs": n_runs,
            "E_fp_mean": float(np.mean([o["Efp"] for o in local_outs if np.isfinite(o["Efp"])])) if any(np.isfinite(o["Efp"]) for o in local_outs) else np.nan,
            "Egibbs": out0["Eg"],
            "|E_fp-Eg|_mean": float(np.mean([
                abs(o["Efp"] - o["Eg"]) for o in local_outs if np.isfinite(o["Efp"])
            ])) if any(np.isfinite(o["Efp"]) for o in local_outs) else np.nan,
            "D_tr(fp,Gibbs)_mean": float(np.mean([
                o["fp_vs_gibbs"]["trace_distance"] for o in local_outs if o["fp_vs_gibbs"] is not None
            ])) if any(o["fp_vs_gibbs"] is not None for o in local_outs) else np.nan,
            "F(fp,Gibbs)_mean": float(np.mean([
                o["fp_vs_gibbs"]["fidelity"] for o in local_outs if o["fp_vs_gibbs"] is not None
            ])) if any(o["fp_vs_gibbs"] is not None for o in local_outs) else np.nan,
            "pop_err_fp_mean": float(np.mean([
                o["fp_vs_gibbs"]["population_error_l1"] for o in local_outs if o["fp_vs_gibbs"] is not None
            ])) if any(o["fp_vs_gibbs"] is not None for o in local_outs) else np.nan,
            coh_table_key: float(np.mean([coh[-1]])),
        }
        rows.append(row)
        outs[name] = local_outs

    # formato de ejes
    axes[0].set_title("Convergencia de la energía")
    axes[0].set_xlabel("reset")
    axes[0].set_ylabel("E")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Distancia a Gibbs (log)")
    axes[1].set_xlabel("reset")
    axes[1].set_ylabel(r"$D_{\rm tr}(\rho_r,\rho_\beta)$")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)

    axes[2].set_title("Coherencia en base de energía")
    axes[2].set_xlabel("reset")
    axes[2].set_ylabel(coh_label)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(rows)
    display(df.round(8))
    return df, outs

def compare_schedule_family(
    compiled_bank,
    schedule_cases,
    rho0_energy,
    rho0_coh,
    Hs=None,
    beta=None,
    random_seed=1234,
    normalize_by_Eg=True,
    title="Comparación de schedules",
):
    """
    schedule_cases = {
        "always_Y": "always_Y",
        "alternate_XY": "alternate_XY",
        "random_XY": "random_XY",
        "custom": lambda r, rng: ...
    }
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    rows = {}
    outs = {}

    for name, schedule in schedule_cases.items():
        out_E = run_scheduled_protocol(
            compiled_bank=compiled_bank,
            schedule=schedule,
            rho0=rho0_energy,
            Hs=Hs,
            beta=beta,
            random_seed=random_seed,
            compute_fp_ref=True,
        )
        out_C = run_scheduled_protocol(
            compiled_bank=compiled_bank,
            schedule=schedule,
            rho0=rho0_coh,
            Hs=Hs,
            beta=beta,
            random_seed=random_seed,
            compute_fp_ref=True,
        )

        Eg = out_E["Eg"]
        yE = out_E["energies"] / Eg if (normalize_by_Eg and abs(Eg) > 1e-14) else out_E["energies"]
        coh = np.array([
            np.abs(energy_basis_matrix(r, out_C["Hs"])[1][0, 1])
            for r in out_C["rhos"]
        ], dtype=float)
        dtr = np.array([
            trace_distance_dm(r, out_C["rho_g"])
            for r in out_C["rhos"]
        ], dtype=float)

        axes[0].plot(yE, lw=2, label=name)
        axes[1].plot(coh, lw=2, label=name)
        axes[2].plot(dtr, lw=2, label=name)

        rows[name] = {
            "schedule": name,
            "Efinal/Eg": out_E["Efinal"] / Eg if abs(Eg) > 1e-14 else np.nan,
            "Dtr(final,Gibbs)": out_C["D_final_gibbs"],
            "F(final,Gibbs)": out_C["F_final_gibbs"],
            "coh_final": coh[-1],
        }
        outs[name] = {"energy": out_E, "coh": out_C}

    axes[0].axhline(1.0, ls="--", color="k", alpha=0.7, label=r"$E/E_\beta=1$")
    axes[0].set_title(r"Energía desde $\rho_0=I/d$")
    axes[0].set_ylabel(r"$E/E_\beta$")
    axes[1].set_title(r"Dephasing desde $\rho_0=|+...+\rangle$")
    axes[1].set_ylabel(r"$|\rho_{01}^{(E)}|$")
    axes[2].set_title(r"Distancia a Gibbs desde $\rho_0=|+...+\rangle$")
    axes[2].set_ylabel(r"$D_{\rm tr}$")
    axes[2].set_yscale("log")

    for ax in axes:
        ax.set_xlabel("ciclo")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(list(rows.values()))
    display(df.round(6))
    return df, outs

def plot_protocol_variants_summary_bars(df, coherence_col=None, title="Resumen de variantes"):
    """
    Dibuja dos barras:
      - distancia traza fp vs Gibbs
      - descomposición población vs coherencia
    """
    if coherence_col is None:
        if "coh_total_final_mean" in df.columns:
            coherence_col = "coh_total_final_mean"
        elif "abs_rho01_final_mean" in df.columns:
            coherence_col = "abs_rho01_final_mean"
        else:
            raise ValueError("No encuentro columna de coherencia en df.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    axes[0].bar(df["variante"], df["D_tr(fp,Gibbs)_mean"])
    axes[0].set_title("Distancia traza al Gibbs en el punto fijo")
    axes[0].tick_params(axis='x', rotation=20)

    x = np.arange(len(df))
    w = 0.35
    axes[1].bar(x - w/2, df["pop_err_fp_mean"], width=w, label="pop error")
    axes[1].bar(x + w/2, df[coherence_col], width=w, label="coherencia")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["variante"], rotation=20)
    axes[1].set_title("Descomposición: poblaciones vs coherencias")
    axes[1].legend()

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


def theta_scaling_stationary_errors(
    runner,
    base_params,
    theta_grid,
    rho0=None,
    fp_mode="auto",
    randomization_cases=None,
    stationary_mode="fp",   # "fp" o "late_time_avg"
    late_time_keep=100,
    title="θ-scaling of stationary population/coherence errors",
    show_table=True,
):
    """
    Estudio tipo Fig. 2a del paper:
    errores estacionarios de poblaciones y coherencias vs θ².

    Parámetros
    ----------
    runner : callable
        Ej. run_single_spin_case
    base_params : dict
        Parámetros base del protocolo
    theta_grid : iterable
        Valores de theta
    rho0 : np.ndarray | None
        Estado inicial
    fp_mode : str
        "auto", "superop", "late_time"
    randomization_cases : dict | None
        Ejemplo:
        {
            "unrandomized": {"randomize": False, "randomization_lambda": 0.0},
            "randomized λ=1": {"randomize": True, "randomization_lambda": 1.0},
        }
    stationary_mode : str
        - "fp": usa out["rho_fp"]
        - "late_time_avg": promedio de las últimas `late_time_keep` matrices
    late_time_keep : int
        Número de estados para el promedio si stationary_mode="late_time_avg"
    """
    if randomization_cases is None:
        randomization_cases = {
            "unrandomized": {"randomize": False, "randomization_lambda": 0.0},
        }

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    for label, extra in randomization_cases.items():
        pop_errs = []
        coh_errs = []

        for th in theta_grid:
            p = base_params.copy()
            p["theta"] = float(th)
            p.update(extra)

            out = runner(p, rho0=rho0, compute_fp=True, fp_mode=fp_mode)

            if stationary_mode == "fp" and out["rho_fp"] is not None:
                rho_ref = out["rho_fp"]
            elif stationary_mode == "late_time_avg":
                rhos = out["res"]["rhos"]
                n_keep = min(late_time_keep, len(rhos))
                rho_ref = project_to_physical_dm(sum(rhos[-n_keep:]) / n_keep)
            else:
                raise ValueError("stationary_mode debe ser 'fp' o 'late_time_avg'")

            summary = state_error_summary(rho_ref, out["rho_g"], out["Hs"])

            pop_err = summary["population_error_l1"]
            coh_err = summary["coherence_norm"]

            pop_errs.append(pop_err)
            coh_errs.append(coh_err)

            rows.append({
                "case": label,
                "theta": th,
                "theta2": th**2,
                "population_error": pop_err,
                "coherence_error": coh_err,
                "trace_distance": summary["trace_distance"],
                "fidelity": summary["fidelity"],
                "fp_mode_used": out["fp_method_used"],
            })

        theta2 = np.array(theta_grid, dtype=float) ** 2
        axes[0].plot(theta2, pop_errs, "o-", label=label)
        axes[1].plot(theta2, coh_errs, "o-", label=label)

    axes[0].set_title("Population error vs $\\theta^2$")
    axes[0].set_xlabel(r"$\theta^2$")
    axes[0].set_ylabel("population error")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Coherence error vs $\\theta^2$")
    axes[1].set_xlabel(r"$\theta^2$")
    axes[1].set_ylabel("coherence error")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(rows)
    if show_table:
        display(df.round(8))
    return df

def randomization_vs_reset_time_scan(
    runner,
    base_params,
    reset_times,
    theta_fixed,
    rho0=None,
    delta=None,
    fp_mode="late_time",
    randomization_cases=None,
    stationary_mode="late_time_avg",
    late_time_keep=200,
    title="Randomized vs unrandomized protocol vs reset time",
    show_table=True,
):
    """
    Estudio tipo Fig. 2b del paper:
    compara randomized vs unrandomized al variar el reset time T = M*delta.

    Parámetros
    ----------
    reset_times : iterable
        Valores de T = M*delta a explorar
    theta_fixed : float
        Theta fijo
    delta : float | None
        Si None, usa base_params["delta"]
    randomization_cases : dict | None
        Ejemplo:
        {
            "unrandomized": {"randomize": False, "randomization_lambda": 0.0},
            "randomized λ=1": {"randomize": True, "randomization_lambda": 1.0},
            "randomized λ=5": {"randomize": True, "randomization_lambda": 5.0},
        }
    """
    if delta is None:
        delta = base_params["delta"]

    if randomization_cases is None:
        randomization_cases = {
            "unrandomized": {"randomize": False, "randomization_lambda": 0.0},
            "randomized λ=1": {"randomize": True, "randomization_lambda": 1.0},
        }

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    for label, extra in randomization_cases.items():
        pop_errs = []
        coh_errs = []

        for T in reset_times:
            p = base_params.copy()
            p["theta"] = float(theta_fixed)
            p["delta"] = float(delta)
            p["MT"] = max(int(round(T / delta)), 2)
            p.update(extra)

            out = runner(p, rho0=rho0, compute_fp=True, fp_mode=fp_mode)

            if stationary_mode == "fp" and out["rho_fp"] is not None:
                rho_ref = out["rho_fp"]
            else:
                rhos = out["res"]["rhos"]
                n_keep = min(late_time_keep, len(rhos))
                rho_ref = project_to_physical_dm(sum(rhos[-n_keep:]) / n_keep)

            summary = state_error_summary(rho_ref, out["rho_g"], out["Hs"])

            pop_err = summary["population_error_l1"]
            coh_err = summary["coherence_norm"]

            pop_errs.append(pop_err)
            coh_errs.append(coh_err)

            rows.append({
                "case": label,
                "T": T,
                "MT": p["MT"],
                "delta": delta,
                "theta": theta_fixed,
                "population_error": pop_err,
                "coherence_error": coh_err,
                "trace_distance": summary["trace_distance"],
                "fidelity": summary["fidelity"],
                "fp_mode_used": out["fp_method_used"],
            })

        axes[0].plot(reset_times, pop_errs, "o-", label=label)
        axes[1].plot(reset_times, coh_errs, "o-", label=label)

    axes[0].set_title("Population error vs reset time $T$")
    axes[0].set_xlabel(r"$T = M\delta$")
    axes[0].set_ylabel("population error")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Coherence error vs reset time $T$")
    axes[1].set_xlabel(r"$T = M\delta$")
    axes[1].set_ylabel("coherence error")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(rows)
    if show_table:
        display(df.round(8))
    return df

def theta_scaling_by_beta(
    runner,
    base_params,
    theta_grid,
    beta_grid,
    rho0=None,
    fp_mode="auto",
    stationary_mode="fp",   # "fp" o "late_time_avg"
    late_time_keep=100,
    randomize=False,
    randomization_lambda=0.0,
    title="θ² scaling of stationary errors for different β",
    show_table=True,
):
    """
    Figura tipo Fig. 2a del paper:
    errores absolutos estacionarios de población y coherencia vs θ²,
    comparando distintos β en los mismos plots.

    Parámetros
    ----------
    runner : callable
        Ej. run_single_spin_case
    base_params : dict
        Parámetros base
    theta_grid : iterable
        Valores de theta
    beta_grid : iterable
        Valores de beta
    rho0 : np.ndarray | None
        Estado inicial
    fp_mode : str
        "auto", "superop", "late_time"
    stationary_mode : str
        - "fp": usa out["rho_fp"]
        - "late_time_avg": promedio de las últimas matrices
    late_time_keep : int
        Número de estados si stationary_mode="late_time_avg"
    randomize : bool
        Si True, activa randomización
    randomization_lambda : float
        Parámetro λ de randomización
    """
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    for beta in beta_grid:
        
        p_beta = base_params.copy()
        p_beta["beta"] = float(beta)

        # --- FIX: MT adaptativo para que el filtro no se trunque ---
        a = np.sqrt(4.0 * p_beta["h"] / beta)
        MT_min = int(np.ceil(5.0 / (p_beta["delta"] * a)))
        p_beta["MT"] = max(base_params["MT"], MT_min)
        # ----------------------------------------------------------

        pop_errs = []
        coh_errs = []

        for th in theta_grid:
            p = p_beta.copy()
           # p = base_params.copy()
            p["beta"] = float(beta)
            p["theta"] = float(th)
            p["randomize"] = bool(randomize)
            p["randomization_lambda"] = float(randomization_lambda)

            out = runner(p, rho0=rho0, compute_fp=True, fp_mode=fp_mode)

            if stationary_mode == "fp" and out["rho_fp"] is not None:
                rho_ref = out["rho_fp"]
            elif stationary_mode == "late_time_avg":
                rhos = out["res"]["rhos"]
                n_keep = min(late_time_keep, len(rhos))
                rho_ref = project_to_physical_dm(sum(rhos[-n_keep:]) / n_keep)
            else:
                raise ValueError("stationary_mode debe ser 'fp' o 'late_time_avg'")

            # Error en base de energía
            _, rho_ref_e = energy_basis_matrix(rho_ref, out["Hs"])
            _, rho_g_e = energy_basis_matrix(out["rho_g"], out["Hs"])
            delta = rho_ref_e - rho_g_e

            # Single spin: análogo a |ζ00| y |ζ01|
            pop_abs = abs(np.real(delta[0, 0]))
            coh_abs = abs(delta[0, 1])

            pop_errs.append(pop_abs)
            coh_errs.append(coh_abs)

            rows.append({
                "beta": beta,
                "theta": th,
                "theta2": th**2,
                "population_abs_error": pop_abs,
                "coherence_abs_error": coh_abs,
                "trace_distance": trace_distance_dm(rho_ref, out["rho_g"]),
                "fidelity": fidelity_dm(rho_ref, out["rho_g"]),
                "randomize": randomize,
                "lambda_rand": randomization_lambda,
                "fp_mode_used": out["fp_method_used"],
            })

        theta2 = np.array(theta_grid, dtype=float) ** 2
        axes[0].plot(theta2, pop_errs, "o--", label=fr"$\beta={beta}$")
        axes[1].plot(theta2, coh_errs, "o--", label=fr"$\beta={beta}$")

    axes[0].set_title(r"Population error vs $\theta^2$")
    axes[0].set_xlabel(r"$\theta^2$")
    axes[0].set_ylabel(r"$|\zeta_{00}|$")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title(r"Coherence error vs $\theta^2$")
    axes[1].set_xlabel(r"$\theta^2$")
    axes[1].set_ylabel(r"$|\zeta_{01}|$")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(rows)
    if show_table:
        display(df.round(8))
    return df


def plot_filter_time_frequency_from_cfg(
    cfg_base,
    betas=(0.5, 1.0, 2.0, 5.0),
    MT=None,
    delta=None,
    h_filter=None,
):
    """
    Visualiza el filtro temporal y frecuencial.

    Acepta:
      - cfg_base como dict, por ejemplo base_single_params;
      - o cfg_base como objeto con atributos tipo ProtocolConfig.

    Usa gaussian_filter_values(cfg), pero construyendo un objeto mínimo
    con los atributos que esa función necesita.
    """

    def get_param(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    MT_base = get_param(cfg_base, "MT")
    delta_base = get_param(cfg_base, "delta")
    h_base = get_param(cfg_base, "h_filter", None)

    # Fallbacks por si en tus params el campo se llama distinto
    if h_base is None:
        h_base = get_param(cfg_base, "h", None)
    if h_base is None:
        h_base = get_param(cfg_base, "h_bath", None)

    if MT_base is None:
        raise KeyError("No encuentro 'MT' en cfg_base.")
    if delta_base is None:
        raise KeyError("No encuentro 'delta' en cfg_base.")
    if h_base is None and h_filter is None:
        raise KeyError("No encuentro 'h_filter', 'h' o 'h_bath' en cfg_base. Pásalo como h_filter=...")

    MT_used = MT_base if MT is None else MT
    delta_used = delta_base if delta is None else delta
    h_used = h_base if h_filter is None else h_filter

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for beta in betas:
        cfg = SimpleNamespace(
            MT=int(MT_used),
            delta=float(delta_used),
            beta=float(beta),
            h_filter=float(h_used),
            filter_values=None,
        )

        taus, fvals, a = gaussian_filter_values(cfg)

        t = cfg.delta * taus

        F = np.fft.fftshift(np.fft.fft(fvals))
        omega = 2 * np.pi * np.fft.fftshift(
            np.fft.fftfreq(len(fvals), d=cfg.delta)
        )

        F_abs = np.abs(F)
        F_abs = F_abs / F_abs.max()

        axes[0].plot(
            t,
            fvals,
            marker="o",
            ms=3,
            lw=1.5,
            label=rf"$\beta={beta}$",
        )

        axes[1].plot(
            omega,
            F_abs,
            lw=1.8,
            label=rf"$\beta={beta}$",
        )

    axes[0].set_title("Filtro temporal")
    axes[0].set_xlabel(r"$t=\delta\tau$")
    axes[0].set_ylabel(r"$f_\tau$")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Filtro frecuencial normalizado")
    axes[1].set_xlabel(r"$\omega$")
    axes[1].set_ylabel(r"$|\tilde f(\omega)|/\max|\tilde f|$")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.suptitle(
        rf"Filtro gaussiano: "
        rf"$f_\tau \propto e^{{-a^2\delta^2\tau^2/2}}$, "
        rf"$a=\sqrt{{4h/\beta}}$, "
        rf"$M_T={MT_used}$, $\delta={delta_used}$, $h={h_used}$"
    )

    plt.tight_layout()
    plt.show()
    
    
def plot_single_spin_compact_dashboard(out, title):
    res = out["res"]
    Hs = out["Hs"]
    rho_g = out["rho_g"]
    rhos = list(res["rhos"])

    cycles = np.arange(len(rhos))
    energies = np.asarray(res["energies"], dtype=float)

    dists = np.array([
        trace_distance_dm(rho, rho_g)
        for rho in rhos
    ])

    pops = []
    cohs = []
    for rho in rhos:
        _, rho_e = energy_basis_matrix(rho, Hs)
        pops.append(np.real(np.diag(rho_e)))
        cohs.append(abs(rho_e[0, 1]))

    pops = np.asarray(pops)
    cohs = np.asarray(cohs)

    fig, axes = plt.subplots(1, 5, figsize=(19, 3.5))

    # Energía
    axes[0].plot(cycles, energies, lw=2)
    axes[0].axhline(expect(Hs, rho_g), ls="--", color="k", alpha=0.6, label=r"$E_\beta$")
    if "rho_fp" in out and out["rho_fp"] is not None:
        axes[0].axhline(expect(Hs, out["rho_fp"]), ls=":", color="tab:red", alpha=0.8, label=r"$E_{\rm fp}$")
    axes[0].set_title("Energía")
    axes[0].set_xlabel("ciclo")
    axes[0].set_ylabel(r"$\langle H_S\rangle$")
    axes[0].legend(fontsize=8)

    # Distancia a Gibbs
    axes[1].plot(cycles, dists, lw=2)
    axes[1].set_yscale("log")
    axes[1].set_title(r"$D_{\rm tr}(\rho_r,\rho_\beta)$")
    axes[1].set_xlabel("ciclo")

    # Poblaciones
    axes[2].plot(cycles, pops[:, 0], lw=2, label=r"$p_0^{(E)}$")
    axes[2].plot(cycles, pops[:, 1], lw=2, label=r"$p_1^{(E)}$")
    axes[2].set_title("Poblaciones")
    axes[2].set_xlabel("ciclo")
    axes[2].set_ylabel(r"$p_i^{(E)}$")
    axes[2].legend(fontsize=8)

    # Coherencia
    axes[3].plot(cycles, cohs, lw=2, color="tab:purple")
    axes[3].set_title("Coherencia")
    axes[3].set_xlabel("ciclo")
    axes[3].set_ylabel(r"$|\rho_{01}^{(E)}|$")
    

    # Matriz final
    _, rho_final_e = energy_basis_matrix(rhos[-1], Hs)
    im = axes[4].imshow(np.abs(rho_final_e), vmin=0, vmax=1)
    axes[4].set_title(r"$|\rho_{\rm final}^{(E)}|$")
    axes[4].set_xticks([0, 1])
    axes[4].set_yticks([0, 1])
    axes[4].set_xlabel("columna")
    axes[4].set_ylabel("fila")
    fig.colorbar(im, ax=axes[4], shrink=0.75)

    fig.suptitle(title, y=1.05)
    plt.tight_layout()
    plt.show()
    
def single_spin_dashboard_stats(out):
    Hs = out["Hs"]
    rhos = list(out["res"]["rhos"])

    pops = []
    cohs = []

    for rho in rhos:
        _, rho_e = energy_basis_matrix(rho, Hs)
        pops.append(np.real(np.diag(rho_e)))
        cohs.append(abs(rho_e[0, 1]))

    pops = np.asarray(pops)
    cohs = np.asarray(cohs)

    return {
        "max_coherence": float(cohs.max()),
        "final_coherence": float(cohs[-1]),
        "final_pop_0": float(pops[-1, 0]),
        "final_pop_1": float(pops[-1, 1]),
    }
    
    
def scan_stationary_off_on_vs_x(
    runner,
    base_params,
    x_grid,
    metrics_fn,
    rho0=None,
    beta=1.0,
    theta=0.25,
    delta=0.20,
    n_cycles=600,
    seed_list=(101, 202, 303, 404, 505),
    randomization_lambda=1.0,
    fp_mode_off="superop",
    fp_mode_on="late_time",
    verbose=False,
):
    """
    Barrido del estado estacionario/punto fijo frente a

        x = T_eff g / pi,    T_eff = M_T delta.

    Compara:
      - OFF: canal determinista, punto fijo exacto por superoperador.
      - ON: canal randomizado, punto fijo aproximado late-time y promediado sobre seeds.

    Devuelve un DataFrame con poblaciones, coherencia y distancia a Gibbs.
    """

    if rho0 is None:
        rho0 = maximally_mixed(1)

    params = dict(base_params)
    params.update({
        "beta": beta,
        "theta": theta,
        "delta": delta,
        "n_cycles": n_cycles,
    })

    g = params["g"]

    rows = []

    for i, x in enumerate(x_grid):
        if verbose:
            print(f"[{i + 1}/{len(x_grid)}] x = {x:.3f}")

        T_eff = x * np.pi / g
        MT_val = max(int(round(T_eff / delta)), 1)

        # -------------------------
        # OFF: punto fijo exacto
        # -------------------------
        p_off = dict(
            params,
            MT=MT_val,
            randomize=False,
            randomization_lambda=0.0,
        )

        out_off = runner(
            p_off,
            rho0=rho0,
            compute_fp=True,
            fp_mode=fp_mode_off,
        )

        met_off = metrics_fn(
            out_off["rho_fp"],
            out_off["Hs"],
            out_off["rho_g"],
        )

        # -------------------------
        # ON: late-time + seeds
        # -------------------------
        mets_on = []

        for sd in seed_list:
            p_on = dict(
                params,
                MT=MT_val,
                randomize=True,
                randomization_lambda=randomization_lambda,
                seed=sd,
            )

            out_on = runner(
                p_on,
                rho0=rho0,
                compute_fp=True,
                fp_mode=fp_mode_on,
            )

            mets_on.append(
                metrics_fn(
                    out_on["rho_fp"],
                    out_on["Hs"],
                    out_on["rho_g"],
                )
            )

        def mean_key(key):
            return float(np.mean([m[key] for m in mets_on]))

        def std_key(key):
            return float(np.std([m[key] for m in mets_on], ddof=1)) if len(mets_on) > 1 else 0.0

        rows.append({
            "x": float(x),
            "MT": int(MT_val),

            "pg_off": float(met_off["p_g"]),
            "pe_off": float(met_off["p_e"]),
            "coh_off": float(met_off["coh_l1"]),
            "dist_off": float(met_off["D_to_gibbs"]),

            "pg_on_mean": mean_key("p_g"),
            "pe_on_mean": mean_key("p_e"),
            "coh_on_mean": mean_key("coh_l1"),
            "dist_on_mean": mean_key("D_to_gibbs"),

            "pg_on_std": std_key("p_g"),
            "pe_on_std": std_key("p_e"),
            "coh_on_std": std_key("coh_l1"),
            "dist_on_std": std_key("D_to_gibbs"),
        })

    return pd.DataFrame(rows).sort_values("x").reset_index(drop=True)

def plot_stationary_off_on_global_and_zoom(
    df_global,
    df_zoom,
    title="Randomización OFF/ON: barrido global y zoom",
    show_zoom_std=True,
):
    """
    Pinta en una única figura:
      fila 1: barrido global;
      fila 2: zoom cerca de la resonancia principal.

    Columnas:
      1) poblaciones
      2) coherencia
      3) distancia a Gibbs
    """

    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), sharex=False)

    # ========================================================
    # Fila 1 — barrido global
    # ========================================================
    ax = axes[0, 0]
    ax.plot(df_global["x"], df_global["pg_off"], lw=2, label=r"$p_g$ OFF")
    ax.plot(df_global["x"], df_global["pe_off"], "--", lw=2, label=r"$p_e$ OFF")
    ax.plot(df_global["x"], df_global["pg_on_mean"], lw=2, label=r"$p_g$ ON")
    ax.plot(df_global["x"], df_global["pe_on_mean"], "--", lw=2, label=r"$p_e$ ON")
    ax.set_title("Global: poblaciones")
    ax.set_xlabel(r"$x=T_{\rm eff}g/\pi$")
    ax.set_ylabel("población estacionaria")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(df_global["x"], df_global["coh_off"], lw=2, label="OFF")
    ax.plot(df_global["x"], df_global["coh_on_mean"], lw=2, label="ON")
    ax.set_title("Global: coherencia")
    ax.set_xlabel(r"$x=T_{\rm eff}g/\pi$")
    ax.set_ylabel(r"$\|\rho_{\rm ss}^{\rm offdiag}\|_1$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(df_global["x"], df_global["dist_off"], lw=2, label="OFF")
    ax.plot(df_global["x"], df_global["dist_on_mean"], lw=2, label="ON")
    ax.set_title("Global: distancia a Gibbs")
    ax.set_xlabel(r"$x=T_{\rm eff}g/\pi$")
    ax.set_ylabel(r"$D(\rho_{\rm ss},\rho_\beta)$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # ========================================================
    # Fila 2 — zoom
    # ========================================================
    ax = axes[1, 0]
    ax.plot(df_zoom["x"], df_zoom["pg_off"], "o-", lw=2, label=r"$p_g$ OFF")
    ax.plot(df_zoom["x"], df_zoom["pg_on_mean"], "s--", lw=2, label=r"$p_g$ ON")

    if show_zoom_std:
        ax.fill_between(
            df_zoom["x"],
            df_zoom["pg_on_mean"] - df_zoom["pg_on_std"],
            df_zoom["pg_on_mean"] + df_zoom["pg_on_std"],
            alpha=0.2,
        )

    ax.set_title(r"Zoom cerca de $x\simeq 2$: población")
    ax.set_xlabel(r"$x=T_{\rm eff}g/\pi$")
    ax.set_ylabel(r"$p_g$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(df_zoom["x"], df_zoom["coh_off"], "o-", lw=2, label="OFF")
    ax.plot(df_zoom["x"], df_zoom["coh_on_mean"], "s--", lw=2, label="ON")

    if show_zoom_std:
        ax.fill_between(
            df_zoom["x"],
            df_zoom["coh_on_mean"] - df_zoom["coh_on_std"],
            df_zoom["coh_on_mean"] + df_zoom["coh_on_std"],
            alpha=0.2,
        )

    ax.set_title(r"Zoom cerca de $x\simeq 2$: coherencia")
    ax.set_xlabel(r"$x=T_{\rm eff}g/\pi$")
    ax.set_ylabel(r"$\|\rho_{\rm ss}^{\rm offdiag}\|_1$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(df_zoom["x"], df_zoom["dist_off"], "o-", lw=2, label="OFF")
    ax.plot(df_zoom["x"], df_zoom["dist_on_mean"], "s--", lw=2, label="ON")

    if show_zoom_std:
        ax.fill_between(
            df_zoom["x"],
            df_zoom["dist_on_mean"] - df_zoom["dist_on_std"],
            df_zoom["dist_on_mean"] + df_zoom["dist_on_std"],
            alpha=0.2,
        )

    ax.set_title(r"Zoom cerca de $x\simeq 2$: distancia")
    ax.set_xlabel(r"$x=T_{\rm eff}g/\pi$")
    ax.set_ylabel(r"$D(\rho_{\rm ss},\rho_\beta)$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

    return fig, axes