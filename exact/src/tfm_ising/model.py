# tfm_ising/model.py
#
# Modelo Quantum Ising (1D y 2D).
#
# Hamiltoniano:
#   H_S = -J sum_{<i,j>} X_i X_j - g sum_i Z_i
#
# En 1D los enlaces son la cadena i <-> i+1.
# En 2D los enlaces son los del retículo cuadrado Lx x Ly.
#
# A diferencia del notebook NI (no-interactuante), el término JX_iX_j introduce
# correlaciones genuinas entre qubits. El Gibbs state ya no es factorizable,
# la mutual information entre mitades es positiva, y a tamaños suficientes
# aparecen transiciones de fase:
#   - 1D: transición cuántica en |J| = g a beta -> inf. Sin transición térmica.
#   - 2D: transición cuántica en J_c ~ 0.33 g; transición térmica para g=0 en
#         beta_c ~ 0.44 / J.
#
# Este módulo reutiliza toda la maquinaria de tfm_shared.core (el builder
# del protocolo es el mismo) y añade:
#   - build_ising_cfg(...)          construye ProtocolConfig para 1D o 2D
#   - run_ising_case(...)           wrapper sobre run_protocol_case
#   - heat_capacity_ising(...)      C_V de variance(H)
#   - susceptibility_ising(...)     chi = beta <M^2> con M = sum Z_i
#   - scan_J_beta_grid(...)         barrido 2D en (J, beta), devuelve observables
#   - mutual_information_halves(...) reusado de nonint para medir correlaciones
#
# Para Fig. 3-5 del paper Lloyd-Abanin (2D Quantum Ising en 4x4) conviene
# fijar g=1 y barrer beta y J por separado. Los plots concretos se hacen en
# el notebook NB03_ising.ipynb.

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm

from tfm_shared.core import (
    ProtocolConfig,
    default_bath_terms,
    make_op,
    mat,
    maximally_mixed,
    compile_protocol,
    run_protocol_case,
    gibbs_state,
    expect,
    trace_distance_dm,
    fidelity_dm,
    ptrace_bath,
    ptrace_system,
    project_to_physical_dm,
    ising_chain_xx_z_terms,
    ising_2d_xx_z_terms,
    heat_capacity_from_variance,
)

from tfm_single_spin.model import one_qubit_pauli_terms


# ============================================================
# CONSTRUCCIÓN DEL CONFIG ISING
# ============================================================

def build_ising_cfg(
    Lx=2, Ly=1,
    J=0.5, g=1.0,
    h=1.0,
    beta=1.0,
    theta=0.15,
    delta=0.08,
    MT=30,
    n_cycles=150,
    sys_A="Y",
    bath_A="Y",
    periodic=False,
    randomize=False,
    randomization_lambda=0.0,
    seed=1234,
    rewind=False,
):
    """
    Configura el protocolo Lloyd-Abanin para Quantum Ising.

    Si Ly == 1, estamos en 1D (cadena de n = Lx sitios).
    Si Ly >= 2, 2D (retículo Lx x Ly).

    H_S = -J sum_{<i,j>} X_i X_j - g sum_i Z_i

    El resto de campos (baño, acoplo, filtro) sigue la misma estructura que
    los modelos anteriores: un auxiliar por qubit del sistema, con acoplo
    A_S^(i) ⊗ A_B^(i) local qubit a qubit.
    """
    n_sys = Lx * Ly
    n_bath = n_sys

    # H_S: 1D o 2D según Ly
    if Ly == 1:
        system_terms = ising_chain_xx_z_terms(n_sys, J=J, g=g, periodic=periodic)
    else:
        system_terms = ising_2d_xx_z_terms(Lx, Ly, J=J, g=g, periodic=periodic)

    # H_B: h/2 Z en cada auxiliar
    bath_terms = default_bath_terms(n_bath, h=h)

    # Acoplos locales: un acoplo por qubit (misma filosofía que NI)
    sys_terms_q0 = one_qubit_pauli_terms(sys_A)
    bath_terms_q0 = one_qubit_pauli_terms(bath_A)

    coupling_specs = []
    for i in range(n_sys):
        sys_terms_i = [(c, {i: list(ops.values())[0]}) for (c, ops) in sys_terms_q0]
        bath_terms_i = [(c, {i: list(ops.values())[0]}) for (c, ops) in bath_terms_q0]
        coupling_specs.append({
            "sys_terms":  sys_terms_i,
            "bath_terms": bath_terms_i,
        })

    return ProtocolConfig(
        n_sys=n_sys,
        n_bath=n_bath,
        system_terms=system_terms,
        bath_terms=bath_terms,
        coupling_specs=coupling_specs,
        beta=beta,
        h_filter=h,
        theta=theta,
        delta=delta,
        MT=MT,
        n_cycles=n_cycles,
        randomize=randomize,
        randomization_lambda=randomization_lambda,
        seed=seed,
        rewind=rewind,
    )


def run_ising_case(
    params,
    rho0=None,
    compute_fp=True,
    fp_mode="auto",
    late_fp_burn=200,
    late_fp_keep=50,
):
    """
    Ejecuta un caso Ising 1D o 2D. Mismo contrato que run_nonint_case o
    run_single_spin_case: params es un dict con los kwargs de build_ising_cfg.
    """
    cfg = build_ising_cfg(**params)
    return run_protocol_case(
        cfg,
        rho0=rho0,
        compute_fp=compute_fp,
        fp_mode=fp_mode,
        late_fp_burn=late_fp_burn,
        late_fp_keep=late_fp_keep,
    )


# ============================================================
# OBSERVABLES TERMODINÁMICOS
# ============================================================

def total_magnetization_X(n_sys):
    """
    M_X = sum_i X_i como matriz densa. El paper usa M_Z; nosotros tenemos
    g Z_i como término longitudinal, de modo que el orden ferromagnético
    aparece en X (dirección de los términos -J X_i X_j).
    """
    return mat(make_op(n_sys, [(1.0, {i: "X"}) for i in range(n_sys)]))


def total_magnetization_Z(n_sys):
    """M_Z = sum_i Z_i."""
    return mat(make_op(n_sys, [(1.0, {i: "Z"}) for i in range(n_sys)]))


def heat_capacity_ising(rho, Hs, beta):
    """C_V = beta^2 Var_rho(H) usando la función utilitaria compartida."""
    return heat_capacity_from_variance(rho, Hs, beta)


def susceptibility_X(rho, n_sys, beta):
    """
    Susceptibilidad magnética en la dirección X:
        chi_X = beta ( <M_X^2> - <M_X>^2 )_rho

    En Ising ferromagnético (J > 0), esta magnitud es la que diverge
    cerca del punto crítico térmico en 2D.
    """
    Mx = total_magnetization_X(n_sys)
    Mx2 = Mx @ Mx
    m1 = float(np.real(np.trace(rho @ Mx)))
    m2 = float(np.real(np.trace(rho @ Mx2)))
    return float(beta * (m2 - m1**2))


def binder_cumulant_X(rho, n_sys):
    """
    Cumulante de Binder U_4 = 1 - <M_X^4> / (3 <M_X^2>^2).

    Es un observable clásico para identificar transiciones de fase:
    las curvas U_4 vs T para distintos L se cruzan en T_c.
    En este TFM lo usamos solo como diagnóstico cualitativo, porque con
    los tamaños accesibles (N_S <= 9 o 16) la señal es ruidosa.
    """
    Mx = total_magnetization_X(n_sys)
    Mx2 = Mx @ Mx
    Mx4 = Mx2 @ Mx2
    m2 = float(np.real(np.trace(rho @ Mx2)))
    m4 = float(np.real(np.trace(rho @ Mx4)))
    if m2 <= 1e-12:
        return 0.0
    return float(1.0 - m4 / (3.0 * m2**2))


def mutual_information_halves(rho, n_sys):
    """
    I(A:A_bar) = S(A) + S(A_bar) - S(rho), partiendo el sistema en dos mitades.
    Es la métrica "estrella" para medir correlaciones globales.
    """
    if n_sys < 2:
        return 0.0

    nA = n_sys // 2
    nB = n_sys - nA

    rho_A = ptrace_bath(rho, nA, nB)
    rho_B = ptrace_system(rho, nA, nB)

    def vn(r):
        r = project_to_physical_dm(r)
        vals = np.linalg.eigvalsh(r)
        vals = np.clip(np.real(vals), 1e-15, None)
        return float(-np.sum(vals * np.log(vals)))

    return vn(rho_A) + vn(rho_B) - vn(rho)


def single_site_entropy(rho, n_sys, site=0):
    """
    S(rho_site) = -Tr(rho_site log rho_site) para el qubit 'site'.
    En el Gibbs factorizado (NI) es log 2; en la fase ordenada 2D baja.
    """
    dims = [2] * n_sys
    rho_t = rho.reshape(dims + dims)
    keep = site
    trace_q = [q for q in range(n_sys) if q != keep]
    for q in sorted(trace_q, reverse=True):
        rho_t = np.trace(rho_t, axis1=q, axis2=q + len(dims))
        dims = dims[:q] + dims[q+1:]
    rho_site = rho_t.reshape(2, 2)
    rho_site = project_to_physical_dm(rho_site)
    vals = np.linalg.eigvalsh(rho_site)
    vals = np.clip(np.real(vals), 1e-15, None)
    return float(-np.sum(vals * np.log(vals)))


# ============================================================
# BARRIDO EN (J, beta) — diagrama de fase Ising 2D
# ============================================================

def scan_J_beta_grid(
    base_params,
    J_values,
    beta_values,
    observables=("E", "C", "chiX", "I_halves"),
    rho0=None,
    fp_mode="auto",
    verbose=False,
):
    """
    Barre una rejilla (J, beta) y devuelve un DataFrame con observables del
    punto fijo en cada celda. Pensado para generar mapas 2D tipo diagrama
    de fase (Fig. 4 paper Lloyd-Abanin).

    observables admite: "E", "E_per_site", "C", "chiX", "I_halves",
                        "Dtr_gibbs", "bias_E", "binder"

    Lo pesado de esta función es que para cada celda construye el cfg y
    ejecuta el protocolo. Pensar los tamaños: con una rejilla 10x10 son
    100 runs. Aceptable para Ising 1D; ajustado para 2D en 3x3.
    """
    rows = []

    for J in J_values:
        for beta in beta_values:
            p = dict(base_params, J=float(J), beta=float(beta))
            cfg = build_ising_cfg(**p)
            n_sys = cfg.n_sys

            out = run_protocol_case(
                cfg, rho0=rho0, compute_fp=True, fp_mode=fp_mode,
            )

            rho_fp = out["rho_fp"]
            if rho_fp is None:
                rho_fp = out["res"]["rhos"][-1]
            Hs = out["Hs"]
            rho_g = out["rho_g"]

            row = {"J": float(J), "beta": float(beta), "n_sys": int(n_sys)}
            if "E" in observables:
                row["E"] = float(expect(Hs, rho_fp))
            if "E_per_site" in observables:
                row["E_per_site"] = float(expect(Hs, rho_fp)) / n_sys
            if "bias_E" in observables:
                row["bias_E"] = abs(float(expect(Hs, rho_fp)) - float(expect(Hs, rho_g)))
            if "C" in observables:
                row["C"] = heat_capacity_ising(rho_fp, Hs, beta)
                row["C_gibbs"] = heat_capacity_ising(rho_g, Hs, beta)
            if "chiX" in observables:
                row["chiX"] = susceptibility_X(rho_fp, n_sys, beta)
            if "I_halves" in observables:
                row["I_halves"] = mutual_information_halves(rho_fp, n_sys)
            if "Dtr_gibbs" in observables:
                row["Dtr_gibbs"] = trace_distance_dm(rho_fp, rho_g)
            if "binder" in observables:
                row["binder"] = binder_cumulant_X(rho_fp, n_sys)

            rows.append(row)

            if verbose:
                print(f"[J={J:.3f}, beta={beta:.3f}] done. "
                      f"n_sys={n_sys}, E={row.get('E', np.nan):.4f}")

    return pd.DataFrame(rows)


# ============================================================
# BARRIDO EN N_S (longitud de la cadena 1D o lado del retículo 2D)
# ============================================================

def scan_1d_sizes(base_params, N_list, rho0=None, fp_mode="auto"):
    """
    Escalado N_S en Ising 1D: fija J, beta, theta, etc. y barre el largo
    de la cadena. Para N pequeños usa superop exacto; para N grandes cae
    automáticamente a late_time.
    """
    rows = []
    trajs = {}
    for N in N_list:
        p = dict(base_params, Lx=int(N), Ly=1)
        cfg = build_ising_cfg(**p)
        out = run_protocol_case(cfg, rho0=None, compute_fp=True, fp_mode=fp_mode)

        rho_fp = out["rho_fp"] if out["rho_fp"] is not None else out["res"]["rhos"][-1]
        Hs = out["Hs"]
        rho_g = out["rho_g"]

        rows.append({
            "N_S":        N,
            "E":          float(expect(Hs, rho_fp)),
            "E_per_site": float(expect(Hs, rho_fp)) / N,
            "E_beta":     float(expect(Hs, rho_g)),
            "bias_per_site": abs(float(expect(Hs, rho_fp)) - float(expect(Hs, rho_g))) / N,
            "Dtr":        trace_distance_dm(rho_fp, rho_g),
            "I_halves":   mutual_information_halves(rho_fp, N),
        })
        trajs[N] = out

    return pd.DataFrame(rows), trajs


def scan_2d_sizes(base_params, LxLy_list, rho0=None, fp_mode="auto"):
    """
    Escalado en Ising 2D: barre lattices cuadradas Lx x Ly = (2x2, 3x3, ...).
    Para 3x3 (9 qubits) el superop ya cuesta; para 4x4 (16 qubits) fuerza
    late_time.
    """
    rows = []
    trajs = {}
    for Lx, Ly in LxLy_list:
        p = dict(base_params, Lx=int(Lx), Ly=int(Ly))
        cfg = build_ising_cfg(**p)
        n = Lx * Ly
        out = run_protocol_case(cfg, rho0=None, compute_fp=True, fp_mode=fp_mode)

        rho_fp = out["rho_fp"] if out["rho_fp"] is not None else out["res"]["rhos"][-1]
        Hs = out["Hs"]
        rho_g = out["rho_g"]

        rows.append({
            "Lx": Lx, "Ly": Ly, "n_sys": n,
            "E":          float(expect(Hs, rho_fp)),
            "E_per_site": float(expect(Hs, rho_fp)) / n,
            "E_beta":     float(expect(Hs, rho_g)),
            "bias_per_site": abs(float(expect(Hs, rho_fp)) - float(expect(Hs, rho_g))) / n,
            "Dtr":        trace_distance_dm(rho_fp, rho_g),
            "I_halves":   mutual_information_halves(rho_fp, n),
            "C":          heat_capacity_ising(rho_fp, Hs, base_params["beta"]) / n,
        })
        trajs[(Lx, Ly)] = out

    return pd.DataFrame(rows), trajs


# ============================================================
# HELPERS NUEVOS PARA NB03
# ============================================================

def plus_density_matrix(n_sys):
    """Estado |+><+|^{\otimes n_sys} como matriz densa."""
    plus_ket_1 = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    plus_ket = plus_ket_1.copy()
    for _ in range(int(n_sys) - 1):
        plus_ket = np.kron(plus_ket, plus_ket_1)
    return np.outer(plus_ket, plus_ket.conj())


def rho_metrics_ising(rho, Hs, rho_g=None):
    """
    Mismas metricas densas que en NB01/NB02, ahora para Ising.
    """
    from tfm_shared.core import (
        coherence_norm_in_energy_basis,
        energy_basis_matrix,
        population_error_l1_in_energy_basis,
    )

    evals, rho_e = energy_basis_matrix(rho, Hs)
    off = rho_e.copy()
    np.fill_diagonal(off, 0.0)

    out = {
        "E": float(expect(Hs, rho)),
        "coh_l1": float(np.sum(np.abs(off))),
        "coh_fro": float(np.linalg.norm(off)),
        "coh_energy_basis": float(coherence_norm_in_energy_basis(rho, Hs, norm="l1")),
        "rho_e": rho_e,
        "evals": evals,
    }

    if rho_g is not None:
        out["D_to_gibbs"] = float(trace_distance_dm(rho, rho_g))
        out["F_to_gibbs"] = float(fidelity_dm(rho, rho_g))
        out["pop_err_l1"] = float(population_error_l1_in_energy_basis(rho, rho_g, Hs))
    else:
        out["D_to_gibbs"] = np.nan
        out["F_to_gibbs"] = np.nan
        out["pop_err_l1"] = np.nan

    return out


def avg_metric_dicts(dicts, keys):
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


def x_to_Teff(x, g):
    return float(x) * np.pi / float(g)


def x_to_MT(x, g, delta):
    return max(int(round(x_to_Teff(x, g) / float(delta))), 1)


def make_xgrid(xmin=0.5, xmax=4.0, n=41):
    return np.linspace(float(xmin), float(xmax), int(n))


def diagnose_ising_attractor_over_T(
    base_params,
    xgrid,
    rho0_a=None,
    rho0_b=None,
    tol_one=1e-8,
    tol_unit=1e-8,
    dyn_tol=1e-3,
):
    """
    Diagnostico espectral del atractor para Ising 1D/2D.
    Adaptacion directa del helper usado en single-spin y NB02.
    """
    from tfm_shared.core import fixed_point_via_superop, maximally_mixed

    n_sys = int(base_params["Lx"]) * int(base_params.get("Ly", 1))
    g = float(base_params["g"])
    delta = float(base_params["delta"])

    if rho0_a is None:
        rho0_a = maximally_mixed(n_sys)
    if rho0_b is None:
        rho0_b = plus_density_matrix(n_sys)

    rows = []
    for x in xgrid:
        MT_val = x_to_MT(x, g=g, delta=delta)
        params = dict(base_params, MT=MT_val, randomize=False, randomization_lambda=0.0)

        out_ref = run_ising_case(
            params,
            rho0=rho0_a,
            compute_fp=False,
        )
        rho_fp, _, evals = fixed_point_via_superop(out_ref["compiled"])

        abs_evals = np.sort(np.abs(evals))[::-1]
        second_abs = float(abs_evals[1]) if len(abs_evals) > 1 else np.nan
        mult_one = int(np.sum(np.abs(evals - 1.0) < tol_one))
        n_unit = int(np.sum(np.abs(np.abs(evals) - 1.0) < tol_unit))

        rho_a_final = out_ref["res"]["rhos"][-1]
        out_b = run_ising_case(
            params,
            rho0=rho0_b,
            compute_fp=False,
        )
        rho_b_final = out_b["res"]["rhos"][-1]

        d_a_fp = float(trace_distance_dm(rho_a_final, rho_fp))
        d_b_fp = float(trace_distance_dm(rho_b_final, rho_fp))
        d_a_b = float(trace_distance_dm(rho_a_final, rho_b_final))

        rows.append({
            "x_Teff_g_over_pi": float(x),
            "MT": int(MT_val),
            "mult_lambda_1": mult_one,
            "n_unit_modulus": n_unit,
            "|lambda2|": second_abs,
            "gap_1-|lambda2|": float(1.0 - second_abs) if np.isfinite(second_abs) else np.nan,
            "D(final_I/d, fp)": d_a_fp,
            "D(final_plus, fp)": d_b_fp,
            "D(final_I/d, final_plus)": d_a_b,
            "unique_attractor_likely": bool((mult_one == 1) and (second_abs < 1.0 - 1e-8)),
            "dynamic_convergence_likely": bool(
                (d_a_fp < dyn_tol) and (d_b_fp < dyn_tol) and (d_a_b < dyn_tol)
            ),
        })

    return pd.DataFrame(rows).sort_values("x_Teff_g_over_pi").reset_index(drop=True)


# Utilidades para A.1.1, A.2.1 y H

def product_basis_dm(bitstring):
    """
    |bitstring><bitstring| en base computacional.
    Ej: '000', '0101', ...
    """
    bits = [int(b) for b in bitstring]
    n = len(bits)
    d = 2**n
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    psi = np.zeros(d, dtype=complex)
    psi[idx] = 1.0
    return np.outer(psi, psi.conj())


def vn_entropy(rho):
    """
    Entropía de von Neumann: S(rho) = -Tr rho log rho
    """
    rho = project_to_physical_dm(rho)
    vals = np.linalg.eigvalsh(rho)
    vals = np.clip(np.real(vals), 1e-15, None)
    return float(-np.sum(vals * np.log(vals)))


def purity_dm(rho):
    """
    Pureza: Tr(rho^2)
    """
    return float(np.real(np.trace(rho @ rho)))


def reduced_dm_keep(rho, keep, n_sys):
    """
    Matriz reducida conservando los qubits de 'keep'.
    keep: lista de índices de qubits a conservar.
    """
    keep = sorted(keep)
    dims = [2] * n_sys
    rho_t = rho.reshape(dims + dims)

    traced = [q for q in range(n_sys) if q not in keep]
    n_cur = n_sys
    for q in sorted(traced, reverse=True):
        rho_t = np.trace(rho_t, axis1=q, axis2=q + n_cur)
        n_cur -= 1

    d = 2 ** len(keep)
    return rho_t.reshape(d, d)


def mutual_information_partition(rho, A, n_sys):
    """
    I(A : A^c) para una partición arbitraria A | A^c.
    """
    A = sorted(A)
    B = [q for q in range(n_sys) if q not in A]
    rho_A = reduced_dm_keep(rho, A, n_sys)
    rho_B = reduced_dm_keep(rho, B, n_sys)
    return vn_entropy(rho_A) + vn_entropy(rho_B) - vn_entropy(rho)


def dephase_computational_basis(rho):
    """
    Desfasa completamente en la base computacional.
    Útil como proxy clásico en base Z.
    """
    return np.diag(np.diag(rho)).astype(complex)


def negativity_twoqubit(rho2):
    """
    Negatividad para un estado de 2 qubits (4x4).
    Si > 0, hay entrelazamiento seguro.
    Si = 0, NO implica ausencia de correlación cuántica.
    """
    rho_pt = rho2.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    rho_pt = 0.5 * (rho_pt + rho_pt.conj().T)
    evals = np.linalg.eigvalsh(rho_pt)
    return float(np.sum(np.abs(evals[evals < 0])))


def run_multistart_ising(case_params, states_dict, fp_mode="superop",
                         late_fp_burn=120, late_fp_keep=30):
    """
    Ejecuta el mismo caso Ising para varios estados iniciales.
    Devuelve:
      - runs: dict con trayectorias y outputs completos
      - df: tabla resumen del estado final tras n_cycles
    """
    runs = {}
    rows = []

    n_sys = case_params["Lx"] * case_params["Ly"]

    for label, rho0 in states_dict.items():
        out = run_ising_case(
            case_params,
            rho0=rho0,
            compute_fp=True,
            fp_mode=fp_mode,
            late_fp_burn=late_fp_burn,
            late_fp_keep=late_fp_keep,
        )

        rhos = out["res"]["rhos"]
        Hs = out["Hs"]
        rho_g = out["rho_g"]
        rho_last = rhos[-1]

        E_traj = np.array(out["res"]["energies"], dtype=float) / n_sys
        Dg_traj = np.array([trace_distance_dm(r, rho_g) for r in rhos], dtype=float)

        runs[label] = {
            "out": out,
            "E_traj_per_site": E_traj,
            "Dg_traj": Dg_traj,
        }

        rows.append({
            "init": label,
            "E0/N": E_traj[0],
            "E_last/N": E_traj[-1],
            "E_Gibbs/N": float(expect(Hs, rho_g)) / n_sys,
            "Dtr(ρ0,ρβ)": Dg_traj[0],
            "Dtr(ρR,ρβ)": Dg_traj[-1],
            "F(ρR,ρβ)": fidelity_dm(rho_last, rho_g),
            "S(ρR)": vn_entropy(rho_last),
            "purity(ρR)": purity_dm(rho_last),
            "Δ_last": trace_distance_dm(rhos[-1], rhos[-2]) if len(rhos) >= 2 else np.nan,
        })

    df = pd.DataFrame(rows)
    return runs, df


def plot_multistart_convergence(runs, title, E_gibbs_per_site, E_fp_per_site=None):
    """
    Plot 3-panel tipo:
      - Energía por qubit vs ciclos
      - Dtr a Gibbs (log)
      - Dtr a Gibbs (lineal)
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    for label, rr in runs.items():
        x = np.arange(len(rr["E_traj_per_site"]))
        axes[0].plot(x, rr["E_traj_per_site"], lw=2, label=label)
        axes[1].plot(x, rr["Dg_traj"], lw=2, label=label)
        axes[2].plot(x, rr["Dg_traj"], lw=2, label=label)

    axes[0].axhline(E_gibbs_per_site, ls="--", color="k", lw=1.8, label=r"$E_\beta/N_S$")
    if E_fp_per_site is not None:
        axes[0].axhline(E_fp_per_site, ls=":", color="tab:red", lw=1.8, label=r"$E_{\rm fp}/N_S$")

    axes[0].set_title("Energía por qubit vs ciclos")
    axes[0].set_xlabel("ciclo")
    axes[0].set_ylabel(r"$\langle H_S\rangle / N_S$")

    axes[1].set_title("Distancia a Gibbs (log)")
    axes[1].set_xlabel("ciclo")
    axes[1].set_ylabel(r"$D_{\rm tr}(\rho_r,\rho_\beta)$")
    axes[1].set_yscale("log")

    axes[2].set_title("Distancia a Gibbs (lineal)")
    axes[2].set_xlabel("ciclo")
    axes[2].set_ylabel(r"$D_{\rm tr}$")

    for ax in axes:
        ax.grid(alpha=0.3)

    axes[0].legend()
    axes[1].legend()
    fig.suptitle(title, y=1.03)
    plt.tight_layout()
    plt.show()



# ============================================================
# Bundle ISA para NB03 (Ising 1D y 2D)
# ============================================================

ISA_CIRCUIT_CACHE_ISING = {}
ISING_NATIVE_BASIS = ["rz", "sx", "x", "cx", "measure", "reset"]
_ISING_BACKEND_BUNDLE = None


def make_compiled_ising(params):
    cfg = build_ising_cfg(**params)
    compiled = compile_protocol(cfg)
    return cfg, compiled


def get_ising_backend_bundle(min_total_qubits=None, force_refresh=False):
    """
    Construye backend ideal/noisy para experimentos ISA.
    Usa FakeSherbrooke si está; si no, GenericBackendV2.
    """
    global _ISING_BACKEND_BUNDLE

    min_total_qubits = 2 if min_total_qubits is None else int(min_total_qubits)
    if (not force_refresh) and (_ISING_BACKEND_BUNDLE is not None):
        backend = _ISING_BACKEND_BUNDLE["hw_backend"]
        if getattr(backend, "num_qubits", min_total_qubits) >= min_total_qubits:
            return _ISING_BACKEND_BUNDLE

    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        hw_backend = FakeSherbrooke()
        label = "FakeSherbrooke (snapshot calibraciones reales)"
    except Exception:
        from qiskit.providers.fake_provider import GenericBackendV2
        n_qubits = max(min_total_qubits, 2)
        hw_backend = GenericBackendV2(
            num_qubits=n_qubits,
            basis_gates=ISING_NATIVE_BASIS,
            coupling_map=[[i, i + 1] for i in range(n_qubits - 1)],
            control_flow=True,
            seed=1234,
        )
        label = f"GenericBackendV2 (sintetico, {n_qubits} qubits)"

    ideal_sim = AerSimulator(seed_simulator=1234)
    try:
        noise_model = NoiseModel.from_backend(hw_backend)
        noisy_sim = AerSimulator(noise_model=noise_model, seed_simulator=4321)
        have_noise_model = True
    except Exception:
        noise_model = None
        noisy_sim = None
        have_noise_model = False

    _ISING_BACKEND_BUNDLE = {
        "hw_backend": hw_backend,
        "label": label,
        "ideal_sim": ideal_sim,
        "noise_model": noise_model,
        "noisy_sim": noisy_sim,
        "have_noise_model": have_noise_model,
    }
    return _ISING_BACKEND_BUNDLE

def mixture_plan_ising(init_mode, shots, n_sys, rng=None):
    """Plan de mezcla clásica para preparar estados iniciales."""
    if init_mode == "mm":
        rng = rng if rng is not None else np.random.default_rng(0)
        samples = rng.integers(0, 2 ** int(n_sys), size=int(shots))
        unique, counts = np.unique(samples, return_counts=True)
        return [
            (format(int(val), f"0{int(n_sys)}b"), int(n))
            for val, n in zip(unique, counts)
        ]
    if init_mode == "0":
        return [("0" * int(n_sys), int(shots))]
    if init_mode in ("+", "plus"):
        return [("plus", int(shots))]
    raise ValueError(f"init_mode no reconocido: {init_mode}")


def build_dynamic_protocol_circuit_ising(compiled, n_cycles, init_label, final_basis="Z"):
    """Circuito dinámico Ising con registros clásicos separados para baño y sistema."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from tfm_shared.qiskit_impl import append_measure_in_basis, append_one_cycle_to_circuit

    cfg = compiled["cfg"]
    n_sys, n_bath = cfg.n_sys, cfg.n_bath

    q_sys = QuantumRegister(n_sys, "s")
    q_bath = QuantumRegister(n_bath, "b")

    n_bath_bits = int(n_cycles) * n_bath
    n_final_bits = n_sys if final_basis is not None else 0
    c = ClassicalRegister(n_bath_bits + n_final_bits, "c")

    qc = QuantumCircuit(q_sys, q_bath, c)

    # Preparación del sistema
    if init_label == "0" * n_sys:
        pass
    elif init_label == "plus":
        for i in range(n_sys):
            qc.h(q_sys[i])
    elif len(init_label) == n_sys and set(init_label).issubset({"0", "1"}):
        for i, bit in enumerate(init_label[::-1]):
            if bit == "1":
                qc.x(q_sys[i])
    else:
        raise ValueError(f"init_label no reconocido: {init_label}")

    # Aplicar n_cycles del protocolo, con medición del baño cada ciclo
    for r in range(int(n_cycles)):
        c_bath_slice = [c[r * n_bath + mu] for mu in range(n_bath)]
        append_one_cycle_to_circuit(
            qc, compiled, q_sys, q_bath,
            c_bath=c_bath_slice,
            measure_bath=True, bath_measure_basis="Z",
        )
        
           # Medida final del sistema
    if final_basis is not None:
        for i in range(n_sys):
            append_measure_in_basis(qc, q_sys[i], c[n_bath_bits + i], basis=final_basis)
    return qc


def get_isa_dynamic_circuit_ising(
    compiled, n_cycles, init_label, final_basis="Z",
    optimization_level=1, backend_bundle=None,
):
    from qiskit.transpiler import generate_preset_pass_manager
    backend_bundle = (
        get_ising_backend_bundle(min_total_qubits=2 * compiled["cfg"].n_sys)
        if backend_bundle is None else backend_bundle
    )
    hw_backend = backend_bundle["hw_backend"]
    cfg = compiled["cfg"]
    key = (
        backend_bundle["label"], cfg.n_sys, cfg.theta, cfg.delta, cfg.MT,
        cfg.beta, n_cycles, init_label, final_basis, int(optimization_level),
    )
    if key not in ISA_CIRCUIT_CACHE_ISING:
        qc = build_dynamic_protocol_circuit_ising(
            compiled, n_cycles=n_cycles,
            init_label=init_label, final_basis=final_basis,
        )
        pm = generate_preset_pass_manager(
            optimization_level=int(optimization_level), backend=hw_backend,
        )
        ISA_CIRCUIT_CACHE_ISING[key] = pm.run(qc)
    return ISA_CIRCUIT_CACHE_ISING[key]


def parse_dynamic_counts_ising(counts, n_cycles, n_sys, n_bath, final_basis="Z"):
    """Reconstruye P(1) por auxiliar y por qubit del sistema."""
    n_bath_bits = int(n_cycles) * int(n_bath)
    n_total = sum(counts.values())

    bath_p1_per_aux = np.zeros((int(n_cycles), int(n_bath)), dtype=float)
    sys_p1_per_qubit = np.zeros(int(n_sys), dtype=float)
    has_final = final_basis is not None
    expected_len = n_bath_bits + (n_sys if has_final else 0)

    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "").zfill(expected_len)
        if has_final:
            sys_bits = bits[:n_sys]
            bath_bits = bits[n_sys:]
            for i in range(n_sys):
                # Qiskit: bit más a la izquierda es el de mayor índice
                if sys_bits[n_sys - 1 - i] == "1":
                    sys_p1_per_qubit[i] += count
        else:
            bath_bits = bits

        for r in range(int(n_cycles)):
            for mu in range(int(n_bath)):
                idx_in_bath = r * n_bath + mu
                if bath_bits[len(bath_bits) - 1 - idx_in_bath] == "1":
                    bath_p1_per_aux[r, mu] += count

    bath_p1_per_aux /= n_total
    sys_p1_per_qubit /= n_total
    return {"bath_p1_per_aux": bath_p1_per_aux,
            "sys_p1_per_qubit": sys_p1_per_qubit}
    
    
def energy_ising_from_z_per_qubit(compiled, sys_p1_per_qubit):
    """
    Reconstruye <H_S> aproximado a partir de P(|1>) por qubit.

    NOTA IMPORTANTE: en Ising H_S = -J sum X_i X_j - g sum Z_i, así que
    medir en base Z solo reconstruye fielmente la parte de campo
    transverso. Para reconstruir <X_i X_j> habría que medir también en
    base X. Aquí devolvemos:
      - E_z: contribución -g sum_i <Z_i> reconstruida desde shots Z
      - E_total_estimate: misma cosa (estimación incompleta)

    Para una reconstrucción completa de la energía habría que separar
    en dos pasadas (una en base Z, otra en X) y combinar. En el plot
    de "energía vs ciclos" lo habitual en este NB es comparar la E_z
    contra el Gibbs, que es físicamente significativo aunque no sea
    la energía total.
    """
    n_sys = compiled["cfg"].n_sys
    z_per_q = 1.0 - 2.0 * np.asarray(sys_p1_per_qubit, dtype=float)

    # Extraer g del Hamiltoniano del sistema. Asumimos que el primer
    # término con peso negativo es la parte -g Z_i (estándar en build_ising_cfg).
    g = 1.0
    for coef, paulis in compiled["cfg"].system_terms:
        if isinstance(paulis, dict) and len(paulis) == 1:
            site, op = next(iter(paulis.items()))
            if op == "Z":
                g = float(-2.0 * coef)
                break

    E_z = -0.5 * g * float(np.sum(z_per_q))
    return E_z, float(g)



def run_isa_dynamic_shots_ising(
    compiled, n_cycles, shots=4096,
    init_mode="mm", sim_mode="ideal", final_basis="Z",
    seed_base=1234, optimization_level=1, backend_bundle=None,
):
    """Ejecuta el circuito ISA dinámico Ising y devuelve E_z + bath."""
    cfg = compiled["cfg"]
    backend_bundle = (
        get_ising_backend_bundle(min_total_qubits=2 * cfg.n_sys)
        if backend_bundle is None else backend_bundle
    )

    rng = np.random.default_rng(seed_base)
    plan = mixture_plan_ising(init_mode, shots, cfg.n_sys, rng=rng)

    if sim_mode == "ideal":
        sim = backend_bundle["ideal_sim"]
    elif sim_mode == "noisy":
        sim = backend_bundle["noisy_sim"]
        if sim is None:
            raise RuntimeError("Noise model no disponible para sim_mode='noisy'.")
    else:
        raise ValueError("sim_mode debe ser 'ideal' o 'noisy'")

    bath_agg = np.zeros((int(n_cycles), cfg.n_bath), dtype=float)
    sys_agg = np.zeros(cfg.n_sys, dtype=float)

    for i, (init_label, n_sh) in enumerate(plan):
        if n_sh <= 0:
            continue
        qc_isa = get_isa_dynamic_circuit_ising(
            compiled, n_cycles=n_cycles,
            init_label=init_label, final_basis=final_basis,
            optimization_level=optimization_level,
            backend_bundle=backend_bundle,
        )
        result = sim.run(
            qc_isa, shots=int(n_sh),
            seed_simulator=int(seed_base + 97 * i + 13 * int(n_cycles)),
        ).result()
        parsed = parse_dynamic_counts_ising(
            result.get_counts(), n_cycles=n_cycles,
            n_sys=cfg.n_sys, n_bath=cfg.n_bath, final_basis=final_basis,
        )
        weight = float(n_sh) / float(shots)
        bath_agg += weight * parsed["bath_p1_per_aux"]
        sys_agg += weight * parsed["sys_p1_per_qubit"]

    bath_p1_per_cycle = np.mean(bath_agg, axis=1)
    E_z, g = energy_ising_from_z_per_qubit(compiled, sys_agg)
    var_per_qubit = sys_agg * (1.0 - sys_agg) / max(int(shots), 1)
    E_sigma = float(g * 0.5 * np.sqrt(np.sum(var_per_qubit)))

    return {
        "bath_p1_per_cycle": bath_p1_per_cycle,
        "bath_p1_per_aux": bath_agg,
        "sys_p1_per_qubit": sys_agg,
        "E_final": float(E_z),
        "E_sigma": float(E_sigma),
        "shots": int(shots),
        "n_cycles": int(n_cycles),
        "backend_label": backend_bundle["label"],
    }


def run_energy_vs_cycles_shots_ising(
    compiled, max_cycles, shots=4096,
    init_mode="mm", sim_mode="ideal",
    seed_base=1234, optimization_level=1, backend_bundle=None,
):
    rows = []
    for ncyc in range(1, int(max_cycles) + 1):
        out = run_isa_dynamic_shots_ising(
            compiled, n_cycles=ncyc, shots=shots,
            init_mode=init_mode, sim_mode=sim_mode,
            seed_base=seed_base + 1000 * ncyc,
            optimization_level=optimization_level,
            backend_bundle=backend_bundle,
        )
        rows.append({
            "cycle": int(ncyc),
            "E_final": out["E_final"],
            "E_sigma": out["E_sigma"],
            "bath_last": float(out["bath_p1_per_cycle"][-1]),
        })
    return pd.DataFrame(rows)

def build_ising_2d_cfg_mix(
    Lx, Ly, sys_op_spec, bath_op_spec="Y",
    J=0.5, g=1.0, h=1.0, beta=1.0,
    theta=0.15, delta=0.15, MT=25, n_cycles=200,
    randomize=False, seed=101, periodic=False,
):
    """
    Ising 2D con acoplos personalizables. sys_op_spec acepta str o lista
    de tuplas (coef, pauli_str) para combinaciones.
    """
    n_sys = Lx * Ly
    system_terms = ising_2d_xx_z_terms(Lx, Ly, J=J, g=g, periodic=periodic)
    bath_terms = default_bath_terms(n_sys, h=h)  # n_bath = n_sys

    coupling_specs = []
    for i in range(n_sys):
        if isinstance(sys_op_spec, str):
            sys_terms_i = [(1.0, {i: sys_op_spec})]
        else:
            sys_terms_i = [(c, {i: p}) for (c, p) in sys_op_spec]
        bath_terms_i = [(1.0, {i: bath_op_spec})]
        coupling_specs.append({
            "sys_terms":  sys_terms_i,
            "bath_terms": bath_terms_i,
        })

    return ProtocolConfig(
        n_sys=n_sys, n_bath=n_sys,
        system_terms=system_terms, bath_terms=bath_terms,
        coupling_specs=coupling_specs,
        beta=beta, h_filter=h,
        theta=theta, delta=delta, MT=MT, n_cycles=n_cycles,
        randomize=randomize, randomization_lambda=0.0, seed=seed,
        rewind=False,
    )