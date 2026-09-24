# tfm_nonint/model.py
#
# Modelo no-interactuante: H_S = -(g/2) sum_{i=0}^{N_S-1} Z_i
# Cada qubit del sistema se acopla a un auxiliar del baño independiente
# con el mismo esquema que single-spin (A_S, A_B), extendido qubit por qubit.
#
# Como los términos del sistema no se acoplan entre sí, el estado de Gibbs
# factoriza como producto tensor sobre qubits:
#   sigma_beta = (sigma_beta^(1))^{⊗ N_S}
# Por eso este modelo es el peldaño natural entre single-spin e Ising.

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from matplotlib.lines import Line2D

from tfm_shared.core import (
    ProtocolConfig,
    default_bath_terms,
    noninteracting_field_terms,
    make_op,
    mat,
    maximally_mixed,
    pure_dm,
    compile_protocol,
    run_protocol_case,
    apply_one_cycle_map,
    gibbs_state,
    expect,
    project_to_physical_dm,
    trace_distance_dm,
    fidelity_dm,
    energy_basis_matrix,
    population_error_l1_in_energy_basis,
    coherence_norm_in_energy_basis,
    fixed_point_via_superop,
)

# Reutilizamos la parsing de operadores del banco single-spin
from tfm_single_spin.model import one_qubit_pauli_terms


# ============================================================
# CONSTRUCCIÓN DEL CONFIG NO-INTERACTUANTE
# ============================================================

def build_nonint_cfg(
    n_sys=2,
    g=1.0,
    h=1.0,
    beta=1.0,
    theta=0.15,
    delta=0.08,
    MT=10,
    n_cycles=80,
    sys_A="Y",
    bath_A="Y",
    randomize=False,
    randomization_lambda=0.0,
    seed=1234,
    rewind=False,
):
    """
    Configura el protocolo para el modelo no-interactuante:
        H_S = -(g/2) sum_{i=0}^{N_S-1} Z_i
        H_B = -(h/2) sum_{mu=0}^{N_S-1} Z_mu       (un auxiliar por qubit del sistema)
        V   = sum_i A_S^(i) ⊗ A_B^(i)              (acoplo diagonal, uno por qubit)

    El acoplo sistema-baño es "uno a uno": el qubit i del sistema se acopla al
    auxiliar i del baño. Como H_S no tiene términos de dos cuerpos, el canal
    factoriza sobre qubits y el punto fijo debe coincidir con el Gibbs factorizado.

    Parámetros
    ----------
    n_sys : int
        Número de qubits del sistema. El baño tiene igual número de auxiliares.
    g, h, beta, theta, delta, MT, n_cycles, ... : ver ProtocolConfig.
    sys_A, bath_A : str | list
        Operador de acoplo sobre cada qubit (se aplica el mismo a todos).
        Admite "X", "Y", "Z" y alias ("YZmix", "(Y+Z)/sqrt2", etc.).
    """
    # H_S no interactuante: suma de -(g/2) Z_i, sin acoplos qubit-qubit
    system_terms = noninteracting_field_terms(
        [g] * n_sys, pauli="Z", half_factor=True,
    )

    # H_B con igual número de auxiliares que qubits del sistema
    n_bath = n_sys
    bath_terms = default_bath_terms(n_bath, h=h)

    # Un acoplo por qubit: A_S^(i) en el qubit i del sistema, A_B^(i) en el auxiliar i.
    # Para eso no usamos one_qubit_pauli_terms (que hardcodea qubit 0) tal cual,
    # sino que reindexamos al qubit i.
    sys_terms_q0 = one_qubit_pauli_terms(sys_A)
    bath_terms_q0 = one_qubit_pauli_terms(bath_A)

    coupling_specs = []
    for i in range(n_sys):
        # Reindexa los term specs del qubit 0 al qubit i
        sys_terms_i = [(c, {i: list(ops.values())[0]}) for (c, ops) in sys_terms_q0]
        bath_terms_i = [(c, {i: list(ops.values())[0]}) for (c, ops) in bath_terms_q0]
        coupling_specs.append({
            "sys_terms": sys_terms_i,
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


def run_nonint_case(params, rho0=None, compute_fp=True, fp_mode="auto", late_fp_burn=200, late_fp_keep=50):
    """
    Ejecuta un caso no-interactuante. Equivalente a run_single_spin_case
    pero aceptando n_sys >= 1 en params.
    """
    cfg = build_nonint_cfg(**params)
    return run_protocol_case(cfg, rho0=rho0, compute_fp=compute_fp, fp_mode=fp_mode, late_fp_burn=late_fp_burn, late_fp_keep=late_fp_keep)


# ============================================================
# OBSERVABLES AGREGADOS Y COMPARACIÓN CON GIBBS FACTORIZADO
# ============================================================

def per_qubit_energy(rho, n_sys, g=1.0):
    """
    Energía por qubit <E_i> = -(g/2) <Z_i> para i = 0,...,N-1.
    Para modelo NI con H_S factorizado, la energía total es la suma:
        <H_S> = sum_i <E_i> = -(g/2) sum_i <Z_i>
    """
    Es = np.zeros(n_sys, dtype=float)
    for i in range(n_sys):
        Zi = make_op(n_sys, [(1.0, {i: "Z"})])
        Es[i] = -0.5 * g * expect(Zi, rho)
    return Es


def factorized_gibbs(n_sys, g=1.0, beta=1.0):
    """
    Gibbs factorizado del modelo NI:
        sigma_beta = ( exp(+(beta g/2) Z) / Z_1 )^{⊗ N_S}

    Lo construimos producto-a-producto. Útil para validar que el canal
    realmente termaliza cada qubit independientemente al mismo Gibbs local.
    """
    from tfm_shared.core import kron_n
    h1_single = -0.5 * g * np.array([[1, 0], [0, -1]], dtype=complex)  # -g/2 Z
    from scipy.linalg import expm
    rho_1 = expm(-beta * h1_single)
    rho_1 = rho_1 / np.trace(rho_1)
    # Producto tensor N_S veces
    out = np.array([[1.0 + 0.0j]])
    for _ in range(n_sys):
        out = np.kron(out, rho_1)
    return out


def factorization_error(rho, n_sys, g=1.0, beta=1.0):
    """
    D_tr( rho , sigma_beta^factorizado ).
    Si el protocolo NI está bien implementado y ha convergido, esto
    tiende a cero con theta^2 (misma escala que single-spin).
    """
    sigma_fac = factorized_gibbs(n_sys, g=g, beta=beta)
    return trace_distance_dm(rho, sigma_fac)


def mutual_information_halves(rho, n_sys):
    """
    Información mutua I(A : A_bar) dividiendo los qubits en dos mitades.
    Para el modelo NI en su Gibbs factorizado, debería ser 0.
    Si sale positiva significativa, algo no está factorizando.

    Usamos S(A) + S(A_bar) - S(rho) con S = -Tr[rho log rho].
    """
    from tfm_shared.core import ptrace_bath, ptrace_system
    if n_sys < 2:
        return 0.0

    nA = n_sys // 2
    nB = n_sys - nA

    # ptrace_bath y ptrace_system en core están pensados para (sys, bath).
    # Reinterpretamos rho como vivo en A ⊗ B con dims (2^nA, 2^nB).
    rho_A = ptrace_bath(rho, nA, nB)
    rho_B = ptrace_system(rho, nA, nB)

    def vn_entropy(r):
        r = project_to_physical_dm(r)
        vals = np.linalg.eigvalsh(r)
        vals = np.clip(np.real(vals), 1e-15, None)
        return float(-np.sum(vals * np.log(vals)))

    S_A = vn_entropy(rho_A)
    S_B = vn_entropy(rho_B)
    S_AB = vn_entropy(rho)

    return S_A + S_B - S_AB


# ============================================================
# ESCALADO CON N_S
# ============================================================


def scan_N_S(base_params, N_S_list, rho0=None, compute_fp=True, fp_mode="auto",
             late_fp_burn=200, late_fp_keep=50):
    """
    Barre N_S: para cada tamaño, ejecuta el protocolo NI y devuelve métricas
    clave. La métrica principal para validar escalado es:
      - |E/N_S - E_beta_single|  -> debe ser independiente de N_S
      - D_tr(rho_fp, sigma_fac)  -> debe ser pequeño y ~constante (régimen perturbativo)

    Si el código y la implementación escalan bien, estas métricas no deben
    degradarse significativamente al subir N_S en este modelo.
    """
    rows = []
    trajectories = {}

    for N_S in N_S_list:
        p = dict(base_params)
        p["n_sys"] = int(N_S)

        if rho0 is None or (rho0 is not None and rho0.shape[0] != 2**N_S):
            rho0_local = maximally_mixed(N_S)
        else:
            rho0_local = rho0

        out = run_nonint_case(
            p, rho0=rho0_local,
            compute_fp=compute_fp, fp_mode=fp_mode,
            late_fp_burn=late_fp_burn,
            late_fp_keep=late_fp_keep,
        )

        g = p.get("g", 1.0)
        beta = p.get("beta", 1.0)

        E_total = out["Efp"] if compute_fp and out.get("rho_fp") is not None else out["Efinal"]
        E_per_qubit = E_total / N_S

        sigma_fac = factorized_gibbs(N_S, g=g, beta=beta)
        Hs_mat = out["compiled"]["Hs_mat"]
        E_beta_total = float(np.real(np.trace(sigma_fac @ Hs_mat)))
        E_beta_per_qubit = E_beta_total / N_S

        if out.get("rho_fp") is not None:
            rho_ref = out["rho_fp"]
        else:
            rho_ref = out["res"]["rhos"][-1]

        Dtr_fac = trace_distance_dm(rho_ref, sigma_fac)
        I_mutual = mutual_information_halves(rho_ref, N_S) if N_S >= 2 else 0.0

        rows.append({
            "N_S": int(N_S),
            "E_total": float(E_total),
            "E_per_qubit": float(E_per_qubit),
            "E_beta_per_qubit": float(E_beta_per_qubit),
            "|E_per_qubit - E_beta_per_qubit|": float(abs(E_per_qubit - E_beta_per_qubit)),
            "Dtr(rho_ref, sigma_fac)": float(Dtr_fac),
            "I_mutual_halves": float(I_mutual),
        })

        trajectories[N_S] = {
            "out": out,
            "rho_ref": rho_ref,
            "sigma_fac": sigma_fac,
        }

    df = pd.DataFrame(rows).sort_values("N_S").reset_index(drop=True)
    return df, trajectories

def plot_NS_scaling(df, trajectories=None, title_prefix="Escalado N_S"):
    """
    Figura estándar del escalado con N_S:
      - E/N_S vs N_S (debe ser plano -> E extensiva)
      - bias E/N_S vs N_S (debe ser pequeño y ~constante)
      - D_tr(rho_fp, sigma_fac) vs N_S  (en log)
      - I_mutual_halves vs N_S (debe quedarse cerca de 0 para NI)
    """
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))

    axes[0].plot(df["N_S"], df["E_per_qubit"], "o-", label="protocolo")
    axes[0].plot(df["N_S"], df["E_beta_per_qubit"], "s--", label="Gibbs factorizado")
    axes[0].set_xlabel("N_S")
    axes[0].set_ylabel("E / N_S")
    axes[0].set_title("Energía por qubit")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["N_S"], df["|E_per_qubit - E_beta_per_qubit|"], "o-")
    axes[1].set_xlabel("N_S")
    axes[1].set_ylabel(r"$|E/N_S - E_\beta/N_S|$")
    axes[1].set_title("Bias energético por qubit")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3, which="both")

    axes[2].plot(df["N_S"], df["Dtr(rho_ref, sigma_fac)"], "o-")
    axes[2].set_xlabel("N_S")
    axes[2].set_ylabel(r"$D_{\rm tr}(\rho_{\rm fp},\sigma_\beta^{\rm fac})$")
    axes[2].set_title("Distancia al Gibbs factorizado")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.3, which="both")

    axes[3].plot(df["N_S"], df["I_mutual_halves"], "o-")
    axes[3].axhline(0, ls="--", color="k", alpha=0.4)
    axes[3].set_xlabel("N_S")
    axes[3].set_ylabel(r"$I(A:\bar A)$")
    axes[3].set_title("Mutual information (debe ~0 en NI)")
    axes[3].grid(alpha=0.3)

    fig.suptitle(title_prefix, y=1.04)
    plt.tight_layout()
    plt.show()

    display(df.round(8))


# ============================================================
# COMPARACIÓN NI vs INTERACTUANTE (mismo N_S, distinto H_S)
# ============================================================

def compare_ni_vs_ising1d(
    base_params,
    n_sys,
    J_list=(0.0, 0.3, 1.0),
    rho0=None,
    compute_fp=True,
    fp_mode="auto",
):
    """
    Compara el modelo NI (J = 0) con Ising 1D a varios acoplos J.
    Esperamos:
      - J = 0: reproduce exactamente el NI (control interno).
      - J > 0: aparece mutual information no nula y modificación del pico de C_V.

    Parámetros
    ----------
    base_params : dict
        Parámetros del protocolo (g, h, beta, theta, delta, MT, n_cycles, ...).
    n_sys : int
        Tamaño común a comparar (>= 2).
    J_list : iterable
        Valores de J a probar. Por convención, J=0 reproduce NI.
    """
    from tfm_shared.core import ising_chain_xx_z_terms, ProtocolConfig

    rows = []
    outputs = {}

    g = base_params.get("g", 1.0)
    beta = base_params.get("beta", 1.0)
    h = base_params.get("h", 1.0)

    if rho0 is None:
        rho0 = maximally_mixed(n_sys)

    # --- Caso NI de referencia ---
    p_ni = dict(base_params, n_sys=n_sys)
    out_ni = run_nonint_case(p_ni, rho0=rho0, compute_fp=compute_fp, fp_mode=fp_mode)
    Hs_ni = out_ni["compiled"]["Hs_mat"]
    rho_ni = out_ni.get("rho_fp")
    if rho_ni is None:
        rho_ni = out_ni["res"]["rhos"][-1]
    sigma_ni = gibbs_state(Hs_ni, beta)

    rows.append({
        "modelo": "NI (J=0)",
        "J": 0.0,
        "E_total": float(np.real(np.trace(rho_ni @ Hs_ni))),
        "E_per_qubit": float(np.real(np.trace(rho_ni @ Hs_ni)) / n_sys),
        "Dtr(rho, Gibbs)": trace_distance_dm(rho_ni, sigma_ni),
        "I_mutual_halves": mutual_information_halves(rho_ni, n_sys),
    })
    outputs["NI (J=0)"] = {"out": out_ni, "Hs": Hs_ni, "rho_ref": rho_ni, "rho_g": sigma_ni}

    # --- Barrido en J de Ising 1D ---
    sys_A_spec = one_qubit_pauli_terms(base_params.get("sys_A", "Y"))
    bath_A_spec = one_qubit_pauli_terms(base_params.get("bath_A", "Y"))

    for J in J_list:
        if float(J) == 0.0 and "NI (J=0)" in outputs:
            continue  # ya está hecho

        cfg_ising = ProtocolConfig(
            n_sys=n_sys,
            n_bath=n_sys,
            system_terms=ising_chain_xx_z_terms(n_sys, J=float(J), g=g),
            bath_terms=default_bath_terms(n_sys, h=h),
            coupling_specs=[
                {
                    "sys_terms":  [(c, {i: list(ops.values())[0]}) for (c, ops) in sys_A_spec],
                    "bath_terms": [(c, {i: list(ops.values())[0]}) for (c, ops) in bath_A_spec],
                }
                for i in range(n_sys)
            ],
            beta=beta,
            h_filter=h,
            theta=base_params.get("theta", 0.15),
            delta=base_params.get("delta", 0.08),
            MT=base_params.get("MT", 10),
            n_cycles=base_params.get("n_cycles", 80),
            randomize=base_params.get("randomize", False),
            randomization_lambda=base_params.get("randomization_lambda", 0.0),
            seed=base_params.get("seed", 1234),
            rewind=base_params.get("rewind", False),
        )

        out_ising = run_protocol_case(
            cfg_ising,
            rho0=rho0,
            compute_fp=compute_fp,
            fp_mode=fp_mode,
        )

        Hs_ising = out_ising["compiled"]["Hs_mat"]
        rho_ising = out_ising.get("rho_fp")
        if rho_ising is None:
            rho_ising = out_ising["res"]["rhos"][-1]
        sigma_ising = gibbs_state(Hs_ising, beta)

        label = f"Ising 1D (J={J})"
        rows.append({
            "modelo": label,
            "J": float(J),
            "E_total": float(np.real(np.trace(rho_ising @ Hs_ising))),
            "E_per_qubit": float(np.real(np.trace(rho_ising @ Hs_ising)) / n_sys),
            "Dtr(rho, Gibbs)": trace_distance_dm(rho_ising, sigma_ising),
            "I_mutual_halves": mutual_information_halves(rho_ising, n_sys),
        })
        outputs[label] = {
            "out": out_ising, "Hs": Hs_ising,
            "rho_ref": rho_ising, "rho_g": sigma_ising,
        }

    df = pd.DataFrame(rows)
    return df, outputs


def plot_ni_vs_ising(df, outputs, title_prefix="NI vs Ising 1D"):
    """
    Figura estándar para la comparación NI / Ising:
      - trayectorias de energía de cada caso
      - I_mutual_halves por modelo  (NI -> 0, Ising -> positivo)
      - Dtr(rho_fp, Gibbs) por modelo
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    for label, pack in outputs.items():
        energies = pack["out"]["res"]["energies"]
        axes[0].plot(energies, lw=1.8, label=label)

    axes[0].set_xlabel("ciclo")
    axes[0].set_ylabel(r"$\langle H_S \rangle$")
    axes[0].set_title("Energía vs ciclo")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    labels = df["modelo"].values
    Imu = df["I_mutual_halves"].values
    Dtr = df["Dtr(rho, Gibbs)"].values

    axes[1].bar(range(len(labels)), Imu, color="tab:orange")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=20, fontsize=9)
    axes[1].set_ylabel(r"$I(A:\bar A)$")
    axes[1].set_title("Mutual info del punto fijo")
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].bar(range(len(labels)), Dtr, color="tab:green")
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=20, fontsize=9)
    axes[2].set_ylabel(r"$D_{\rm tr}(\rho_{\rm fp},\rho_\beta)$")
    axes[2].set_title("Distancia a Gibbs")
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle(title_prefix, y=1.04)
    plt.tight_layout()
    plt.show()

    display(df.round(8))


# ============================================================
# HELPERS NUEVOS PARA NB02 / SCRIPTS NOCTURNOS
# ============================================================

ISA_CIRCUIT_CACHE_NONINT = {}
NONINT_NATIVE_BASIS = ["rz", "sx", "x", "cx", "measure", "reset"]
_NONINT_BACKEND_BUNDLE = None


def plus_density_matrix(n_sys):
    """Estado |+><+|^{\otimes n_sys} como matriz densa."""
    plus_ket_1 = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    plus_ket = plus_ket_1.copy()
    for _ in range(int(n_sys) - 1):
        plus_ket = np.kron(plus_ket, plus_ket_1)
    return pure_dm(plus_ket)


def rho_metrics_nonint(rho, Hs, rho_g=None):
    """
    Métricas densas del estado final en la base de energía.
    Adaptación directa del helper usado en single-spin.
    """
    evals, rho_e = energy_basis_matrix(rho, Hs)
    off = rho_e.copy()
    np.fill_diagonal(off, 0.0)

    out = {
        "E": float(expect(Hs, rho)),
        "coh_l1": float(np.sum(np.abs(off))),
        "coh_fro": float(np.linalg.norm(off)),
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


def diagnose_nonint_attractor_over_T(
    base_params,
    xgrid,
    rho0_a=None,
    rho0_b=None,
    tol_one=1e-8,
    tol_unit=1e-8,
    dyn_tol=1e-3,
):
    """
    Diagnóstico espectral del atractor para el modelo no-interactuante.
    Es la adaptación mecánica del helper de single-spin a n_sys >= 1.
    """
    n_sys = int(base_params["n_sys"])
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

        out_ref = run_nonint_case(
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
        out_b = run_nonint_case(
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


def make_compiled_nonint(params):
    cfg = build_nonint_cfg(**params)
    compiled = compile_protocol(cfg)
    return cfg, compiled


def get_nonint_backend_bundle(min_total_qubits=None, force_refresh=False):
    """
    Construye backend ideal/noisy para los experimentos ISA de NB02.
    Usa FakeSherbrooke cuando está disponible; si no, cae a un backend sintético.
    """
    global _NONINT_BACKEND_BUNDLE

    min_total_qubits = 2 if min_total_qubits is None else int(min_total_qubits)
    if (not force_refresh) and (_NONINT_BACKEND_BUNDLE is not None):
        backend = _NONINT_BACKEND_BUNDLE["hw_backend"]
        if getattr(backend, "num_qubits", min_total_qubits) >= min_total_qubits:
            return _NONINT_BACKEND_BUNDLE

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
            basis_gates=NONINT_NATIVE_BASIS,
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

    _NONINT_BACKEND_BUNDLE = {
        "hw_backend": hw_backend,
        "label": label,
        "ideal_sim": ideal_sim,
        "noise_model": noise_model,
        "noisy_sim": noisy_sim,
        "have_noise_model": have_noise_model,
    }
    return _NONINT_BACKEND_BUNDLE


def mixture_plan_nonint(init_mode, shots, n_sys, rng=None):
    """
    Plan de mezcla clásica hardware-like para preparar estados iniciales.
    """
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


def build_dynamic_protocol_circuit_nonint(compiled, n_cycles, init_label, final_basis="Z"):
    """
    Circuito dinámico con registros clásicos separados para baño y lectura final.
    """
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

    for r in range(int(n_cycles)):
        c_bath_slice = [c[r * n_bath + mu] for mu in range(n_bath)]
        append_one_cycle_to_circuit(
            qc,
            compiled,
            q_sys,
            q_bath,
            c_bath=c_bath_slice,
            measure_bath=True,
            bath_measure_basis="Z",
        )

    if final_basis is not None:
        for i in range(n_sys):
            append_measure_in_basis(qc, q_sys[i], c[n_bath_bits + i], basis=final_basis)

    return qc


def get_isa_dynamic_circuit_nonint(
    compiled,
    n_cycles,
    init_label,
    final_basis="Z",
    optimization_level=1,
    backend_bundle=None,
):
    from qiskit.transpiler import generate_preset_pass_manager

    backend_bundle = (
        get_nonint_backend_bundle(min_total_qubits=2 * compiled["cfg"].n_sys)
        if backend_bundle is None else backend_bundle
    )
    hw_backend = backend_bundle["hw_backend"]
    cfg = compiled["cfg"]
    key = (
        backend_bundle["label"],
        cfg.n_sys,
        cfg.theta,
        cfg.delta,
        cfg.MT,
        cfg.beta,
        n_cycles,
        init_label,
        final_basis,
        int(optimization_level),
    )
    if key not in ISA_CIRCUIT_CACHE_NONINT:
        qc = build_dynamic_protocol_circuit_nonint(
            compiled,
            n_cycles=n_cycles,
            init_label=init_label,
            final_basis=final_basis,
        )
        pm = generate_preset_pass_manager(
            optimization_level=int(optimization_level),
            backend=hw_backend,
        )
        ISA_CIRCUIT_CACHE_NONINT[key] = pm.run(qc)
    return ISA_CIRCUIT_CACHE_NONINT[key]


def parse_dynamic_counts_nonint(counts, n_cycles, n_sys, n_bath, final_basis="Z"):
    """
    Reconstruye P(1) por auxiliar y por qubit del sistema a partir de counts.
    """
    n_bath_bits = int(n_cycles) * int(n_bath)
    n_total = sum(counts.values())

    bath_counts = np.zeros((int(n_cycles), int(n_bath)), dtype=float)
    sys_p1 = np.zeros(int(n_sys), dtype=float)

    for bitstr, ncount in counts.items():
        bs = bitstr.replace(" ", "")
        bs_rev = bs[::-1]
        for r in range(int(n_cycles)):
            for mu in range(int(n_bath)):
                if bs_rev[r * int(n_bath) + mu] == "1":
                    bath_counts[r, mu] += ncount
        if final_basis is not None:
            for i in range(int(n_sys)):
                if bs_rev[n_bath_bits + i] == "1":
                    sys_p1[i] += ncount

    return {
        "bath_p1_per_aux": bath_counts / max(n_total, 1),
        "sys_p1_per_qubit": sys_p1 / max(n_total, 1),
        "n_total": int(n_total),
    }


def energy_nonint_from_z_per_qubit(compiled, sys_p1):
    """
    Reconstruye E = -(g/2) sum_i <Z_i> a partir de P(|1>) por qubit.
    """
    g = compiled["cfg"].system_terms[0][0] * (-2.0)
    z_per_q = 1.0 - 2.0 * np.asarray(sys_p1, dtype=float)
    return -0.5 * g * float(np.sum(z_per_q)), float(g)


def run_isa_dynamic_shots_nonint(
    compiled,
    n_cycles,
    shots=4096,
    init_mode="mm",
    sim_mode="ideal",
    final_basis="Z",
    seed_base=1234,
    optimization_level=1,
    backend_bundle=None,
):
    """
    Ejecuta el circuito ISA dinámico ya transpilado y devuelve energía + error shot.
    """
    cfg = compiled["cfg"]
    backend_bundle = (
        get_nonint_backend_bundle(min_total_qubits=2 * cfg.n_sys)
        if backend_bundle is None else backend_bundle
    )

    rng = np.random.default_rng(seed_base)
    plan = mixture_plan_nonint(init_mode, shots, cfg.n_sys, rng=rng)

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
        qc_isa = get_isa_dynamic_circuit_nonint(
            compiled,
            n_cycles=n_cycles,
            init_label=init_label,
            final_basis=final_basis,
            optimization_level=optimization_level,
            backend_bundle=backend_bundle,
        )
        result = sim.run(
            qc_isa,
            shots=int(n_sh),
            seed_simulator=int(seed_base + 97 * i + 13 * int(n_cycles)),
        ).result()
        parsed = parse_dynamic_counts_nonint(
            result.get_counts(),
            n_cycles=n_cycles,
            n_sys=cfg.n_sys,
            n_bath=cfg.n_bath,
            final_basis=final_basis,
        )

        weight = float(n_sh) / float(shots)
        bath_agg += weight * parsed["bath_p1_per_aux"]
        sys_agg += weight * parsed["sys_p1_per_qubit"]

    bath_p1_per_cycle = np.mean(bath_agg, axis=1)
    E_final, g = energy_nonint_from_z_per_qubit(compiled, sys_agg)
    var_per_qubit = sys_agg * (1.0 - sys_agg) / max(int(shots), 1)
    E_sigma = float(g * 0.5 * np.sqrt(np.sum(var_per_qubit)))

    return {
        "bath_p1_per_cycle": bath_p1_per_cycle,
        "bath_p1_per_aux": bath_agg,
        "sys_p1_per_qubit": sys_agg,
        "E_final": float(E_final),
        "E_sigma": float(E_sigma),
        "shots": int(shots),
        "n_cycles": int(n_cycles),
        "backend_label": backend_bundle["label"],
        "optimization_level": int(optimization_level),
    }


def run_energy_vs_cycles_shots_nonint(
    compiled,
    max_cycles,
    shots=4096,
    init_mode="mm",
    sim_mode="ideal",
    seed_base=1234,
    optimization_level=1,
    backend_bundle=None,
):
    rows = []
    for ncyc in range(1, int(max_cycles) + 1):
        out = run_isa_dynamic_shots_nonint(
            compiled,
            n_cycles=ncyc,
            shots=shots,
            init_mode=init_mode,
            sim_mode=sim_mode,
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


