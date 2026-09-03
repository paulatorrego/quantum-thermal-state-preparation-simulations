# tfm_single_spin/model.py

from __future__ import annotations

from typing import Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

from tfm_shared.core import (
    ProtocolConfig,
    default_bath_terms,
    make_op,
    mat,
    maximally_mixed,
    all_plus,
    pure_dm,
    compile_protocol,
    run_protocol_case,
    apply_one_cycle_map,
    gibbs_state,
    expect,
    project_to_physical_dm,
    run_exact_trajectory,
    trace_distance_dm,
    fidelity_dm,
    energy_basis_matrix,
    population_error_l1_in_energy_basis,
    coherence_norm_in_energy_basis,
    fixed_point_via_superop,
    ptrace_bath,
    ptrace_system,
    cycle_unitary_matrix,
    gaussian_filter_values,
    heat_capacity_from_variance,
)

from tfm_shared.plots import (
    theta_scaling_study,
)

from tfm_shared.qiskit_impl import (
    run_qiskit_density_matrix,
    run_qiskit_noisy,
    transpiled_circuit_stats,
    build_protocol_circuit,
    append_one_cycle_to_circuit,
    append_measure_in_basis,
    make_depol_noise,
)

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import DensityMatrix, Operator
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit.transpiler import generate_preset_pass_manager

    HAVE_QISKIT = True

except Exception:
    HAVE_QISKIT = False



def _require_qiskit():
    if not HAVE_QISKIT:
        raise RuntimeError("Qiskit no está disponible en este entorno.")
    
# ============================================================
# Constantes Qiskit / ISA
# ============================================================

HW_BACKEND_LABEL = "AerSimulator"
OPT_LEVEL_ISA = 0

ISA_CIRCUIT_CACHE = {}
PASS_MANAGER_CACHE = {}

NATIVE_BASIS = ["rz", "sx", "x", "cx", "measure", "reset"]

if HAVE_QISKIT:
    hw_backend = AerSimulator()
    ideal_sim = AerSimulator()

    _single_spin_noise = NoiseModel()
    _single_spin_noise.add_all_qubit_quantum_error(
        depolarizing_error(1e-3, 1),
        ["u1", "u2", "u3", "rz", "sx", "x", "id"],
    )
    _single_spin_noise.add_all_qubit_quantum_error(
        depolarizing_error(1e-2, 2),
        ["cx"],
    )

    noisy_sim = AerSimulator(noise_model=_single_spin_noise)

else:
    hw_backend = None
    ideal_sim = None
    noisy_sim = None

# ============================================================
# FUNCIONES SINGLE SPIN — VERSIÓN LIMPIA
# ============================================================
def one_qubit_pauli_terms(spec):
    """
    spec puede ser:
      - "X", "Y", "Z"
      - alias como "YZmix", "ZYmix", "(Y+Z)/sqrt2"
      - lista de términos tipo [(coef, {0:'Y'}), ...]
    """
    if isinstance(spec, str):
        key = spec.replace(" ", "").upper()

        aliases = {
            "YZMIX": [
                (1.0 / np.sqrt(2), {0: "Y"}),
                (1.0 / np.sqrt(2), {0: "Z"}),
            ],
            "ZYMIX": [
                (1.0 / np.sqrt(2), {0: "Z"}),
                (1.0 / np.sqrt(2), {0: "Y"}),
            ],
            "1/SQRT(2)*(Y+Z)": [
                (1.0 / np.sqrt(2), {0: "Y"}),
                (1.0 / np.sqrt(2), {0: "Z"}),
            ],
            "1/SQRT(2)*(Z+Y)": [
                (1.0 / np.sqrt(2), {0: "Z"}),
                (1.0 / np.sqrt(2), {0: "Y"}),
            ],
            "(Y+Z)/SQRT2": [
                (1.0 / np.sqrt(2), {0: "Y"}),
                (1.0 / np.sqrt(2), {0: "Z"}),
            ],
            "(Z+Y)/SQRT2": [
                (1.0 / np.sqrt(2), {0: "Z"}),
                (1.0 / np.sqrt(2), {0: "Y"}),
            ],
        }

        if key in {"X", "Y", "Z"}:
            return [(1.0, {0: key})]
        if key in aliases:
            return aliases[key]

        raise ValueError(f"spec string no reconocida: {spec}")

    elif isinstance(spec, list):
        return spec

    else:
        raise ValueError("spec debe ser str o lista de term_specs")

def pauli_spec(combo):
    """'X','Y','Z' → pauli puro;   'XY','YZ','XZ','XYZ' → combinación normalizada."""
    combo = combo.upper()
    if len(combo) == 1:
        return combo
    norm = 1.0 / np.sqrt(len(combo))
    return [(norm, {0: c}) for c in combo]


def build_single_spin_cfg(
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
    return ProtocolConfig(
        n_sys=1,
        n_bath=1,
        system_terms=[(-0.5 * g, {0: "Z"})],
        bath_terms=default_bath_terms(1, h=h),
        coupling_specs=[
            {
                "sys_terms": one_qubit_pauli_terms(sys_A),
                "bath_terms": one_qubit_pauli_terms(bath_A),
            }
        ],
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


def run_single_spin_case(params, rho0=None, compute_fp=True, fp_mode="auto"):
    cfg = build_single_spin_cfg(**params)
    return run_protocol_case(cfg, rho0=rho0, compute_fp=compute_fp, fp_mode=fp_mode)

# ============================================================
# OBJETIVO 1.2 — COMPARAR A = X, Y, Z (CORREGIDO)
# ============================================================



# ============================================================
# OBJETIVO 1.3 — VARIANTES DE SCHEDULE DE PAULIS
# ============================================================

def single_spin_compiled_bank(base_params):
    """Banco compilado con todos los operadores usados en las 13 variantes."""
    def _compile(sys_A_spec):
        p = base_params.copy()
        p["sys_A"] = sys_A_spec
        return compile_protocol(build_single_spin_cfg(**p))

    mix2 = lambda a, b: [(1.0/np.sqrt(2), {0: a}), (1.0/np.sqrt(2), {0: b})]
    mix3 = [(1.0/np.sqrt(3), {0: "X"}),
            (1.0/np.sqrt(3), {0: "Y"}),
            (1.0/np.sqrt(3), {0: "Z"})]

    bank = {
        "X":      _compile("X"),
        "Y":      _compile("Y"),
        "Z":      _compile("Z"),
        "XYmix":  _compile(mix2("X", "Y")),
        "YZmix":  _compile(mix2("Y", "Z")),
        "XZmix":  _compile(mix2("X", "Z")),
        "XYZmix": _compile(mix3),
    }
    return bank

_SCHEDULE_DISPATCHER = {
    "always_X":     lambda r, rng: "X",
    "always_Y":     lambda r, rng: "Y",
    "always_Z":     lambda r, rng: "Z",
    "alternate_XY": lambda r, rng: "X" if r % 2 == 0 else "Y",
    "alternate_YZ": lambda r, rng: "Y" if r % 2 == 0 else "Z",
    "alternate_XZ": lambda r, rng: "X" if r % 2 == 0 else "Z",
    "random_XY":    lambda r, rng: rng.choice(["X", "Y"]),
    "random_YZ":    lambda r, rng: rng.choice(["Y", "Z"]),
    "random_XZ":    lambda r, rng: rng.choice(["X", "Z"]),
    "XYmix":        lambda r, rng: "XYmix",
    "YZmix":        lambda r, rng: "YZmix",
    "XZmix":        lambda r, rng: "XZmix",
    "XYZmix":       lambda r, rng: "XYZmix",
}


def run_single_spin_schedule(base_params, variant, rho0=None, random_seed=1234):
    bank = single_spin_compiled_bank(base_params)
    rng = np.random.default_rng(random_seed)

    cfg_ref = build_single_spin_cfg(**base_params)
    Hs = make_op(cfg_ref.n_sys, cfg_ref.system_terms)
    rho_g = gibbs_state(Hs, cfg_ref.beta)

    if rho0 is None:
        rho0 = maximally_mixed(1)

    rho = rho0.copy()
    rhos = [rho.copy()]
    energies = [expect(Hs, rho)]
    chosen_labels = []

    if variant not in _SCHEDULE_DISPATCHER:
        raise ValueError(
            f"Variante '{variant}' no reconocida. "
            f"Opciones: {sorted(_SCHEDULE_DISPATCHER.keys())}"
        )
    label_picker = _SCHEDULE_DISPATCHER[variant]

    for r in range(cfg_ref.n_cycles):
        label = label_picker(r, rng)
        step = apply_one_cycle_map(rho, bank[label], mr=0, return_bath=False)
        rho = step["rho_s_next"]

        rhos.append(rho.copy())
        energies.append(expect(Hs, rho))
        chosen_labels.append(label)

    out = {
        "variant": variant,
        "Hs": Hs,
        "rho_g": rho_g,
        "rhos": rhos,
        "energies": np.array(energies, dtype=float),
        "rho_final": rhos[-1],
        "Efinal": energies[-1],
        "Eg": expect(Hs, rho_g),
        "chosen_labels": chosen_labels,
        "final_fidelity": fidelity_dm(rhos[-1], rho_g),
        "final_trace_distance": trace_distance_dm(rhos[-1], rho_g),
        "final_pop_error": population_error_l1_in_energy_basis(rhos[-1], rho_g, Hs),
        "final_coh_error": coherence_norm_in_energy_basis(rhos[-1], Hs),
    }
    return out





# ============================================================
# CAPACIDAD CALORÍFICA SINGLE SPIN — CORREGIDA
# ============================================================

def single_spin_heat_capacity_study(base_params, beta_grid, rho0=None, use_fp=True):
    rows = []

    for beta in beta_grid:
        params = base_params.copy()
        params["beta"] = float(beta)

        out = run_single_spin_case(params, rho0=rho0, compute_fp=True)

        rows.append({
            "beta": beta,
            "T": 1.0 / beta,
            "Egibbs": out["Eg"],
            "Efinal": out["Efinal"],
            "Efp": out["Efp"],
            "E_bias": abs((out["Efp"] if use_fp else out["Efinal"]) - out["Eg"]),
        })

    # ESTO VA FUERA DEL FOR
    df = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

    # capacidad calorífica por derivada numérica dE/dT
    energy_col = "Efp" if use_fp else "Efinal"
    df["Cv_gibbs_fd"] = np.gradient(df["Egibbs"], df["T"])
    df["Cv_proto_fd"] = np.gradient(df[energy_col], df["T"])

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    ax[0].plot(df["T"], df["Egibbs"], 'o-', label=r"$E_{\rm Gibbs}$")
    ax[0].plot(df["T"], df[energy_col], 's--', label=r"$E_{\rm proto}$")
    ax[0].set_xlabel("T")
    ax[0].set_ylabel("E")
    ax[0].set_title("Energía vs temperatura")
    ax[0].legend()

    ax[1].plot(df["T"], df["Cv_gibbs_fd"], 'o-', label=r"$C_V$ exacta")
    ax[1].plot(df["T"], df["Cv_proto_fd"], 's--', label=r"$C_V$ protocolo")
    ax[1].set_xlabel("T")
    ax[1].set_ylabel(r"$C_V$")
    ax[1].set_title("Capacidad calorífica efectiva vs temperatura")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    display(df.round(8))
    return df

# ============================================================
# PLOTS SINGLE SPIN
# ============================================================

# BARRIDOS
from matplotlib.lines import Line2D

# Plot 2x3 con barridos de beta, theta, h, g, delta, MT
def plot_single_spin_energy_sweeps(base_params, sweep_values, rho0=None, figsize=(18, 10)):
    panels = [
        ("beta",  sweep_values["beta"],  r"Barrido en $\beta$"),
        ("theta", sweep_values["theta"], r"Barrido en $\theta$"),
        ("h",     sweep_values["h"],     r"Barrido en $h$"),
        ("g",     sweep_values["g"],     r"Barrido en $g$"),
        ("delta", sweep_values["delta"], r"Barrido en $\delta$"),
        ("MT",    sweep_values["MT"],    r"Barrido en $M_T$"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    for ax, (param_name, values, panel_title) in zip(axes, panels):
        for val in values:
            params = base_params.copy()
            params[param_name] = val

            out = run_single_spin_case(params, rho0=rho0)

            line, = ax.plot(out["res"]["energies"], label=f"{param_name}={val}")
            c = line.get_color()

            ax.axhline(out["Eg"], color=c, ls="--", alpha=0.85)
            ax.axhline(out["Efp"], color=c, ls=":", alpha=0.85)

        ax.set_title(panel_title)
        ax.set_xlabel("reset")
        ax.set_ylabel("E")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    style_handles = [
        Line2D([0], [0], color="black", lw=2, ls="-",  label="E(r)"),
        Line2D([0], [0], color="black", lw=2, ls="--", label=r"$E_{\rm Gibbs}$"),
        Line2D([0], [0], color="black", lw=2, ls=":",  label=r"$E_{\rm fp}$"),
    ]

    fig.legend(
        handles=style_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02)
    )

    fig.suptitle("Single spin — barridos de energía vs resets", y=1.06)
    plt.tight_layout()
    plt.show()
    
# sesgo |Efp-Egibbs| Para optimizar parámetros, esto suele ser aún más útil que las trayectorias.
def plot_energy_bias_scan(param_name, values, base_params, rho0=None):
    biases = []
    Eg_list = []
    Efp_list = []

    for val in values:
        params = base_params.copy()
        params[param_name] = val
        out = run_single_spin_case(params, rho0=rho0)

        Eg_list.append(out["Eg"])
        Efp_list.append(out["Efp"])
        biases.append(abs(out["Efp"] - out["Eg"]))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(values, Eg_list, marker="o", label=r"$E_{\rm Gibbs}$")
    ax[0].plot(values, Efp_list, marker="s", label=r"$E_{\rm fp}$")
    ax[0].set_xlabel(param_name)
    ax[0].set_ylabel("energía")
    ax[0].set_title(f"Energías vs {param_name}")
    ax[0].legend()

    ax[1].plot(values, biases, marker="o")
    ax[1].set_xlabel(param_name)
    ax[1].set_ylabel(r"$|E_{\rm fp}-E_{\rm Gibbs}|$")
    ax[1].set_title(f"Sesgo energético vs {param_name}")

    plt.tight_layout()
    plt.show()
    
#============================================================
# PLOT SINGLE SPIN EXTRAS — VERSIÓN COMPLETA
# ============================================================

def purity(rho):
    return float(np.real(np.trace(rho @ rho)))

def von_neumann_entropy(rho, eps=1e-15):
    vals = np.linalg.eigvalsh(project_to_physical_dm(rho))
    vals = np.clip(np.real(vals), eps, None)
    return float(-np.sum(vals * np.log(vals)))


    
def scan_fp_energy_table(base_params, param_name, values, rho0=None, fp_mode="superop"):
    rows = []
    for val in values:
        p = base_params.copy()
        p[param_name] = val
        out = run_single_spin_case(p, rho0=rho0, compute_fp=True, fp_mode=fp_mode)

        rows.append({
            param_name: val,
            "E_gibbs": out["Eg"],
            "E_fp": out["Efp"],
            "E_final": out["Efinal"],
            "|E_fp-Eg|": abs(out["Efp"] - out["Eg"]),
            "|E_final-Eg|": abs(out["Efinal"] - out["Eg"]),
            "Dtr(fp,Gibbs)": out["fp_vs_gibbs"]["trace_distance"],
        })

    df = pd.DataFrame(rows)
    display(df.round(8))
    return df
    
def rewind_random_block(rho0, label_init, study_pars):
    variants = {
        "sin nada":      dict(randomize=False, randomization_lambda=0.0, rewind=False),
        "solo random":   dict(randomize=True,  randomization_lambda=1.0, rewind=False),
        "solo rewind":   dict(randomize=False, randomization_lambda=0.0, rewind=True),
        "ambos":         dict(randomize=True,  randomization_lambda=1.0, rewind=True),
    }

    rows, trajs = [], {}

    for name, extra in variants.items():
        p = study_pars.copy()
        p.update(extra)
        out = run_single_spin_case(p, rho0=rho0, compute_fp=False)

        rhos = out["res"]["rhos"]
        rho_fp = project_to_physical_dm(sum(rhos[-100:]) / 100)

        rho_g = out["rho_g"]
        Hs = out["Hs"]
        _, rho_fp_e = energy_basis_matrix(rho_fp, Hs)
        _, rho_g_e = energy_basis_matrix(rho_g, Hs)

        off = rho_fp_e - np.diag(np.diag(rho_fp_e))
        coh_total = np.sum(np.abs(off))
        pop_err = np.sum(np.abs(np.diag(rho_fp_e) - np.diag(rho_g_e)))

        D_traj = np.array([trace_distance_dm(r, rho_g) for r in rhos], dtype=float)

        rows.append({
            "variante": name,
            "E_fp": float(expect(Hs, rho_fp)),
            "Egibbs": out["Eg"],
            "|E_fp-Eg|": abs(float(expect(Hs, rho_fp)) - out["Eg"]),
            "D_tr(fp,Gibbs)": trace_distance_dm(rho_fp, rho_g),
            "F(fp,Gibbs)": fidelity_dm(rho_fp, rho_g),
            "pop_err_fp": pop_err,
            "coh_total_fp": coh_total,
        })

        trajs[name] = {
            "out": out,
            "rho_fp": rho_fp,
            "D_traj": D_traj,
            "coh_traj": np.array([
                coherence_norm_in_energy_basis(r, Hs, norm="l1")
                for r in rhos
            ], dtype=float),
        }

    df = pd.DataFrame(rows)
    display(df.round(8))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for name, pack in trajs.items():
        axes[0].plot(pack["out"]["res"]["energies"], label=name)
        axes[1].plot(pack["D_traj"], label=name)
        axes[2].plot(pack["D_traj"], label=name)

    axes[0].axhline(trajs["sin nada"]["out"]["Eg"], ls="--", color="k", label="E_Gibbs")
    axes[0].set_title("Convergencia de la energía")
    axes[1].set_title("Distancia a Gibbs (log)")
    axes[1].set_yscale("log")
    axes[2].set_title("Distancia a Gibbs (lineal)")

    for ax in axes:
        ax.set_xlabel("reset")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Randomización y rewind — {label_init}", y=1.02)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].bar(df["variante"], df["D_tr(fp,Gibbs)"])
    axes[0].set_title("Distancia traza al Gibbs en el punto fijo")
    axes[0].tick_params(axis='x', rotation=20)

    x = np.arange(len(df))
    w = 0.35
    axes[1].bar(x - w/2, df["pop_err_fp"], width=w, label="pop error")
    axes[1].bar(x + w/2, df["coh_total_fp"], width=w, label="|coh|")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["variante"], rotation=20)
    axes[1].set_title("Descomposición: poblaciones vs coherencias")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return df, trajs



def resource_scan(base_params, param_name, values, n_cycles_for_circuit=None):
    rows = []

    for val in values:
        p = base_params.copy()
        p[param_name] = val

        cfg = build_single_spin_cfg(**p)
        compiled = compile_protocol(cfg)

        ncyc = p["n_cycles"] if n_cycles_for_circuit is None else n_cycles_for_circuit
        st = transpiled_circuit_stats(compiled, n_cycles=ncyc, measure_bath=True)

        rows.append({
            param_name: val,
            "depth": st["depth"],
            "size": st["size"],
            "width": st["width"],
            "cx": st["cx"],
            "measure": st["measure"],
            "reset": st["reset"],
            "rz": st["rz"],
            "sx": st["sx"],
            "x": st["x"],
        })

    df = pd.DataFrame(rows)
    return df

# ============================================================
# 3.4 — Comparación Qiskit MT=10 vs MT=30
# ============================================================

def compare_qiskit_MT_cases(cases, rho0_sys, n_cycles_plot=40, title_prefix=""):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    rows = []

    for label, pars in cases.items():
        cfg = build_single_spin_cfg(**pars)
        compiled = compile_protocol(cfg)

        out_exact = run_single_spin_case(pars, rho0=rho0_sys, compute_fp=False)
        res_dm = run_qiskit_density_matrix(compiled, n_cycles=n_cycles_plot, rho0_sys=rho0_sys)

        d_exact = np.array([
            trace_distance_dm(r, out_exact["rho_g"])
            for r in out_exact["res"]["rhos"][1:n_cycles_plot+1]
        ], dtype=float)

        fids = np.array([
            fidelity_dm(r, out_exact["rho_g"])
            for r in out_exact["res"]["rhos"][1:n_cycles_plot+1]
        ], dtype=float)

        axes[0, 0].plot(out_exact["res"]["energies"][:n_cycles_plot+1], label=f"NumPy {label}", ls="--")
        axes[0, 0].plot(np.arange(1, n_cycles_plot+1), res_dm["energies"], label=f"Qiskit {label}")

        axes[0, 1].plot(np.arange(1, n_cycles_plot+1), d_exact, ls="--", label=f"NumPy {label}")
        axes[0, 1].plot(np.arange(1, n_cycles_plot+1), res_dm["trace_dists"], label=f"Qiskit {label}")

        axes[1, 0].plot(np.arange(1, n_cycles_plot+1), res_dm["bath_p1"], label=label)
        axes[1, 1].plot(np.arange(1, n_cycles_plot+1), fids[:n_cycles_plot], label=label)

        st = transpiled_circuit_stats(compiled, n_cycles=4, measure_bath=True)

        rows.append({
            "label": label,
            "MT": pars["MT"],
            "delta": pars["delta"],
            "T_eff": pars["MT"] * pars["delta"],
            "E_final_qiskit": res_dm["energies"][-1],
            "D_final_qiskit": res_dm["trace_dists"][-1],
            "bath_p1_final": res_dm["bath_p1"][-1],
            "depth_4cyc": st["depth"],
            "cx_4cyc": st["cx"],
            "size_4cyc": st["size"],
        })

    axes[0, 0].axhline(res_dm["E_gibbs"], color="k", ls=":", label=r"$E_\beta$")
    axes[0, 0].set_title("Energía")
    axes[0, 1].set_title("Distancia a Gibbs")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_title("Excitación del baño")
    axes[1, 1].set_title("Fidelidad con Gibbs")

    for ax in axes.ravel():
        ax.set_xlabel("ciclo")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(title_prefix)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(rows)
    display(df.round(8))
    return df

def plot_single_spin_energy_basis_observables(res, Hs, title=""):
    """
    Versión específica para single spin:
    - poblaciones rho_00, rho_11
    - coherencias Re, Im y |rho_01|
    """
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
    ax[0].grid(alpha=0.3)

    ax[1].plot(coh_re, label=r"$\Re(\rho_{01})$")
    ax[1].plot(coh_im, label=r"$\Im(\rho_{01})$")
    ax[1].plot(coh_abs, label=r"$|\rho_{01}|$")
    ax[1].set_xlabel("reset")
    ax[1].set_ylabel("coherencia")
    ax[1].set_title("Coherencias en base de energía")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    


# ============================================================
# Helpers internos — single spin
# ============================================================

def _single_spin_stationary_rho(out, stationary_mode="fp", late_time_keep=200):
    """
    Devuelve el estado estacionario de referencia:
      - "fp"            -> out["rho_fp"]
      - "late_time_avg" -> promedio de las últimas matrices
    """
    if stationary_mode == "fp":
        if out["rho_fp"] is None:
            raise ValueError("rho_fp es None; usa stationary_mode='late_time_avg' o compute_fp=True.")
        return out["rho_fp"]

    if stationary_mode == "late_time_avg":
        rhos = out["res"]["rhos"]
        n_keep = min(late_time_keep, len(rhos))
        return project_to_physical_dm(sum(rhos[-n_keep:]) / n_keep)

    raise ValueError("stationary_mode debe ser 'fp' o 'late_time_avg'")


def _single_spin_zeta_components(rho_ref, rho_g, Hs):
    """
    Devuelve los dos errores absolutos que usa la Fig. 2 del paper:
      |ζ00| = error absoluto en la población del GS en base de energía
      |ζ01| = error absoluto en la coherencia off-diagonal en base de energía
    """
    _, rho_ref_e = energy_basis_matrix(rho_ref, Hs)
    _, rho_g_e = energy_basis_matrix(rho_g, Hs)
    delta = rho_ref_e - rho_g_e

    zeta_pop = abs(np.real(delta[0, 0]))
    zeta_coh = abs(delta[0, 1])

    return zeta_pop, zeta_coh


def _bohr_frequency_from_Hs(Hs):
    evals = np.linalg.eigvalsh(Hs.to_matrix() if hasattr(Hs, "to_matrix") else Hs)
    diffs = sorted({
        round(abs(evals[j] - evals[i]), 12)
        for i in range(len(evals))
        for j in range(i + 1, len(evals))
        if abs(evals[j] - evals[i]) > 1e-12
    })
    if not diffs:
        raise ValueError("No se encontraron frecuencias de Bohr no triviales.")
    return float(diffs[0])


# ============================================================
# FIGURA 2(a)-LIKE — un único plot, varias betas, sin randomización
# ============================================================

def figure2a_single_spin_by_beta(
    runner,
    base_params,
    theta_grid,
    beta_grid,
    MT_by_beta=None,          # <-- NUEVO: dict {beta: MT} o None
    rho0=None,
    fp_mode="superop",
    stationary_mode="fp",
    late_time_keep=200,
    title="Fig. 2(a)-like — single spin cooling",
    show_table=True,
    show_audit=True,          # <-- NUEVO: imprime tabla de MT/T/n_sigma usados
):
    """
    Reproduce la Fig. 2(a) del paper.

    Parámetros
    ----------
    MT_by_beta : dict o None
        Si es dict {β: MT}, usa ese MT para cada β. Falta uno → usa base_params['MT'].
        Si es None, usa el MT de base_params para todos.

    Ejemplos de uso
    ---------------
    # (1) mismo MT para todos
    figure2a_single_spin_by_beta(..., MT_by_beta=None)   # usa base_params['MT']

    # (2) MT único, pero elegido tú
    figure2a_single_spin_by_beta(..., MT_by_beta={0.3: 59, 1.0: 59, 2.0: 59})

    # (3) Un MT por β (por ejemplo, aumentando con β)
    figure2a_single_spin_by_beta(..., MT_by_beta={0.3: 30, 1.0: 45, 2.0: 70})
    """
    rows, audit_rows = [], []

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, beta in enumerate(beta_grid):
        color = colors[i % len(colors)]

        # ---- resolver MT para esta β ----
        if MT_by_beta is None:
            MT_eff = int(base_params["MT"])
        else:
            MT_eff = int(MT_by_beta.get(beta, base_params["MT"]))

        # diagnóstico para esta β
        a_beta  = np.sqrt(4.0 * base_params["h"] / float(beta))
        T_eff   = MT_eff * base_params["delta"]
        n_sigma = T_eff * a_beta
        # distancia a la resonancia más cercana kπ/ω (con ω=g)
        omega   = base_params["g"]
        k_near  = round(T_eff * omega / np.pi)
        dist_res = abs(T_eff - k_near * np.pi / omega)

        audit_rows.append(dict(
            beta=beta, MT=MT_eff, T_reset=T_eff, a=a_beta,
            n_sigma_filter=n_sigma,
            dist_to_resonance=dist_res,
            near_resonance="⚠️" if dist_res < 0.5 else "ok",
            filter_coverage="⚠️" if n_sigma < 5.0 else "ok",
        ))

        pop_vals, coh_vals = [], []

        for th in theta_grid:
            p = base_params.copy()
            p["beta"] = float(beta)
            p["theta"] = float(th)
            p["MT"] = MT_eff
            p["randomize"] = False
            p["randomization_lambda"] = 0.0

            out = runner(p, rho0=rho0, compute_fp=True, fp_mode=fp_mode)
            rho_ref = _single_spin_stationary_rho(
                out,
                stationary_mode=stationary_mode,
                late_time_keep=late_time_keep,
            )

            zeta_pop, zeta_coh = _single_spin_zeta_components(
                rho_ref, out["rho_g"], out["Hs"]
            )
            pop_vals.append(zeta_pop)
            coh_vals.append(zeta_coh)

            rows.append(dict(
                beta=beta, theta=th, theta2=th**2,
                MT_used=MT_eff, T_reset=T_eff, n_sigma_filter=n_sigma,
                abs_zeta_pop=zeta_pop, abs_zeta_coh=zeta_coh,
                trace_distance=trace_distance_dm(rho_ref, out["rho_g"]),
                fidelity=fidelity_dm(rho_ref, out["rho_g"]),
            ))

        theta2 = np.array(theta_grid, dtype=float) ** 2
        ax.plot(theta2, pop_vals, ":", color=color, lw=2.2, marker="o", ms=4,
                label=fr"$\zeta^{{\rm pop}},\ \beta={beta}$ (MT={MT_eff})")
        ax.plot(theta2, coh_vals, "--", color=color, lw=2.2, marker="s", ms=4,
                label=fr"$\zeta^{{\rm coh}},\ \beta={beta}$ (MT={MT_eff})")

    ax.set_xlabel(r"$\theta^2$")
    ax.set_ylabel(r"$|\zeta_{ab}|$")
    ax.set_title("Single spin cooling")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    plt.suptitle(title, y=1.01)
    plt.tight_layout()
    plt.show()

    df_audit = pd.DataFrame(audit_rows)
    if show_audit:
        print("=== auditoría MT por β ===")
        print(df_audit.to_string(index=False))
        print()
        print("Criterios:")
        print("  n_sigma_filter ≥ 5  → filtro Gaussiano bien cubierto")
        print("  dist_to_resonance ≥ 0.5  → T lejos de la resonancia kπ/ω")

    df = pd.DataFrame(rows)
    if show_table:
        display(df.round(8))

    return df, df_audit

def figure2a_single_spin_by_beta_m(
    runner,
    base_params,
    theta_grid,
    beta_grid,
    rho0=None,
    fp_mode="superop",
    stationary_mode="fp",
    late_time_keep=200,
    n_sigma_min=6.0,  
    title="Fig. 2(a)-like — single spin cooling",
    show_table=True,
):
    """
    Reproduce la lógica de la Fig. 2(a) del paper:
    un único plot con |ζ00| y |ζ01| vs θ², para varias β,
    usando el protocolo NO randomizado.
    """
    rows = []

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, beta in enumerate(beta_grid):
        color = colors[i % len(colors)]
        pop_vals = []
        coh_vals = []
        
        # ------------------------------------------------------------
        # FIX: MT adaptativo para que el filtro Gaussiano NO se trunque.
        # Ancho del filtro: a = sqrt(4h/β). Cubrimos n_sigma_min·σ.
        # Requiere MT·δ·a >= n_sigma_min  =>  MT >= n_sigma_min/(δ·a).
        # Solo aumentamos MT respecto al base; nunca lo reducimos.
        # ------------------------------------------------------------
        a_beta = np.sqrt(4.0 * base_params["h"] / float(beta))
        MT_min = int(np.ceil(n_sigma_min / (base_params["delta"] * a_beta)))
        MT_eff = max(int(base_params["MT"]), MT_min)


        for th in theta_grid:
            p = base_params.copy()
            p["beta"] = float(beta)
            p["theta"] = float(th)
            p["MT"] = MT_eff   
            p["randomize"] = False
            p["randomization_lambda"] = 0.0

            out = runner(p, rho0=rho0, compute_fp=True, fp_mode=fp_mode)
            rho_ref = _single_spin_stationary_rho(
                out,
                stationary_mode=stationary_mode,
                late_time_keep=late_time_keep,
            )

            zeta_pop, zeta_coh = _single_spin_zeta_components(rho_ref, out["rho_g"], out["Hs"])
            pop_vals.append(zeta_pop)
            coh_vals.append(zeta_coh)

            rows.append({
                "beta": beta,
                "theta": th,
                "theta2": th**2,
                "MT_used": MT_eff,      
                "abs_zeta_pop": zeta_pop,
                "abs_zeta_coh": zeta_coh,
                "trace_distance": trace_distance_dm(rho_ref, out["rho_g"]),
                "fidelity": fidelity_dm(rho_ref, out["rho_g"]),
                "fp_mode_used": out["fp_method_used"],
            })

        theta2 = np.array(theta_grid, dtype=float) ** 2

        # estilo parecido al paper: mismo color para beta, distinto estilo para pop/coh
        ax.plot(theta2, pop_vals, ":", color=color, lw=2.2,
                label=fr"$\zeta^{{\rm pop}},\ \beta={beta}$")
        ax.plot(theta2, coh_vals, "--", color=color, lw=2.2,
                label=fr"$\zeta^{{\rm coh}},\ \beta={beta}$")

    ax.set_xlabel(r"$\theta^2$")
    ax.set_ylabel(r"$|\zeta_{ab}|$")
    ax.set_title("Single spin cooling")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    plt.suptitle(title, y=1.01)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(rows)
    if show_table:
        display(df.round(8))
    return df


# ============================================================
# FIGURA 2(b)-LIKE — randomized vs unrandomized vs T
# con líneas perturbativas para coherencias
# ============================================================

def figure2b_single_spin_reset_scan(
    runner,
    base_params,
    T_over_pi_grid,
    rho0=None,
    beta_fixed=1.0,
    theta_fixed=0.25,
    delta=None,
    lambda_rand=1.0,
    stationary_mode="late_time_avg",
    late_time_keep=300,
    n_cycles=4000,
    fit_exclude_near_integer=0.06,
    title="Fig. 2(b)-like — randomized vs unrandomized",
    show_table=True,
):
    """
    Reproduce la lógica de la Fig. 2(b) del paper:
    - unrandomized vs randomized
    - eje y = |ζ_ab|
    - líneas perturbativas para coherencias usando la forma de Eq. (43)
      con amplitud ajustada globalmente.

    Devuelve DOS panels:
      izquierda: x = T
      derecha:   x = T / (π/ω_Bohr)  ~ T [π/g] para single spin
    """
    if delta is None:
        delta = base_params["delta"]

    rows = []

    # ω de Bohr del sistema
    p_tmp = base_params.copy()
    p_tmp["beta"] = beta_fixed
    p_tmp["theta"] = theta_fixed
    p_tmp["randomize"] = False
    p_tmp["randomization_lambda"] = 0.0

    out_tmp = runner(p_tmp, rho0=rho0, compute_fp=False)
    omega = _bohr_frequency_from_Hs(out_tmp["Hs"])

    T_grid = np.array(T_over_pi_grid, dtype=float) * (np.pi / omega)

    zpop_un, zcoh_un = [], []
    zpop_ra, zcoh_ra = [], []

    # ---------------------------
    # barrido numérico
    # ---------------------------
    for T_over_pi, T in zip(T_over_pi_grid, T_grid):
        MT = max(int(round(T / delta)), 2)

        # unrandomized
        p_un = base_params.copy()
        p_un.update({
            "beta": beta_fixed,
            "theta": theta_fixed,
            "delta": delta,
            "MT": MT,
            "n_cycles": n_cycles,
            "randomize": False,
            "randomization_lambda": 0.0,
        })
        out_un = runner(p_un, rho0=rho0, compute_fp=True, fp_mode="late_time")
        rho_un = _single_spin_stationary_rho(out_un, stationary_mode=stationary_mode, late_time_keep=late_time_keep)
        zp_un, zc_un = _single_spin_zeta_components(rho_un, out_un["rho_g"], out_un["Hs"])
        zpop_un.append(zp_un)
        zcoh_un.append(zc_un)

        rows.append({
            "case": "unrandomized",
            "beta": beta_fixed,
            "theta": theta_fixed,
            "lambda_rand": 0.0,
            "T": T,
            "T_over_pi": T_over_pi,
            "MT": MT,
            "delta": delta,
            "abs_zeta_pop": zp_un,
            "abs_zeta_coh": zc_un,
            "trace_distance": trace_distance_dm(rho_un, out_un["rho_g"]),
            "fidelity": fidelity_dm(rho_un, out_un["rho_g"]),
        })

        # randomized
        p_ra = base_params.copy()
        p_ra.update({
            "beta": beta_fixed,
            "theta": theta_fixed,
            "delta": delta,
            "MT": MT,
            "n_cycles": n_cycles,
            "randomize": True,
            "randomization_lambda": lambda_rand,
        })
        out_ra = runner(p_ra, rho0=rho0, compute_fp=True, fp_mode="late_time")
        rho_ra = _single_spin_stationary_rho(out_ra, stationary_mode=stationary_mode, late_time_keep=late_time_keep)
        zp_ra, zc_ra = _single_spin_zeta_components(rho_ra, out_ra["rho_g"], out_ra["Hs"])
        zpop_ra.append(zp_ra)
        zcoh_ra.append(zc_ra)

        rows.append({
            "case": f"randomized λ={lambda_rand}",
            "beta": beta_fixed,
            "theta": theta_fixed,
            "lambda_rand": lambda_rand,
            "T": T,
            "T_over_pi": T_over_pi,
            "MT": MT,
            "delta": delta,
            "abs_zeta_pop": zp_ra,
            "abs_zeta_coh": zc_ra,
            "trace_distance": trace_distance_dm(rho_ra, out_ra["rho_g"]),
            "fidelity": fidelity_dm(rho_ra, out_ra["rho_g"]),
        })

    zpop_un = np.array(zpop_un, dtype=float)
    zcoh_un = np.array(zcoh_un, dtype=float)
    zpop_ra = np.array(zpop_ra, dtype=float)
    zcoh_ra = np.array(zcoh_ra, dtype=float)

    # ---------------------------
    # solución perturbativa para coherencias
    # Eq. (43): |ζ01| ~ A / |1 - exp(2 i ω T) - i ω λ T|
    # Ajustamos SOLO el prefactor A; la dependencia en T es exactamente la del paper.
    # ---------------------------
    def denom_abs(T, lam):
        return np.abs(1.0 - np.exp(2j * omega * T) - 1j * omega * lam * T)

    def fit_prefactor(T, y, lam):
        x = 1.0 / denom_abs(T, lam)

        # excluir vecindad demasiado cercana a resonancias si se quiere
        if fit_exclude_near_integer is not None and fit_exclude_near_integer > 0:
            xaxis = T * omega / np.pi   # T en unidades π/ω
            dist_to_int = np.abs(xaxis - np.round(xaxis))
            mask = dist_to_int > fit_exclude_near_integer
            if np.sum(mask) >= 3:
                x = x[mask]
                y = y[mask]

        # ajuste lineal y = A x
        A = np.dot(x, y) / np.dot(x, x)
        return A

    A_un = fit_prefactor(T_grid, zcoh_un.copy(), lam=0.0)
    A_ra = fit_prefactor(T_grid, zcoh_ra.copy(), lam=lambda_rand)

    zcoh_un_pert = A_un / denom_abs(T_grid, 0.0)
    zcoh_ra_pert = A_ra / denom_abs(T_grid, lambda_rand)

    # ---------------------------
    # Plots
    # ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), sharey=True)

    # estilos cercanos al paper
    # unrandomized: rojo(pop), azul(coh)
    # randomized:  magenta(pop), naranja(coh)
    for ax, xvals, xlabel in [
        (axes[0], T_grid, r"$T$"),
        (axes[1], np.array(T_over_pi_grid, dtype=float), r"$T\,[\pi/g]$"),
    ]:
        ax.plot(xvals, zpop_un, color="red", lw=2.0, label=r"$\zeta^{\rm pop}$")
        ax.plot(xvals, zcoh_un, color="blue", lw=2.0, label=r"$\zeta^{\rm coh}$")

        ax.plot(xvals, zpop_ra, color="magenta", lw=1.8, label=r"$\zeta^{\rm pop}\ {\rm (rand.)}$")
        ax.plot(xvals, zcoh_ra, color="orange", lw=1.8, label=r"$\zeta^{\rm coh}\ {\rm (rand.)}$")

        # perturbativas de coherencia
        ax.plot(xvals, zcoh_un_pert, "k-.", lw=1.8, label=r"pert. $\zeta^{\rm coh}$")
        ax.plot(xvals, zcoh_ra_pert, color="gray", ls="-.", lw=1.8, label=r"pert. $\zeta^{\rm coh}$ (rand.)")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$|\zeta_{ab}|$")
        ax.grid(alpha=0.3)

    axes[0].set_title("(b1) vs reset time $T$")
    axes[1].set_title(r"(b2) vs reset time in units $\pi/g$")

    # una sola leyenda
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, fontsize=9, loc="upper left")

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(rows)
    if show_table:
        display(df.round(8))
    return df, {
        "T_grid": T_grid,
        "T_over_pi_grid": np.array(T_over_pi_grid, dtype=float),
        "zpop_un": zpop_un,
        "zcoh_un": zcoh_un,
        "zpop_ra": zpop_ra,
        "zcoh_ra": zcoh_ra,
        "zcoh_un_pert": zcoh_un_pert,
        "zcoh_ra_pert": zcoh_ra_pert,
        "omega": omega,
        "A_un": A_un,
        "A_ra": A_ra,
    }
    
# ============================================================
# Helpers para single spin
# ============================================================

def rho_metrics_single_spin(rho, Hs, rho_g=None):
    evals, rho_e = energy_basis_matrix(rho, Hs)   # base de energía
    pops = np.real(np.diag(rho_e))
    off = rho_e.copy()
    np.fill_diagonal(off, 0.0)

    out = {
        "E": float(expect(Hs, rho)),
        "p_g": float(pops[0]),   # nivel de menor energía
        "p_e": float(pops[1]),   # nivel excitado
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


def avg_dicts(dicts, keys):
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


def rho_to_array(rho):
    return rho.to_matrix() if hasattr(rho, "to_matrix") else np.array(rho, dtype=complex)


# ============================================================
# Helpers para single spin
# ============================================================

def _rho_metrics_single_spin(rho, Hs, rho_g=None):
    evals, rho_e = energy_basis_matrix(rho, Hs)   # base de energía
    pops = np.real(np.diag(rho_e))
    off = rho_e.copy()
    np.fill_diagonal(off, 0.0)

    out = {
        "E": float(expect(Hs, rho)),
        "p_g": float(pops[0]),   # nivel de menor energía
        "p_e": float(pops[1]),   # nivel excitado
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


def _avg_dicts(dicts, keys):
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


def _rho_to_array(rho):
    return rho.to_matrix() if hasattr(rho, "to_matrix") else np.array(rho, dtype=complex)


def x_to_Teff(x, g):
    return float(x) * np.pi / float(g)

def x_to_MT(x, g, delta):
    T_eff = x_to_Teff(x, g)
    return max(int(round(T_eff / delta)), 1)

def make_xgrid(xmin=0.5, xmax=4.0, n=41):
    return np.linspace(float(xmin), float(xmax), int(n))


def diagnose_single_spin_attractor_over_T(
    base_params,
    xgrid,
    rho0_a=None,
    rho0_b=None,
    tol_one=1e-8,
    tol_unit=1e-8,
    dyn_tol=1e-3,
    plot=True,
):
    """
    Diagnóstico completo del atractor del canal:
      - espectro del superoperador
      - multiplicidad de lambda=1
      - convergencia dinámica desde dos estados iniciales distintos
    """
    if rho0_a is None:
        rho0_a = maximally_mixed(1)

    if rho0_b is None:
        rho0_b = pure_dm(np.array([1, 1], dtype=complex) / np.sqrt(2))

    g = base_params["g"]
    delta = base_params["delta"]

    rows = []

    for x in xgrid:
        MT_val = x_to_MT(x, g=g, delta=delta)
        params = dict(base_params, MT=MT_val, randomize=False, randomization_lambda=0.0)

        # Un caso para obtener compiled, Hs, etc.
        out_ref = run_single_spin_case(
            params,
            rho0=rho0_a,
            compute_fp=False,
        )

        rho_fp, S, evals = fixed_point_via_superop(out_ref["compiled"])

        abs_evals = np.sort(np.abs(evals))[::-1]
        second_abs = float(abs_evals[1]) if len(abs_evals) > 1 else np.nan

        mult_one = int(np.sum(np.abs(evals - 1.0) < tol_one))
        n_unit = int(np.sum(np.abs(np.abs(evals) - 1.0) < tol_unit))

        # Trayectoria desde I/2
        out_a = run_single_spin_case(
            params,
            rho0=rho0_a,
            compute_fp=False,
        )
        rho_a_final = out_a["res"]["rhos"][-1]

        # Trayectoria desde |+>
        out_b = run_single_spin_case(
            params,
            rho0=rho0_b,
            compute_fp=False,
        )
        rho_b_final = out_b["res"]["rhos"][-1]

        d_a_fp = float(trace_distance_dm(rho_a_final, rho_fp))
        d_b_fp = float(trace_distance_dm(rho_b_final, rho_fp))
        d_a_b  = float(trace_distance_dm(rho_a_final, rho_b_final))

        rows.append({
            "x_Teff_g_over_pi": float(x),
            "MT": int(MT_val),

            "mult_lambda_1": mult_one,
            "n_unit_modulus": n_unit,
            "|lambda2|": second_abs,
            "gap_1-|lambda2|": float(1.0 - second_abs) if np.isfinite(second_abs) else np.nan,

            "D(final_I2, fp)": d_a_fp,
            "D(final_plus, fp)": d_b_fp,
            "D(final_I2, final_plus)": d_a_b,

            "unique_attractor_likely": bool((mult_one == 1) and (second_abs < 1 - 1e-8)),
            "dynamic_convergence_likely": bool((d_a_fp < dyn_tol) and (d_b_fp < dyn_tol) and (d_a_b < dyn_tol)),
        })

    df = pd.DataFrame(rows).sort_values("x_Teff_g_over_pi").reset_index(drop=True)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

        # Panel 1
        axes[0].plot(df["x_Teff_g_over_pi"], df["|lambda2|"], "o-")
        axes[0].axhline(1.0, ls="--", color="k")
        axes[0].set_xlabel(r"$x \equiv T_{\rm eff} g/\pi$")
        axes[0].set_ylabel(r"$|\lambda_2|$")
        axes[0].set_title("Segundo mayor módulo espectral")

        # Panel 2
        axes[1].plot(df["x_Teff_g_over_pi"], df["gap_1-|lambda2|"], "o-")
        axes[1].set_xlabel(r"$x \equiv T_{\rm eff} g/\pi$")
        axes[1].set_ylabel(r"$1-|\lambda_2|$")
        axes[1].set_title("Gap espectral del canal")

        # Panel 3
        axes[2].plot(df["x_Teff_g_over_pi"], df["D(final_I2, fp)"], "o-", label=r"$D(\rho_r^{I/2},\rho_{fp})$")
        axes[2].plot(df["x_Teff_g_over_pi"], df["D(final_plus, fp)"], "s--", label=r"$D(\rho_r^{+},\rho_{fp})$")
        axes[2].plot(df["x_Teff_g_over_pi"], df["D(final_I2, final_plus)"], "^-.", label=r"$D(\rho_r^{I/2},\rho_r^{+})$")
        axes[2].set_xlabel(r"$x \equiv T_{\rm eff} g/\pi$")
        axes[2].set_ylabel("distancia de traza")
        axes[2].set_yscale("log")
        axes[2].set_title("Convergencia dinámica al mismo atractor")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plt.show()

    cols_show = [
        "x_Teff_g_over_pi", "MT",
        "mult_lambda_1", "n_unit_modulus",
        "|lambda2|", "gap_1-|lambda2|",
        "D(final_I2, fp)", "D(final_plus, fp)", "D(final_I2, final_plus)",
        "unique_attractor_likely", "dynamic_convergence_likely",
    ]
    display(df[cols_show].round(6))
    return df


# =====================================================================
# Protocolo ANALÓGICO exacto (paper Eq. 11 v2): "exact real-time evolution"
# dU/dt = -i H(t) U(t),  con H(t) = H_S + H_B + θ f(t) A_S ⊗ A_B
# Integración continua vía Runge-Kutta de orden 8 (DOP853). Sin Trotter.
# =====================================================================
from scipy.integrate import solve_ivp

def _op(name):
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]],    dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]],   dtype=complex)
    if name == "X": return X
    if name == "Y": return Y
    if name == "Z": return Z
    if name in ("ZYmix","YZmix"): return (Y + Z) / np.sqrt(2)
    raise ValueError(name)

def analog_single_spin_fp(g, h, beta, theta, T, sys_A="ZYmix", bath_A="Y",
                          rtol=1e-10, atol=1e-12):
    """
    Devuelve (rho_fp, Hs, Q, unitarity_err) para el protocolo analógico
    con H_S = -g/2 Z, H_B = -h/2 Z, V(t) = f(t)·A_S⊗A_B, filtro Gaussiano
    normalizado ∫|f|dt = 1 con ancho a = √(4h/β).
    """
    I2 = np.eye(2, dtype=complex)
    Z  = np.array([[1,0],[0,-1]], dtype=complex)
    Hs = -0.5 * g * Z
    Hb = -0.5 * h * Z
    a  = np.sqrt(4.0 * h / beta)
    f_norm = a / np.sqrt(2 * np.pi)   # normaliza f para que ∫|f|dt = 1

    H0   = np.kron(Hs, I2) + np.kron(I2, Hb)
    V_op = np.kron(_op(sys_A), _op(bath_A))

    def rhs(t, Uflat):
        U = Uflat.reshape(4, 4)
        ft = f_norm * np.exp(-0.5 * (a * t)**2)
        H  = H0 + theta * ft * V_op
        return (-1j * H @ U).flatten()

    sol = solve_ivp(
        rhs, (-T, +T),
        np.eye(4, dtype=complex).flatten(),
        method="DOP853", rtol=rtol, atol=atol,
    )
    Q = sol.y[:, -1].reshape(4, 4)
    u_err = np.linalg.norm(Q.conj().T @ Q - np.eye(4))

    # superop del canal E(ρ) = tr_B[Q (ρ ⊗ |0⟩⟨0|) Q†]
    phi0 = np.array([[1,0],[0,0]], dtype=complex)
    S = np.zeros((4, 4), dtype=complex)
    for i in range(2):
        for j in range(2):
            rs = np.zeros((2,2), dtype=complex); rs[i,j] = 1
            out = (Q @ np.kron(rs, phi0) @ Q.conj().T).reshape(2,2,2,2)
            S[:, i + 2*j] = np.einsum('sbpb->sp', out).flatten(order='F')

    # fixed point por eigendecomposición
    ev, vec = np.linalg.eig(S)
    idx = np.argmin(np.abs(ev - 1.0))
    rho_fp = vec[:, idx].reshape(2, 2, order='F')
    rho_fp = 0.5 * (rho_fp + rho_fp.conj().T)
    rho_fp = rho_fp / np.trace(rho_fp)
    return rho_fp, Hs, Q, u_err


def figure2a_single_spin_ANALOG(
    theta_grid,
    beta_grid,
    g=1.0, h=1.0,
    sys_A="ZYmix", bath_A="Y",
    T_over_a=5.0,            # T = T_over_a / a, exactamente como paper v1
    title="Fig. 2(a) — protocolo analógico del paper",
):
    """
    Reproduce la Fig. 2(a) del paper con integración continua.
    Para cada β, usa T = T_over_a / a (con a=√(4h/β)), siguiendo la
    receta del paper v1 "T ≈ 5a^-1 sufficient for effective cooling".
    """
    rows = []
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = ['C3', 'C0', 'C2']  # rojo, azul, verde como en el paper

    for i, beta in enumerate(beta_grid):
        a = np.sqrt(4.0 * h / beta)
        T = T_over_a / a
        pops, cohs = [], []
        for th in theta_grid:
            rho_fp, Hs, _, u_err = analog_single_spin_fp(
                g=g, h=h, beta=beta, theta=th, T=T,
                sys_A=sys_A, bath_A=bath_A,
            )
            w, V = np.linalg.eigh(Hs)
            rg = V @ np.diag(np.exp(-beta*w)) @ V.conj().T; rg /= np.trace(rg)
            d  = (V.conj().T @ rho_fp @ V) - (V.conj().T @ rg @ V)
            zp, zc = abs(np.real(d[0,0])), abs(d[0,1])
            pops.append(zp); cohs.append(zc)
            rows.append(dict(
                beta=beta, theta=th, theta2=th**2, T=T, Ta=T*a,
                abs_zeta_pop=zp, abs_zeta_coh=zc, unitarity_err=u_err,
            ))
        ax.plot(np.array(theta_grid)**2, pops, ":", color=colors[i % 3],
                lw=2.3, marker='o', ms=4,
                label=fr"$\zeta^{{\rm pop}},\ \beta={beta}$")
        ax.plot(np.array(theta_grid)**2, cohs, "--", color=colors[i % 3],
                lw=2.3, marker='s', ms=4,
                label=fr"$\zeta^{{\rm coh}},\ \beta={beta}$")

    ax.set_xlabel(r"$\theta^2$", fontsize=12)
    ax.set_ylabel(r"$|\zeta_{ab}|$", fontsize=12)
    ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout(); plt.show()

    return pd.DataFrame(rows)

def single_spin_fp_vs_gibbs_Tscan(base_params, T_over_pi_grid, rho0=None,
                                  fp_mode="superop", plot=True):
    if rho0 is None:
        rho0 = maximally_mixed(1)

    g = base_params["g"]
    delta = base_params["delta"]

    rows = []
    snapshots = {}

    for x in T_over_pi_grid:
        T_eff = x * np.pi / g
        MT_val = max(int(round(T_eff / delta)), 1)

        params = dict(base_params, MT=MT_val)

        out = run_single_spin_case(
            params,
            rho0=rho0,
            compute_fp=True,
            fp_mode=fp_mode,
        )

        Hs = out["Hs"]
        rho_fp = out["rho_fp"]
        rho_g  = out["rho_g"]

        evals, rho_fp_e = energy_basis_matrix(rho_fp, Hs)
        _,     rho_g_e  = energy_basis_matrix(rho_g, Hs)

        rows.append({
            "T_over_pi_g": float(x),
            "T_eff": float(T_eff),
            "MT": int(MT_val),

            "Egibbs": float(out["Eg"]),
            "Efp": float(out["Efp"]),
            "E_bias_abs": float(abs(out["Efp"] - out["Eg"])),

            "p_g_fp": float(np.real(rho_fp_e[0, 0])),
            "p_e_fp": float(np.real(rho_fp_e[1, 1])),
            "p_g_gibbs": float(np.real(rho_g_e[0, 0])),
            "p_e_gibbs": float(np.real(rho_g_e[1, 1])),

            "abs_rho01_fp": float(np.abs(rho_fp_e[0, 1])),
            "abs_rho01_gibbs": float(np.abs(rho_g_e[0, 1])),

            "Dtr_fp_gibbs": float(trace_distance_dm(rho_fp, rho_g)),
            "F_fp_gibbs": float(fidelity_dm(rho_fp, rho_g)),
        })

        snapshots[float(x)] = {
            "MT": int(MT_val),
            "evals": np.array(evals, dtype=float),
            "rho_fp_e": np.array(rho_fp_e, dtype=complex),
            "rho_g_e": np.array(rho_g_e, dtype=complex),
            "rho_fp": np.array(rho_fp, dtype=complex),
            "rho_g": np.array(rho_g, dtype=complex),
        }

    df = pd.DataFrame(rows).sort_values("T_over_pi_g").reset_index(drop=True)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

        # 1) Energías
        axes[0].plot(df["T_over_pi_g"], df["Efp"], "o-", label=r"$E_{fp}$")
        axes[0].plot(df["T_over_pi_g"], df["Egibbs"], "--", label=r"$E_{\rm Gibbs}$")
        axes[0].set_xlabel(r"$T_{\rm eff} g/\pi$")
        axes[0].set_ylabel("E")
        axes[0].set_title("Energía del punto fijo vs Gibbs")
        axes[0].legend()

        # 2) Elementos de rho en base de energía
        axes[1].plot(df["T_over_pi_g"], df["p_g_fp"], "o-", label=r"$(\rho_{fp}^{(E)})_{00}$")
        axes[1].plot(df["T_over_pi_g"], df["p_g_gibbs"], "--", label=r"$(\rho_{\beta}^{(E)})_{00}$")
        axes[1].plot(df["T_over_pi_g"], df["abs_rho01_fp"], "s-", label=r"$|(\rho_{fp}^{(E)})_{01}|$")
        axes[1].plot(df["T_over_pi_g"], df["abs_rho01_gibbs"], ":", label=r"$|(\rho_{\beta}^{(E)})_{01}|$")
        axes[1].set_xlabel(r"$T_{\rm eff} g/\pi$")
        axes[1].set_ylabel("magnitud")
        axes[1].set_title(r"Comparación de $\rho_{fp}$ y $\rho_\beta$")
        axes[1].legend(fontsize=8)

        # 3) Distancia entre estados
        axes[2].plot(df["T_over_pi_g"], df["Dtr_fp_gibbs"], "o-", label=r"$D_{\rm tr}(\rho_{fp},\rho_\beta)$")
        axes[2].plot(df["T_over_pi_g"], 1.0 - df["F_fp_gibbs"], "s--", label=r"$1-F(\rho_{fp},\rho_\beta)$")
        axes[2].set_xlabel(r"$T_{\rm eff} g/\pi$")
        axes[2].set_ylabel("error")
        axes[2].set_title("Qué tan térmico es el punto fijo")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plt.show()

    display(df.round(8))
    return {"df": df, "snapshots": snapshots}


# DE NB01

def sigma_binomial(p, shots):
    p = np.asarray(p, dtype=float)
    return np.sqrt(np.maximum(p * (1.0 - p), 0.0) / shots)

def shot_band_95(p, shots):
    sig = sigma_binomial(p, shots)
    lo = np.clip(p - 1.96 * sig, 0.0, 1.0)
    hi = np.clip(p + 1.96 * sig, 0.0, 1.0)
    return lo, hi, sig

def inverse_sqrt_guide(shots_grid, ref_shots, ref_value):
    shots_grid = np.asarray(shots_grid, dtype=float)
    return ref_value * np.sqrt(ref_shots / shots_grid)

def noisy_run_summary(params, rho0, noise_model=None, *, n_cycles=None):
    p = params.copy()
    n_use = p["n_cycles"] if n_cycles is None else int(n_cycles)
    _, compiled = make_compiled_single_spin(p)

    if noise_model is None:
        out = run_qiskit_density_matrix(compiled, n_cycles=n_use, rho0_sys=rho0)
        energies = np.asarray(out["energies"], dtype=float)
        dists = np.asarray(out["trace_dists"], dtype=float)
        E_gibbs = float(out["E_gibbs"])
    else:
        out = run_qiskit_noisy(compiled, n_cycles=n_use, noise_model=noise_model, rho0_sys=rho0)
        energies = np.asarray(out["energies"], dtype=float)
        dists = np.asarray(out["trace_dists"], dtype=float)

        Hs_mat = compiled["Hs_mat"]
        rho_g = gibbs_state(Hs_mat, p["beta"])
        E_gibbs = float(np.real(np.trace(rho_g @ Hs_mat)))

    idx_best = int(np.argmin(dists))
    return {
        "energies": energies,
        "trace_dists": dists,
        "E_gibbs": E_gibbs,
        "D_min": float(dists[idx_best]),
        "cycle_best": idx_best + 1,
        "D_final": float(dists[-1]),
        "E_final": float(energies[-1]),
    }
    
    
def make_compiled_single_spin(params):
    """
    Helper local del notebook:
    construye la cfg single-spin y compila el protocolo exacto.
    """
    cfg = build_single_spin_cfg(**params)
    compiled = compile_protocol(cfg)
    return cfg, compiled

def exact_and_dm_single_spin(params, rho0):
    cfg, compiled = make_compiled_single_spin(params)
    out_exact = run_single_spin_case(params, rho0=rho0, compute_fp=False)
    out_dm = run_qiskit_density_matrix(
        compiled,
        n_cycles=params["n_cycles"],
        rho0_sys=rho0,
    )
    d_exact = np.array(
        [trace_distance_dm(r, out_exact["rho_g"]) for r in out_exact["res"]["rhos"][1:]],
        dtype=float,
    )
    return cfg, compiled, out_exact, out_dm, d_exact

def resource_scan_single_spin(base_params, sweep_name, values, *, n_cycles_for_circuit=4, optimization_level=0):
    rows = []
    backend = AerSimulator()

    for val in values:
        p = base_params.copy()
        p[sweep_name] = val

        _, compiled = make_compiled_single_spin(p)
        n_cycles_circ = p["n_cycles"] if n_cycles_for_circuit is None else int(n_cycles_for_circuit)

        qc = build_protocol_circuit(
            compiled,
            n_cycles=n_cycles_circ,
            measure_bath=True,
            bath_measure_basis="Z",
        )

        qc_t = transpile(
            qc,
            backend=backend,
            basis_gates=NATIVE_BASIS,
            optimization_level=optimization_level,
        )
        ops = dict(qc_t.count_ops())

        rows.append({
            sweep_name: val,
            "depth": qc_t.depth(),
            "size": qc_t.size(),
            "width": qc_t.width(),
            "cx": ops.get("cx", 0),
            "measure": ops.get("measure", 0),
            "reset": ops.get("reset", 0),
            "rz": ops.get("rz", 0),
            "sx": ops.get("sx", 0),
            "x": ops.get("x", 0),
        })

    return pd.DataFrame(rows)


# PARTE MAS QISKIT, 

# ============================================================
# C.1 — helpers hardware-like
# ============================================================


def mixture_plan(init_mode, shots):
    """
    Implementación hardware-like del estado inicial.
    - 'mm'   -> mezcla clásica exacta de |0> y |1>
    - '0','1','plus','minus' -> estado puro
    """
    if init_mode == "mm":
        n0 = shots // 2
        n1 = shots - n0
        return [("0", n0), ("1", n1)]
    elif init_mode in ["0", "1", "plus", "+", "minus", "-"]:
        label = {"+" : "plus", "-" : "minus"}.get(init_mode, init_mode)
        return [(label, shots)]
    else:
        raise ValueError("init_mode debe ser 'mm', '0', '1', 'plus', '+', 'minus' o '-'.")

def build_dynamic_protocol_circuit(
    compiled,
    n_cycles,
    init_label="0",
    final_basis="Z",
):
    """
    Circuito dinámico hardware-like:
      - prepara el sistema en un estado puro
      - aplica n_cycles veces: U_cycle + measure/reset del baño
      - mide el sistema al final si final_basis no es None

    Un solo ClassicalRegister:
      c[0],...,c[n_cycles-1] -> lecturas del baño
      c[n_cycles]            -> lectura final del sistema
    """
    cfg = compiled["cfg"]
    if cfg.n_sys != 1 or cfg.n_bath != 1:
        raise ValueError("Estas celdas están preparadas para single-spin con n_sys=1, n_bath=1.")

    q_sys = QuantumRegister(cfg.n_sys, "s")
    q_bath = QuantumRegister(cfg.n_bath, "b")
    n_cbits = n_cycles + (1 if final_basis is not None else 0)
    c = ClassicalRegister(n_cbits, "c")

    qc = QuantumCircuit(q_sys, q_bath, c)

    # Preparación física del sistema
    if init_label == "0":
        pass
    elif init_label == "1":
        qc.x(q_sys[0])
    elif init_label in ["plus", "+"]:
        qc.h(q_sys[0])
    elif init_label in ["minus", "-"]:
        qc.x(q_sys[0])
        qc.h(q_sys[0])
    else:
        raise ValueError("init_label no reconocido.")

    for r in range(n_cycles):
        append_one_cycle_to_circuit(
            qc,
            compiled,
            q_sys,
            q_bath,
            c_bath=[c[r]],
            measure_bath=True,
            bath_measure_basis="Z",
        )

    if final_basis is not None:
        append_measure_in_basis(qc, q_sys[0], c[n_cycles], basis=final_basis)

    return qc

def get_isa_dynamic_circuit(
    compiled,
    n_cycles,
    init_label="0",
    final_basis="Z",
    optimization_level=OPT_LEVEL_ISA,
):
    """
    Transpila el circuito dinámico contra el backend elegido.
    """
    cfg = compiled["cfg"]
    key = (
        HW_BACKEND_LABEL,
        cfg.theta,
        cfg.delta,
        cfg.MT,
        cfg.beta,
        cfg.randomize,
        cfg.randomization_lambda,
        cfg.rewind,
        n_cycles,
        init_label,
        final_basis,
        optimization_level,
    )

    if key not in ISA_CIRCUIT_CACHE:
        qc = build_dynamic_protocol_circuit(
            compiled,
            n_cycles=n_cycles,
            init_label=init_label,
            final_basis=final_basis,
        )
        pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            backend=hw_backend,
        )
        qc_isa = pm.run(qc)
        ISA_CIRCUIT_CACHE[key] = qc_isa

    return ISA_CIRCUIT_CACHE[key]

def parse_dynamic_counts(counts, n_cycles, final_basis="Z"):
    """
    Parse robusto de counts para el ClassicalRegister único c.
    En bitstrings de Qiskit, el bit más a la derecha es c[0].
    """
    shots = sum(counts.values())
    bath_p1 = np.zeros(n_cycles, dtype=float)
    sys_p1 = None
    has_final = final_basis is not None

    if has_final:
        sys_p1_acc = 0.0

    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        expected_len = n_cycles + (1 if has_final else 0)
        bits = bits.zfill(expected_len)

        if has_final:
            sys_bit = int(bits[0])    # c[n_cycles]
            bath_bits = bits[1:]      # c[n_cycles-1] ... c[0]
            sys_p1_acc += count * sys_bit
        else:
            bath_bits = bits

        for r in range(n_cycles):
            bit_r = int(bath_bits[-1 - r])  # c[r]
            bath_p1[r] += count * bit_r

    bath_p1 /= shots

    if has_final:
        sys_p1 = sys_p1_acc / shots

    return {
        "bath_p1": bath_p1,
        "sys_p1": sys_p1,
        "shots": shots,
    }

def single_spin_energy_from_z(compiled, z_est):
    """
    Para H = a I + b Z, E = a + b <Z>.
    En este notebook single-spin esto es la reconstrucción natural.
    """
    H = np.asarray(compiled["Hs_mat"], dtype=complex)
    a = 0.5 * np.real(H[0, 0] + H[1, 1])
    b = 0.5 * np.real(H[0, 0] - H[1, 1])
    return a + b * z_est

def run_isa_dynamic_shots(
    compiled,
    n_cycles,
    shots=4096,
    init_mode="mm",
    final_basis="Z",
    sim_mode="ideal",   # 'ideal' o 'noisy'
    optimization_level=OPT_LEVEL_ISA,
    seed_base=1234,
):
    """
    Ejecuta el protocolo shot-based, ya transpileado al backend, en:
      - Aer ideal
      - Aer con noise_model del backend
    """
    if sim_mode == "ideal":
        simulator = AerSimulator(seed_simulator=seed_base)
    elif sim_mode == "noisy":
        noise_model = make_depol_noise()
        simulator = AerSimulator(
            noise_model=noise_model,
            seed_simulator=seed_base,
        )
    else:
        raise ValueError("sim_mode debe ser 'ideal' o 'noisy'")
    plan = mixture_plan(init_mode, shots)

    bath_mean = np.zeros(n_cycles, dtype=float)
    bath_var = np.zeros(n_cycles, dtype=float)

    sys_p1_mean = 0.0
    sys_p1_var = 0.0
    total_shots = 0

    for i, (init_label, n_sh) in enumerate(plan):
        if n_sh <= 0:
            continue

        qc_isa = get_isa_dynamic_circuit(
            compiled,
            n_cycles=n_cycles,
            init_label=init_label,
            final_basis=final_basis,
            optimization_level=optimization_level,
        )

        result = simulator.run(
            qc_isa,
            shots=n_sh,
            seed_simulator=seed_base + 97 * i + 13 * n_cycles,
        ).result()

        counts = result.get_counts()
        parsed = parse_dynamic_counts(counts, n_cycles=n_cycles, final_basis=final_basis)

        w = n_sh / shots
        total_shots += n_sh

        bath_mean += w * parsed["bath_p1"]
        bath_var += (w ** 2) * parsed["bath_p1"] * (1.0 - parsed["bath_p1"]) / max(n_sh, 1)

        if final_basis is not None:
            sys_p1_mean += w * parsed["sys_p1"]
            sys_p1_var += (w ** 2) * parsed["sys_p1"] * (1.0 - parsed["sys_p1"]) / max(n_sh, 1)

    bath_sigma = np.sqrt(np.maximum(bath_var, 0.0))
    bath_lo95 = np.clip(bath_mean - 1.96 * bath_sigma, 0.0, 1.0)
    bath_hi95 = np.clip(bath_mean + 1.96 * bath_sigma, 0.0, 1.0)

    out = {
        "bath_p1": bath_mean,
        "bath_sigma": bath_sigma,
        "bath_lo95": bath_lo95,
        "bath_hi95": bath_hi95,
        "shots": total_shots,
        "init_mode": init_mode,
        "n_cycles": n_cycles,
        "sim_mode": sim_mode,
    }

    if final_basis == "Z":
        z_final = 1.0 - 2.0 * sys_p1_mean
        z_sigma = 2.0 * np.sqrt(max(sys_p1_var, 0.0))

        E_final = single_spin_energy_from_z(compiled, z_final)

        H = np.asarray(compiled["Hs_mat"], dtype=complex)
        b = 0.5 * np.real(H[0, 0] - H[1, 1])
        E_sigma = abs(b) * z_sigma

        out.update({
            "sys_p1_final": sys_p1_mean,
            "z_final": z_final,
            "z_sigma": z_sigma,
            "E_final": E_final,
            "E_sigma": E_sigma,
        })

    return out

def run_energy_vs_cycles_shots(
    compiled,
    max_cycles,
    shots=4096,
    init_mode="mm",
    sim_mode="ideal",
    optimization_level=OPT_LEVEL_ISA,
    seed_base=1234,
):
    rows = []
    for ncyc in range(1, max_cycles + 1):
        res = run_isa_dynamic_shots(
            compiled,
            n_cycles=ncyc,
            shots=shots,
            init_mode=init_mode,
            final_basis="Z",
            sim_mode=sim_mode,
            optimization_level=optimization_level,
            seed_base=seed_base + 1000 * ncyc,
        )
        rows.append({
            "cycle": ncyc,
            "E_final": res["E_final"],
            "E_sigma": res["E_sigma"],
            "z_final": res["z_final"],
            "z_sigma": res["z_sigma"],
        })
    return pd.DataFrame(rows)

def repeat_dynamic_runs(
    compiled,
    n_cycles,
    shots,
    init_mode="mm",
    final_basis="Z",
    sim_mode="ideal",
    optimization_level=OPT_LEVEL_ISA,
    n_reps=8,
    seed_base=1234,
):
    curves = []
    sigmas = []

    for rep in range(n_reps):
        res = run_isa_dynamic_shots(
            compiled,
            n_cycles=n_cycles,
            shots=shots,
            init_mode=init_mode,
            final_basis=final_basis,
            sim_mode=sim_mode,
            optimization_level=optimization_level,
            seed_base=seed_base + 1000 * rep,
        )
        curves.append(res["bath_p1"])
        sigmas.append(res["bath_sigma"])

    curves = np.asarray(curves, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)

    return {
        "curves": curves,
        "sigmas": sigmas,
        "mean_curve": curves.mean(axis=0),
        "std_curve": curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros(curves.shape[1]),
    }

def robust_signal_span(curve, q_low=0.10, q_high=0.90):
    curve = np.asarray(curve, dtype=float)
    return float(np.quantile(curve, q_high) - np.quantile(curve, q_low))

def effective_sigma_from_trace(sig_trace):
    sig_trace = np.asarray(sig_trace, dtype=float)
    return float(np.sqrt(np.mean(sig_trace**2)))

def inverse_sqrt_guide(shots_grid, ref_shots, ref_value):
    shots_grid = np.asarray(shots_grid, dtype=float)
    return ref_value * np.sqrt(ref_shots / shots_grid)