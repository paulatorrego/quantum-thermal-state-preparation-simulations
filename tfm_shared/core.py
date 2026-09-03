# tfm_shared/core.py

from __future__ import annotations

from dataclasses import dataclass
from typing import  Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from scipy.linalg import expm, sqrtm, svdvals
from qiskit.quantum_info import SparsePauliOp


# ============================================================
# Álgebra básica
# ============================================================

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI_MAP = {
    "I": I2,
    "X": X,
    "Y": Y,
    "Z": Z,
}


def dagger(a: np.ndarray) -> np.ndarray:
    return np.conjugate(a.T)


def kron_n(*ops: np.ndarray) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


# ============================================================
# UTILIDADES BÁSICAS - FUNCIONES GLOBALES
# ============================================================

def term_label(n_qubits, ops):
    """
    Convierte un diccionario {qubit: 'X'/'Y'/'Z'} en una etiqueta Pauli.
    Convención: el qubit 0 es el de la derecha en la string de Qiskit.
    Ejemplo n=3, {0:'X', 2:'Z'} -> 'ZIX'
    """
    chars = ["I"] * n_qubits
    for q, p in ops.items():
        if not (0 <= q < n_qubits):
            raise ValueError(f"Qubit {q} fuera de rango para n_qubits={n_qubits}")
        chars[n_qubits - 1 - q] = p.upper()
    return "".join(chars)


def make_op(n_qubits, term_specs):
    """
    term_specs = [(coef, {q:'P', ...}), ...]
    Devuelve SparsePauliOp.
    """
    if not term_specs:
        return SparsePauliOp.from_list([("I" * n_qubits, 0.0)])
    pairs = [(term_label(n_qubits, ops), complex(coeff)) for coeff, ops in term_specs]
    return SparsePauliOp.from_list(pairs).simplify()


def identity_op(n_qubits):
    return SparsePauliOp.from_list([("I" * n_qubits, 1.0)])


def mat(op_or_matrix):
    return op_or_matrix.to_matrix() if hasattr(op_or_matrix, "to_matrix") else np.asarray(op_or_matrix, dtype=complex)

# I/d
def maximally_mixed(n_qubits):
    d = 2 ** n_qubits
    return np.eye(d, dtype=complex) / d

# |+⟩⟨+|
def all_plus(n_sys):
    """
    Estado |+⟩⟨+|^⊗n  —  producto tensorial de |+⟩ en n qubits.
    
    Drop-in replacement de maximally_mixed(n_sys): misma firma, misma forma
    (matriz densidad 2^n × 2^n), pero estado puro con coherencias máximas
    en la base computacional.
    
    Todos los elementos valen 1/d con d = 2^n.
    """
    d = 2**n_sys
    psi = np.ones(d, dtype=complex) / np.sqrt(d)
    return np.outer(psi, psi.conj())


def expect(op_or_matrix, rho):
    O = mat(op_or_matrix)
    return float(np.real(np.trace(rho @ O)))


def gibbs_state(H, beta):
    """
    rho_beta = exp(-beta H)/Z
    """
    Hm = mat(H)
    d = Hm.shape[0]
    if abs(beta) < 1e-15:
        return np.eye(d, dtype=complex) / d
    rho = expm(-beta * Hm)
    rho /= np.trace(rho)
    return rho


def fidelity_dm(rho, sigma):
    """
    Fidelidad de Uhlmann para matrices densidad.
    """
    s = sqrtm(rho)
    val = np.trace(sqrtm(s @ sigma @ s))
    return float(np.real(val * np.conjugate(val)))


def trace_distance_dm(rho, sigma):
    """
    D( rho, sigma ) = 1/2 ||rho - sigma||_1
    """
    return 0.5 * float(np.sum(svdvals(rho - sigma)))


def project_to_physical_dm(rho):
    """
    Simetriza, recorta autovalores negativos pequeños y normaliza.
    Útil para limpiar ruido numérico.
    """
    rho = 0.5 * (rho + rho.conj().T)
    vals, vecs = np.linalg.eigh(rho)
    vals = np.clip(np.real(vals), 0.0, None)
    rho = (vecs * vals) @ vecs.conj().T
    tr = np.trace(rho)
    if abs(tr) < 1e-14:
        raise ValueError("La matriz quedó con traza ~0 al proyectar.")
    return rho / tr


def ptrace_bath(rho_tot, n_sys, n_bath):
    """
    Traza parcial del baño.
    rhotot vive en H_sys otimes H_bath
    """
    dS, dB = 2 ** n_sys, 2 ** n_bath
    resh = rho_tot.reshape(dS, dB, dS, dB)
    return np.trace(resh, axis1=1, axis2=3)


def ptrace_system(rho_tot, n_sys, n_bath):
    """
    Traza parcial del sistema.
    """
    dS, dB = 2 ** n_sys, 2 ** n_bath
    resh = rho_tot.reshape(dS, dB, dS, dB)
    return np.trace(resh, axis1=0, axis2=2)

def pure_dm(ket):
    ket = np.asarray(ket, dtype=complex)
    ket = ket / np.linalg.norm(ket)
    return np.outer(ket, ket.conj())

# ============================================================
# CONSTRUCTORES DE MODELOS
# ============================================================

def default_bath_terms(n_bath, h):
    """
    H_B = -(h/2) sum_mu Z_mu
    """
    return [(-0.5 * h, {mu: "Z"}) for mu in range(n_bath)]


def noninteracting_field_terms(fields, pauli="Z", half_factor=False):
    """
    H = -sum_i field_i * P_i
    o, si half_factor=True, H = -(1/2) sum_i field_i * P_i
    """
    terms = []
    factor = 0.5 if half_factor else 1.0
    for i, h_i in enumerate(fields):
        terms.append((-factor * h_i, {i: pauli}))
    return terms


def ising_chain_xx_z_terms(n, J, g, periodic=False):
    """
    H = -J sum X_i X_{i+1} - g sum Z_i
    """
    terms = []
    for i in range(n - 1):
        terms.append((-J, {i: "X", i + 1: "X"}))
    if periodic and n > 2:
        terms.append((-J, {n - 1: "X", 0: "X"}))
    for i in range(n):
        terms.append((-g, {i: "Z"}))
    return terms


def square_lattice_edges(Lx, Ly, periodic=False):
    def idx(x, y):
        return y * Lx + x

    edges = []
    for y in range(Ly):
        for x in range(Lx):
            if x + 1 < Lx:
                edges.append((idx(x, y), idx(x + 1, y)))
            elif periodic and Lx > 2:
                edges.append((idx(x, y), idx(0, y)))

            if y + 1 < Ly:
                edges.append((idx(x, y), idx(x, y + 1)))
            elif periodic and Ly > 2:
                edges.append((idx(x, y), idx(x, 0)))
    return edges


def ising_2d_xx_z_terms(Lx, Ly, J, g, periodic=False):
    """
    H = -J sum_{<i,j>} X_i X_j - g sum_i Z_i
    """
    n = Lx * Ly
    terms = [(-J, {i: "X", j: "X"}) for i, j in square_lattice_edges(Lx, Ly, periodic=periodic)]
    for i in range(n):
        terms.append((-g, {i: "Z"}))
    return terms


@dataclass
class ProtocolConfig:
    n_sys: int
    n_bath: int

    # H_S y H_B como listas de términos [(coef, {q:'P', ...}), ...]
    system_terms: list
    bath_terms: Optional[list]

    # lista de acoplos; cada uno es un dict con:
    # {
    #   "sys_terms":  [(coef, {q:'P', ...}), ...],
    #   "bath_terms": [(coef, {q:'P', ...}), ...]
    # }
    coupling_specs: list

    beta: float
    h_filter: float    # h que entra en a = sqrt(4 h / beta)
    theta: float
    delta: float
    MT: int            # número de capas a cada lado: tau in [-MT, ..., MT]
    n_cycles: int

    randomize: bool = False
    randomization_lambda: float = 0.0
    seed: int = 1234
    
    # Rewind paper 2 (Hahn-Parameswaran-Placke)
    rewind: bool = False

    # Si quieres sobreescribir el filtro:
    # array real de longitud 2*MT+1
    filter_values: Optional[np.ndarray] = None


def gaussian_filter_values(cfg: ProtocolConfig):
    """
    f(tau) gaussiano, normalizado para que
    delta * sum_tau |f(tau)| = 1
    """
    taus = np.arange(-cfg.MT, cfg.MT + 1)

    if cfg.filter_values is not None:
        f = np.asarray(cfg.filter_values, dtype=float)
        if len(f) != len(taus):
            raise ValueError("filter_values debe tener longitud 2*MT + 1")
    else:
        if cfg.beta <= 1e-15:
            f = np.ones_like(taus, dtype=float)
            a = np.nan
        else:
            a = np.sqrt(4.0 * cfg.h_filter / cfg.beta)
            f = np.exp(-0.5 * (a * cfg.delta * taus) ** 2)

    f = f / (cfg.delta * np.sum(np.abs(f)))
    a = np.nan if cfg.beta <= 1e-15 else np.sqrt(4.0 * cfg.h_filter / cfg.beta)
    return taus, f, a

# ============================================================
# MOTOR EXACTO DEL CANAL: UN RESET
# ============================================================

def compile_protocol(cfg: ProtocolConfig):
    Hs = make_op(cfg.n_sys, cfg.system_terms)

    if cfg.bath_terms is None:
        Hb_terms = default_bath_terms(cfg.n_bath, cfg.h_filter)
    else:
        Hb_terms = cfg.bath_terms
    Hb = make_op(cfg.n_bath, Hb_terms)

    I_sys = identity_op(cfg.n_sys)
    I_bath = identity_op(cfg.n_bath)

    # Operadores embebidos en sistema+baño
    Hs_full = Hs.tensor(I_bath)
    Hb_full = I_sys.tensor(Hb)

    # V_base = sum_k A_k \otimes B_k
    V_full = SparsePauliOp.from_list([("I" * (cfg.n_sys + cfg.n_bath), 0.0)])
    for spec in cfg.coupling_specs:
        As = make_op(cfg.n_sys, spec["sys_terms"])
        Bb = make_op(cfg.n_bath, spec["bath_terms"])
        V_full = (V_full + As.tensor(Bb)).simplify()

    taus, fvals, a = gaussian_filter_values(cfg)

    dS = 2 ** cfg.n_sys
    dB = 2 ** cfg.n_bath

    bath0 = np.zeros((dB, dB), dtype=complex)
    bath0[0, 0] = 1.0   # |0...0><0...0|

    compiled = {
        "cfg": cfg,
        "Hs": Hs,
        "Hb": Hb,
        "Hs_mat": mat(Hs),
        "Hb_mat": mat(Hb),
        "Hs_full": Hs_full,
        "Hb_full": Hb_full,
        "V_full": V_full,
        "Hs_full_mat": mat(Hs_full),
        "Hb_full_mat": mat(Hb_full),
        "V_full_mat": mat(V_full),
        "taus": taus,
        "fvals": fvals,
        "a": a,
        "dS": dS,
        "dB": dB,
        "bath0": bath0,
    }
    return compiled


def sample_mr(cfg: ProtocolConfig, rng):
    """
    Aproximación discreta del randomizado exponencial del paper.
    Si randomize=False o lambda=0, devuelve 0.
    """
    if (not cfg.randomize) or cfg.randomization_lambda <= 0:
        return 0

    scale = cfg.randomization_lambda * max(cfg.MT, 1)
    return int(np.round(rng.exponential(scale=scale)))


def cycle_unitary_matrix(compiled, mr=0):
    """
    Construye la unidad de un ciclo:
    Q = R * U(M) ... U(0) ... U(-M)
    """
    cfg = compiled["cfg"]
    d = compiled["dS"] * compiled["dB"]

    U_s = expm(-1j * cfg.delta * compiled["Hs_full_mat"])
    U_b = expm(-1j * cfg.delta * compiled["Hb_full_mat"])

    U_total = np.eye(d, dtype=complex)

    for f_tau in compiled["fvals"]:
        U_sb = expm(-1j * cfg.delta * cfg.theta * f_tau * compiled["V_full_mat"])
        U_layer = U_sb @ U_b @ U_s
        U_total = U_layer @ U_total
        
    # Rewind (paper 2): U_S^{-(2MT+1)} aplicado al final del ciclo
    if getattr(cfg, "rewind", False):
        n_layers = 2 * cfg.MT + 1
        U_rewind = expm(+1j * n_layers * cfg.delta * compiled["Hs_full_mat"])
        U_total = U_rewind @ U_total

    if mr > 0:
        U_r = expm(-1j * cfg.delta * mr * compiled["Hs_full_mat"])
        U_total = U_r @ U_total

    return U_total


def apply_one_cycle_map(rho_s, compiled, mr=0, return_bath=False, return_joint=False):
    """
    Aplica un reset exacto al estado del sistema rho_s.
    rho_s puede ser una matriz densidad física o una matriz cualquiera
    (sirve también para construir el superoperador).
    """
    U = cycle_unitary_matrix(compiled, mr=mr)

    rho_tot = np.kron(rho_s, compiled["bath0"])
    rho_after = U @ rho_tot @ U.conj().T

    rho_s_next = ptrace_bath(rho_after, compiled["cfg"].n_sys, compiled["cfg"].n_bath)

    out = {"rho_s_next": rho_s_next}

    if return_bath:
        out["rho_bath_pre_reset"] = ptrace_system(
            rho_after, compiled["cfg"].n_sys, compiled["cfg"].n_bath
        )

    if return_joint:
        out["rho_joint_pre_reset"] = rho_after

    return out

# ============================================================
# TRAYECTORIA COMPLETA Y OBSERVABLES 
# ============================================================

def qubit_pauli_expectations(rho_subsystem, n_qubits, pauli="Z"):
    vals = []
    for q in range(n_qubits):
        op_q = make_op(n_qubits, [(1.0, {q: pauli})])
        vals.append(expect(op_q, rho_subsystem))
    return np.array(vals, dtype=float)


def run_exact_trajectory(
    rho0_s,
    compiled,
    n_cycles=None,
    sample_randomization=False,
    store_rhos=True,
):
    """
    Propaga el canal reset a reset.
    Devuelve:
      - rhos[r] = rho del sistema tras r resets
      - energies[r] = Tr[rho_r Hs]
      - bath_{x,y,z}[r,mu] = <P_mu> del baño justo antes del reset
    """
    cfg = compiled["cfg"]
    n_cycles = cfg.n_cycles if n_cycles is None else n_cycles
    rng = np.random.default_rng(cfg.seed)

    rho = np.array(rho0_s, dtype=complex)

    rhos = [rho.copy()] if store_rhos else None
    energies = [expect(compiled["Hs_mat"], rho)]

    bath_x = []
    bath_y = []
    bath_z = []
    mr_list = []

    for _ in range(n_cycles):
        mr = sample_mr(cfg, rng) if sample_randomization else 0

        step = apply_one_cycle_map(
            rho, compiled, mr=mr, return_bath=True, return_joint=False
        )
        rho = step["rho_s_next"]
        rho_b = step["rho_bath_pre_reset"]

        if store_rhos:
            rhos.append(rho.copy())

        energies.append(expect(compiled["Hs_mat"], rho))
        bath_x.append(qubit_pauli_expectations(rho_b, cfg.n_bath, pauli="X"))
        bath_y.append(qubit_pauli_expectations(rho_b, cfg.n_bath, pauli="Y"))
        bath_z.append(qubit_pauli_expectations(rho_b, cfg.n_bath, pauli="Z"))
        mr_list.append(mr)

    return {
        "rhos": rhos,
        "energies": np.array(energies, dtype=float),
        "bath_x": np.array(bath_x, dtype=float),
        "bath_y": np.array(bath_y, dtype=float),
        "bath_z": np.array(bath_z, dtype=float),
        "mrs": np.array(mr_list, dtype=int),
    }


def late_time_fixed_point(
    compiled,
    rho0_s=None,
    n_burn=200,
    n_keep=50,
    sample_randomization=False,
):
    """
    Aproximación práctica al punto fijo:
    media temporal de las últimas n_keep matrices.
    """
    dS = compiled["dS"]
    if rho0_s is None:
        rho0_s = np.eye(dS, dtype=complex) / dS

    res = run_exact_trajectory(
        rho0_s,
        compiled,
        n_cycles=n_burn + n_keep,
        sample_randomization=sample_randomization,
        store_rhos=True,
    )
    rho_fp = sum(res["rhos"][-n_keep:]) / n_keep
    rho_fp = project_to_physical_dm(rho_fp)
    return rho_fp, res


def fixed_point_via_superop(compiled, mr_max=None):
    """
    Punto fijo exacto del canal promedio.
    Recomendado solo para sistemas pequeños.
    """
    cfg = compiled["cfg"]
    d = compiled["dS"]

    if cfg.randomize and cfg.randomization_lambda > 0:
        mean = cfg.randomization_lambda * max(cfg.MT, 1)
        if mr_max is None:
            mr_max = min(int(np.ceil(3 * mean)) + 5, 200)
        ks = np.arange(mr_max + 1)
        probs = np.exp(-ks / mean)
        probs = probs / probs.sum()
    else:
        ks = np.array([0])
        probs = np.array([1.0])

    S = np.zeros((d * d, d * d), dtype=complex)

    for j in range(d):
        for i in range(d):
            Eij = np.zeros((d, d), dtype=complex)
            Eij[i, j] = 1.0

            out = np.zeros((d, d), dtype=complex)
            for k, p in zip(ks, probs):
                out += p * apply_one_cycle_map(Eij, compiled, mr=int(k))["rho_s_next"]

            S[:, i + j * d] = out.reshape(d * d, order="F")

    evals, evecs = np.linalg.eig(S)
    idx = np.argmin(np.abs(evals - 1.0))
    rho_fp = evecs[:, idx].reshape((d, d), order="F")
    rho_fp = project_to_physical_dm(rho_fp)
    return rho_fp, S, evals


def energy_basis_matrix(rho, Hs):
    """
    rho en la base propia de Hs
    """
    Hm = mat(Hs)
    evals, evecs = np.linalg.eigh(Hm)
    rho_e = evecs.conj().T @ rho @ evecs
    return evals, rho_e


def metrics_vs_target(rhos, rho_target):
    fidelities = np.array([fidelity_dm(r, rho_target) for r in rhos], dtype=float)
    trace_dists = np.array([trace_distance_dm(r, rho_target) for r in rhos], dtype=float)
    return fidelities, trace_dists



# ============================================================
# MÉTRICAS GENÉRICAS EN BASE DE ENERGÍA
# ============================================================

def population_error_l1_in_energy_basis(rho, rho_target, Hs):
    _, rho_e = energy_basis_matrix(rho, Hs)
    _, tgt_e = energy_basis_matrix(rho_target, Hs)
    p = np.real(np.diag(rho_e))
    q = np.real(np.diag(tgt_e))
    return float(np.sum(np.abs(p - q)))

def coherence_norm_in_energy_basis(rho, Hs, norm="l1"):
    _, rho_e = energy_basis_matrix(rho, Hs)
    off = rho_e.copy()
    np.fill_diagonal(off, 0.0)
    if norm == "fro":
        return float(np.linalg.norm(off))
    elif norm == "l1":
        return float(np.sum(np.abs(off)))
    else:
        raise ValueError("norm debe ser 'fro' o 'l1'")
    
def total_coherence_vs_resets(rho0, compiled, Hs=None, n_cycles=None):
    """
    Devuelve la coherencia total en base de energía a lo largo de la trayectoria:
        C_r = sum_{a != b} |rho^{(E)}_{ab}(r)|

    Parámetros
    ----------
    rho0 : np.ndarray
        Estado inicial del sistema.
    compiled : dict
        Objeto compilado del protocolo.
    Hs : np.ndarray | None
        Hamiltoniano del sistema. Si es None, se toma de compiled.
    n_cycles : int | None
        Número de ciclos. Si es None, se usa compiled["cfg"].n_cycles.

    Returns
    -------
    coh : np.ndarray
        Array de longitud n_cycles + 1 con la coherencia total por ciclo.
    res : dict
        Salida completa de run_exact_trajectory.
    """
    if Hs is None:
        Hs = compiled.get("Hs")
        if Hs is None:
            Hs = compiled.get("Hs_mat")
        if Hs is None:
            raise KeyError("No encuentro Hs ni Hs_mat dentro de compiled.")

    if n_cycles is None:
        n_cycles = compiled["cfg"].n_cycles

    res = run_exact_trajectory(rho0, compiled, n_cycles=n_cycles)

    coh = np.array(
        [coherence_norm_in_energy_basis(rho, Hs) for rho in res["rhos"]],
        dtype=float
    )

    return coh, res

def state_error_summary(rho_ss, rho_gibbs, Hs):
    return {
        "population_error_l1": population_error_l1_in_energy_basis(rho_ss, rho_gibbs, Hs),
        "coherence_norm": coherence_norm_in_energy_basis(rho_ss, Hs, norm="fro"),
        "trace_distance": trace_distance_dm(rho_ss, rho_gibbs),
        "fidelity": fidelity_dm(rho_ss, rho_gibbs),
    }

def trajectory_error_arrays(rhos, rho_gibbs, Hs):
    pop_err = np.array([population_error_l1_in_energy_basis(r, rho_gibbs, Hs) for r in rhos], dtype=float)
    coh_err = np.array([coherence_norm_in_energy_basis(r, Hs, norm="fro") for r in rhos], dtype=float)
    dtr = np.array([trace_distance_dm(r, rho_gibbs) for r in rhos], dtype=float)
    fid = np.array([fidelity_dm(r, rho_gibbs) for r in rhos], dtype=float)
    return pop_err, coh_err, dtr, fid


# ============================================================
# RUN GENÉRICO DE UN CASO
# ============================================================

def run_protocol_case(
    cfg,
    rho0=None,
    compute_fp=True,
    fp_mode="auto",
    late_fp_burn=200,
    late_fp_keep=50,
):
    compiled = compile_protocol(cfg)

    if rho0 is None:
        rho0 = maximally_mixed(cfg.n_sys)

    res = run_exact_trajectory(
        rho0,
        compiled,
        n_cycles=cfg.n_cycles,
        sample_randomization=cfg.randomize,
        store_rhos=True,
    )

    Hs = make_op(cfg.n_sys, cfg.system_terms)
    rho_g = gibbs_state(Hs, cfg.beta)

    rho_fp = None
    fp_method_used = None

    if compute_fp:
        dS = compiled["dS"]

        if fp_mode == "auto":
            if dS <= 4 and (not cfg.randomize):
                fp_mode_eff = "superop"
            else:
                fp_mode_eff = "late_time"
        else:
            fp_mode_eff = fp_mode

        if fp_mode_eff == "superop":
            rho_fp, _, _ = fixed_point_via_superop(compiled)
            fp_method_used = "superop"
        elif fp_mode_eff == "late_time":
            rho_fp, _ = late_time_fixed_point(
                compiled,
                rho0_s=rho0,
                n_burn=late_fp_burn,
                n_keep=late_fp_keep,
                sample_randomization=cfg.randomize,
            )
            fp_method_used = "late_time"
        else:
            raise ValueError("fp_mode debe ser 'auto', 'superop' o 'late_time'")

    out = {
        "cfg": cfg,
        "compiled": compiled,
        "rho0": rho0,
        "res": res,
        "Hs": Hs,
        "rho_g": rho_g,
        "rho_fp": rho_fp,
        "fp_method_used": fp_method_used,
        "Eg": expect(Hs, rho_g),
        "E0": expect(Hs, rho0),
        "Efinal": float(res["energies"][-1]),
    }

    if rho_fp is not None:
        out["Efp"] = expect(Hs, rho_fp)
        out["fp_vs_gibbs"] = state_error_summary(rho_fp, rho_g, Hs)
    else:
        out["Efp"] = np.nan
        out["fp_vs_gibbs"] = None

    out["final_vs_gibbs"] = state_error_summary(res["rhos"][-1], rho_g, Hs)
    return out

def run_scheduled_protocol(
    compiled_bank,
    schedule,
    rho0,
    Hs=None,
    beta=None,
    rho_g=None,
    random_seed=1234,
    n_cycles=None,
    compute_fp_ref=True,
):
    """
    Ejecuta una familia de protocolos donde en cada ciclo eliges qué bloque compilado aplicar.

    Parámetros
    ----------
    compiled_bank : dict[str, compiled]
        Banco de protocolos compilados. Todas las entradas deben compartir n_sys, n_bath y Hs.
    schedule : str | list[str] | callable
        - str: variantes estándar ("always_X", "alternate_XY", "random_XY", etc.)
        - list[str]: etiqueta explícita por ciclo
        - callable: fn(r, rng) -> label
    rho0 : np.ndarray
        Estado inicial del sistema.
    Hs, beta, rho_g : opcionales
        Si no se pasan, se infieren del primer elemento de compiled_bank.
    random_seed : int
        Semilla para schedules aleatorios.
    n_cycles : int | None
        Si None, se usa el n_cycles del cfg del primer compiled.
    compute_fp_ref : bool
        Si True, devuelve promedio late-time de las últimas matrices como referencia estacionaria.

    Devuelve
    --------
    dict con trayectoria, métricas, labels usados y punto fijo de referencia.
    """
    if not compiled_bank:
        raise ValueError("compiled_bank está vacío.")

    first = next(iter(compiled_bank.values()))
    cfg_ref = first["cfg"]

    if n_cycles is None:
        n_cycles = cfg_ref.n_cycles

    if Hs is None:
        Hs = first["Hs"]

    if rho_g is None:
        if beta is None:
            beta = cfg_ref.beta
        rho_g = gibbs_state(Hs, beta)

    rng = np.random.default_rng(random_seed)

    def choose_label(r):
        if isinstance(schedule, str):
            if schedule.startswith("always_"):
                return schedule.replace("always_", "")
            elif schedule == "alternate_XY":
                return "X" if (r % 2 == 0) else "Y"
            elif schedule == "alternate_YZ":
                return "Y" if (r % 2 == 0) else "Z"
            elif schedule == "alternate_XZ":
                return "X" if (r % 2 == 0) else "Z"
            elif schedule == "random_XY":
                return rng.choice(["X", "Y"])
            elif schedule == "random_YZ":
                return rng.choice(["Y", "Z"])
            elif schedule == "random_XZ":
                return rng.choice(["X", "Z"])
            else:
                raise ValueError(f"schedule string no reconocida: {schedule}")
        elif callable(schedule):
            return schedule(r, rng)
        else:
            # secuencia explícita
            if r >= len(schedule):
                raise ValueError("La secuencia explícita de labels es más corta que n_cycles.")
            return schedule[r]

    rho = np.array(rho0, dtype=complex)
    rhos = [rho.copy()]
    energies = [expect(Hs, rho)]
    chosen_labels = []

    for r in range(n_cycles):
        label = choose_label(r)
        if label not in compiled_bank:
            raise KeyError(f"Label '{label}' no está en compiled_bank.")
        step = apply_one_cycle_map(rho, compiled_bank[label], mr=0, return_bath=False)
        rho = step["rho_s_next"]

        rhos.append(rho.copy())
        energies.append(expect(Hs, rho))
        chosen_labels.append(label)

    rho_final = rhos[-1]

    if compute_fp_ref:
        n_keep = min(50, len(rhos))
        rho_fp_ref = project_to_physical_dm(sum(rhos[-n_keep:]) / n_keep)
    else:
        rho_fp_ref = None

    return {
        "Hs": Hs,
        "rho_g": rho_g,
        "rhos": rhos,
        "energies": np.array(energies, dtype=float),
        "rho_final": rho_final,
        "rho_fp_ref": rho_fp_ref,
        "chosen_labels": chosen_labels,
        "Efinal": float(energies[-1]),
        "Eg": expect(Hs, rho_g),
        "D_final_gibbs": trace_distance_dm(rho_final, rho_g),
        "F_final_gibbs": fidelity_dm(rho_final, rho_g),
        "PopErr_final": population_error_l1_in_energy_basis(rho_final, rho_g, Hs),
        "CohErr_final": coherence_norm_in_energy_basis(rho_final, Hs, norm="l1"),
    }

def heat_capacity_from_variance(rho, H, beta):
    """
    C_V = β² (⟨H²⟩ − ⟨H⟩²)_ρ  calculada sobre un estado arbitrario.

    Casos de uso:
      - Si ρ = ρ_β (Gibbs): reproduce la capacidad calorífica termodinámica exacta.
      - Si ρ = ρ_fp (punto fijo del protocolo): mide si el punto fijo 
        satisface la relación varianza-capacidad propia de un estado térmico.
      - Si ρ ≠ Gibbs, no hay garantía de que esta cantidad sea positiva o
        tenga 
        significado termodinámico: es un diagnóstico, no una identidad.

    Parámetros
    ----------
    rho : np.ndarray
        Matriz densidad del sistema.
    H : np.ndarray | SparsePauliOp
        Hamiltoniano del sistema. Puede ser matriz o SparsePauliOp.
    beta : float
        Inverso de la temperatura termodinámica.

    Devuelve
    --------
    float
        Valor de β² Var_ρ(H). No se fuerza a ser positivo.
    """
    Hm = mat(H)
    E  = float(np.real(np.trace(rho @ Hm)))
    E2 = float(np.real(np.trace(rho @ (Hm @ Hm))))
    return float(beta**2 * (E2 - E**2))

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


######################
# AÑADIDOS DE NB01
########################

def first_hitting_time(values, eps):
    values = np.asarray(values, dtype=float)
    hits = np.where(values <= eps)[0]
    return int(hits[0]) if len(hits) else np.nan


def spectral_gap_from_compiled(compiled):
    """
    Gap espectral del canal de un ciclo:
        gap = 1 - |lambda_2|.
    """
    _, _, evals = fixed_point_via_superop(compiled)

    idx_one = np.argmin(np.abs(evals - 1.0))
    evals_rest = np.delete(evals, idx_one)

    if len(evals_rest) == 0:
        return np.nan, np.nan

    lambda2 = float(np.max(np.abs(evals_rest)))
    gap = float(1.0 - lambda2)
    return lambda2, gap


def classify_regime(theta):
    if theta <= 0.10:
        return "débil / perturbativo"
    elif theta <= 0.35:
        return "intermedio"
    else:
        return "fuerte / τJ≈1"


def gibbs_status(nstar_gibbs, D_fp_gibbs, eps_gibbs, n_cycles):
    """
    Clasifica por qué aparece o no aparece un tiempo de llegada a Gibbs.
    """
    if np.isfinite(nstar_gibbs):
        return "alcanzado"

    if D_fp_gibbs > eps_gibbs:
        return "no alcanzable: sesgo estacionario"

    return f"no observado: lento >{n_cycles}"


def gibbs_from_H(H, beta):
    """
    Estado de Gibbs exp(-beta H)/Z calculado de forma estable.
    """
    evals, evecs = np.linalg.eigh(H)
    weights = np.exp(-beta * (evals - evals.min()))
    probs = weights / weights.sum()
    return evecs @ np.diag(probs) @ evecs.conj().T


def first_hitting_time(values, eps):
    values = np.asarray(values, dtype=float)
    hits = np.where(values <= eps)[0]
    return int(hits[0]) if len(hits) else np.nan