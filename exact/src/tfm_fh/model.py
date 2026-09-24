"""
tfm_fh.model
============

Modelo de Fermi-Hubbard con Jordan-Wigner intercalado y bundle ISA
para experimentos shot-based con backend FakeSherbrooke.

Convención: orden de qubits  up0, down0, up1, down1, ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy import kron
from functools import reduce
from scipy.linalg import expm, eigh, block_dia
from scipy.sparse.linalg import LinearOperator, lgmres
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import itertools
import networkx as nx
from matplotlib.lines import Line2D
import warnings, time
from itertools import product
import gc

from qiskit.quantum_info import SparsePauliOp

warnings.filterwarnings('ignore')



# ============================================================
# Pauli y utilidades básicas
# ============================================================

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)


def pauli_string(ops, n_qubits):
    return reduce(kron, [PAULI.get(ops.get(q,'I'),I2) for q in range(n_qubits)])

def trace_distance(rho, sigma):
    return 0.5 * np.sum(np.linalg.svd(rho - sigma, compute_uv=False))

def partial_trace_B(rho_tot, dim_S, dim_B):
    return np.trace(rho_tot.reshape(dim_S, dim_B, dim_S, dim_B), axis1=1, axis2=3)

def von_neumann_entropy(rho):
    ev = np.linalg.eigvalsh(rho); ev = ev[ev > 1e-15]
    return -np.sum(ev * np.log(ev))

def off_diag_norm(rho, eigvecs):
    re = eigvecs.conj().T @ rho @ eigvecs
    return np.linalg.norm(re - np.diag(np.diag(re)), 'fro')

def reduced_dm_qubits(rho, keep, n_qubits):
    """
    Matriz reducida manteniendo los qubits indicados en keep.

    Convención:
      q=0 es el primer factor tensorial usado en pauli_string.
    """
    keep = sorted(set(int(q) for q in keep))

    if any(q < 0 or q >= n_qubits for q in keep):
        raise ValueError(f"keep={keep} contiene qubits fuera de rango")

    trace_out = [q for q in range(n_qubits) if q not in keep]

    tensor = rho.reshape([2] * n_qubits + [2] * n_qubits)
    current_n = n_qubits

    for q in sorted(trace_out, reverse=True):
        tensor = np.trace(tensor, axis1=q, axis2=q + current_n)
        current_n -= 1

    d_keep = 2 ** len(keep)
    return tensor.reshape(d_keep, d_keep)


def qiskit_label(ops, n_qubits):
    """Pauli string en convención Qiskit (qubit más alto a la izquierda)."""
    chars = ["I"] * n_qubits
    for q, p in ops.items():
        chars[q] = p
    return "".join(reversed(chars))


def trace_distance_fh(rho, sigma):
    svals = np.linalg.svd(rho - sigma, compute_uv=False)
    return 0.5 * float(np.sum(svals))


def partial_trace_bath_fh(rho_tot, d_sys, d_bath):
    return np.trace(
        rho_tot.reshape(d_sys, d_bath, d_sys, d_bath), axis1=1, axis2=3
    )


def suggest_MT(h, beta, delta, n_sigma=4.5):
    a = np.sqrt(4.0 * h / beta)
    width_tau = 1.0 / (a * delta)
    return max(4, int(np.ceil(n_sigma * width_tau)))


def gaussian_filter_values(h, beta, delta, MT):
    a = np.sqrt(4.0 * h / beta)
    taus = np.arange(-MT, MT + 1)
    raw = np.exp(-0.5 * (a * delta * taus) ** 2)
    fvals = raw / (delta * np.sum(raw))
    return taus, fvals


def sample_random_depths(n_cycles, MT, mode="exp", lambda_rand=1.0, seed=1234):
    rng = np.random.default_rng(seed)
    if mode == "none":
        return np.zeros(n_cycles, dtype=int)
    if mode == "uniform":
        return rng.integers(0, 2 * MT + 1, size=n_cycles, endpoint=False)
    if mode == "exp":
        cutoff = max(4 * MT, 1)
        m_vals = np.arange(0, cutoff + 1)
        scale = max(lambda_rand * MT, 1e-12)
        probs = np.exp(-m_vals / scale)
        probs = probs / probs.sum()
        return rng.choice(m_vals, size=n_cycles, p=probs)
    raise ValueError(f"Unknown randomization mode: {mode}")


# ============================================================
# Modelo Fermi-Hubbard con JW intercalado
# ============================================================



def paper_local_operators(model):
    """
    Operadores de acoplo del paper Lloyd-Abanin: A_q = (Z_q + Y_q)/sqrt(2)
    con un baño por qubit del sistema.
    """
    return [
        (pauli_string({q: "Z"}, model.n_qubits)
         + pauli_string({q: "Y"}, model.n_qubits)) / np.sqrt(2.0)
        for q in range(model.n_qubits)
    ]


# ============================================================
# Bundle ISA: backend, transpilación, shots
# ============================================================

ISA_CIRCUIT_CACHE_FH = {}
FH_NATIVE_BASIS = ["rz", "sx", "x", "cx", "measure", "reset"]
_FH_BACKEND_BUNDLE = None


def get_fh_backend_bundle(min_total_qubits=None, force_refresh=False):
    """
    Construye backend ideal/noisy para experimentos ISA en FH.
    Usa FakeSherbrooke si está disponible.
    """
    global _FH_BACKEND_BUNDLE

    min_total_qubits = 8 if min_total_qubits is None else int(min_total_qubits)
    if (not force_refresh) and (_FH_BACKEND_BUNDLE is not None):
        backend = _FH_BACKEND_BUNDLE["hw_backend"]
        if getattr(backend, "num_qubits", min_total_qubits) >= min_total_qubits:
            return _FH_BACKEND_BUNDLE

    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        hw_backend = FakeSherbrooke()
        label = "FakeSherbrooke (snapshot calibraciones reales)"
    except Exception:
        from qiskit.providers.fake_provider import GenericBackendV2
        n_qubits = max(min_total_qubits, 8)
        hw_backend = GenericBackendV2(
            num_qubits=n_qubits,
            basis_gates=FH_NATIVE_BASIS,
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

    _FH_BACKEND_BUNDLE = {
        "hw_backend": hw_backend,
        "label": label,
        "ideal_sim": ideal_sim,
        "noise_model": noise_model,
        "noisy_sim": noisy_sim,
        "have_noise_model": have_noise_model,
    }
    return _FH_BACKEND_BUNDLE


def mixture_plan_fh(init_mode, shots, n_sys, rng=None):
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
    if init_mode == "half_filling":
        # Estado con ocupación 1 por sitio (alternando up/down): |1010 1010...>
        # cada sitio empieza con un electrón.
        # En orden interleaved up0,down0,up1,down1,...:
        # alternamos un sitio con up ocupado, otro con down ocupado
        bitstring = ""
        for site in range(int(n_sys) // 2):
            if site % 2 == 0:
                bitstring += "10"  # up ocupado, down vacio
            else:
                bitstring += "01"
        # padding por si n_sys es impar
        bitstring = bitstring.ljust(int(n_sys), "0")
        return [(bitstring, int(shots))]
    raise ValueError(f"init_mode no reconocido: {init_mode}")


# ============================================================
# Construcción del circuito Qiskit con baño dinámico
# ============================================================

def _bath_hamiltonian_sparse_fh(n_bath, h):
    from qiskit.quantum_info import SparsePauliOp
    terms = [(qiskit_label({mu: "Z"}, n_bath), -0.5 * h) for mu in range(n_bath)]
    return SparsePauliOp.from_list(terms).simplify()


def _system_hamiltonian_sparse_fh(model):
    """Hamiltoniano del sistema FH como SparsePauliOp."""
    from qiskit.quantum_info import SparsePauliOp
    n = model.n_qubits
    terms = []

    for i, j in model.bonds:
        for spin in ("up", "down"):
            p = model.orb_index(i, spin)
            q = model.orb_index(j, spin)
            if p > q:
                p, q = q, p
            xx = {p: "X", q: "X"}
            yy = {p: "Y", q: "Y"}
            for m in range(p + 1, q):
                xx[m] = "Z"
                yy[m] = "Z"
            terms.append((qiskit_label(xx, n), -0.5 * model.t))
            terms.append((qiskit_label(yy, n), -0.5 * model.t))

    for i in range(model.n_sites):
        pu = model.orb_index(i, "up")
        pd = model.orb_index(i, "down")
        terms.append((qiskit_label({pu: "Z"}, n), -0.25 * model.U))
        terms.append((qiskit_label({pd: "Z"}, n), -0.25 * model.U))
        terms.append((qiskit_label({pu: "Z", pd: "Z"}, n), +0.25 * model.U))

    return SparsePauliOp.from_list(terms).simplify()


def _interaction_sparse_fh(n_sys):
    """V = (1/sqrt(2)) sum_q (Z_q + Y_q) Y_{baño_q}, con baño separado."""
    from qiskit.quantum_info import SparsePauliOp
    n_bath = n_sys
    n_tot = n_sys + n_bath
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    terms = []
    for q in range(n_sys):
        bq = n_sys + q
        terms.append((qiskit_label({q: "Z", bq: "Y"}, n_tot), inv_sqrt2))
        terms.append((qiskit_label({q: "Y", bq: "Y"}, n_tot), inv_sqrt2))
    return SparsePauliOp.from_list(terms).simplify()


def build_fh_dynamic_circuit(
    model, n_cycles, h=1.0, beta=2.0, theta=0.35, delta=0.30, MT=None,
    trotter_order=2, trotter_reps=2,
    init_label=None, final_basis="Z",
):
    """
    Circuito dinámico FH:
      - prepara el sistema en init_label
      - aplica n_cycles de (Trotter ciclo + medida baño + reset baño)
      - mide el sistema al final si final_basis no es None

    Registros:
      QuantumRegister "sys"  (n_sys = 2*n_sites)
      QuantumRegister "bath" (n_bath = n_sys)
      ClassicalRegister "c" con n_cycles*n_bath bits + n_sys bits finales
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import SuzukiTrotter

    n_sys = model.n_qubits
    n_bath = n_sys
    n_tot = n_sys + n_bath

    MT = suggest_MT(h, beta, delta) if MT is None else int(MT)

    # Operadores Pauli del Hamiltoniano y acoplo
    H_sys = _system_hamiltonian_sparse_fh(model)
    H_bath = _bath_hamiltonian_sparse_fh(n_bath, h)
    V_int = _interaction_sparse_fh(n_sys)
    _, fvals = gaussian_filter_values(h, beta, delta, MT)

    # Registros
    q_sys = QuantumRegister(n_sys, "sys")
    q_bath = QuantumRegister(n_bath, "bath")

    n_bath_bits = int(n_cycles) * n_bath
    n_final_bits = n_sys if final_basis is not None else 0
    c = ClassicalRegister(n_bath_bits + n_final_bits, "c")

    qc = QuantumCircuit(q_sys, q_bath, c)

    # Preparación del sistema
    if init_label is None or init_label == "0" * n_sys:
        pass
    elif len(init_label) == n_sys and set(init_label).issubset({"0", "1"}):
        # Convención: bit 0 -> qubit 0, bit n-1 -> qubit n-1 (lectura de izquierda a derecha)
        for i, bit in enumerate(init_label):
            if bit == "1":
                qc.x(q_sys[i])
    else:
        raise ValueError(f"init_label no reconocido: {init_label}")

    # Construir el ciclo Trotterizado
    synth = SuzukiTrotter(order=trotter_order, reps=trotter_reps)
    sys_qubits = list(q_sys)
    bath_qubits = list(q_bath)
    all_qubits = sys_qubits + bath_qubits

    for r in range(int(n_cycles)):
        # Trotter del ciclo: para cada tau aplicar U_S, U_B, U_int
        for f_tau in fvals:
            qc.append(
                PauliEvolutionGate(H_sys, time=delta, synthesis=synth),
                sys_qubits,
            )
            qc.append(
                PauliEvolutionGate(H_bath, time=delta, synthesis=synth),
                bath_qubits,
            )
            if abs(theta * f_tau) > 1e-15:
                qc.append(
                    PauliEvolutionGate(
                        V_int, time=delta * theta * f_tau, synthesis=synth
                    ),
                    all_qubits,
                )

        # Medida del baño antes del reset
        for mu in range(n_bath):
            qc.measure(q_bath[mu], c[r * n_bath + mu])

        # Reset
        for mu in range(n_bath):
            qc.reset(q_bath[mu])

    # Medida final del sistema
    if final_basis is not None:
        if final_basis == "Z":
            for i in range(n_sys):
                qc.measure(q_sys[i], c[n_bath_bits + i])
        elif final_basis == "X":
            for i in range(n_sys):
                qc.h(q_sys[i])
                qc.measure(q_sys[i], c[n_bath_bits + i])
        else:
            raise ValueError(f"final_basis no soportada: {final_basis}")

    return qc


def get_isa_dynamic_circuit_fh(
    model, n_cycles, h=1.0, beta=2.0, theta=0.35, delta=0.30, MT=None,
    trotter_order=2, trotter_reps=2,
    init_label=None, final_basis="Z",
    optimization_level=1, backend_bundle=None,
):
    """Transpila el circuito dinámico FH contra el backend ISA."""
    from qiskit.transpiler import generate_preset_pass_manager

    n_sys = model.n_qubits
    n_bath = n_sys
    n_tot = n_sys + n_bath

    backend_bundle = (
        get_fh_backend_bundle(min_total_qubits=n_tot)
        if backend_bundle is None else backend_bundle
    )
    hw_backend = backend_bundle["hw_backend"]

    MT_use = suggest_MT(h, beta, delta) if MT is None else int(MT)
    key = (
        backend_bundle["label"], n_sys, model.t, model.U,
        h, beta, theta, delta, MT_use,
        trotter_order, trotter_reps,
        n_cycles, init_label or "default", final_basis,
        int(optimization_level),
    )
    if key not in ISA_CIRCUIT_CACHE_FH:
        qc = build_fh_dynamic_circuit(
            model, n_cycles=n_cycles,
            h=h, beta=beta, theta=theta, delta=delta, MT=MT_use,
            trotter_order=trotter_order, trotter_reps=trotter_reps,
            init_label=init_label, final_basis=final_basis,
        )
        pm = generate_preset_pass_manager(
            optimization_level=int(optimization_level), backend=hw_backend,
        )
        ISA_CIRCUIT_CACHE_FH[key] = pm.run(qc)
    return ISA_CIRCUIT_CACHE_FH[key]


def parse_dynamic_counts_fh(counts, n_cycles, n_sys, n_bath, final_basis="Z"):
    """Reconstruye P(1) por auxiliar y por qubit del sistema."""
    n_bath_bits = int(n_cycles) * int(n_bath)
    n_total = sum(counts.values())

    bath_p1_per_aux = np.zeros((int(n_cycles), int(n_bath)), dtype=float)
    sys_p1_per_qubit = np.zeros(int(n_sys), dtype=float)
    sys_zz_pairs = {}  # para extraer <Z_p Z_q>
    has_final = final_basis is not None
    expected_len = n_bath_bits + (n_sys if has_final else 0)

    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "").zfill(expected_len)
        # En Qiskit, el bit más a la izquierda es el último creg
        # Si tenemos final + bath: bits = [sys (n_sys)][bath (n_bath_bits)]
        if has_final:
            sys_bits = bits[:n_sys]
            bath_bits = bits[n_sys:]
            # sys_bits leído de izq a der corresponde a c[n_bath_bits+n_sys-1] ... c[n_bath_bits]
            # c[n_bath_bits + i] es la medida del qubit i
            for i in range(n_sys):
                if sys_bits[n_sys - 1 - i] == "1":
                    sys_p1_per_qubit[i] += count
            # ZZ correlations
            for p in range(n_sys):
                for q in range(p + 1, n_sys):
                    bp = sys_bits[n_sys - 1 - p]
                    bq = sys_bits[n_sys - 1 - q]
                    z_p = 1 if bp == "0" else -1
                    z_q = 1 if bq == "0" else -1
                    sys_zz_pairs[(p, q)] = sys_zz_pairs.get((p, q), 0.0) + z_p * z_q * count
        else:
            bath_bits = bits

        for r in range(int(n_cycles)):
            for mu in range(int(n_bath)):
                idx_in_bath = r * n_bath + mu
                # bath_bits leído de izq a der: bath_bits[0] es el creg más alto (el último escrito)
                # Queremos c[idx_in_bath]
                pos = len(bath_bits) - 1 - idx_in_bath
                if bath_bits[pos] == "1":
                    bath_p1_per_aux[r, mu] += count

    bath_p1_per_aux /= n_total
    sys_p1_per_qubit /= n_total
    for k in sys_zz_pairs:
        sys_zz_pairs[k] /= n_total

    return {
        "bath_p1_per_aux": bath_p1_per_aux,
        "sys_p1_per_qubit": sys_p1_per_qubit,
        "sys_zz_pairs": sys_zz_pairs,
    }


def energy_fh_z_only(model, sys_p1_per_qubit, sys_zz_pairs):
    """
    Reconstruye SOLO la parte diagonal-en-Z del Hamiltoniano FH:
      E_Z = (U/4) sum_i (1 - <Z_up_i> - <Z_down_i> + <Z_up_i Z_down_i>)

    Es decir: E_Z = U * <D> donde D = avg_double_occupancy.
    NOTA: la parte cinética -t sum c† c NO se reconstruye; requiere medir
    en bases conjugadas con cuerdas Z de Jordan-Wigner.

    Devuelve: (E_Z, sigma_E_Z_estimate)
    """
    n_sys = model.n_qubits
    z_per_q = 1.0 - 2.0 * np.asarray(sys_p1_per_qubit, dtype=float)

    E_z = 0.0
    for i in range(model.n_sites):
        pu = model.orb_index(i, "up")
        pd = model.orb_index(i, "down")
        pair = (min(pu, pd), max(pu, pd))
        zz = sys_zz_pairs.get(pair, 0.0)
        E_z += (model.U / 4.0) * (1.0 - z_per_q[pu] - z_per_q[pd] + zz)
    return float(E_z)


def energy_fh_z_total_estimate(model, sys_p1_per_qubit, sys_zz_pairs):
    """
    Estimación incompleta de la energía total: E_Z + 0 para hopping.
    Es lo que se compara contra <H>_Gibbs en el plot, asumiendo que
    el lector entiende que es solo la contribución diagonal en Z.
    """
    return energy_fh_z_only(model, sys_p1_per_qubit, sys_zz_pairs)


def run_isa_dynamic_shots_fh(
    model, n_cycles,
    h=1.0, beta=2.0, theta=0.35, delta=0.30, MT=None,
    trotter_order=2, trotter_reps=2,
    shots=4096, init_mode="mm", sim_mode="ideal",
    final_basis="Z",
    seed_base=1234, optimization_level=1,
    backend_bundle=None,
):
    """
    Ejecuta el protocolo FH dinámico shot-based.
    Devuelve E_Z reconstruida + bath_p1_per_cycle.
    """
    n_sys = model.n_qubits
    n_bath = n_sys

    backend_bundle = (
        get_fh_backend_bundle(min_total_qubits=2 * n_sys)
        if backend_bundle is None else backend_bundle
    )

    rng = np.random.default_rng(seed_base)
    plan = mixture_plan_fh(init_mode, shots, n_sys, rng=rng)

    if sim_mode == "ideal":
        sim = backend_bundle["ideal_sim"]
    elif sim_mode == "noisy":
        sim = backend_bundle["noisy_sim"]
        if sim is None:
            raise RuntimeError("Noise model no disponible para sim_mode='noisy'.")
    else:
        raise ValueError("sim_mode debe ser 'ideal' o 'noisy'")

    bath_agg = np.zeros((int(n_cycles), n_bath), dtype=float)
    sys_agg = np.zeros(n_sys, dtype=float)
    zz_agg = {}

    for i, (init_label, n_sh) in enumerate(plan):
        if n_sh <= 0:
            continue
        qc_isa = get_isa_dynamic_circuit_fh(
            model, n_cycles=n_cycles,
            h=h, beta=beta, theta=theta, delta=delta, MT=MT,
            trotter_order=trotter_order, trotter_reps=trotter_reps,
            init_label=init_label, final_basis=final_basis,
            optimization_level=optimization_level,
            backend_bundle=backend_bundle,
        )
        result = sim.run(
            qc_isa, shots=int(n_sh),
            seed_simulator=int(seed_base + 97 * i + 13 * int(n_cycles)),
        ).result()
        parsed = parse_dynamic_counts_fh(
            result.get_counts(), n_cycles=n_cycles,
            n_sys=n_sys, n_bath=n_bath, final_basis=final_basis,
        )
        weight = float(n_sh) / float(shots)
        bath_agg += weight * parsed["bath_p1_per_aux"]
        sys_agg += weight * parsed["sys_p1_per_qubit"]
        for k, v in parsed["sys_zz_pairs"].items():
            zz_agg[k] = zz_agg.get(k, 0.0) + weight * v

    bath_p1_per_cycle = np.mean(bath_agg, axis=1)
    E_z = energy_fh_z_only(model, sys_agg, zz_agg)
    # Estimación grosera de sigma_E (binomial sobre cada P(1))
    var_per_qubit = sys_agg * (1.0 - sys_agg) / max(int(shots), 1)
    E_sigma = float(0.5 * model.U * np.sqrt(np.sum(var_per_qubit)))

    return {
        "bath_p1_per_cycle": bath_p1_per_cycle,
        "bath_p1_per_aux": bath_agg,
        "sys_p1_per_qubit": sys_agg,
        "sys_zz_pairs": zz_agg,
        "E_final": float(E_z),
        "E_sigma": float(E_sigma),
        "shots": int(shots),
        "n_cycles": int(n_cycles),
        "backend_label": backend_bundle["label"],
    }


def run_energy_vs_cycles_shots_fh(
    model, max_cycles,
    h=1.0, beta=2.0, theta=0.35, delta=0.30, MT=None,
    trotter_order=2, trotter_reps=2,
    shots=4096, init_mode="mm", sim_mode="ideal",
    seed_base=1234, optimization_level=1, backend_bundle=None,
):
    """Barrido de energía Z vs número de ciclos."""
    rows = []
    for ncyc in range(1, int(max_cycles) + 1):
        out = run_isa_dynamic_shots_fh(
            model, n_cycles=ncyc,
            h=h, beta=beta, theta=theta, delta=delta, MT=MT,
            trotter_order=trotter_order, trotter_reps=trotter_reps,
            shots=shots, init_mode=init_mode, sim_mode=sim_mode,
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


def gibbs_E_z_fh(model, beta):
    """
    E_Z del estado de Gibbs: solo la parte de interacción U,
    sumada sobre todos los sitios (no promediada).
        E_Z = sum_i U <n_up,i n_down,i> = U * n_sites * <D>_avg
    Esta es la magnitud comparable a la reconstrucción shots de
    energy_fh_z_only(), que ya suma sobre sitios.
    """
    rho_g = model.gibbs_state(beta)
    return float(model.U * model.n_sites * model.avg_double_occupancy(rho_g))


def gaussian_filter(M_T, delta, a):
    taus = np.arange(-M_T, M_T+1)
    raw = np.exp(-0.5*a**2*delta**2*taus**2)
    return taus, raw/(delta*np.sum(raw))

def compute_a(h_bath, beta):
    return np.sqrt(4.0*h_bath/beta)


class FermiHubbardJW:
    """
    Fermi-Hubbard con Jordan-Wigner intercalado.

    Hamiltoniano:
        H = -t sum_<ij>,sigma (c†_{i,sigma} c_{j,sigma} + h.c.)
            + U sum_i n_{i,up} n_{i,down}
            - mu sum_i,sigma n_{i,sigma}
            - h_mag sum_i S^z_i

    Orden de qubits:
        up0, down0, up1, down1, ...

    Es decir:
        orbital_index(site, "up")   = 2*site
        orbital_index(site, "down") = 2*site + 1

    Esta clase sirve para:
      - ED exacta NumPy;
      - construcción de observables;
      - validación de simetrías;
      - generación de términos Pauli para Qiskit.
    """

    def __init__(
        self,
        n_sites,
        bonds=None,
        t=1.0,
        U=4.0,
        *,
        t_hop=None,
        U_int=None,
        mu=0.0,
        h_mag=0.0,
    ):
        self.n_sites = int(n_sites)

        if bonds is None:
            bonds = [(i, i + 1) for i in range(self.n_sites - 1)]

        self.bonds = [(int(i), int(j)) for i, j in bonds]

        # Compatibilidad con nombres antiguos
        self.t = float(t if t_hop is None else t_hop)
        self.U = float(U if U_int is None else U_int)

        self.mu = float(mu)
        self.h_mag = float(h_mag)

        self.n_qubits = 2 * self.n_sites
        self.dim = 2 ** self.n_qubits

        self._ed = None
        self.H = self.hamiltonian_matrix(include_identity=True)

    # --------------------------------------------------------
    # Índices y Jordan-Wigner
    # --------------------------------------------------------

    def orbital_index(self, site, spin):
        site = int(site)
        if not (0 <= site < self.n_sites):
            raise ValueError(f"site={site} fuera de rango")

        if spin in ("up", "u", 0):
            return 2 * site
        if spin in ("down", "d", 1):
            return 2 * site + 1

        raise ValueError("spin debe ser 'up', 'down', 0 o 1")

    def orb_index(self, site, spin):
        """
        Alias para compatibilidad con celdas antiguas.
        """
        return self.orbital_index(site, spin)

    def jw_annihilation(self, q):
        """
        Operador c_q con Jordan-Wigner:
            c_q = Z_0 ... Z_{q-1} sigma^-_q
        """
        q = int(q)
        if not (0 <= q < self.n_qubits):
            raise ValueError(f"q={q} fuera de rango")

        ops_z = {m: "Z" for m in range(q)}
        Z_string = pauli_string(ops_z, self.n_qubits)

        sigma_minus = np.zeros((self.dim, self.dim), dtype=complex)
        sigma_minus += 0.5 * pauli_string({q: "X"}, self.n_qubits)
        sigma_minus += 0.5j * pauli_string({q: "Y"}, self.n_qubits)

        return Z_string @ sigma_minus

    def jw_creation(self, q):
        return self.jw_annihilation(q).conj().T

    def check_jw_anticommutation(self, atol=1e-10):
        """
        Comprueba {c_p, c_q†}=delta_pq y {c_p,c_q}=0.
        """
        d = self.dim
        I = np.eye(d, dtype=complex)

        max_err = 0.0
        cs = [self.jw_annihilation(q) for q in range(self.n_qubits)]

        for p in range(self.n_qubits):
            for q in range(self.n_qubits):
                anti = cs[p] @ cs[q].conj().T + cs[q].conj().T @ cs[p]
                target = I if p == q else np.zeros((d, d), dtype=complex)
                max_err = max(max_err, np.max(np.abs(anti - target)))

                anti_cc = cs[p] @ cs[q] + cs[q] @ cs[p]
                max_err = max(max_err, np.max(np.abs(anti_cc)))

        return {
            "ok": bool(max_err < atol),
            "max_error": float(max_err),
        }

    # --------------------------------------------------------
    # Términos Pauli del Hamiltoniano
    # --------------------------------------------------------

    def _jw_string_ops(self, p, q, pauli_p, pauli_q):
        """
        String Pauli entre orbitales p<q para hopping JW.
        """
        p, q = int(p), int(q)
        if p > q:
            p, q = q, p

        ops = {p: pauli_p, q: pauli_q}
        for m in range(p + 1, q):
            ops[m] = "Z"
        return ops
    
    def jw_hopping_operator(self, p, q):
        """
        Operador hermítico de hopping JW:
            c_p† c_q + c_q† c_p

        Para p < q:
            1/2 * (X_p Z_{p+1}...Z_{q-1} X_q
                 + Y_p Z_{p+1}...Z_{q-1} Y_q)
        """
        p, q = int(p), int(q)

        if p == q:
            raise ValueError("p y q deben ser distintos")

        if p > q:
            p, q = q, p

        ops_xx = {p: "X", q: "X"}
        ops_yy = {p: "Y", q: "Y"}

        for m in range(p + 1, q):
            ops_xx[m] = "Z"
            ops_yy[m] = "Z"

        A = 0.5 * (
            pauli_string(ops_xx, self.n_qubits)
            + pauli_string(ops_yy, self.n_qubits)
        )

        return 0.5 * (A + A.conj().T)


    def _jw_hopping(self, p, q):
        """
        Alias antiguo para compatibilidad con notebooks viejos.
        """
        return self.jw_hopping_operator(p, q)


    def jw_pairing_operator(self, p, q):
        """
        Operador hermítico tipo pairing:
            c_p† c_q† + c_q c_p

        No forma parte del Hubbard estándar, pero puede ser útil como
        operador de acoplo auxiliar para probar ergodicidad.
        """
        p, q = int(p), int(q)

        if p == q:
            raise ValueError("p y q deben ser distintos")

        if p > q:
            p, q = q, p

        ops_xx = {p: "X", q: "X"}
        ops_yy = {p: "Y", q: "Y"}

        for m in range(p + 1, q):
            ops_xx[m] = "Z"
            ops_yy[m] = "Z"

        A = 0.5 * (
            pauli_string(ops_xx, self.n_qubits)
            - pauli_string(ops_yy, self.n_qubits)
        )

        return 0.5 * (A + A.conj().T)
    

    def hopping_pauli_terms(self):
        """
        Devuelve términos Pauli de la parte cinética:
            -t/2 * (X Z...Z X + Y Z...Z Y)
        """
        terms = []

        for i, j in self.bonds:
            for spin in ("up", "down"):
                p = self.orbital_index(i, spin)
                q = self.orbital_index(j, spin)

                if p > q:
                    p, q = q, p

                terms.append((-0.5 * self.t, self._jw_string_ops(p, q, "X", "X")))
                terms.append((-0.5 * self.t, self._jw_string_ops(p, q, "Y", "Y")))

        return terms

    def interaction_pauli_terms(self, include_identity=True):
        """
        U n_up n_down = U/4 * (I - Z_up - Z_down + Z_up Z_down)
        """
        terms = []

        for i in range(self.n_sites):
            pu = self.orbital_index(i, "up")
            pd = self.orbital_index(i, "down")

            if include_identity:
                terms.append((+0.25 * self.U, {}))

            terms.append((-0.25 * self.U, {pu: "Z"}))
            terms.append((-0.25 * self.U, {pd: "Z"}))
            terms.append((+0.25 * self.U, {pu: "Z", pd: "Z"}))

        return terms

    def chemical_potential_pauli_terms(self, include_identity=True):
        """
        -mu n = -mu/2 * (I - Z)
        """
        terms = []

        if abs(self.mu) < 1e-15:
            return terms

        for q in range(self.n_qubits):
            if include_identity:
                terms.append((-0.5 * self.mu, {}))
            terms.append((+0.5 * self.mu, {q: "Z"}))

        return terms

    def magnetic_field_pauli_terms(self):
        """
        -h_mag S^z, con S^z_i = (n_up - n_down)/2
                 = (Z_down - Z_up)/4
        """
        terms = []

        if abs(self.h_mag) < 1e-15:
            return terms

        for i in range(self.n_sites):
            pu = self.orbital_index(i, "up")
            pd = self.orbital_index(i, "down")

            terms.append((+0.25 * self.h_mag, {pu: "Z"}))
            terms.append((-0.25 * self.h_mag, {pd: "Z"}))

        return terms

    def hamiltonian_pauli_terms(self, include_identity=True):
        return (
            self.hopping_pauli_terms()
            + self.interaction_pauli_terms(include_identity=include_identity)
            + self.chemical_potential_pauli_terms(include_identity=include_identity)
            + self.magnetic_field_pauli_terms()
        )

    def hamiltonian_matrix(self, include_identity=True):
        H = np.zeros((self.dim, self.dim), dtype=complex)

        for coeff, ops in self.hamiltonian_pauli_terms(include_identity=include_identity):
            H += coeff * pauli_string(ops, self.n_qubits)

        return 0.5 * (H + H.conj().T)

    def as_sparse_pauli_op(self, include_identity=False):
        """
        SparsePauliOp para Qiskit.

        Para dinámica Qiskit se suele usar include_identity=False porque la
        identidad solo añade fase global. Para medir energía total, usa True.
        """
        from qiskit.quantum_info import SparsePauliOp

        pairs = [
            (qiskit_label(ops, self.n_qubits), complex(coeff))
            for coeff, ops in self.hamiltonian_pauli_terms(
                include_identity=include_identity
            )
        ]

        return SparsePauliOp.from_list(pairs).simplify()

    # --------------------------------------------------------
    # Diagonalización y estados térmicos
    # --------------------------------------------------------

    def eigensystem(self):
        if self._ed is None:
            self._ed = eigh(self.H)
        return self._ed

    def eig(self):
        """
        Alias para compatibilidad con código antiguo.
        """
        return self.eigensystem()

    def gibbs_state(self, beta, N_sector=None):
        """
        Estado de Gibbs.

        Si N_sector=None:
            Gibbs grand-canonical de todo el Hilbert.

        Si N_sector=int:
            Gibbs canónico restringido al sector con N partículas.
            Esto es útil para half-filling.
        """
        beta = float(beta)

        if N_sector is None:
            evals, evecs = self.eigensystem()
            w = np.exp(-beta * (evals - np.min(evals)))
            w /= np.sum(w)
            return (evecs * w) @ evecs.conj().T

        idx = self.basis_indices_number_sector(N_sector)
        Hs = self.H[np.ix_(idx, idx)]

        evals, evecs = eigh(Hs)
        w = np.exp(-beta * (evals - np.min(evals)))
        w /= np.sum(w)

        rho_s = (evecs * w) @ evecs.conj().T

        rho = np.zeros((self.dim, self.dim), dtype=complex)
        rho[np.ix_(idx, idx)] = rho_s
        return rho

    def basis_indices_number_sector(self, N):
        """
        Índices de la base computacional con N orbitales ocupados.
        """
        N = int(N)
        return [
            state
            for state in range(self.dim)
            if int(state).bit_count() == N
        ]

    # --------------------------------------------------------
    # Operadores físicos
    # --------------------------------------------------------

    def number_op_qubit(self, q):
        q = int(q)
        return 0.5 * (
            np.eye(self.dim, dtype=complex)
            - pauli_string({q: "Z"}, self.n_qubits)
        )

    def orbital_number_operator(self, site, spin):
        return self.number_op_qubit(self.orbital_index(site, spin))

    def number_operator(self):
        return sum(
            self.number_op_qubit(q)
            for q in range(self.n_qubits)
        )

    def site_density_op(self, site):
        return (
            self.orbital_number_operator(site, "up")
            + self.orbital_number_operator(site, "down")
        )

    def site_spin_z_op(self, site):
        return 0.5 * (
            self.orbital_number_operator(site, "up")
            - self.orbital_number_operator(site, "down")
        )

    def sz_total_operator(self):
        return sum(
            self.site_spin_z_op(i)
            for i in range(self.n_sites)
        )

    def double_occupancy_op(self, site):
        pu = self.orbital_index(site, "up")
        pd = self.orbital_index(site, "down")

        return 0.25 * (
            np.eye(self.dim, dtype=complex)
            - pauli_string({pu: "Z"}, self.n_qubits)
            - pauli_string({pd: "Z"}, self.n_qubits)
            + pauli_string({pu: "Z", pd: "Z"}, self.n_qubits)
        )

    def kinetic_operator(self):
        K = np.zeros((self.dim, self.dim), dtype=complex)

        for coeff, ops in self.hopping_pauli_terms():
            K += coeff * pauli_string(ops, self.n_qubits)

        return 0.5 * (K + K.conj().T)

    def interaction_operator(self):
        V = np.zeros((self.dim, self.dim), dtype=complex)

        for coeff, ops in self.interaction_pauli_terms(include_identity=True):
            V += coeff * pauli_string(ops, self.n_qubits)

        return 0.5 * (V + V.conj().T)

    # --------------------------------------------------------
    # Valores esperados
    # --------------------------------------------------------

    def energy(self, rho):
        return float(np.real(np.trace(self.H @ rho)))

    def kinetic_energy(self, rho):
        return float(np.real(np.trace(self.kinetic_operator() @ rho)))

    def interaction_energy(self, rho):
        return float(np.real(np.trace(self.interaction_operator() @ rho)))

    def total_number(self, rho):
        return float(np.real(np.trace(self.number_operator() @ rho)))

    def total_sz(self, rho):
        return float(np.real(np.trace(self.sz_total_operator() @ rho)))

    def occupations(self, rho):
        return np.array([
            np.real(np.trace(self.number_op_qubit(q) @ rho))
            for q in range(self.n_qubits)
        ], dtype=float)

    def site_densities(self, rho):
        return np.array([
            np.real(np.trace(self.site_density_op(i) @ rho))
            for i in range(self.n_sites)
        ], dtype=float)

    def spin_z_profile(self, rho):
        return np.array([
            np.real(np.trace(self.site_spin_z_op(i) @ rho))
            for i in range(self.n_sites)
        ], dtype=float)

    def double_occupancies(self, rho):
        return np.array([
            np.real(np.trace(self.double_occupancy_op(i) @ rho))
            for i in range(self.n_sites)
        ], dtype=float)

    def avg_double_occupancy(self, rho):
        return float(np.mean(self.double_occupancies(rho)))

    def spin_correlation(self, rho, i, j, connected=False):
        Si = self.site_spin_z_op(i)
        Sj = self.site_spin_z_op(j)

        val = np.real(np.trace(Si @ Sj @ rho))

        if connected:
            val -= (
                np.real(np.trace(Si @ rho))
                * np.real(np.trace(Sj @ rho))
            )

        return float(val)

    def density_correlation(self, rho, i, j, connected=True):
        ni = self.site_density_op(i)
        nj = self.site_density_op(j)

        val = np.real(np.trace(ni @ nj @ rho))

        if connected:
            val -= (
                np.real(np.trace(ni @ rho))
                * np.real(np.trace(nj @ rho))
            )

        return float(val)

    def site_mutual_information(self, rho, i, j):
        """
        Información mutua entre dos sitios Hubbard.
        Cada sitio contiene dos qubits: up/down.
        """
        qi = [
            self.orbital_index(i, "up"),
            self.orbital_index(i, "down"),
        ]
        qj = [
            self.orbital_index(j, "up"),
            self.orbital_index(j, "down"),
        ]

        rho_i = reduced_dm_qubits(rho, qi, self.n_qubits)
        rho_j = reduced_dm_qubits(rho, qj, self.n_qubits)
        rho_ij = reduced_dm_qubits(rho, qi + qj, self.n_qubits)

        return (
            von_neumann_entropy(rho_i)
            + von_neumann_entropy(rho_j)
            - von_neumann_entropy(rho_ij)
        )

    # --------------------------------------------------------
    # Diagnósticos compactos
    # --------------------------------------------------------

    def validate_symmetries(self):
        N_op = self.number_operator()
        Sz_op = self.sz_total_operator()

        return {
            "H_hermitian": float(np.max(np.abs(self.H - self.H.conj().T))),
            "comm_H_N": float(np.max(np.abs(self.H @ N_op - N_op @ self.H))),
            "comm_H_Sz": float(np.max(np.abs(self.H @ Sz_op - Sz_op @ self.H))),
        }

    def summary_row(self, rho, label="state"):
        row = {
            "label": label,
            "E": self.energy(rho),
            "K": self.kinetic_energy(rho),
            "U_D": self.interaction_energy(rho),
            "N": self.total_number(rho),
            "Sz": self.total_sz(rho),
            "D_avg": self.avg_double_occupancy(rho),
            "S": von_neumann_entropy(rho),
        }

        if self.n_sites >= 2:
            row["Czz_01"] = self.spin_correlation(rho, 0, 1)
            row["Czz_01_conn"] = self.spin_correlation(rho, 0, 1, connected=True)
            row["Cnn_01_conn"] = self.density_correlation(rho, 0, 1, connected=True)
            row["MI_01"] = self.site_mutual_information(rho, 0, 1)

        return row
    
# ============================================================
# Protocolos térmicos exactos
# ============================================================

def _coerce_coupling_ops(A_ops, labels=None):
    """
    Acepta:
      - A_ops como lista de matrices
      - (A_ops, labels) como devuelven los builders
    """
    if isinstance(A_ops, tuple) and len(A_ops) == 2:
        A_ops, labels_tuple = A_ops
        if labels is None:
            labels = labels_tuple

    A_ops = list(A_ops)

    if labels is None:
        labels = [f"A_{i}" for i in range(len(A_ops))]
    else:
        labels = list(labels)

    if len(labels) != len(A_ops):
        raise ValueError("labels y A_ops deben tener la misma longitud")

    return A_ops, labels


def _project_density_matrix(rho):
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.real(np.trace(rho))
    if abs(tr) < 1e-15:
        raise ValueError("La matriz de densidad tiene traza casi cero")
    return rho / tr


class ThermalProtocolFullBath:
    """
    Protocolo tipo paper con baño simultáneo.

    Para el dímero con operadores Paper:
        n_sys = 4
        n_bath = n_ops = 4
        dim_S = 16
        dim_B = 16
        dim_tot = 256

    Canal de un ciclo:

        rho -> Tr_B[ Q (rho ⊗ |0...0><0...0|) Q† ]

    con

        Q = producto_tau exp(-i delta theta f_tau V)
                         exp(-i delta H_B)
                         exp(-i delta H_S)

    y

        V = sum_mu A_mu ⊗ Y_mu.
    """

    def __init__(
        self,
        model,
        A_ops=None,
        labels=None,
        h_bath=1.0,
        beta=1.0,
        theta=0.35,
        delta=0.30,
        M_T=12,
        random_evolution=True,
        use_rand=None,
    ):
        if A_ops is None:
            A_ops, labels = build_coupling_ops_paper(model)

        A_ops, labels = _coerce_coupling_ops(A_ops, labels)

        if use_rand is not None:
            random_evolution = bool(use_rand)

        self.model = model
        self.A_ops = A_ops
        self.labels = labels

        self.h = float(h_bath)
        self.beta = float(beta)
        self.theta = float(theta)
        self.delta = float(delta)
        self.M_T = int(M_T)
        self.random_evolution = bool(random_evolution)

        self.dS = int(model.dim)
        self.n_bath = len(A_ops)
        self.dB = 2 ** self.n_bath
        self.dT = self.dS * self.dB

        self.metadata = {
            "protocol": "paper_full_bath",
            "n_system_qubits": model.n_qubits,
            "n_bath_qubits": self.n_bath,
            "n_ops": len(A_ops),
            "dim_system": self.dS,
            "dim_bath": self.dB,
            "dim_total": self.dT,
        }

        self._build()

    def _build_bath_operator(self, single_op, mu):
        ops = [I2] * self.n_bath
        ops[mu] = single_op
        return reduce(kron, ops)

    def _build(self):
        t0 = time.time()

        IS = np.eye(self.dS, dtype=complex)
        IB = np.eye(self.dB, dtype=complex)

        # Hamiltoniano del sistema y del baño en el espacio total
        self.H_S_tot = kron(self.model.H, IB)

        H_B = np.zeros((self.dB, self.dB), dtype=complex)
        for mu in range(self.n_bath):
            H_B += -0.5 * self.h * self._build_bath_operator(Z, mu)

        self.H_B_tot = kron(IS, H_B)

        # Interacción simultánea V = sum_mu A_mu ⊗ Y_mu
        self.V_tot = np.zeros((self.dT, self.dT), dtype=complex)

        for mu, A in enumerate(self.A_ops):
            if A.shape != (self.dS, self.dS):
                raise ValueError(
                    f"A_ops[{mu}] tiene shape {A.shape}, esperado {(self.dS, self.dS)}"
                )

            Y_mu = self._build_bath_operator(Y, mu)
            self.V_tot += kron(A, Y_mu)

        # Estado inicial del baño |0...0><0...0|
        psi0 = np.zeros(self.dB, dtype=complex)
        psi0[0] = 1.0
        self.phi = np.outer(psi0, psi0.conj())

        # Filtro temporal
        a = compute_a(self.h, self.beta)
        _, self.fvals = gaussian_filter(self.M_T, self.delta, a)

        # Unitaries elementales
        eHS = expm(-1j * self.delta * self.H_S_tot)
        eHB = expm(-1j * self.delta * self.H_B_tot)
        eF = eHB @ eHS

        Q = np.eye(self.dT, dtype=complex)

        for f in self.fvals:
            if abs(f) > 1e-15:
                eV = expm(-1j * self.delta * self.theta * f * self.V_tot)
                Q = (eV @ eF) @ Q
            else:
                Q = eF @ Q

        self.Q_base = Q

        # Para randomización por evolución libre del sistema
        self._evals_S, self._evecs_S = self.model.eigensystem()

        self.build_time = time.time() - t0
        self.metadata["build_time_sec"] = self.build_time

    def _system_free_evolution(self, m):
        U = (
            self._evecs_S
            @ np.diag(np.exp(-1j * self.delta * int(m) * self._evals_S))
            @ self._evecs_S.conj().T
        )
        return U

    def cycle_unitary(self, random_depth=0):
        Q = self.Q_base

        if random_depth and random_depth > 0:
            R = kron(self._system_free_evolution(random_depth), np.eye(self.dB))
            Q = R @ Q

        return Q

    def apply_channel(self, rho):
        if self.random_evolution:
            MR = np.random.randint(1, 2 * self.M_T + 1)
        else:
            MR = 0

        Q = self.cycle_unitary(random_depth=MR)

        rho_tot = Q @ kron(rho, self.phi) @ Q.conj().T
        rho_out = partial_trace_B(rho_tot, self.dS, self.dB)

        return _project_density_matrix(rho_out)

    def run(self, n_cycles, rho_init=None, verbose=True, record_every=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        rho_g = self.model.gibbs_state(self.beta)

        if rho_init is None:
            rho = np.eye(self.dS, dtype=complex) / self.dS
        else:
            rho = rho_init.copy()

        if record_every is None:
            record_every = max(1, int(n_cycles) // 20)

        res = {
            "states": [],
            "cycles": [],
            "trace_dist": [],
            "energies": [],
            "double_occ": [],
        }

        def record(n, r):
            res["states"].append(r.copy())
            res["cycles"].append(n)
            res["trace_dist"].append(trace_distance(r, rho_g))
            res["energies"].append(self.model.energy(r))
            res["double_occ"].append(self.model.avg_double_occupancy(r))

        record(0, rho)

        for n in range(1, int(n_cycles) + 1):
            rho = self.apply_channel(rho)

            if n % record_every == 0 or n == n_cycles:
                record(n, rho)

            if verbose and n % max(1, int(n_cycles) // 5) == 0:
                print(
                    f"  Ciclo {n:4d}/{n_cycles}: "
                    f"D={res['trace_dist'][-1]:.6f}, "
                    f"E={res['energies'][-1]:.6f}"
                )

        res["rho_final"] = rho.copy()
        res["metadata"] = self.metadata.copy()

        for k in ["cycles", "trace_dist", "energies", "double_occ"]:
            res[k] = np.array(res[k])

        return res

    def fixed_point(self):
        """
        Punto fijo exacto del canal determinista sin random_evolution.
        """
        Q = self.cycle_unitary(random_depth=0)

        d2 = self.dS ** 2
        S = np.zeros((d2, d2), dtype=complex)

        for c in range(self.dS):
            for d in range(self.dS):
                rho_cd = np.zeros((self.dS, self.dS), dtype=complex)
                rho_cd[c, d] = 1.0

                rho_tot = Q @ kron(rho_cd, self.phi) @ Q.conj().T
                rho_out = partial_trace_B(rho_tot, self.dS, self.dB)

                S[:, c * self.dS + d] = rho_out.reshape(-1)

        eigvals, eigvecs = np.linalg.eig(S)
        idx = np.argmin(np.abs(eigvals - 1.0))

        rho_fp = eigvecs[:, idx].reshape(self.dS, self.dS)
        return _project_density_matrix(rho_fp)


class ThermalProtocolSequential:
    """
    Protocolo secuencial con un único qubit de baño reutilizado.

    Si A_ops tiene 4 operadores, el ciclo hace:

        A_0 con 1 baño -> trace/reset
        A_1 con 1 baño -> trace/reset
        A_2 con 1 baño -> trace/reset
        A_3 con 1 baño -> trace/reset

    Es más barato que ThermalProtocolPaper, pero no es el baño simultáneo
    N_B = N_ops.
    """

    def __init__(
        self,
        model,
        A_ops=None,
        labels=None,
        h_bath=1.0,
        beta=1.0,
        theta=0.35,
        delta=0.30,
        M_T=12,
        random_order=True,
        random_evolution=True,
        use_rand=None,
    ):
        if A_ops is None:
            A_ops, labels = build_coupling_ops_paper(model)

        A_ops, labels = _coerce_coupling_ops(A_ops, labels)

        if use_rand is not None:
            random_order = bool(use_rand)
            random_evolution = bool(use_rand)

        self.model = model
        self.A_ops = A_ops
        self.labels = labels

        self.h = float(h_bath)
        self.beta = float(beta)
        self.theta = float(theta)
        self.delta = float(delta)
        self.M_T = int(M_T)

        self.random_order = bool(random_order)
        self.random_evolution = bool(random_evolution)

        self.dS = int(model.dim)
        self.dB = 2
        self.dT = self.dS * self.dB

        self.metadata = {
            "protocol": "sequential_reused_bath",
            "n_system_qubits": model.n_qubits,
            "n_bath_qubits": 1,
            "n_ops": len(A_ops),
            "dim_system": self.dS,
            "dim_bath": self.dB,
            "dim_total": self.dT,
        }

        self._build()

    def _build(self):
        t0 = time.time()

        IS = np.eye(self.dS, dtype=complex)

        a = compute_a(self.h, self.beta)
        _, self.fvals = gaussian_filter(self.M_T, self.delta, a)

        self.phi = np.array([[1, 0], [0, 0]], dtype=complex)

        eHS = expm(-1j * self.delta * kron(self.model.H, I2))
        eHB = expm(-1j * self.delta * kron(IS, -0.5 * self.h * Z))
        eF = eHB @ eHS

        self.Qs = []

        for mu, A in enumerate(self.A_ops):
            if A.shape != (self.dS, self.dS):
                raise ValueError(
                    f"A_ops[{mu}] tiene shape {A.shape}, esperado {(self.dS, self.dS)}"
                )

            V = kron(A, Y)
            Q = np.eye(self.dT, dtype=complex)

            for f in self.fvals:
                if abs(f) > 1e-15:
                    eV = expm(-1j * self.delta * self.theta * f * V)
                    Q = (eV @ eF) @ Q
                else:
                    Q = eF @ Q

            self.Qs.append(Q)

        self._evals_S, self._evecs_S = self.model.eigensystem()

        self.build_time = time.time() - t0
        self.metadata["build_time_sec"] = self.build_time

    def _system_free_evolution(self, m):
        U = (
            self._evecs_S
            @ np.diag(np.exp(-1j * self.delta * int(m) * self._evals_S))
            @ self._evecs_S.conj().T
        )
        return U

    def apply_channel(self, rho):
        if self.random_order:
            order = np.random.permutation(len(self.A_ops))
        else:
            order = range(len(self.A_ops))

        for mu in order:
            Q = self.Qs[mu]

            if self.random_evolution:
                MR = np.random.randint(1, 2 * self.M_T + 1)
                R = kron(self._system_free_evolution(MR), I2)
                Q_eff = R @ Q
            else:
                Q_eff = Q

            rho_tot = Q_eff @ kron(rho, self.phi) @ Q_eff.conj().T
            rho = partial_trace_B(rho_tot, self.dS, 2)
            rho = _project_density_matrix(rho)

        return rho

    def run(self, n_cycles, rho_init=None, verbose=True, record_every=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        rho_g = self.model.gibbs_state(self.beta)

        if rho_init is None:
            rho = np.eye(self.dS, dtype=complex) / self.dS
        else:
            rho = rho_init.copy()

        if record_every is None:
            record_every = max(1, int(n_cycles) // 20)

        res = {
            "states": [],
            "cycles": [],
            "trace_dist": [],
            "energies": [],
            "double_occ": [],
        }

        def record(n, r):
            res["states"].append(r.copy())
            res["cycles"].append(n)
            res["trace_dist"].append(trace_distance(r, rho_g))
            res["energies"].append(self.model.energy(r))
            res["double_occ"].append(self.model.avg_double_occupancy(r))

        record(0, rho)

        for n in range(1, int(n_cycles) + 1):
            rho = self.apply_channel(rho)

            if n % record_every == 0 or n == n_cycles:
                record(n, rho)

            if verbose and n % max(1, int(n_cycles) // 5) == 0:
                print(
                    f"  Ciclo {n:4d}/{n_cycles}: "
                    f"D={res['trace_dist'][-1]:.6f}, "
                    f"E={res['energies'][-1]:.6f}"
                )

        res["rho_final"] = rho.copy()
        res["metadata"] = self.metadata.copy()

        for k in ["cycles", "trace_dist", "energies", "double_occ"]:
            res[k] = np.array(res[k])

        return res

    def fixed_point(self):
        """
        Punto fijo exacto del canal secuencial determinista.

        Importante:
        al construir el superoperador no se puede normalizar la salida de
        cada operador base |c><d|, porque esos operadores no son matrices
        de densidad y pueden tener traza cero.

        Primero se construye el canal lineal completo. Solo al final se
        normaliza el autovector estacionario.
        """
        d2 = self.dS ** 2
        S = np.zeros((d2, d2), dtype=complex)

        for c in range(self.dS):
            for d in range(self.dS):
                X = np.zeros((self.dS, self.dS), dtype=complex)
                X[c, d] = 1.0

                for Q in self.Qs:
                    X_tot = Q @ kron(X, self.phi) @ Q.conj().T
                    X = partial_trace_B(X_tot, self.dS, 2)

                S[:, c * self.dS + d] = X.reshape(-1)

        eigvals, eigvecs = np.linalg.eig(S)

        # Elegimos un autovector con autovalor cercano a 1 y traza no nula.
        order = np.argsort(np.abs(eigvals - 1.0))

        for idx in order:
            rho_fp = eigvecs[:, idx].reshape(self.dS, self.dS)
            tr = np.trace(rho_fp)

            if abs(tr) > 1e-12:
                rho_fp = rho_fp / tr
                rho_fp = 0.5 * (rho_fp + rho_fp.conj().T)
                rho_fp = rho_fp / np.real(np.trace(rho_fp))
                return rho_fp

        raise RuntimeError(
            "No se encontró un punto fijo con traza no nula "
            "para el canal secuencial."
        )
    
    def fixed_point_sequential_linear(self):
        dS = self.dS
        d2 = dS**2
        S = np.zeros((d2, d2), dtype=complex)

        for c in range(dS):
            for d in range(dS):
                X_cd = np.zeros((dS, dS), dtype=complex)
                X_cd[c, d] = 1.0

                X = X_cd

                for Q in self.Qs:
                    X_tot = Q @ kron(X, self.phi) @ Q.conj().T
                    X = partial_trace_B(X_tot, dS, 2)

                S[:, c * dS + d] = X.reshape(-1)

        eigvals, eigvecs = np.linalg.eig(S)
        order = np.argsort(np.abs(eigvals - 1.0))

        for idx in order:
            rho_fp = eigvecs[:, idx].reshape(dS, dS)
            tr = np.trace(rho_fp)

            if abs(tr) > 1e-12:
                rho_fp = rho_fp / tr
                rho_fp = 0.5 * (rho_fp + rho_fp.conj().T)
                rho_fp = rho_fp / np.real(np.trace(rho_fp))
                return rho_fp
        raise RuntimeError("No se encontró punto fijo con traza no nula.")


   
    
# ═══════════════════════════════════════════════════════
# FAMILIA 1: Paper — (Z+Y)/√2 por qubit
# ═══════════════════════════════════════════════════════
def build_coupling_ops_paper(model):
    """A_q = (Z_q + Y_q)/√2  por qubit del sistema.
    Z modula el acoplo según el estado, Y induce transiciones."""
    n = model.n_qubits
    ops = [(pauli_string({q:'Z'},n) + pauli_string({q:'Y'},n))/np.sqrt(2) for q in range(n)]
    labels = [f'(Z+Y)_{q}/√2' for q in range(n)]
    return ops, labels

# ═══════════════════════════════════════════════════════
# FAMILIA 2: Hubbard físicos (carga, espín, salto)
# ═══════════════════════════════════════════════════════
def build_coupling_operators(model):
    """Operadores de acoplo físicos de Fermi-Hubbard."""
    n, dim = model.n_qubits, model.dim
    A_ops, labels = [], []
    for i in range(model.n_sites):
        pu,pd = model.orbital_index(i,'up'), model.orbital_index(i,'down')
        A_ops.append(-0.5*(pauli_string({pu:'Z'},n)+pauli_string({pd:'Z'},n)))
        labels.append(f"charge_{i}")
    for i in range(model.n_sites):
        pu,pd = model.orbital_index(i,'up'), model.orbital_index(i,'down')
        A_ops.append(0.5*(pauli_string({pd:'Z'},n)-pauli_string({pu:'Z'},n)))
        labels.append(f"spin_{i}")
    for i,j in model.bonds:
        for spin in ('up','down'):
            p,q = model.orbital_index(i,spin), model.orbital_index(j,spin)
            if p>q: p,q=q,p
            A_ops.append(model._jw_hopping(p,q))
            labels.append(f"hop_{i}{j}_{'↑' if spin=='up' else '↓'}")
    return A_ops, labels

# ═══════════════════════════════════════════════════════
# FAMILIA 3: Hubbard "vestido" — operadores físicos + Y
# ═══════════════════════════════════════════════════════
def build_coupling_ops_dressed(model):
    """Hubbard con componente Y añadida para ergodicidad.
    charge_dressed = (1/√2)[-½(Z_up+Z_dn) + ½(Y_up+Y_dn)]
    spin_dressed   = (1/√2)[½(Z_dn-Z_up) + ½(Y_up-Y_dn)]
    hop            = sin cambio (ya es off-diagonal)"""
    n = model.n_qubits
    ops, labels = [], []
    for i in range(model.n_sites):
        pu,pd = model.orbital_index(i,'up'), model.orbital_index(i,'down')
        diag = -0.5*(pauli_string({pu:'Z'},n)+pauli_string({pd:'Z'},n))
        offdiag = 0.5*(pauli_string({pu:'Y'},n)+pauli_string({pd:'Y'},n))
        ops.append((diag+offdiag)/np.sqrt(2)); labels.append(f"charge+Y_{i}")
    for i in range(model.n_sites):
        pu,pd = model.orbital_index(i,'up'), model.orbital_index(i,'down')
        diag = 0.5*(pauli_string({pd:'Z'},n)-pauli_string({pu:'Z'},n))
        offdiag = 0.5*(pauli_string({pu:'Y'},n)-pauli_string({pd:'Y'},n))
        ops.append((diag+offdiag)/np.sqrt(2)); labels.append(f"spin+Y_{i}")
    for i,j in model.bonds:
        for spin in ('up','down'):
            p,q = model.orbital_index(i,spin), model.orbital_index(j,spin)
            if p>q: p,q=q,p
            ops.append(model._jw_hopping(p,q))
            labels.append(f"hop_{i}{j}_{'↑' if spin=='up' else '↓'}")
    return ops, labels

# ═══════════════════════════════════════════════════════
# FAMILIA 4: Pauli individual por qubit
# ═══════════════════════════════════════════════════════
def build_coupling_ops_pauli(model, pauli_type='Y'):
    """Un Pauli por qubit: 'X', 'Y', 'Z', o combinaciones 'ZY','ZX','XY'."""
    n = model.n_qubits; ops, labels = [], []
    if len(pauli_type)==1:
        for q in range(n):
            ops.append(pauli_string({q:pauli_type},n)); labels.append(f'{pauli_type}_{q}')
    else:
        P1,P2 = pauli_type[0],pauli_type[1]
        for q in range(n):
            ops.append((pauli_string({q:P1},n)+pauli_string({q:P2},n))/np.sqrt(2))
            labels.append(f'({P1}+{P2})_{q}/√2')
    return ops, labels


def jw_c(p, nq):
    ops = [I2]*nq
    for m in range(p): ops[m] = Z
    ops[p] = (X + 1j*Y)/2
    return reduce(kron, ops)








# ============================================================
# Qiskit equivalents of ThermalProtocolFullBath / Sequential
# Paste this block in tfm_fh/model.py after the NumPy protocols.
#
# Requires existing objects in model.py:
#   np, pd, time
#   pauli_string, qiskit_label, gaussian_filter_values, suggest_MT
#   mixture_plan_fh, get_fh_backend_bundle
#   FermiHubbardJW
# ============================================================

def fh_terms_to_sparse_pauli_op(terms, n_qubits, drop_identity=False):
    """Convert list[(coeff, {qubit:'X/Y/Z'})] to Qiskit's SparsePauliOp."""
    from qiskit.quantum_info import SparsePauliOp

    pairs = []
    for coeff, ops in terms:
        ops = dict(ops)
        if drop_identity and len(ops) == 0:
            continue
        pairs.append((qiskit_label(ops, n_qubits), complex(coeff)))

    if not pairs:
        pairs = [("I" * int(n_qubits), 0.0)]

    return SparsePauliOp.from_list(pairs).simplify()


def fh_coupling_term_sets(model, family="paper", pauli_type="ZY"):
    """
    Coupling operators A_mu as Pauli-term lists, not dense matrices.

    Returns
    -------
    term_sets : list[list[(coeff, ops_dict)]]
        Each entry represents one A_mu.
    labels : list[str]
    """
    n = model.n_qubits
    term_sets, labels = [], []

    if family == "paper":
        for q in range(n):
            term_sets.append([
                (1.0 / np.sqrt(2.0), {q: "Z"}),
                (1.0 / np.sqrt(2.0), {q: "Y"}),
            ])
            labels.append(f"(Z+Y)_{q}/sqrt2")
        return term_sets, labels

    if family == "pauli":
        if len(pauli_type) == 1:
            for q in range(n):
                term_sets.append([(1.0, {q: pauli_type})])
                labels.append(f"{pauli_type}_{q}")
        else:
            p1, p2 = pauli_type[0], pauli_type[1]
            for q in range(n):
                term_sets.append([
                    (1.0 / np.sqrt(2.0), {q: p1}),
                    (1.0 / np.sqrt(2.0), {q: p2}),
                ])
                labels.append(f"({p1}+{p2})_{q}/sqrt2")
        return term_sets, labels

    if family == "hubbard":
        # charge_i = -1/2 (Z_up + Z_down)
        for i in range(model.n_sites):
            pu = model.orbital_index(i, "up")
            pd = model.orbital_index(i, "down")
            term_sets.append([(-0.5, {pu: "Z"}), (-0.5, {pd: "Z"})])
            labels.append(f"charge_{i}")

        # spin_i = 1/2 (Z_down - Z_up)
        for i in range(model.n_sites):
            pu = model.orbital_index(i, "up")
            pd = model.orbital_index(i, "down")
            term_sets.append([(-0.5, {pu: "Z"}), (+0.5, {pd: "Z"})])
            labels.append(f"spin_{i}")

        # each hopping Pauli string as a separate coupling operator
        for coeff, ops in model.hopping_pauli_terms():
            sign = float(np.sign(np.real(coeff))) if abs(coeff) > 0 else 1.0
            term_sets.append([(sign, dict(ops))])
            labels.append("hop_" + "".join(f"{q}{p}" for q, p in sorted(ops.items())))

        return term_sets, labels

    if family == "dressed":
        # charge/spin + local Y terms for ergodicity
        for i in range(model.n_sites):
            pu = model.orbital_index(i, "up")
            pd = model.orbital_index(i, "down")
            term_sets.append([
                (-0.5 / np.sqrt(2.0), {pu: "Z"}),
                (-0.5 / np.sqrt(2.0), {pd: "Z"}),
                (+0.5 / np.sqrt(2.0), {pu: "Y"}),
                (+0.5 / np.sqrt(2.0), {pd: "Y"}),
            ])
            labels.append(f"charge+Y_{i}")

        for i in range(model.n_sites):
            pu = model.orbital_index(i, "up")
            pd = model.orbital_index(i, "down")
            term_sets.append([
                (-0.5 / np.sqrt(2.0), {pu: "Z"}),
                (+0.5 / np.sqrt(2.0), {pd: "Z"}),
                (+0.5 / np.sqrt(2.0), {pu: "Y"}),
                (-0.5 / np.sqrt(2.0), {pd: "Y"}),
            ])
            labels.append(f"spin+Y_{i}")

        for coeff, ops in model.hopping_pauli_terms():
            sign = float(np.sign(np.real(coeff))) if abs(coeff) > 0 else 1.0
            term_sets.append([(sign, dict(ops))])
            labels.append("hop_" + "".join(f"{q}{p}" for q, p in sorted(ops.items())))

        return term_sets, labels

    raise ValueError(f"Unknown coupling family: {family}")


def fh_term_sets_to_numpy_ops(model, term_sets):
    """Convert fh_coupling_term_sets(...) output to dense NumPy A_mu matrices."""
    A_ops = []
    for terms in term_sets:
        A = np.zeros((model.dim, model.dim), dtype=complex)
        for coeff, ops in terms:
            A += coeff * pauli_string(ops, model.n_qubits)
        A_ops.append(0.5 * (A + A.conj().T))
    return A_ops


def validate_fh_pauli_decomposition(model, atol=1e-10):
    """Check that model.H equals the Pauli-string decomposition."""
    H2 = np.zeros((model.dim, model.dim), dtype=complex)
    for coeff, ops in model.hamiltonian_pauli_terms(include_identity=True):
        H2 += coeff * pauli_string(ops, model.n_qubits)

    err = float(np.max(np.abs(model.H - H2)))
    return {"ok": bool(err < atol), "max_error": err}


def _cbit_from_flat_bitstring(bitstring, n_clbits, cbit_index):
    """
    Return classical bit c[cbit_index] from a Qiskit counts bitstring.
    Assumes a single ClassicalRegister c.
    """
    bits = bitstring.replace(" ", "").zfill(int(n_clbits))
    return bits[int(n_clbits) - 1 - int(cbit_index)]


def _rotate_for_pauli_measurement(qc, qreg, pauli_ops):
    """Rotate so computational-basis measurement estimates the requested Pauli string."""
    for q, p in sorted(pauli_ops.items()):
        if p == "X":
            qc.h(qreg[q])
        elif p == "Y":
            qc.sdg(qreg[q])
            qc.h(qreg[q])
        elif p in ("Z", "I"):
            pass
        else:
            raise ValueError(f"Unsupported Pauli: {p}")


class _ThermalProtocolQiskitBase:
    """Common base class. Do not instantiate directly."""

    protocol_name = "qiskit_base"

    def __init__(
        self,
        model,
        *,
        h_bath=1.0,
        beta=2.0,
        theta=0.35,
        delta=0.30,
        M_T=None,
        coupling_family="paper",
        pauli_type="ZY",
        trotter_order=2,
        trotter_reps=1,
        backend_bundle=None,
        optimization_level=1,
    ):
        self.model = model
        self.h = float(h_bath)
        self.beta = float(beta)
        self.theta = float(theta)
        self.delta = float(delta)
        self.M_T = suggest_MT(self.h, self.beta, self.delta) if M_T is None else int(M_T)

        self.coupling_family = coupling_family
        self.pauli_type = pauli_type
        self.term_sets, self.labels = fh_coupling_term_sets(
            model, family=coupling_family, pauli_type=pauli_type
        )

        self.trotter_order = int(trotter_order)
        self.trotter_reps = int(trotter_reps)
        self.optimization_level = int(optimization_level)

        self.n_sys = int(model.n_qubits)
        self.n_ops = len(self.term_sets)

        _, self.fvals = gaussian_filter_values(self.h, self.beta, self.delta, self.M_T)

        self.backend_bundle = backend_bundle
        self._transpile_cache = {}

    # ---------- Sparse operators ----------

    def _get_backend_bundle(self):
        if self.backend_bundle is None:
            self.backend_bundle = get_fh_backend_bundle(min_total_qubits=self.n_total)
        return self.backend_bundle

    def _system_sparse(self):
        return fh_terms_to_sparse_pauli_op(
            self.model.hamiltonian_pauli_terms(include_identity=False),
            self.n_sys,
            drop_identity=True,
        )

    def _bath_sparse(self, n_bath):
        terms = [(-0.5 * self.h, {mu: "Z"}) for mu in range(int(n_bath))]
        return fh_terms_to_sparse_pauli_op(terms, int(n_bath), drop_identity=True)

    def _append_filter_block(self, qc, sys_qubits, bath_qubits, V_sparse):
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.synthesis import SuzukiTrotter

        synth = SuzukiTrotter(order=self.trotter_order, reps=self.trotter_reps)
        Hs = self._system_sparse()
        Hb = self._bath_sparse(len(bath_qubits))

        all_qubits = list(sys_qubits) + list(bath_qubits)

        for f_tau in self.fvals:
            qc.append(PauliEvolutionGate(Hs, time=self.delta, synthesis=synth), list(sys_qubits))
            qc.append(PauliEvolutionGate(Hb, time=self.delta, synthesis=synth), list(bath_qubits))

            angle = self.delta * self.theta * f_tau
            if abs(angle) > 1e-15:
                qc.append(PauliEvolutionGate(V_sparse, time=angle, synthesis=synth), all_qubits)

    # ---------- Circuit building utilities ----------

    def _prepare_initial_state(self, qc, q_sys, init_label):
        if init_label is None or init_label == "0" * self.n_sys:
            return

        if len(init_label) != self.n_sys or not set(init_label).issubset({"0", "1"}):
            raise ValueError(f"Invalid init_label: {init_label}")

        # Convention: init_label[q] prepares system qubit q.
        for q, bit in enumerate(init_label):
            if bit == "1":
                qc.x(q_sys[q])

    def _append_final_measurement(self, qc, q_sys, c_reg, final_start, final_pauli):
        if final_pauli is None:
            return

        _rotate_for_pauli_measurement(qc, q_sys, final_pauli)

        # Always measure all system qubits. The parser only multiplies the requested support.
        for q in range(self.n_sys):
            qc.measure(q_sys[q], c_reg[final_start + q])

    def transpile(self, qc):
        from qiskit.transpiler import generate_preset_pass_manager

        bundle = self._get_backend_bundle()
        key = (qc.qasm() if hasattr(qc, "qasm") else str(qc), self.optimization_level, bundle["label"])

        if key not in self._transpile_cache:
            pm = generate_preset_pass_manager(
                optimization_level=self.optimization_level,
                backend=bundle["hw_backend"],
            )
            self._transpile_cache[key] = pm.run(qc)

        return self._transpile_cache[key]

    # ---------- Cost and shots ----------

    def resources(self, n_cycles=1, final_pauli=None, init_label=None, transpiled=True):
        qc = self.build_circuit(
            n_cycles=n_cycles,
            init_label=init_label,
            final_pauli=final_pauli,
        )
        qc_eval = self.transpile(qc) if transpiled else qc
        ops = dict(qc_eval.count_ops())

        return {
            "protocol": self.protocol_name,
            "n_cycles": int(n_cycles),
            "n_sys": self.n_sys,
            "n_bath": self.n_bath,
            "n_ops": self.n_ops,
            "n_qubits_total": qc_eval.num_qubits,
            "n_clbits": qc_eval.num_clbits,
            "depth": qc_eval.depth(),
            "size": qc_eval.size(),
            "cx": int(ops.get("cx", 0)),
            "ecr": int(ops.get("ecr", 0)),
            "reset": int(ops.get("reset", 0)),
            "measure": int(ops.get("measure", 0)),
            "ops": ops,
        }

    def _parse_counts(self, counts, n_cycles, n_clbits, n_bath_bits, final_pauli):
        bath = np.zeros(n_bath_bits, dtype=float)
        p1 = np.zeros(self.n_sys, dtype=float)
        zz_pairs = {}
        exp_acc = 0.0
        total = sum(counts.values())

        final_start = n_bath_bits
        support = {} if final_pauli is None else {
            int(q): p for q, p in final_pauli.items() if p != "I"
        }

        for bitstring, count in counts.items():
            # Bath bits
            for k in range(n_bath_bits):
                if _cbit_from_flat_bitstring(bitstring, n_clbits, k) == "1":
                    bath[k] += count

            if final_pauli is None:
                continue

            # Final system bits
            zvals = np.ones(self.n_sys, dtype=float)
            eig = 1.0

            for q in range(self.n_sys):
                b = _cbit_from_flat_bitstring(bitstring, n_clbits, final_start + q)
                if b == "1":
                    p1[q] += count
                    zvals[q] = -1.0

                if q in support and b == "1":
                    eig *= -1.0

            # Exact ZZ correlations in the final measurement basis.
            # Only physically meaningful as Z-basis ZZ if final_pauli is all Z/None.
            for p in range(self.n_sys):
                for q in range(p + 1, self.n_sys):
                    zz_pairs[(p, q)] = zz_pairs.get((p, q), 0.0) + count * zvals[p] * zvals[q]

            exp_acc += count * eig

        out = {
            "bath_p1_flat": bath / total if total else bath,
        }

        if final_pauli is not None:
            out["sys_p1"] = p1 / total if total else p1
            out["zz_pairs"] = {k: v / total for k, v in zz_pairs.items()}
            out["expectation"] = float(exp_acc / total) if total else np.nan

        return out

    def run_counts(
        self,
        *,
        n_cycles,
        shots=1024,
        init_mode="mm",
        init_label=None,
        final_pauli=None,
        sim_mode="ideal",
        seed_base=1234,
    ):
        bundle = self._get_backend_bundle()

        if sim_mode == "ideal":
            sim = bundle["ideal_sim"]
        elif sim_mode == "noisy":
            sim = bundle["noisy_sim"]
            if sim is None:
                raise RuntimeError("No noise model available for sim_mode='noisy'.")
        else:
            raise ValueError("sim_mode must be 'ideal' or 'noisy'.")

        if init_label is not None:
            plan = [(init_label, int(shots))]
        else:
            plan = mixture_plan_fh(
                init_mode,
                shots,
                self.n_sys,
                rng=np.random.default_rng(seed_base),
            )

        n_bath_bits = self.n_bath_bits_per_cycle * int(n_cycles)
        n_final_bits = self.n_sys if final_pauli is not None else 0
        n_clbits = n_bath_bits + n_final_bits

        bath_acc = np.zeros(n_bath_bits, dtype=float)
        p1_acc = np.zeros(self.n_sys, dtype=float)
        zz_acc = {}
        exp_acc = 0.0

        wall0 = time.time()

        for k, (label, n_sh) in enumerate(plan):
            if int(n_sh) <= 0:
                continue

            qc = self.build_circuit(
                n_cycles=n_cycles,
                init_label=label,
                final_pauli=final_pauli,
            )
            tqc = self.transpile(qc)

            result = sim.run(
                tqc,
                shots=int(n_sh),
                seed_simulator=int(seed_base + 97 * k),
            ).result()

            parsed = self._parse_counts(
                result.get_counts(),
                n_cycles=n_cycles,
                n_clbits=n_clbits,
                n_bath_bits=n_bath_bits,
                final_pauli=final_pauli,
            )

            weight = float(n_sh) / float(shots)
            bath_acc += weight * parsed["bath_p1_flat"]

            if final_pauli is not None:
                p1_acc += weight * parsed["sys_p1"]
                exp_acc += weight * parsed["expectation"]
                for pair, val in parsed["zz_pairs"].items():
                    zz_acc[pair] = zz_acc.get(pair, 0.0) + weight * val

        out = {
            "protocol": self.protocol_name,
            "shots": int(shots),
            "n_cycles": int(n_cycles),
            "bath_p1_flat": bath_acc,
            "wall_time_sec": float(time.time() - wall0),
            "backend_label": bundle["label"],
        }

        if final_pauli is not None:
            out["sys_p1"] = p1_acc
            out["zz_pairs"] = zz_acc
            out["expectation"] = float(exp_acc)

        return out

    def estimate_Ez_from_z_counts(self, sys_p1, zz_pairs):
        """
        Exact shot estimator for the diagonal Hubbard interaction energy:
            E_Z = U sum_i <n_up,i n_down,i>
        using <Z_up>, <Z_down>, and <Z_up Z_down>.
        """
        z = 1.0 - 2.0 * np.asarray(sys_p1, dtype=float)

        E = 0.0
        for i in range(self.model.n_sites):
            pu = self.model.orbital_index(i, "up")
            pd = self.model.orbital_index(i, "down")
            pair = (min(pu, pd), max(pu, pd))
            zz = float(zz_pairs.get(pair, 0.0))
            E += (self.model.U / 4.0) * (1.0 - z[pu] - z[pd] + zz)

        return float(E)

    def run_bath_and_z(
        self,
        *,
        n_cycles,
        shots=1024,
        init_mode="mm",
        sim_mode="ideal",
        seed_base=1234,
    ):
        final_pauli = {q: "Z" for q in range(self.n_sys)}

        out = self.run_counts(
            n_cycles=n_cycles,
            shots=shots,
            init_mode=init_mode,
            final_pauli=final_pauli,
            sim_mode=sim_mode,
            seed_base=seed_base,
        )

        out["E_Z"] = self.estimate_Ez_from_z_counts(out["sys_p1"], out["zz_pairs"])
        return out

    def energy_from_pauli_strings(
        self,
        *,
        n_cycles,
        shots=1024,
        init_mode="mm",
        sim_mode="ideal",
        seed_base=1234,
        include_identity=True,
    ):
        """
        Total Hubbard energy from Pauli-string measurements.

        This runs one circuit per Hamiltonian Pauli string. It is intentionally
        not optimized/grouped; use it only for the dimer proof-of-concept.
        """
        rows = []
        E = 0.0

        for idx, (coeff, ops) in enumerate(
            self.model.hamiltonian_pauli_terms(include_identity=include_identity)
        ):
            coeff = float(np.real(coeff))

            if len(ops) == 0:
                expval = 1.0
                wall = 0.0
            else:
                out = self.run_counts(
                    n_cycles=n_cycles,
                    shots=shots,
                    init_mode=init_mode,
                    final_pauli=ops,
                    sim_mode=sim_mode,
                    seed_base=seed_base + 1009 * idx,
                )
                expval = float(out["expectation"])
                wall = float(out["wall_time_sec"])

            contribution = coeff * expval
            E += contribution

            rows.append({
                "coeff": coeff,
                "pauli_ops": dict(ops),
                "expectation": expval,
                "contribution": contribution,
                "wall_time_sec": wall,
            })

        return {
            "E_total": float(E),
            "terms": pd.DataFrame(rows),
        }


class ThermalProtocolFullBathQiskit(_ThermalProtocolQiskitBase):
    """
    Qiskit equivalent of ThermalProtocolFullBath.

    One bath qubit per coupling operator A_mu.
    One cycle:
        product_tau [ U_int(tau) U_B U_S ]
        measure all bath qubits
        reset all bath qubits
    """

    protocol_name = "qiskit_full_bath"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bath = self.n_ops
        self.n_total = self.n_sys + self.n_bath
        self.n_bath_bits_per_cycle = self.n_bath

    def _interaction_sparse_full(self):
        n_tot = self.n_sys + self.n_bath
        terms = []

        for mu, A_terms in enumerate(self.term_sets):
            for coeff, ops in A_terms:
                ops_tot = {int(q): p for q, p in ops.items()}
                ops_tot[self.n_sys + mu] = "Y"
                terms.append((coeff, ops_tot))

        return fh_terms_to_sparse_pauli_op(terms, n_tot, drop_identity=True)

    def build_circuit(self, *, n_cycles, init_label=None, final_pauli=None):
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

        q_sys = QuantumRegister(self.n_sys, "sys")
        q_bath = QuantumRegister(self.n_bath, "bath")

        n_bath_bits = int(n_cycles) * self.n_bath
        n_final_bits = self.n_sys if final_pauli is not None else 0

        c = ClassicalRegister(n_bath_bits + n_final_bits, "c")
        qc = QuantumCircuit(q_sys, q_bath, c)

        self._prepare_initial_state(qc, q_sys, init_label)

        V = self._interaction_sparse_full()

        for r in range(int(n_cycles)):
            self._append_filter_block(qc, q_sys, q_bath, V)

            for mu in range(self.n_bath):
                qc.measure(q_bath[mu], c[r * self.n_bath + mu])

            for mu in range(self.n_bath):
                qc.reset(q_bath[mu])

        self._append_final_measurement(qc, q_sys, c, n_bath_bits, final_pauli)
        return qc


class ThermalProtocolSequentialQiskit(_ThermalProtocolQiskitBase):
    """
    Qiskit equivalent of ThermalProtocolSequential.

    One bath qubit is reused. One cycle loops over A_mu:
        product_tau [ U_int_mu(tau) U_B U_S ]
        measure bath
        reset bath
    """

    protocol_name = "qiskit_sequential"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bath = 1
        self.n_total = self.n_sys + 1
        self.n_bath_bits_per_cycle = self.n_ops

    def _interaction_sparse_one(self, A_terms):
        n_tot = self.n_sys + 1
        terms = []

        for coeff, ops in A_terms:
            ops_tot = {int(q): p for q, p in ops.items()}
            ops_tot[self.n_sys] = "Y"
            terms.append((coeff, ops_tot))

        return fh_terms_to_sparse_pauli_op(terms, n_tot, drop_identity=True)

    def build_circuit(self, *, n_cycles, init_label=None, final_pauli=None):
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

        q_sys = QuantumRegister(self.n_sys, "sys")
        q_bath = QuantumRegister(1, "bath")

        n_bath_bits = int(n_cycles) * self.n_ops
        n_final_bits = self.n_sys if final_pauli is not None else 0

        c = ClassicalRegister(n_bath_bits + n_final_bits, "c")
        qc = QuantumCircuit(q_sys, q_bath, c)

        self._prepare_initial_state(qc, q_sys, init_label)

        cbit = 0
        for _r in range(int(n_cycles)):
            for A_terms in self.term_sets:
                V = self._interaction_sparse_one(A_terms)
                self._append_filter_block(qc, q_sys, q_bath, V)

                qc.measure(q_bath[0], c[cbit])
                qc.reset(q_bath[0])
                cbit += 1

        self._append_final_measurement(qc, q_sys, c, n_bath_bits, final_pauli)
        return qc


# desde NB04



def make_fh_dimer(t=1.0, U=4.0):
    return FermiHubbardJW(n_sites=2, bonds=[(0, 1)], t=t, U=U)

def project_density(rho):
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.real(np.trace(rho))
    if abs(tr) < 1e-14:
        raise ValueError("Traza casi cero.")
    return rho / tr

def energy_basis_offdiag_norm(model, rho):
    evals, evecs = model.eigensystem()
    rho_e = evecs.conj().T @ rho @ evecs
    off = rho_e - np.diag(np.diag(rho_e))
    return float(np.linalg.norm(off, ord="fro"))

def exact_heat_capacity(model, T):
    beta = 1.0 / float(T)
    rho = model.gibbs_state(beta)
    H = model.H
    E = np.real(np.trace(H @ rho))
    E2 = np.real(np.trace(H @ H @ rho))
    return float(beta**2 * (E2 - E**2))

def build_protocol(protocol_cls, model, A_ops, labels, beta, h_bath, theta, delta, M_T, random=False):
    if protocol_cls is ThermalProtocolSequential:
        return protocol_cls(
            model,
            A_ops=A_ops,
            labels=labels,
            h_bath=h_bath,
            beta=beta,
            theta=theta,
            delta=delta,
            M_T=int(M_T),
            random_order=False,
            random_evolution=bool(random),
        )

    return protocol_cls(
        model,
        A_ops=A_ops,
        labels=labels,
        h_bath=h_bath,
        beta=beta,
        theta=theta,
        delta=delta,
        M_T=int(M_T),
        random_evolution=bool(random),
    )

def protocol_fixed_point(protocol_cls, model, A_ops, labels, beta, h_bath, theta, delta, M_T):
    proto = build_protocol(
        protocol_cls,
        model,
        A_ops,
        labels,
        beta=beta,
        h_bath=h_bath,
        theta=theta,
        delta=delta,
        M_T=M_T,
        random=False,
    )
    return proto.fixed_point(), proto

def protocol_run(protocol_cls, model, A_ops, labels, beta, h_bath, theta, delta, M_T, n_cycles, seed=1, record_every=25, random=False):
    proto = build_protocol(
        protocol_cls,
        model,
        A_ops,
        labels,
        beta=beta,
        h_bath=h_bath,
        theta=theta,
        delta=delta,
        M_T=M_T,
        random=random,
    )
    out = proto.run(
        n_cycles=n_cycles,
        verbose=False,
        record_every=record_every,
        seed=seed,
    )
    return out, proto

def state_metrics(model, rho, rho_g, label):
    return {
        "label": label,
        "Dtr": trace_distance(rho, rho_g),
        "E": model.energy(rho),
        "E_g": model.energy(rho_g),
        "Delta_E": model.energy(rho) - model.energy(rho_g),
        "abs_Delta_E": abs(model.energy(rho) - model.energy(rho_g)),
        "K": model.kinetic_energy(rho),
        "U_D": model.interaction_energy(rho),
        "D_avg": model.avg_double_occupancy(rho),
        "D_g": model.avg_double_occupancy(rho_g),
        "abs_Delta_D": abs(model.avg_double_occupancy(rho) - model.avg_double_occupancy(rho_g)),
        "Czz_01": model.spin_correlation(rho, 0, 1),
        "Czz_g": model.spin_correlation(rho_g, 0, 1),
        "abs_Delta_Czz": abs(model.spin_correlation(rho, 0, 1) - model.spin_correlation(rho_g, 0, 1)),
        "MI_01": model.site_mutual_information(rho, 0, 1),
        "MI_g": model.site_mutual_information(rho_g, 0, 1),
        "abs_Delta_MI": abs(model.site_mutual_information(rho, 0, 1) - model.site_mutual_information(rho_g, 0, 1)),
        "S": von_neumann_entropy(rho),
        "S_g": von_neumann_entropy(rho_g),
        "coh_E": energy_basis_offdiag_norm(model, rho),
    }
    
    
    
    
    
# ============================================================
# Sequential: variantes randomización / rewind
# Usado en NB04, sección A.5.5
# ============================================================

def project_density_safe(rho):
    """
    Hermitiza y renormaliza una matriz de densidad.

    Es un wrapper público de _project_density_matrix para usar en análisis
    de trayectorias y promedios estacionarios.
    """
    return _project_density_matrix(rho)


def system_free_unitary(model, time):
    """
    Evolución libre del sistema bajo H_S durante un tiempo dado.
    """
    evals, evecs = model.eigensystem()
    phases = np.exp(-1j * float(time) * evals)
    return evecs @ np.diag(phases) @ evecs.conj().T


def energy_basis_offdiag_norm_model(model, rho):
    """
    Norma de la parte off-diagonal de rho en la base de energía de H_S.
    """
    evals, evecs = model.eigensystem()
    rho_e = evecs.conj().T @ rho @ evecs
    off = rho_e - np.diag(np.diag(rho_e))
    return float(np.linalg.norm(off, ord="fro"))


def sequential_Q_eff_variant(
    proto,
    Q,
    random_evolution=False,
    rewind=False,
    rng=None,
):
    """
    Construye el unitario efectivo de un subpaso secuencial.

    Parameters
    ----------
    proto:
        Instancia de ThermalProtocolSequential.
    Q:
        Un subpaso unitario ya construido en proto.Qs.
    random_evolution:
        Si True, añade una evolución libre aleatoria del sistema después
        del subpaso Q.
    rewind:
        Si True, aplica una evolución libre inversa aproximada durante
        la ventana temporal completa del filtro.
    rng:
        Generador np.random.default_rng.
    """
    IB = np.eye(2, dtype=complex)
    Q_eff = Q

    if rewind:
        T_window = (2 * proto.M_T + 1) * proto.delta
        U_rew = system_free_unitary(proto.model, -T_window)
        Q_eff = kron(U_rew, IB) @ Q_eff

    if random_evolution:
        if rng is None:
            rng = np.random.default_rng()

        m_rand = rng.integers(1, 2 * proto.M_T + 1)
        U_rand = system_free_unitary(
            proto.model,
            proto.delta * int(m_rand),
        )
        Q_eff = kron(U_rand, IB) @ Q_eff

    return Q_eff


def apply_sequential_cycle_variant(
    proto,
    rho,
    random_order=False,
    random_evolution=False,
    rewind=False,
    rng=None,
):
    """
    Aplica un ciclo macro del protocolo secuencial modificado.

    El ciclo secuencial aplica:
        A_0, reset;
        A_1, reset;
        ...
        A_mu, reset.

    random_order=True cambia el orden de los operadores A_mu.
    random_evolution=True añade evolución libre aleatoria.
    rewind=True añade evolución libre inversa aproximada.
    """
    if rng is None:
        rng = np.random.default_rng()

    if random_order:
        order = rng.permutation(len(proto.Qs))
    else:
        order = range(len(proto.Qs))

    for mu in order:
        Q_eff = sequential_Q_eff_variant(
            proto,
            proto.Qs[mu],
            random_evolution=random_evolution,
            rewind=rewind,
            rng=rng,
        )

        rho_tot = Q_eff @ kron(rho, proto.phi) @ Q_eff.conj().T
        rho = partial_trace_B(rho_tot, proto.dS, 2)
        rho = project_density_safe(rho)

    return rho


def run_sequential_variant_trajectory(
    proto,
    rho_target,
    n_cycles=700,
    burn=450,
    keep=150,
    record_every=25,
    random_order=False,
    random_evolution=False,
    rewind=False,
    seed=123,
):
    """
    Ejecuta una trayectoria finita del protocolo secuencial modificado.

    Devuelve:
    - rho_final;
    - rho_stationary: promedio de estados después del burn-in;
    - history: DataFrame con distancia traza y observables registrados.
    """
    rng = np.random.default_rng(seed)

    rho = np.eye(proto.dS, dtype=complex) / proto.dS
    kept_states = []
    history = []

    def record(n, rho_current):
        history.append({
            "cycle": int(n),
            "D_trace": trace_distance(rho_current, rho_target),
            "E": proto.model.energy(rho_current),
            "D_avg": proto.model.avg_double_occupancy(rho_current),
            "Czz": proto.model.spin_correlation(rho_current, 0, 1),
            "MI_01": proto.model.site_mutual_information(rho_current, 0, 1),
            "offdiag_E": energy_basis_offdiag_norm_model(
                proto.model,
                rho_current,
            ),
        })

    record(0, rho)

    for n in range(1, int(n_cycles) + 1):
        rho = apply_sequential_cycle_variant(
            proto,
            rho,
            random_order=random_order,
            random_evolution=random_evolution,
            rewind=rewind,
            rng=rng,
        )

        if n % int(record_every) == 0 or n == int(n_cycles):
            record(n, rho)

        if n > int(burn) and len(kept_states) < int(keep):
            kept_states.append(rho.copy())

    if kept_states:
        rho_stationary = project_density_safe(np.mean(kept_states, axis=0))
    else:
        rho_stationary = rho.copy()

    return {
        "rho_final": rho.copy(),
        "rho_stationary": rho_stationary,
        "history": pd.DataFrame(history),
        "seed": int(seed),
    }


def summarize_state_against_target(model, rho, rho_target):
    """
    Resume un estado frente al Gibbs exacto.

    Incluye distancia traza, energía, doble ocupación, correlación de spin,
    información mutua y coherencia residual en la base de energía.
    """
    E = model.energy(rho)
    E_target = model.energy(rho_target)

    D_avg = model.avg_double_occupancy(rho)
    D_avg_target = model.avg_double_occupancy(rho_target)

    Czz = model.spin_correlation(rho, 0, 1)
    Czz_target = model.spin_correlation(rho_target, 0, 1)

    MI = model.site_mutual_information(rho, 0, 1)
    MI_target = model.site_mutual_information(rho_target, 0, 1)

    return {
        "D_trace": trace_distance(rho, rho_target),

        "E": E,
        "E_target": E_target,
        "Delta_E": E - E_target,
        "abs_Delta_E": abs(E - E_target),

        "D_avg": D_avg,
        "D_avg_target": D_avg_target,
        "Delta_D_avg": D_avg - D_avg_target,
        "abs_Delta_D_avg": abs(D_avg - D_avg_target),

        "Czz": Czz,
        "Czz_target": Czz_target,
        "Delta_Czz": Czz - Czz_target,
        "abs_Delta_Czz": abs(Czz - Czz_target),

        "MI_01": MI,
        "MI_01_target": MI_target,
        "Delta_MI_01": MI - MI_target,
        "abs_Delta_MI_01": abs(MI - MI_target),

        "offdiag_E": energy_basis_offdiag_norm_model(model, rho),
    }


def run_sequential_random_rewind_case(
    MT,
    delta=0.30,
    theta=0.35,
    U=4.0,
    beta=2.0,
    h=1.0,
    n_cycles=700,
    burn=450,
    keep=150,
    record_every=25,
    random_seeds=(101, 202, 303, 404, 505),
):
    """
    Ejecuta las cuatro variantes de A.5.5:

    - sin nada: orden fijo y evolución determinista;
    - solo random: orden aleatorio + evolución libre aleatoria;
    - solo rewind: evolución libre inversa aproximada;
    - ambos: randomización + rewind.

    El target es el Gibbs global del dímero Fermi-Hubbard.
    """
    model = FermiHubbardJW(
        n_sites=2,
        bonds=[(0, 1)],
        t=1.0,
        U=float(U),
        mu=0.0,
    )

    rho_target = model.gibbs_state(beta)
    A_ops, labels = build_coupling_ops_paper(model)

    proto = ThermalProtocolSequential(
        model,
        A_ops=A_ops,
        labels=labels,
        h_bath=float(h),
        beta=float(beta),
        theta=float(theta),
        delta=float(delta),
        M_T=int(MT),
        random_order=False,
        random_evolution=False,
    )

    variants = {
        "sin nada": {
            "random_order": False,
            "random_evolution": False,
            "rewind": False,
            "seeds": [7],
        },
        "solo random": {
            "random_order": True,
            "random_evolution": True,
            "rewind": False,
            "seeds": list(random_seeds),
        },
        "solo rewind": {
            "random_order": False,
            "random_evolution": False,
            "rewind": True,
            "seeds": [7],
        },
        "ambos": {
            "random_order": True,
            "random_evolution": True,
            "rewind": True,
            "seeds": list(random_seeds),
        },
    }

    outs = {}
    summary_rows = []

    for name, cfg in variants.items():
        print(f"M_T={MT:2d} | running: {name}")

        outs[name] = []

        for seed in cfg["seeds"]:
            out = run_sequential_variant_trajectory(
                proto,
                rho_target=rho_target,
                n_cycles=n_cycles,
                burn=burn,
                keep=keep,
                record_every=record_every,
                random_order=cfg["random_order"],
                random_evolution=cfg["random_evolution"],
                rewind=cfg["rewind"],
                seed=seed,
            )

            outs[name].append(out)

        rho_stationary = project_density_safe(
            np.mean(
                [out["rho_stationary"] for out in outs[name]],
                axis=0,
            )
        )

        row = {
            "M_T": int(MT),
            "M_T_delta": float(MT * delta),
            "variant": name,
            "n_seeds": len(cfg["seeds"]),
            "random": bool(cfg["random_order"] or cfg["random_evolution"]),
            "rewind": bool(cfg["rewind"]),
        }

        row.update(
            summarize_state_against_target(
                model,
                rho_stationary,
                rho_target,
            )
        )

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    return {
        "model": model,
        "rho_target": rho_target,
        "proto": proto,
        "outs": outs,
        "df_summary": df_summary,
        "params": {
            "U": float(U),
            "beta": float(beta),
            "h": float(h),
            "theta": float(theta),
            "delta": float(delta),
            "M_T": int(MT),
            "n_cycles": int(n_cycles),
            "burn": int(burn),
            "keep": int(keep),
            "record_every": int(record_every),
        },
    }


def run_sequential_random_rewind_cases(
    MT_values=(12, 10),
    delta=0.30,
    theta=0.35,
    U=4.0,
    beta=2.0,
    h=1.0,
    n_cycles=700,
    burn=450,
    keep=150,
    record_every=25,
    random_seeds=(101, 202, 303, 404, 505),
):
    """
    Ejecuta run_sequential_random_rewind_case para varios M_T.

    Devuelve:
    - cases: diccionario con resultados por M_T;
    - df_all: tabla concatenada de resultados estacionarios.
    """
    cases = {}

    for MT in MT_values:
        label = f"M_T={int(MT)}"

        cases[label] = run_sequential_random_rewind_case(
            MT=int(MT),
            delta=delta,
            theta=theta,
            U=U,
            beta=beta,
            h=h,
            n_cycles=n_cycles,
            burn=burn,
            keep=keep,
            record_every=record_every,
            random_seeds=random_seeds,
        )

    df_all = pd.concat(
        [
            res["df_summary"].assign(case=case_name)
            for case_name, res in cases.items()
        ],
        ignore_index=True,
    )

    return cases, df_all


# AÑADIDOS PARA DEJAR LIMPIO EL NOTEBOOK

def build_ising_cfg_multi_bath(
    Lx, Ly, target_sites,
    J=0.5, g=1.0, h=1.0, beta=1.0,
    theta=0.15, delta=0.15, MT=15, n_cycles=1,
    sys_A="Y", bath_A="Y",
    randomize=False, randomization_lambda=0.0, seed=101,
    rewind=False, periodic=False,
):
    """
    Ising 1D (Ly=1) o 2D con N_B = len(target_sites) qubits baño.
    Cada qubit baño acopla a un sitio distinto del sistema.
    """
    n_sys = Lx * Ly
    n_bath = len(target_sites)

    if Ly == 1:
        system_terms = ising_chain_xx_z_terms(n_sys, J=J, g=g, periodic=periodic)
    else:
        system_terms = ising_2d_xx_z_terms(Lx, Ly, J=J, g=g, periodic=periodic)

    bath_terms = default_bath_terms(n_bath, h=h)

    coupling_specs = []
    for bath_idx, site in enumerate(target_sites):
        sys_terms_i  = [(1.0, {site: sys_A})]
        bath_terms_i = [(1.0, {bath_idx: bath_A})]
        coupling_specs.append({
            "sys_terms":  sys_terms_i,
            "bath_terms": bath_terms_i,
        })

    return ProtocolConfig(
        n_sys=n_sys, n_bath=n_bath,
        system_terms=system_terms, bath_terms=bath_terms,
        coupling_specs=coupling_specs,
        beta=beta, h_filter=h,
        theta=theta, delta=delta, MT=MT, n_cycles=n_cycles,
        randomize=randomize, randomization_lambda=randomization_lambda,
        seed=seed, rewind=rewind,
    )
    
def local_Z_matrix(n_sys, site):
    return mat(make_op(n_sys, [(1.0, {site: "Z"})]))

def build_ising_cfg_single_bath(
    Lx, Ly, target_site,
    J=0.5, g=1.0, h=1.0, beta=1.0,
    theta=0.15, delta=0.15, MT=25, n_cycles=200,
    sys_A="Y", bath_A="Y",
    randomize=False, randomization_lambda=0.0, seed=101,
    rewind=False, periodic=False,
):
    """
    Ising 1D (Ly=1) o 2D con N_B = 1 qubit baño, acoplado al sitio
    'target_site' del sistema. Wrapper de build_ising_cfg_multi_bath.
    """
    return build_ising_cfg_multi_bath(
        Lx=Lx, Ly=Ly,
        target_sites=[target_site],
        J=J, g=g, h=h, beta=beta,
        theta=theta, delta=delta, MT=MT, n_cycles=n_cycles,
        sys_A=sys_A, bath_A=bath_A,
        randomize=randomize, randomization_lambda=randomization_lambda,
        seed=seed, rewind=rewind, periodic=periodic,
    )
    
    
    
#— Geometría del baño con N_B=1, control OBC vs PBC
# Cuatro schedules de geometría × dos condiciones de contorno.
# Métricas: σ del perfil (homogeneidad) y Δ_perfil (distancia al Gibbs).


def local_Z_matrix(n_sys, site):
    return mat(make_op(n_sys, [(1.0, {site: "Z"})]))


def run_with_geometry_schedule(
    Lx, Ly, schedule_fn, periodic,
    J=0.5, g=1.0, h=1.0, beta=1.0,
    theta=0.15, delta=0.15, MT=25, n_cycles=400,
    rng_seed=12345,
):
    """
    Ejecuta el protocolo con geometría de acoplo variable entre ciclos.
    schedule_fn(cycle_idx, rng) -> int (sitio donde se acopla el N_B=1 baño).
    """
    n_sys = Lx * Ly
    rng = np.random.default_rng(rng_seed)
    rho_sys = maximally_mixed(n_sys)

    compiled_per_site = {}
    for site in range(n_sys):
        cfg = build_ising_cfg_single_bath(
            Lx=Lx, Ly=Ly, target_site=site,
            J=J, g=g, h=h, beta=beta,
            theta=theta, delta=delta, MT=MT, n_cycles=1,
            sys_A="Y", bath_A="Y", periodic=periodic,
        )
        compiled_per_site[site] = compile_protocol(cfg)

    n_bath = 1
    Hs = compiled_per_site[0]["Hs_mat"]
    phi_B = compiled_per_site[0]["bath0"]

    for c in range(n_cycles):
        site = schedule_fn(c, rng)
        Q = cycle_unitary_matrix(compiled_per_site[site], mr=0)
        rho_total = np.kron(rho_sys, phi_B)
        rho_total = Q @ rho_total @ Q.conj().T
        rho_sys = ptrace_bath(rho_total, n_sys, n_bath)

    return rho_sys, Hs



def run_with_random_geometry_NB(
    Lx, Ly, NB,
    J=0.5, g=1.0, h=1.0, beta=1.0,
    theta=0.15, delta=0.15, MT=25, n_cycles=400,
    rng_seed=12345,
):
    """
    Ejecuta el protocolo con N_B qubits baño y rotación aleatoria de
    geometría: en cada ciclo sortea N_B sitios distintos sin repetición
    del sistema y los acopla a los N_B qubits baño. Devuelve la matriz
    densidad final del sistema y el Hamiltoniano del sistema.
    """
    n_sys = Lx * Ly
    rng = np.random.default_rng(rng_seed)
    rho_sys = maximally_mixed(n_sys)

    # Cache de compilados: una entrada por tupla ordenada de sitios.
    cache = {}

    def get_compiled(target_sites_tuple):
        if target_sites_tuple not in cache:
            cfg = build_ising_cfg_multi_bath(
                Lx=Lx, Ly=Ly,
                target_sites=list(target_sites_tuple),
                J=J, g=g, h=h, beta=beta,
                theta=theta, delta=delta, MT=MT, n_cycles=1,
                sys_A="Y", bath_A="Y",
            )
            cache[target_sites_tuple] = compile_protocol(cfg)
        return cache[target_sites_tuple]

    # Inicializar Hs y phi_B con un primer sorteo
    first_sites = tuple(sorted(
        rng.choice(n_sys, size=NB, replace=False).tolist()
    ))
    compiled_first = get_compiled(first_sites)
    Hs = compiled_first["Hs_mat"]
    phi_B = compiled_first["bath0"]

    # Loop de ciclos
    for c in range(n_cycles):
        if c == 0:
            sites_tuple = first_sites
        else:
            sites_tuple = tuple(sorted(
                rng.choice(n_sys, size=NB, replace=False).tolist()
            ))
        compiled = get_compiled(sites_tuple)
        Q = cycle_unitary_matrix(compiled, mr=0)
        rho_total = np.kron(rho_sys, phi_B)
        rho_total = Q @ rho_total @ Q.conj().T
        rho_sys = ptrace_bath(rho_total, n_sys, NB)

    return rho_sys, Hs


# ============================================================
# NB03 — Helpers numpy para mapa (theta, MT) en Ising interactuante
# ============================================================


def fixed_point_from_compiled_kraus(compiled, mr=0):
    """
    Punto fijo exacto del canal usando Kraus obtenidos de una sola unidad U.
    Mucho más rápido que construir el superoperador llamando al canal base a base.

    Funciona para canal determinista sin randomización explícita.
    """
    U = cycle_unitary_matrix(compiled, mr=mr)

    dS = compiled["dS"]
    dB = compiled["dB"]

    # Índices: |s,b>. U[s_out,b_out,s_in,b_in]
    U4 = U.reshape(dS, dB, dS, dB)

    # Baño inicial |0...0>. Kraus K_b = <b|U|0>
    K_list = [U4[:, b, :, 0] for b in range(dB)]

    # Superoperador con vectorización column-major:
    # vec(K rho K†) = (K.conj() ⊗ K) vec(rho)
    S = np.zeros((dS*dS, dS*dS), dtype=complex)
    for K in K_list:
        S += np.kron(K.conj(), K)

    evals, evecs = np.linalg.eig(S)
    idx = np.argmin(np.abs(evals - 1.0))

    rho_fp = evecs[:, idx].reshape((dS, dS), order="F")
    rho_fp = project_to_physical_dm(rho_fp)
    return rho_fp, evals


def system_A_ops_from_cfg(cfg):
    """
    Extrae los operadores A_S locales del config Ising.
    Sirve para estimar qué gaps de Bohr están más acoplados.
    """
    A_ops = []
    for spec in cfg.coupling_specs:
        A_ops.append(mat(make_op(cfg.n_sys, spec["sys_terms"])))
    return A_ops


def dominant_bohr_frequency(H, A_ops, tol_gap=1e-9, decimals=8):
    """
    Frecuencia de Bohr dominante ponderada por elementos de matriz del acoplo:

        peso(omega_ab) = sum_mu |<a|A_mu|b>|^2.

    En sistemas interactuantes hay varias; devolvemos la más pesada como guía
    para dibujar las líneas T*omega = pi, 2pi.
    """
    E, V = np.linalg.eigh(H)
    weights = {}

    for A in A_ops:
        Ae = V.conj().T @ A @ V
        for a in range(len(E)):
            for b in range(len(E)):
                w = abs(E[a] - E[b])
                if w > tol_gap:
                    key = np.round(w, decimals)
                    weights[key] = weights.get(key, 0.0) + abs(Ae[a, b])**2

    df = (
        pd.DataFrame(
            [{"omega": float(k), "weight": float(v)} for k, v in weights.items()]
        )
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )

    omega_ref = float(df.loc[0, "omega"])
    return omega_ref, df


def ising_bias_fp_fast(params):
    """
    Ejecuta un punto del mapa:
    - compila el protocolo,
    - obtiene rho_fp por Kraus exactos,
    - compara energía con Gibbs exacto.
    """
    cfg = build_ising_cfg(**params)
    compiled = compile_protocol(cfg)

    rho_fp, evals = fixed_point_from_compiled_kraus(compiled)
    rho_g = gibbs_state(compiled["Hs"], cfg.beta)

    E_fp = expect(compiled["Hs"], rho_fp)
    E_g  = expect(compiled["Hs"], rho_g)

    bias = abs(E_fp - E_g) / cfg.n_sys
    return float(bias), rho_fp, rho_g, compiled, evals

def add_harmonic_guides(ax, omega_ref, delta, kmax=5, color="red"):
    """
    Dibuja guías T omega_ref = k pi.
    """
    for k in range(1, kmax + 1):
        mt_k = k * np.pi / (delta * omega_ref)

        if k == 1:
            ls = "--"
        elif k == 2:
            ls = ":"
        else:
            ls = "-."

        ax.axvline(
            mt_k,
            color=color,
            ls=ls,
            alpha=0.45,
            lw=1.5,
            label=rf"$T\omega_\star={k}\pi$" if k <= 2 else None,
        )
        
def grid_edges(x):
    """
    Devuelve bordes de celdas para un grid no uniformemente espaciado.
    """
    x = np.asarray(x, dtype=float)
    edges = np.zeros(len(x) + 1)

    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])

    return edges


def plot_regime_heatmap_real_axes(
    Z,
    theta_grid,
    MT_grid,
    title,
    omega_ref,
    delta,
    kmax=4,
):
    theta_edges = grid_edges(theta_grid)
    MT_edges = grid_edges(MT_grid)

    fig, ax = plt.subplots(figsize=(9, 6))

    vmin = max(np.nanmin(Z), 1e-8)
    vmax = np.nanmax(Z)

    im = ax.pcolormesh(
        theta_edges,
        MT_edges,
        Z,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
        shading="auto",
    )

    for k in range(1, kmax + 1):
        mt_k = k * np.pi / (delta * omega_ref)

        if k == 1:
            ls = "--"
        elif k == 2:
            ls = ":"
        else:
            ls = "-."

        ax.axhline(
            mt_k,
            color="red",
            lw=1.7,
            ls=ls,
            alpha=0.65,
            label=rf"$T\omega_\star={k}\pi$" if k <= 2 else None,
        )

    mt_valley = 1.5 * np.pi / (delta * omega_ref)
    ax.axhline(
        mt_valley,
        color="white",
        lw=2,
        ls="-.",
        alpha=0.85,
        label=rf"valle $T\omega_\star=1.5\pi$",
    )

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$M_T$")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    plt.colorbar(
        im,
        ax=ax,
        fraction=0.04,
        label=r"$|E_{\rm fp}-E_\beta|/N_S$",
    )

    plt.tight_layout()
    plt.show()
    
def product_of_single_qubit_marginals(rho, n_sys):
    """ρ_1 ⊗ ρ_2 ⊗ ... ⊗ ρ_N construido desde las marginales de ρ."""
    marginals = [reduced_dm_keep(rho, [i], n_sys) for i in range(n_sys)]
    rho_prod = marginals[0]
    for r in marginals[1:]:
        rho_prod = np.kron(rho_prod, r)
    return project_to_physical_dm(rho_prod)

def total_correlation_distance(rho, n_sys):
    """
    Distancia entre ρ y el producto de sus marginales.
    En NI ideal debe ser ~0.
    En interactuante mide cuánta correlación total contiene ρ.
    """
    rho_prod = product_of_single_qubit_marginals(rho, n_sys)
    return trace_distance_dm(rho, rho_prod)

def connected_corr_vector_ising(rho, n_sys, pairs=None, paulis=("X", "Z")):
    """
    Vector de correladores conectados:
        C_ij^P = <P_i P_j> - <P_i><P_j>
    """
    if pairs is None:
        pairs = [(i, i + 1) for i in range(n_sys - 1)]

    vals = []
    for i, j in pairs:
        for P in paulis:
            Pi = make_op(n_sys, [(1.0, {i: P})])
            Pj = make_op(n_sys, [(1.0, {j: P})])
            PiPj = make_op(n_sys, [(1.0, {i: P, j: P})])

            ci = expect(Pi, rho)
            cj = expect(Pj, rho)
            cij = expect(PiPj, rho)
            vals.append(float(np.real(cij - ci * cj)))

    return np.array(vals, dtype=float)



# A.7.0 — Helpers para tiempo de mezcla y test Mpemba-like


def first_hitting_time_A7(values, eps):
    """Primer índice n tal que values[n] <= eps. Devuelve NaN si no cruza."""
    values = np.asarray(values, dtype=float)
    hits = np.where(values <= float(eps))[0]
    return int(hits[0]) if len(hits) else np.nan


def make_params_A7(
    beta,
    N=4,
    J=0.5,
    theta=0.25,
    MT=30,
    n_cycles=300,
):
    """
    Usa la misma convención que make_params_1d del notebook:
    h = max(2g, 4J), delta = pi/40, A_S = Y, A_B = Y.
    """
    return make_params_1d(
        Lx=int(N),
        J=float(J),
        beta=float(beta),
        MT=int(MT),
        n_cycles=int(n_cycles),
        theta=float(theta),
    )


def run_single_beta_A7(
    beta,
    *,
    N=4,
    J=0.5,
    theta=0.25,
    MT=30,
    n_cycles=300,
    eps=0.05,
):
    """
    Ejecuta una trayectoria exacta desde rho0=I/d y calcula:
      - distancia traza a Gibbs por ciclo;
      - tiempo de mezcla n_star;
      - energía por ciclo;
      - sesgo energético final.
    """
    params = make_params_A7(
        beta=beta,
        N=N,
        J=J,
        theta=theta,
        MT=MT,
        n_cycles=n_cycles,
    )

    out = run_ising_case(
        params,
        rho0=maximally_mixed(N),
        compute_fp=False,
    )

    rho_g = out["rho_g"]
    Hs = out["Hs"]
    rhos = out["res"]["rhos"]

    dtr = np.array(
        [trace_distance_dm(rho, rho_g) for rho in rhos],
        dtype=float,
    )

    energies = np.array(
        [expect(Hs, rho) for rho in rhos],
        dtype=float,
    )

    E_g = float(expect(Hs, rho_g))
    denom_E = max(abs(energies[0] - E_g), 1e-15)
    err_E_rel = np.abs(energies - E_g) / denom_E

    n_star = first_hitting_time_A7(dtr, eps)

    row = {
        "geometry": f"1d{N}",
        "rho0": "I/d",
        "N_S": int(N),
        "J": float(J),
        "beta": float(beta),
        "theta": float(theta),
        "MT": int(MT),
        "h": float(params["h"]),
        "n_cycles": int(n_cycles),
        "eps": float(eps),
        "n_star": n_star,
        "D_initial": float(dtr[0]),
        "D_final": float(dtr[-1]),
        "E_initial": float(energies[0]),
        "E_final": float(energies[-1]),
        "E_gibbs": float(E_g),
        "bias_E_final": float(abs(energies[-1] - E_g)),
        "bias_E_final_per_site": float(abs(energies[-1] - E_g) / N),
    }

    traj = {
        "beta": float(beta),
        "params": params,
        "dtr": dtr,
        "energies": energies,
        "err_E_rel": err_E_rel,
        "E_gibbs": E_g,
    }

    return row, traj


def scan_beta_A7(
    betas,
    *,
    N=4,
    J=0.5,
    theta=0.25,
    MT=30,
    n_cycles=300,
    eps=0.05,
):
    rows = []
    trajectories = {}

    for beta in betas:
        print(f"[A.7] beta={beta:.3f}")
        row, traj = run_single_beta_A7(
            beta,
            N=N,
            J=J,
            theta=theta,
            MT=MT,
            n_cycles=n_cycles,
            eps=eps,
        )
        rows.append(row)
        trajectories[float(beta)] = traj

    return pd.DataFrame(rows), trajectories


def mpemba_inversions_A7(df):
    """
    Busca pares beta_cold > beta_hot con n_star(cold) < n_star(hot).
    Ignora puntos que no hayan alcanzado la tolerancia.
    """
    rows = []
    clean = df.dropna(subset=["n_star"]).copy()

    for _, hot in clean.iterrows():
        for _, cold in clean.iterrows():
            if cold["beta"] <= hot["beta"]:
                continue

            if cold["n_star"] < hot["n_star"]:
                rows.append({
                    "beta_hot": hot["beta"],
                    "beta_cold": cold["beta"],
                    "n_hot": hot["n_star"],
                    "n_cold": cold["n_star"],
                    "delta_n": cold["n_star"] - hot["n_star"],
                })

    return pd.DataFrame(rows)

df_beta_mix = pd.DataFrame([
    dict(
        geometry="1d4", rho0="I/d", beta=0.4, J=0.5, theta=0.25,
        MT=30, n_cycles=300, eps=0.05,
        n_star=22,
        D_initial=0.336474, D_final=0.009908,
        E_initial=0.0, E_final=-1.767200, E_gibbs=-1.761692,
    ),
    dict(
        geometry="1d4", rho0="I/d", beta=0.6, J=0.5, theta=0.25,
        MT=30, n_cycles=300, eps=0.05,
        n_star=29,
        D_initial=0.464815, D_final=0.014047,
        E_initial=0.0, E_final=-2.439391, E_gibbs=-2.433439,
    ),
    dict(
        geometry="1d4", rho0="I/d", beta=0.8, J=0.5, theta=0.25,
        MT=30, n_cycles=300, eps=0.05,
        n_star=38,
        D_initial=0.571586, D_final=0.016030,
        E_initial=0.0, E_final=-2.945838, E_gibbs=-2.939970,
    ),
    dict(
        geometry="1d4", rho0="I/d", beta=1.0, J=0.5, theta=0.25,
        MT=30, n_cycles=300, eps=0.05,
        n_star=45,
        D_initial=0.655115, D_final=0.016185,
        E_initial=0.0, E_final=-3.310881, E_gibbs=-3.305535,
    ),
    dict(
        geometry="1d4", rho0="I/d", beta=1.3, J=0.5, theta=0.25,
        MT=30, n_cycles=300, eps=0.05,
        n_star=54,
        D_initial=0.736037, D_final=0.014472,
        E_initial=0.0, E_final=-3.665542, E_gibbs=-3.661530,
    ),
    dict(
        geometry="1d4", rho0="I/d", beta=1.7, J=0.5, theta=0.25,
        MT=30, n_cycles=300, eps=0.05,
        n_star=63,
        D_initial=0.810242, D_final=0.011161,
        E_initial=0.0, E_final=-3.919763, E_gibbs=-3.917611,
    ),
    dict(
        geometry="1d4", rho0="I/d", beta=2.2, J=0.5, theta=0.25,
        MT=30, n_cycles=300, eps=0.05,
        n_star=70,
        D_initial=0.855763, D_final=0.007609,
        E_initial=0.0, E_final=-4.065906, E_gibbs=-4.065257,
    ),
    dict(
        geometry="1d4", rho0="I/d", beta=2.8, J=0.5, theta=0.25,
        MT=30, n_cycles=300, eps=0.05,
        n_star=76,
        D_initial=0.901599, D_final=0.004795,
        E_initial=0.0, E_final=-4.137754, E_gibbs=-4.137841,
    ),
    dict(
        geometry="1d4", rho0="I/d", beta=3.5, J=0.5, theta=0.25,
        MT=30, n_cycles=300, eps=0.05,
        n_star=80,
        D_initial=0.923468, D_final=0.002970,
        E_initial=0.0, E_final=-4.169204, E_gibbs=-4.169508,
    ),
])


# — Helpers gaussianos para free-fermion chain
#
# Esta celda implementa la versión free-fermion del protocolo:
# - H = (i/4) gamma^T A gamma
# - estado gaussiano representado por matriz de covarianza C
# - un ciclo de cooling se implementa como:
#     sistema + bath reset -> evolución cuadrática modulada -> traza del bath
# - opcionalmente se aplica una dephasing/randomización en la base de energía.


def antisymmetrize(M):
    """Proyecta una matriz real a antisimétrica."""
    M = np.asarray(M)
    return 0.5 * (M - M.T)


def add_majorana_term(A, i, j, coeff):
    """
    Añade un término i * coeff * gamma_i gamma_j al Hamiltoniano.

    Convención:
        H = (i/4) gamma^T A gamma

    Un término i*c*gamma_i*gamma_j implica:
        A_ij += 2c
        A_ji -= 2c
    """
    A[i, j] += 2.0 * coeff
    A[j, i] -= 2.0 * coeff


def majorana_chain_A(N, J=1.0, g=1.0):
    """
    Matriz A_S para la cadena free-fermion/Majorana del paper:

        H_S = (i g / 2) sum_j gamma_{2j} gamma_{2j+1}
              - (i J / 2) sum_j gamma_{2j+1} gamma_{2j+2}

    Cadena abierta.
    """
    N = int(N)
    A = np.zeros((2 * N, 2 * N), dtype=float)

    for j in range(N):
        add_majorana_term(A, 2 * j, 2 * j + 1, +0.5 * g)

    for j in range(N - 1):
        add_majorana_term(A, 2 * j + 1, 2 * j + 2, -0.5 * J)

    return antisymmetrize(A)


def bath_majorana_A(N, h=2.0):
    """
    Bath de N modos Majorana independientes.

    H_B = (i h / 2) sum_mu eta_{2mu} eta_{2mu+1}
    """
    N = int(N)
    A = np.zeros((2 * N, 2 * N), dtype=float)

    for mu in range(N):
        add_majorana_term(A, 2 * mu, 2 * mu + 1, +0.5 * h)

    return antisymmetrize(A)


def coupling_majorana_A(N):
    """
    Acoplo cuadrático sistema-bath.

    Para cada sitio s usamos el operador lineal del paper:

        A_s = (gamma_{2s} - gamma_{2s+1}) / sqrt(2)

    y lo acoplamos a una Majorana del bath eta_{2s}:

        H_SB ∝ i A_s eta_{2s}

    Devuelve A_coup sobre el espacio total sistema+bath.
    """
    N = int(N)
    dim_s = 2 * N
    dim_b = 2 * N
    dim = dim_s + dim_b

    A = np.zeros((dim, dim), dtype=float)
    bath0 = dim_s

    for s in range(N):
        eta = bath0 + 2 * s

        # i * (gamma_2s / sqrt(2)) * eta
        add_majorana_term(A, 2 * s, eta, +1.0 / np.sqrt(2.0))

        # i * (-gamma_{2s+1} / sqrt(2)) * eta
        add_majorana_term(A, 2 * s + 1, eta, -1.0 / np.sqrt(2.0))

    return antisymmetrize(A)


def canonical_energies(A):
    """
    Energías single-particle positivas de iA.
    """
    vals = np.linalg.eigvalsh(1j * A)
    vals = np.real(vals)
    vals_pos = vals[vals > 1e-12]
    return np.sort(vals_pos)


def thermal_covariance_majorana(A, beta):
    """
    Covarianza gaussiana térmica para H = (i/4) gamma^T A gamma.

    Convención:
        C_ij = (i/2) <[gamma_i, gamma_j]>

    Fórmula:
        C_beta = -i tanh(i beta A / 2)
    """
    A = np.asarray(A, dtype=float)
    w, V = np.linalg.eigh(1j * A)

    tanhK = (V * np.tanh(0.5 * beta * w)) @ V.conj().T
    C = -1j * tanhK
    C = np.real_if_close(C, tol=1000).real

    return antisymmetrize(C)


def gaussian_entropy(C):
    """
    Entropía de von Neumann de un estado gaussiano fermiónico.
    Los autovalores de iC vienen en pares ±nu_k, con 0 <= nu_k <= 1.
    """
    vals = np.linalg.eigvalsh(1j * antisymmetrize(C))
    vals_abs = np.sort(np.abs(np.real(vals)))

    # Tomamos un representante por par degenerado.
    nus = vals_abs[::2]
    nus = np.clip(nus, 0.0, 1.0 - 1e-14)

    p = 0.5 * (1.0 + nus)
    q = 0.5 * (1.0 - nus)

    return float(-np.sum(p * np.log(p) + q * np.log(q)))


def gaussian_energy(A, C):
    """
    Energía E = Tr(H rho) = 1/4 Tr(A C).
    """
    return float(np.real(0.25 * np.trace(A @ antisymmetrize(C))))


def gaussian_relative_entropy_to_thermal(C, C_beta, A, beta):
    """
    S(rho || rho_beta) usando la identidad de energía libre:

        S(rho || rho_beta) =
            beta (E(rho) - E(rho_beta)) - (S(rho) - S(rho_beta))
    """
    E = gaussian_energy(A, C)
    E_beta = gaussian_energy(A, C_beta)

    S = gaussian_entropy(C)
    S_beta = gaussian_entropy(C_beta)

    Srel = beta * (E - E_beta) - (S - S_beta)
    return float(max(Srel, 0.0))


def ff_metrics(C, C_beta, A, beta):
    """
    Métricas principales frente al Gibbs gaussiano.
    """
    N = A.shape[0] // 2

    E = gaussian_energy(A, C)
    E_beta = gaussian_energy(A, C_beta)
    Srel = gaussian_relative_entropy_to_thermal(C, C_beta, A, beta)

    return {
        "E": float(E),
        "E_beta": float(E_beta),
        "E_per_site": float(E / N),
        "E_beta_per_site": float(E_beta / N),
        "bias_E": float(abs(E - E_beta)),
        "bias_E_per_site": float(abs(E - E_beta) / N),
        "cov_fro": float(np.linalg.norm(C - C_beta) / np.sqrt(C.size)),
        "sqrt_rel_entropy_bound": float(np.sqrt(0.5 * Srel)),
        "entropy": float(gaussian_entropy(C)),
        "entropy_beta": float(gaussian_entropy(C_beta)),
    }


def gaussian_filter_values(beta, h, delta, MT):
    """
    Filtro gaussiano discretizado con normalización:
        delta * sum_tau |f_tau| = 1
    """
    a = np.sqrt(4.0 * h / beta)
    taus = np.arange(-int(MT), int(MT) + 1)
    t = delta * taus

    f = np.exp(-0.5 * (a * t) ** 2)
    f = f / (delta * np.sum(np.abs(f)))

    return taus, f, a


def random_dephase_covariance(C, A, T, lam):
    """
    Randomización en la base de energía de iA.

    Implementa la idea de la Eq. (A5) del paper:
        C_ab -> C_ab / (1 - i lambda T omega_ab)

    En la representación Majorana usamos omega_ab = eps_a + eps_b.
    """
    if lam is None or lam <= 0:
        return antisymmetrize(C)

    w, V = np.linalg.eigh(1j * A)

    # Transformación congruente al basis de modos.
    C_e = V.conj().T @ C @ V.conj()

    omega = w[:, None] + w[None, :]
    denom = 1.0 - 1j * float(lam) * float(T) * omega

    C_e = C_e / denom

    C_back = V @ C_e @ V.T
    C_back = np.real_if_close(C_back, tol=1000).real

    return antisymmetrize(C_back)


def prepare_ff_cycle_blocks(
    N,
    J=1.0,
    g=1.0,
    beta=1.0,
    theta=0.05,
    delta=0.08,
    MT=None,
    h=None,
    T_factor=4.0,
):
    """
    Precalcula el mapa afín de un ciclo:

        C -> O_SS C O_SS^T + O_SB C_B O_SB^T

    Esto evita aplicar todos los pasos de tiempo en cada ciclo.
    """
    N = int(N)

    if h is None:
        h = 2.0 * max(float(g), float(J))

    A_S = majorana_chain_A(N, J=J, g=g)
    A_B = bath_majorana_A(N, h=h)
    A_coup = coupling_majorana_A(N)

    a = np.sqrt(4.0 * h / beta)

    if MT is None:
        T = float(T_factor) / a
        MT = max(int(round(T / float(delta))), 1)
    else:
        MT = int(MT)

    T_eff = MT * float(delta)

    taus, fvals, a = gaussian_filter_values(beta=beta, h=h, delta=delta, MT=MT)

    A_free = block_diag(A_S, A_B)

    dim_total = 4 * N
    O_cycle = np.eye(dim_total)

    for f_tau in fvals:
        A_tau = A_free + float(theta) * float(f_tau) * A_coup
        O_tau = expm(float(delta) * A_tau)
        O_cycle = O_tau @ O_cycle

    dim_s = 2 * N

    O_SS = O_cycle[:dim_s, :dim_s]
    O_SB = O_cycle[:dim_s, dim_s:]

    # Bath reset a su estado fundamental aproximado.
    C_B_reset = thermal_covariance_majorana(A_B, beta=80.0 / max(h, 1e-12))

    Y = O_SB @ C_B_reset @ O_SB.T

    return {
        "A_S": A_S,
        "A_B": A_B,
        "O_SS": O_SS,
        "Y": antisymmetrize(Y),
        "N": N,
        "J": float(J),
        "g": float(g),
        "h": float(h),
        "beta": float(beta),
        "theta": float(theta),
        "delta": float(delta),
        "MT": int(MT),
        "T_eff": float(T_eff),
        "a": float(a),
    }


def apply_ff_cycle(C, blocks, randomization_lambda=0.0):
    """
    Aplica un ciclo de cooling al estado gaussiano del sistema.
    """
    C_next = blocks["O_SS"] @ C @ blocks["O_SS"].T + blocks["Y"]
    C_next = antisymmetrize(C_next)

    C_next = random_dephase_covariance(
        C_next,
        A=blocks["A_S"],
        T=blocks["T_eff"],
        lam=randomization_lambda,
    )

    return antisymmetrize(C_next)


def run_ff_protocol_case(
    N=40,
    J=1.0,
    g=1.0,
    beta=1.0,
    theta=0.05,
    delta=0.08,
    MT=None,
    h=None,
    T_factor=4.0,
    n_cycles=600,
    randomization_lambda=2.0,
    store_every=10,
):
    """
    Ejecuta el protocolo free-fermion gaussiano desde temperatura infinita:
        C_0 = 0

    Devuelve:
        - C_final
        - C_beta
        - tabla de trayectoria
        - métricas finales
    """
    blocks = prepare_ff_cycle_blocks(
        N=N,
        J=J,
        g=g,
        beta=beta,
        theta=theta,
        delta=delta,
        MT=MT,
        h=h,
        T_factor=T_factor,
    )

    A_S = blocks["A_S"]
    C_beta = thermal_covariance_majorana(A_S, beta=beta)

    C = np.zeros_like(A_S)

    rows = []
    for n in range(int(n_cycles) + 1):
        if n % int(store_every) == 0 or n == int(n_cycles):
            m = ff_metrics(C, C_beta, A_S, beta)
            rows.append({
                "cycle": int(n),
                **m,
            })

        if n < int(n_cycles):
            C = apply_ff_cycle(
                C,
                blocks,
                randomization_lambda=randomization_lambda,
            )

    df_traj = pd.DataFrame(rows)

    final_metrics = ff_metrics(C, C_beta, A_S, beta)
    final_metrics.update({
        "N": int(N),
        "J": float(J),
        "g": float(g),
        "beta": float(beta),
        "theta": float(theta),
        "theta2": float(theta) ** 2,
        "delta": float(delta),
        "MT": int(blocks["MT"]),
        "T_eff": float(blocks["T_eff"]),
        "h": float(blocks["h"]),
        "lambda": float(randomization_lambda),
        "n_cycles": int(n_cycles),
    })

    return {
        "C_final": C,
        "C_beta": C_beta,
        "A_S": A_S,
        "blocks": blocks,
        "traj": df_traj,
        "metrics": final_metrics,
    }
    
# A. — Helpers corregidos: aplicación rápida y convergencia estacionaria

def prepare_ff_cycle_blocks_cached(
    N,
    J=1.0,
    g=1.0,
    beta=1.0,
    theta=0.05,
    delta=np.pi / 80,
    MT=None,
    h=None,
    T_factor=10.0,
    randomization_lambda=2.0,
):
    """
    Prepara bloques de un ciclo y cachea la diagonalización usada por la randomización.

    En el paper se fija normalmente:
        h = 2 max(g,J)
        T = 10 / a
        a = sqrt(4h/beta)

    En el notebook elegimos T mediante T_factor/a y después:
        MT = round(T/delta)
    """
    blocks = prepare_ff_cycle_blocks(
        N=N,
        J=J,
        g=g,
        beta=beta,
        theta=theta,
        delta=delta,
        MT=MT,
        h=h,
        T_factor=T_factor,
    )

    A = blocks["A_S"]
    w, V = np.linalg.eigh(1j * A)

    blocks["eig_w"] = w
    blocks["eig_V"] = V
    blocks["randomization_lambda"] = float(randomization_lambda)

    if randomization_lambda is not None and randomization_lambda > 0:
        omega = w[:, None] + w[None, :]
        blocks["dephase_denom"] = 1.0 - 1j * float(randomization_lambda) * blocks["T_eff"] * omega
    else:
        blocks["dephase_denom"] = None

    return blocks


def dephase_covariance_cached(C, blocks):
    """
    Randomización/dephasing usando la diagonalización cacheada.
    """
    lam = blocks.get("randomization_lambda", 0.0)
    if lam is None or lam <= 0:
        return antisymmetrize(C)

    V = blocks["eig_V"]
    denom = blocks["dephase_denom"]

    C_e = V.conj().T @ C @ V.conj()
    C_e = C_e / denom
    C_back = V @ C_e @ V.T
    C_back = np.real_if_close(C_back, tol=1000).real

    return antisymmetrize(C_back)


def apply_ff_cycle_cached(C, blocks):
    """
    Ciclo rápido:
        C -> dephase( O_SS C O_SS^T + Y )
    """
    C_next = blocks["O_SS"] @ C @ blocks["O_SS"].T + blocks["Y"]
    C_next = antisymmetrize(C_next)
    C_next = dephase_covariance_cached(C_next, blocks)
    return antisymmetrize(C_next)


def run_ff_protocol_to_stationary(
    N=32,
    J=1.0,
    g=1.0,
    beta=1.0,
    theta=0.05,
    delta=np.pi / 80,
    MT=None,
    h=None,
    T_factor=10.0,
    randomization_lambda=2.0,
    max_cycles=20000,
    min_cycles=500,
    check_every=100,
    tol_step=1e-10,
    tol_metric=1e-7,
    store_trajectory=True,
):
    """
    Ejecuta hasta convergencia aproximada del estado gaussiano.

    Esto es lo que hay que usar para A.10.3:
    comparar errores estacionarios, no errores a número fijo de ciclos.
    """
    blocks = prepare_ff_cycle_blocks_cached(
        N=N,
        J=J,
        g=g,
        beta=beta,
        theta=theta,
        delta=delta,
        MT=MT,
        h=h,
        T_factor=T_factor,
        randomization_lambda=randomization_lambda,
    )

    A_S = blocks["A_S"]
    C_beta = thermal_covariance_majorana(A_S, beta=beta)

    C = np.zeros_like(A_S)
    rows = []
    prev_cov_err = None
    converged = False

    for n in range(int(max_cycles) + 1):
        if n % int(check_every) == 0 or n == int(max_cycles):
            m = ff_metrics(C, C_beta, A_S, beta)
            rows.append({"cycle": int(n), **m})

            cov_err = m["cov_fro"]
            if prev_cov_err is not None:
                rel_change = abs(cov_err - prev_cov_err) / max(abs(prev_cov_err), 1e-15)
            else:
                rel_change = np.inf

            if n >= int(min_cycles):
                C_next_test = apply_ff_cycle_cached(C, blocks)
                step = np.linalg.norm(C_next_test - C) / np.sqrt(C.size)

                if step < tol_step and rel_change < tol_metric:
                    converged = True
                    break

            prev_cov_err = cov_err

        if n < int(max_cycles):
            C = apply_ff_cycle_cached(C, blocks)

    final_metrics = ff_metrics(C, C_beta, A_S, beta)
    final_metrics.update({
        "N": int(N),
        "J": float(J),
        "g": float(g),
        "beta": float(beta),
        "theta": float(theta),
        "theta2": float(theta) ** 2,
        "delta": float(delta),
        "MT": int(blocks["MT"]),
        "T_eff": float(blocks["T_eff"]),
        "T_factor": float(T_factor),
        "a": float(blocks["a"]),
        "h": float(blocks["h"]),
        "lambda": float(randomization_lambda),
        "n_cycles_used": int(n),
        "converged": bool(converged),
    })

    return {
        "C_final": C,
        "C_beta": C_beta,
        "A_S": A_S,
        "blocks": blocks,
        "traj": pd.DataFrame(rows),
        "metrics": final_metrics,
    }


def scaled_max_cycles(theta, theta_ref=0.10, cycles_ref=1200, max_cap=40000):
    """
    Como la mezcla se ralentiza aproximadamente como 1/theta^2,
    damos más ciclos a theta pequeño.
    """
    return int(min(max_cap, max(cycles_ref, cycles_ref * (theta_ref / float(theta)) ** 2)))


def loglog_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return np.nan
    p = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(p[0])

# — Punto fijo afín del mapa gaussiano
#
# El mapa gaussiano tiene la forma:
#     C -> L(C) + b
#
# En vez de iterar miles de ciclos, resolvemos:
#     (I - L) vec(C*) = vec(b)
#
# Esto es mucho mejor para estudiar el escalado estacionario con θ.

from scipy.sparse.linalg import LinearOperator, lgmres


def ff_affine_fixed_point(
    blocks,
    randomization_lambda=2.0,
    rtol=1e-10,
    atol=1e-12,
    maxiter=800,
):
    """
    Calcula el punto fijo C* del mapa gaussiano:
        C_{n+1} = dephase(O C_n O^T + Y)

    usando un solver iterativo sobre vec(C).

    Requiere:
        prepare_ff_cycle_blocks_cached(...)
        dephase_covariance_cached(...)
    definidos en A.10.0.
    """
    d = blocks["A_S"].shape[0]

    blocks = dict(blocks)
    blocks["randomization_lambda"] = float(randomization_lambda)

    # Recalcular denominador de dephasing si hace falta.
    A = blocks["A_S"]
    w, V = np.linalg.eigh(1j * A)
    blocks["eig_w"] = w
    blocks["eig_V"] = V

    if randomization_lambda is not None and randomization_lambda > 0:
        omega = w[:, None] + w[None, :]
        blocks["dephase_denom"] = 1.0 - 1j * float(randomization_lambda) * blocks["T_eff"] * omega
    else:
        blocks["dephase_denom"] = None

    def L_of_C(C):
        C = C.reshape(d, d)
        C = antisymmetrize(C)
        C_next = blocks["O_SS"] @ C @ blocks["O_SS"].T
        C_next = antisymmetrize(C_next)
        C_next = dephase_covariance_cached(C_next, blocks)
        return antisymmetrize(C_next)

    # b = map(0)
    b = dephase_covariance_cached(blocks["Y"], blocks)
    b_vec = b.reshape(-1)

    def matvec(v):
        C = v.reshape(d, d)
        LC = L_of_C(C)
        return (C - LC).reshape(-1)

    Aop = LinearOperator(
        shape=(d * d, d * d),
        matvec=matvec,
        dtype=float,
    )

    x0 = np.zeros(d * d, dtype=float)

    sol, info = lgmres(
        Aop,
        b_vec,
        x0=x0,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
    )

    C_fp = sol.reshape(d, d)
    C_fp = antisymmetrize(C_fp)

    residual = np.linalg.norm(matvec(sol) - b_vec) / max(np.linalg.norm(b_vec), 1e-15)

    return C_fp, {
        "solver_info": int(info),
        "residual": float(residual),
    }


def run_ff_fixed_point_case(
    N=32,
    J=0.7,
    g=1.0,
    beta=1.0,
    theta=0.08,
    delta=np.pi / 80,
    T_factor=15.0,
    h=None,
    randomization_lambda=2.0,
    rtol=1e-10,
    maxiter=800,
):
    """
    Caso estacionario para A.10.3.
    """
    blocks = prepare_ff_cycle_blocks_cached(
        N=N,
        J=J,
        g=g,
        beta=beta,
        theta=theta,
        delta=delta,
        MT=None,
        h=h,
        T_factor=T_factor,
        randomization_lambda=randomization_lambda,
    )

    C_fp, info = ff_affine_fixed_point(
        blocks,
        randomization_lambda=randomization_lambda,
        rtol=rtol,
        maxiter=maxiter,
    )

    A_S = blocks["A_S"]
    C_beta = thermal_covariance_majorana(A_S, beta=beta)

    m = ff_metrics(C_fp, C_beta, A_S, beta)
    m.update({
        "N": int(N),
        "J": float(J),
        "g": float(g),
        "beta": float(beta),
        "theta": float(theta),
        "theta2": float(theta) ** 2,
        "delta": float(delta),
        "h": float(blocks["h"]),
        "a": float(blocks["a"]),
        "T_factor": float(T_factor),
        "T_eff": float(blocks["T_eff"]),
        "MT": int(blocks["MT"]),
        "lambda": float(randomization_lambda),
        "solver_info": int(info["solver_info"]),
        "solver_residual": float(info["residual"]),
    })

    return {
        "C_final": C_fp,
        "C_beta": C_beta,
        "A_S": A_S,
        "blocks": blocks,
        "metrics": m,
    }


def fit_power_law_vs_theta2(df, ycol, floor_factor=4.0, theta_max=None):
    """
    Fit y ~ (theta^2)^p eliminando puntos dominados por el piso numérico.

    Si y está demasiado cerca del mínimo, el fit se aplana artificialmente.
    """
    dff = df.copy().sort_values("theta2")
    y = np.asarray(dff[ycol], dtype=float)
    x = np.asarray(dff["theta2"], dtype=float)

    floor = np.nanmin(y)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

    # quitar puntos demasiado cerca del piso
    mask &= y > floor_factor * floor

    if theta_max is not None:
        mask &= np.asarray(dff["theta"], dtype=float) <= float(theta_max)

    # fallback si quedan pocos
    if np.sum(mask) < 3:
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

    p = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)

    return {
        "slope": float(p[0]),
        "intercept": float(p[1]),
        "mask": mask,
        "floor": float(floor),
    }
    
    
# A.8.3.0 — Helpers para punto fijo gaussiano y ajuste con piso



def ff_affine_fixed_point(
    blocks,
    randomization_lambda=2.0,
    rtol=1e-10,
    atol=1e-12,
    maxiter=1000,
):
    """
    Calcula el punto fijo C* del mapa gaussiano

        C -> dephase(O_SS C O_SS^T + Y)

    resolviendo

        (I - L) vec(C*) = vec(b).

    Esto evita confundir error estacionario con falta de ciclos.
    """
    d = blocks["A_S"].shape[0]

    blocks = dict(blocks)
    blocks["randomization_lambda"] = float(randomization_lambda)

    A = blocks["A_S"]
    w, V = np.linalg.eigh(1j * A)

    blocks["eig_w"] = w
    blocks["eig_V"] = V

    if randomization_lambda is not None and randomization_lambda > 0:
        omega = w[:, None] + w[None, :]
        blocks["dephase_denom"] = (
            1.0 - 1j * float(randomization_lambda) * blocks["T_eff"] * omega
        )
    else:
        blocks["dephase_denom"] = None

    def L_of_C(C):
        C = C.reshape(d, d)
        C = antisymmetrize(C)

        C_next = blocks["O_SS"] @ C @ blocks["O_SS"].T
        C_next = antisymmetrize(C_next)
        C_next = dephase_covariance_cached(C_next, blocks)

        return antisymmetrize(C_next)

    # b = map(0)
    b = dephase_covariance_cached(blocks["Y"], blocks)
    b_vec = b.reshape(-1)

    def matvec(v):
        C = v.reshape(d, d)
        LC = L_of_C(C)
        return (C - LC).reshape(-1)

    Aop = LinearOperator(
        shape=(d * d, d * d),
        matvec=matvec,
        dtype=float,
    )

    x0 = np.zeros(d * d, dtype=float)

    sol, info = lgmres(
        Aop,
        b_vec,
        x0=x0,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
    )

    C_fp = sol.reshape(d, d)
    C_fp = antisymmetrize(C_fp)

    residual = np.linalg.norm(matvec(sol) - b_vec) / max(np.linalg.norm(b_vec), 1e-15)

    return C_fp, {
        "solver_info": int(info),
        "residual": float(residual),
    }


def run_ff_fixed_point_case(
    N=28,
    J=0.7,
    g=1.0,
    beta=1.0,
    theta=0.08,
    delta=np.pi / 80,
    T_factor=15.0,
    h=None,
    randomization_lambda=2.0,
    rtol=2e-10,
    maxiter=1000,
):
    """
    Ejecuta un caso estacionario para el protocolo gaussiano.
    """
    blocks = prepare_ff_cycle_blocks_cached(
        N=N,
        J=J,
        g=g,
        beta=beta,
        theta=theta,
        delta=delta,
        MT=None,
        h=h,
        T_factor=T_factor,
        randomization_lambda=randomization_lambda,
    )

    C_fp, info = ff_affine_fixed_point(
        blocks,
        randomization_lambda=randomization_lambda,
        rtol=rtol,
        maxiter=maxiter,
    )

    A_S = blocks["A_S"]
    C_beta = thermal_covariance_majorana(A_S, beta=beta)

    metrics = ff_metrics(C_fp, C_beta, A_S, beta)

    metrics.update({
        "N": int(N),
        "J": float(J),
        "g": float(g),
        "beta": float(beta),
        "theta": float(theta),
        "theta2": float(theta) ** 2,
        "delta": float(delta),
        "h": float(blocks["h"]),
        "a": float(blocks["a"]),
        "T_factor": float(T_factor),
        "T_eff": float(blocks["T_eff"]),
        "MT": int(blocks["MT"]),
        "lambda": float(randomization_lambda),
        "solver_info": int(info["solver_info"]),
        "solver_residual": float(info["residual"]),
    })

    return {
        "C_final": C_fp,
        "C_beta": C_beta,
        "A_S": A_S,
        "blocks": blocks,
        "metrics": metrics,
    }


def offset_power_law(x, y0, A, p):
    """
    Modelo:
        y = y0 + A x^p
    con x = theta^2.
    """
    return y0 + A * x**p


def fit_offset_power_law(df, ycol):
    """
    Ajusta y(theta^2) = y0 + A (theta^2)^p.
    """
    x = np.asarray(df["theta2"], dtype=float)
    y = np.asarray(df[ycol], dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    y0_guess = 0.8 * np.min(y)
    A_guess = max((np.max(y) - np.min(y)) / max(np.max(x), 1e-15), 1e-15)
    p_guess = 1.0

    popt, pcov = curve_fit(
        offset_power_law,
        x,
        y,
        p0=[y0_guess, A_guess, p_guess],
        bounds=(
            [0.0, 0.0, 0.0],
            [np.min(y) * 1.05, np.inf, 4.0],
        ),
        maxfev=20000,
    )

    y0, A, p = popt

    y_corr = y - y0
    mask_corr = y_corr > 0

    if np.sum(mask_corr) >= 2:
        p_corr = np.polyfit(
            np.log(x[mask_corr]),
            np.log(y_corr[mask_corr]),
            1,
        )[0]
    else:
        p_corr = np.nan

    return {
        "x": x,
        "y": y,
        "y0": float(y0),
        "A": float(A),
        "p_offset_model": float(p),
        "p_after_floor_subtraction": float(p_corr),
        "y_corr": y_corr,
        "mask_corr": mask_corr,
    }
    

def covariance_error_energy_basis_fig6a(out, subtract_population_blocks=True):
    """
    Devuelve puntos para una figura estilo Fig. 6a del paper.

    x = omega_ab / omega_T
    y = |Delta C_ab|

    Delta C = C_sigma - C_beta en la base de energía de iA.

    Si subtract_population_blocks=True:
        se eliminan los elementos estrictamente "diagonales" de la matriz
        en la base de energía para reducir el peso visual del sector poblacional.
        El sector cerca de omega=0 puede seguir presente porque hay pares
        casi degenerados y bloques Majorana.
    """
    A = out["A_S"]
    C = out["C_final"]
    Cb = out["C_beta"]
    T_eff = out["blocks"]["T_eff"]

    w, V = np.linalg.eigh(1j * A)

    Delta = C - Cb
    Delta_e = V.conj().T @ Delta @ V.conj()

    if subtract_population_blocks:
        Delta_e = Delta_e.copy()
        np.fill_diagonal(Delta_e, 0.0)

    # En nuestra convención de covarianzas, la frecuencia relevante
    # para un elemento C_ab es aproximadamente w_a + w_b.
    omega = w[:, None] + w[None, :]
    omega_T = 2.0 * np.pi / T_eff

    x = np.real(omega / omega_T).ravel()
    y = np.abs(Delta_e).ravel()

    mask = np.isfinite(x) & np.isfinite(y) & (y > 1e-15)

    return x[mask], y[mask]


def qiskit_label_from_ops_B(ops, n_qubits):
    """
    Qiskit usa etiquetas Pauli con orden q_{n-1} ... q_0.
    ops usa índices lógicos q -> 'X','Y','Z'.
    """
    chars = ["I"] * n_qubits

    for q, p in ops.items():
        chars[n_qubits - 1 - int(q)] = p

    return "".join(chars)


def jw_majorana_ops_B(j, kind):
    """
    Majoranas JW:
        gamma_{2j}   = Z_0 ... Z_{j-1} X_j
        gamma_{2j+1} = Z_0 ... Z_{j-1} Y_j
    """
    ops = {k: "Z" for k in range(j)}

    if kind == "X":
        ops[j] = "X"
    elif kind == "Y":
        ops[j] = "Y"
    else:
        raise ValueError("kind debe ser 'X' o 'Y'.")

    return ops


def free_fermion_system_terms_B(N, J=1.0, g=1.0):
    """
    Hamiltoniano free-fermion del paper en JW, hasta convención de signo:

        H_S ≈ -g/2 sum Z_j + J/2 sum X_j X_{j+1}

    Para recursos, los signos no afectan al conteo.
    """
    terms = []

    for j in range(N):
        terms.append((-0.5 * g, {j: "Z"}))

    for j in range(N - 1):
        terms.append((0.5 * J, {j: "X", j + 1: "X"}))

    return terms


def free_fermion_bath_terms_B(N, h=2.0):
    """
    Un bath por sitio, colocado en qubits N ... 2N-1.
    """
    terms = []

    for j in range(N):
        b = N + j
        terms.append((-0.5 * h, {b: "Z"}))

    return terms


def free_fermion_coupling_terms_B(N, theta=0.05):
    """
    A_s = (gamma_{2s} - gamma_{2s+1})/sqrt(2)
    acoplado a Y_b.

    Bajo JW, A_s contiene strings de Z crecientes. Esto es justo lo
    que encarece la implementación Qiskit.
    """
    terms = []

    for j in range(N):
        b = N + j

        ops_x = jw_majorana_ops_B(j, "X")
        ops_x[b] = "Y"

        ops_y = jw_majorana_ops_B(j, "Y")
        ops_y[b] = "Y"

        terms.append((theta / np.sqrt(2.0), ops_x))
        terms.append((-theta / np.sqrt(2.0), ops_y))

    return terms


def sparse_pauli_from_terms_B(terms, n_qubits):
    paulis = []

    for coeff, ops in terms:
        paulis.append(
            (qiskit_label_from_ops_B(ops, n_qubits), complex(coeff))
        )

    return SparsePauliOp.from_list(paulis)


def append_pauli_evolution_B(qc, terms, n_qubits, time_value):
    if len(terms) == 0:
        return

    op = sparse_pauli_from_terms_B(terms, n_qubits)

    gate = PauliEvolutionGate(
        op,
        time=float(time_value),
        synthesis=SuzukiTrotter(order=1, reps=1),
    )

    qc.append(gate, range(n_qubits))


def build_free_fermion_protocol_resource_circuit_B(
    N,
    J=1.0,
    g=1.0,
    h=2.0,
    beta=1.0,
    theta=0.05,
    delta=np.pi / 40.0,
    MT=2,
    n_cycles=1,
    measure_bath=True,
):
    """
    Circuito de recursos para un protocolo free-fermion JW.

    No pretende ser una simulación física completa de A.10.
    Sirve para contar coste Qiskit de implementar:
        H_S + H_B + A_s ⊗ Y_b
    con strings JW.
    """
    n_sys = int(N)
    n_bath = int(N)
    n_total = n_sys + n_bath

    n_clbits = n_bath * n_cycles if measure_bath else 0
    qc = QuantumCircuit(n_total, n_clbits)

    system_terms = free_fermion_system_terms_B(N, J=J, g=g)
    bath_terms = free_fermion_bath_terms_B(N, h=h)
    coupling_terms = free_fermion_coupling_terms_B(N, theta=theta)

    # Preparación aproximada de bath térmico como rotación pura.
    # Para recursos, lo relevante es que hay coste local de preparación.
    p1 = 1.0 / (1.0 + np.exp(beta * h))
    angle = 2.0 * np.arcsin(np.sqrt(p1))

    for r in range(n_cycles):
        for j in range(N):
            qc.ry(angle, N + j)

        # Bloque trotterizado: varias capas de filtro.
        for _ in range(2 * MT + 1):
            append_pauli_evolution_B(qc, system_terms, n_total, delta)
            append_pauli_evolution_B(qc, bath_terms, n_total, delta)
            append_pauli_evolution_B(qc, coupling_terms, n_total, delta)

        if measure_bath:
            for j in range(N):
                qc.measure(N + j, r * N + j)
                qc.reset(N + j)

    return qc


def safe_transpile_B(qc, backend, optimization_level=1):
    try:
        return transpile(
            qc,
            backend=backend,
            optimization_level=optimization_level,
        )
    except Exception as exc:
        print("Transpile estándar falló; pruebo con decompose + optimization_level=0")
        print(type(exc).__name__, exc)

        return transpile(
            qc.decompose(reps=4),
            backend=backend,
            optimization_level=0,
        )


# C.0.0 — Helpers de geometría hardware en FakeSherbrooke




def load_fake_sherbrooke_C0():
    """
    Carga FakeSherbrooke en versiones nuevas o antiguas de Qiskit.
    """
    try:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        return FakeSherbrooke()
    except Exception:
        try:
            from qiskit.providers.fake_provider import FakeSherbrooke
            return FakeSherbrooke()
        except Exception as exc:
            raise ImportError(
                "No encuentro FakeSherbrooke. Revisa qiskit-ibm-runtime/qiskit."
            ) from exc


def backend_name_C0(backend):
    name = getattr(backend, "name", None)
    return name() if callable(name) else str(name)


def backend_num_qubits_C0(backend):
    if hasattr(backend, "num_qubits"):
        return int(backend.num_qubits)
    return int(backend.configuration().n_qubits)


def coupling_edges_C0(backend):
    """
    Extrae coupling map del backend y lo convierte en aristas no dirigidas.
    """
    edges = []

    if hasattr(backend, "target"):
        try:
            cmap = backend.target.build_coupling_map()
            if cmap is not None:
                edges = list(cmap.get_edges())
        except Exception:
            pass

    if len(edges) == 0 and hasattr(backend, "coupling_map"):
        try:
            edges = list(backend.coupling_map.get_edges())
        except Exception:
            pass

    if len(edges) == 0:
        try:
            edges = list(backend.configuration().coupling_map)
        except Exception as exc:
            raise RuntimeError("No pude extraer coupling_map de FakeSherbrooke.") from exc

    undirected = sorted({tuple(sorted((int(a), int(b)))) for a, b in edges if a != b})
    return undirected


def backend_graph_C0(backend):
    n = backend_num_qubits_C0(backend)
    edges = coupling_edges_C0(backend)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    return G


def shortest_path_edges_C0(path):
    return list(zip(path[:-1], path[1:]))


def canonical_path_C0(path):
    path = tuple(map(int, path))
    rev = tuple(reversed(path))
    return min(path, rev)


def find_simple_chains_C0(G, n_sys):
    """
    Encuentra cadenas simples de longitud n_sys.
    """
    chains = set()

    for start in G.nodes:
        stack = [(int(start), [int(start)])]

        while stack:
            current, path = stack.pop()

            if len(path) == n_sys:
                chains.add(canonical_path_C0(path))
                continue

            for nb in G.neighbors(current):
                nb = int(nb)
                if nb not in path:
                    stack.append((nb, path + [nb]))

    return sorted(chains)


def connected_sets_C0(G, size):
    """
    Genera conjuntos conectados de tamaño 'size'.
    """
    sets = {frozenset([int(n)]) for n in G.nodes}

    for _ in range(size - 1):
        new_sets = set()

        for S in sets:
            neigh = set()
            for q in S:
                neigh.update(int(nb) for nb in G.neighbors(q))

            for nb in neigh:
                if nb not in S:
                    new_sets.add(frozenset(set(S) | {nb}))

        sets = new_sets

    return sorted([tuple(sorted(S)) for S in sets])


def assign_baths_C0(G, system_qubits, max_distance=2, max_candidates_per_site=8):
    """
    Asigna un bath distinto a cada qubit de sistema.
    Prioriza baths cercanos.
    """
    system_qubits = list(map(int, system_qubits))
    system_set = set(system_qubits)

    candidates_per_site = []

    for s in system_qubits:
        lengths = nx.single_source_shortest_path_length(
            G,
            s,
            cutoff=max_distance,
        )

        candidates = [
            (int(q), int(d))
            for q, d in lengths.items()
            if q not in system_set and d > 0 and d <= max_distance
        ]

        if len(candidates) == 0:
            return None

        # Orden: menor distancia, luego mayor grado para no bloquear demasiado.
        candidates = sorted(
            candidates,
            key=lambda x: (x[1], -G.degree[x[0]], x[0]),
        )[:max_candidates_per_site]

        candidates_per_site.append(candidates)

    best = None

    for combo in itertools.product(*candidates_per_site):
        baths = [q for q, d in combo]

        if len(set(baths)) != len(baths):
            continue

        distances = np.array([d for q, d in combo], dtype=int)
        swaps = int(np.sum(np.maximum(distances - 1, 0)))

        score = (
            int(np.max(distances)),
            swaps,
            float(np.mean(distances)),
        )

        if best is None or score < best["score"]:
            best = {
                "bath_qubits": tuple(map(int, baths)),
                "bath_map_by_site": {
                    i: int(b)
                    for i, b in enumerate(baths)
                },
                "distances": tuple(map(int, distances)),
                "max_distance": int(np.max(distances)),
                "mean_distance": float(np.mean(distances)),
                "total_swaps_one_way": swaps,
                "score": score,
            }

    return best


def make_embedding_C0(case, system_qubits, logical_edges, bath_assignment):
    system_qubits = list(map(int, system_qubits))
    bath_qubits = [
        int(bath_assignment["bath_map_by_site"][i])
        for i in range(len(system_qubits))
    ]

    native_logical_edges = sum(
        G_sherbrooke_C0.has_edge(system_qubits[i], system_qubits[j])
        for i, j in logical_edges
    )

    return {
        "case": case,
        "system_qubits": system_qubits,
        "bath_qubits": bath_qubits,
        "bath_map_by_site": bath_assignment["bath_map_by_site"],
        "logical_edges": list(logical_edges),
        "bath_distances": bath_assignment["distances"],
        "max_bath_distance": bath_assignment["max_distance"],
        "mean_bath_distance": bath_assignment["mean_distance"],
        "total_swaps_one_way": bath_assignment["total_swaps_one_way"],
        "native_logical_edges": int(native_logical_edges),
    }


def find_1d_embedding_C0(G, n_sys=4):
    """
    Busca una cadena 1D con bath por sitio.
    """
    logical_edges = [(i, i + 1) for i in range(n_sys - 1)]
    chains = find_simple_chains_C0(G, n_sys=n_sys)

    best = None

    for max_distance in [1, 2]:
        for chain in chains:
            bath_assignment = assign_baths_C0(
                G,
                chain,
                max_distance=max_distance,
            )

            if bath_assignment is None:
                continue

            emb = make_embedding_C0(
                case=f"Ising 1D N={n_sys}",
                system_qubits=chain,
                logical_edges=logical_edges,
                bath_assignment=bath_assignment,
            )

            score = (
                emb["max_bath_distance"],
                emb["total_swaps_one_way"],
                emb["mean_bath_distance"],
            )

            if best is None or score < best["score"]:
                emb["score"] = score
                best = emb

        if best is not None and best["max_bath_distance"] == 1:
            break

    if best is None:
        raise RuntimeError("No encontré embedding 1D sistema-bath.")

    return best


def find_2d2x2_embedding_C0(G):
    """
    Busca un embedding para 2x2 lógico:

        0 -- 1
        |    |
        2 -- 3

    En heavy-hex no siempre existe un cuadrado nativo perfecto. Por eso
    maximizamos el número de aristas lógicas nativas y luego minimizamos
    distancia sistema-bath.
    """
    logical_edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    patches = connected_sets_C0(G, size=4)

    best = None

    for patch in patches:
        for perm in itertools.permutations(patch):
            native_edges = sum(
                G.has_edge(perm[i], perm[j])
                for i, j in logical_edges
            )

            bath_assignment = assign_baths_C0(
                G,
                perm,
                max_distance=2,
            )

            if bath_assignment is None:
                continue

            emb = make_embedding_C0(
                case="Ising 2D 2x2",
                system_qubits=perm,
                logical_edges=logical_edges,
                bath_assignment=bath_assignment,
            )

            score = (
                -native_edges,
                emb["max_bath_distance"],
                emb["total_swaps_one_way"],
                emb["mean_bath_distance"],
            )

            if best is None or score < best["score"]:
                emb["score"] = score
                best = emb

    if best is None:
        raise RuntimeError("No encontré embedding 2D 2x2 sistema-bath.")

    return best


def embedding_summary_C0(embedding):
    n_sys = len(embedding["system_qubits"])
    direct_baths = sum(d == 1 for d in embedding["bath_distances"])

    return {
        "caso": embedding["case"],
        "system_qubits": tuple(embedding["system_qubits"]),
        "bath_qubits": tuple(embedding["bath_qubits"]),
        "logical_edges_native": embedding["native_logical_edges"],
        "logical_edges_total": len(embedding["logical_edges"]),
        "bath_distances": tuple(embedding["bath_distances"]),
        "direct_baths": f"{direct_baths}/{n_sys}",
        "max_bath_distance": embedding["max_bath_distance"],
        "total_swaps_one_way": embedding["total_swaps_one_way"],
    }


def nodes_and_paths_C0(G, embedding):
    """
    Nodos y caminos relevantes para el zoom.
    """
    sys_nodes = embedding["system_qubits"]
    bath_nodes = embedding["bath_qubits"]
    logical_edges = embedding["logical_edges"]

    nodes = set(sys_nodes) | set(bath_nodes)
    path_edges = []

    # Caminos de acoplo lógico del sistema.
    for i, j in logical_edges:
        a = sys_nodes[i]
        b = sys_nodes[j]
        path = nx.shortest_path(G, a, b)
        nodes.update(path)
        path_edges.extend(shortest_path_edges_C0(path))

    # Caminos sistema-bath.
    for i, s in enumerate(sys_nodes):
        b = embedding["bath_map_by_site"][i]
        path = nx.shortest_path(G, s, b)
        nodes.update(path)
        path_edges.extend(shortest_path_edges_C0(path))

    # Añadimos vecinos inmediatos para que el zoom no quede flotando.
    for q in list(nodes):
        nodes.update(G.neighbors(q))

    return nodes, path_edges


def draw_embedding_C0(
    ax,
    G,
    embedding,
    pos,
    full_chip=True,
    title="",
):
    system_nodes = embedding["system_qubits"]
    bath_nodes = embedding["bath_qubits"]
    logical_edges = embedding["logical_edges"]

    if full_chip:
        H = G
    else:
        nodes, _ = nodes_and_paths_C0(G, embedding)
        H = G.subgraph(sorted(nodes)).copy()

    # Base chip
    nx.draw_networkx_edges(
        H,
        pos,
        ax=ax,
        edge_color="lightgray",
        width=0.7 if full_chip else 1.0,
        alpha=0.35 if full_chip else 0.65,
    )

    other_nodes = [
        q for q in H.nodes
        if q not in system_nodes and q not in bath_nodes
    ]

    nx.draw_networkx_nodes(
        H,
        pos,
        nodelist=other_nodes,
        ax=ax,
        node_color="lightgray",
        node_size=22 if full_chip else 70,
        alpha=0.65,
    )

    # Caminos/acoplos lógicos del sistema
    for i, j in logical_edges:
        a = system_nodes[i]
        b = system_nodes[j]
        path = nx.shortest_path(G, a, b)
        edges = shortest_path_edges_C0(path)

        nx.draw_networkx_edges(
            H,
            pos,
            edgelist=edges,
            ax=ax,
            edge_color="tab:green",
            width=2.4 if full_chip else 3.0,
            alpha=0.95,
            style="-" if len(path) == 2 else "--",
        )

    # Caminos sistema-bath
    for i, s in enumerate(system_nodes):
        b = embedding["bath_map_by_site"][i]
        path = nx.shortest_path(G, s, b)
        edges = shortest_path_edges_C0(path)

        nx.draw_networkx_edges(
            H,
            pos,
            edgelist=edges,
            ax=ax,
            edge_color="tab:orange",
            width=2.1 if full_chip else 2.8,
            alpha=0.95,
            style="-" if len(path) == 2 else "--",
        )

    # Sistema
    nx.draw_networkx_nodes(
        H,
        pos,
        nodelist=system_nodes,
        ax=ax,
        node_color="tab:green",
        node_size=180 if full_chip else 460,
        edgecolors="black",
        linewidths=0.8,
        label="sistema",
    )

    # Baths
    nx.draw_networkx_nodes(
        H,
        pos,
        nodelist=bath_nodes,
        ax=ax,
        node_color="tab:orange",
        node_size=150 if full_chip else 400,
        edgecolors="black",
        linewidths=0.8,
        label="baths",
    )

    labels = {}
    for i, q in enumerate(system_nodes):
        labels[q] = f"S{i}\nq{q}" if not full_chip else f"S{i}"
    for i, q in enumerate(bath_nodes):
        labels[q] = f"B{i}\nq{q}" if not full_chip else f"B{i}"

    nx.draw_networkx_labels(
        H,
        pos,
        labels=labels,
        ax=ax,
        font_size=7 if full_chip else 8,
    )

    ax.set_title(title)
    ax.axis("off")


def make_zoom_pos_C0(G, embedding):
    nodes, _ = nodes_and_paths_C0(G, embedding)
    H = G.subgraph(sorted(nodes)).copy()
    return nx.spring_layout(H, seed=11, iterations=200)


def make_manual_embedding_C0(G, case, system_qubits, bath_qubits, logical_edges):
    system_qubits = list(map(int, system_qubits))
    bath_qubits = list(map(int, bath_qubits))

    if len(system_qubits) != len(bath_qubits):
        raise ValueError("system_qubits y bath_qubits deben tener la misma longitud")

    bath_map_by_site = {i: bath_qubits[i] for i in range(len(system_qubits))}
    bath_distances = tuple(
        nx.shortest_path_length(G, source=s, target=b)
        for s, b in zip(system_qubits, bath_qubits)
    )

    native_logical_edges = sum(
        G.has_edge(system_qubits[i], system_qubits[j])
        for i, j in logical_edges
    )

    return {
        "case": case,
        "system_qubits": system_qubits,
        "bath_qubits": bath_qubits,
        "bath_map_by_site": bath_map_by_site,
        "logical_edges": list(logical_edges),
        "bath_distances": bath_distances,
        "max_bath_distance": int(max(bath_distances)),
        "mean_bath_distance": float(np.mean(bath_distances)),
        "total_swaps_one_way": int(sum(max(d - 1, 0) for d in bath_distances)),
        "native_logical_edges": int(native_logical_edges),
    }
    
    

def extract_ising_terms_C3(cfg):
    """
    Extrae términos Z_i y X_i X_j del Hamiltoniano Ising guardado en cfg.system_terms.
    """
    z_terms = []
    xx_terms = []

    for coef, paulis in cfg.system_terms:
        coef = float(np.real(coef))

        if len(paulis) == 1:
            i, op = next(iter(paulis.items()))
            if op == "Z":
                z_terms.append((int(i), coef))

        elif len(paulis) == 2:
            items = sorted(paulis.items())
            (i, op_i), (j, op_j) = items

            if op_i == "X" and op_j == "X":
                xx_terms.append((int(i), int(j), coef))

    return z_terms, xx_terms


def hadamard_n_C3(n):
    H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)

    H = np.array([[1.0]], dtype=complex)
    for _ in range(n):
        H = np.kron(H, H1)

    return H


def bit_eigs_C3(indices, n, endian="big"):
    """
    Devuelve eigenvalores ±1 de los bits computacionales.

    endian="big": qubit 0 es el bit más significativo.
    endian="little": qubit 0 es el bit menos significativo.

    Elegimos automáticamente el endian que reproduce mejor la energía exacta.
    """
    indices = np.asarray(indices, dtype=int)
    eigs = np.empty((len(indices), n), dtype=float)

    for q in range(n):
        if endian == "big":
            bitpos = n - 1 - q
        elif endian == "little":
            bitpos = q
        else:
            raise ValueError("endian debe ser 'big' o 'little'.")

        bits = (indices >> bitpos) & 1
        eigs[:, q] = 1.0 - 2.0 * bits

    return eigs


def exact_energy_from_probs_C3(probs_Z, probs_X, cfg, endian):
    """
    Energía exacta estimada desde distribuciones completas de medida Z y X.
    """
    n = cfg.n_sys
    z_terms, xx_terms = extract_ising_terms_C3(cfg)

    indices = np.arange(2**n)

    eig_Z = bit_eigs_C3(indices, n, endian=endian)
    eig_X = bit_eigs_C3(indices, n, endian=endian)

    E_Z = 0.0
    for i, coef in z_terms:
        E_Z += coef * np.sum(probs_Z * eig_Z[:, i])

    E_X = 0.0
    for i, j, coef in xx_terms:
        E_X += coef * np.sum(probs_X * eig_X[:, i] * eig_X[:, j])

    return float(E_Z + E_X)


def choose_endian_C3(probs_Z, probs_X, cfg, E_exact):
    """
    Escoge la convención de bits que reproduce mejor la energía exacta.
    """
    candidates = []

    for endian in ["big", "little"]:
        E_est = exact_energy_from_probs_C3(
            probs_Z=probs_Z,
            probs_X=probs_X,
            cfg=cfg,
            endian=endian,
        )
        candidates.append((abs(E_est - E_exact), endian, E_est))

    candidates = sorted(candidates, key=lambda x: x[0])
    return candidates[0][1], candidates


def energy_from_sampled_indices_C3(indices_Z, indices_X, cfg, endian):
    """
    Reconstruye energía usando muestras en base Z y base X.
    """
    n = cfg.n_sys
    z_terms, xx_terms = extract_ising_terms_C3(cfg)

    eig_Z = bit_eigs_C3(indices_Z, n, endian=endian)
    eig_X = bit_eigs_C3(indices_X, n, endian=endian)

    E_Z = 0.0
    for i, coef in z_terms:
        E_Z += coef * np.mean(eig_Z[:, i])

    E_X = 0.0
    for i, j, coef in xx_terms:
        E_X += coef * np.mean(eig_X[:, i] * eig_X[:, j])

    return float(E_Z), float(E_X), float(E_Z + E_X)


def sample_energy_shots_C3(rho, cfg, shots, n_reps=80, seed=1234):
    """
    Simula shots de medida en Z y X desde rho.
    """
    rng = np.random.default_rng(seed)
    n = cfg.n_sys

    Hn = hadamard_n_C3(n)

    probs_Z = np.real(np.diag(rho)).copy()
    probs_Z = np.maximum(probs_Z, 0.0)
    probs_Z /= np.sum(probs_Z)

    rho_X = Hn @ rho @ Hn.conj().T
    probs_X = np.real(np.diag(rho_X)).copy()
    probs_X = np.maximum(probs_X, 0.0)
    probs_X /= np.sum(probs_X)

    # Elegimos endian usando la energía exacta.
    E_exact = float(expect(out_exact_C3["Hs"], rho))
    endian, endian_candidates = choose_endian_C3(
        probs_Z=probs_Z,
        probs_X=probs_X,
        cfg=cfg,
        E_exact=E_exact,
    )

    rows = []

    for rep in range(n_reps):
        idx_Z = rng.choice(2**n, size=int(shots), replace=True, p=probs_Z)
        idx_X = rng.choice(2**n, size=int(shots), replace=True, p=probs_X)

        E_Z, E_X, E_total = energy_from_sampled_indices_C3(
            idx_Z,
            idx_X,
            cfg=cfg,
            endian=endian,
        )

        rows.append({
            "rep": rep,
            "shots_por_base": int(shots),
            "E_Z": E_Z,
            "E_X": E_X,
            "E_total": E_total,
            "endian": endian,
        })

    return pd.DataFrame(rows), endian_candidates




def count_twoq_like_C4(ops):
    """
    En circuitos lógicos puede haber unitary grande en vez de puertas CX.
    Contamos twoq si existen, pero también mostramos unitary.
    """
    return int(
        ops.get("cx", 0)
        + ops.get("ecr", 0)
        + ops.get("cz", 0)
        + ops.get("swap", 0)
    )


def circuit_resource_row_C4(qc, label, model, N_S, n_cycles, MT):
    ops = dict(qc.count_ops())

    return {
        "label": label,
        "model": model,
        "N_S": int(N_S),
        "N_total_logical": int(2 * N_S),
        "n_cycles": int(n_cycles),
        "MT": int(MT),
        "n_qubits": int(qc.num_qubits),
        "n_clbits": int(qc.num_clbits),
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "twoq_like": count_twoq_like_C4(ops),
        "unitary": int(ops.get("unitary", 0)),
        "measure": int(ops.get("measure", 0)),
        "reset": int(ops.get("reset", 0)),
        "barrier": int(ops.get("barrier", 0)),
    }


def make_params_C4_resource(Lx, Ly=1, n_cycles=2, MT=3):
    """
    Parámetros ligeros para conteo de recursos.
    """
    J = 0.5
    g = 1.0

    return dict(
        Lx=int(Lx),
        Ly=int(Ly),
        J=J,
        g=g,
        h=max(2.0 * g, 4.0 * J),
        beta=1.0,
        theta=0.25,
        delta=np.pi / 40.0,
        MT=int(MT),
        n_cycles=int(n_cycles),
        sys_A="Y",
        bath_A="Y",
        periodic=False,
        randomize=False,
        randomization_lambda=0.0,
        seed=1234,
        rewind=False,
    )
    
    

def extract_ising_terms_D(cfg):
    """Extrae términos Z_i y X_i X_j del Hamiltoniano Ising."""
    z_terms = []
    xx_terms = []

    for coef, paulis in cfg.system_terms:
        coef = float(np.real(coef))

        if len(paulis) == 1:
            i, op = next(iter(paulis.items()))
            if op == "Z":
                z_terms.append((int(i), coef))

        elif len(paulis) == 2:
            items = sorted(paulis.items())
            (i, op_i), (j, op_j) = items
            if op_i == "X" and op_j == "X":
                xx_terms.append((int(i), int(j), coef))

    return z_terms, xx_terms


def parse_counts_final_eigs_D(counts, n_cycles, n_sys, n_bath):
    """
    Devuelve eigenvalores ±1 de la medida final del sistema.

    Convención consistente con parse_dynamic_counts_ising:
    dentro del substring de sistema, qubit i se lee como sys_bits[n_sys-1-i].
    """
    n_bath_bits = int(n_cycles) * int(n_bath)
    expected_len = n_bath_bits + int(n_sys)

    eig_rows = []
    weights = []

    bath_p1 = np.zeros((int(n_cycles), int(n_bath)), dtype=float)
    n_total = sum(counts.values())

    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "").zfill(expected_len)

        sys_bits = bits[:n_sys]
        bath_bits = bits[n_sys:]

        eig = np.empty(int(n_sys), dtype=float)
        for i in range(int(n_sys)):
            bit = sys_bits[int(n_sys) - 1 - i]
            eig[i] = 1.0 if bit == "0" else -1.0

        eig_rows.append(eig)
        weights.append(float(count))

        for r in range(int(n_cycles)):
            for mu in range(int(n_bath)):
                idx = r * int(n_bath) + mu
                if bath_bits[len(bath_bits) - 1 - idx] == "1":
                    bath_p1[r, mu] += count

    eig_rows = np.asarray(eig_rows, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights /= np.sum(weights)

    bath_p1 /= n_total

    return eig_rows, weights, bath_p1


def run_ising_total_energy_shots_D(
    compiled,
    n_cycles,
    shots=1024,
    init_mode="mm",
    sim_mode="ideal",
    seed_base=1234,
    optimization_level=0,
    backend_bundle=None,
):
    """
    Ejecuta dos pasadas del circuito dinámico:
      - final_basis='Z'
      - final_basis='X'

    Devuelve energía total Ising y señal del baño.
    """
    cfg = compiled["cfg"]

    if backend_bundle is None:
        backend_bundle = get_ising_backend_bundle(
            min_total_qubits=2 * cfg.n_sys,
        )

    if sim_mode == "ideal":
        sim = backend_bundle["ideal_sim"]
    elif sim_mode == "noisy":
        if not backend_bundle["have_noise_model"]:
            raise RuntimeError("No hay noise model disponible para sim_mode='noisy'.")
        sim = backend_bundle["noisy_sim"]
    else:
        raise ValueError("sim_mode debe ser 'ideal' o 'noisy'.")

    z_terms, xx_terms = extract_ising_terms_D(cfg)

    rng = np.random.default_rng(seed_base)
    plan = mixture_plan_ising(init_mode, shots, cfg.n_sys, rng=rng)

    z_means = np.zeros(cfg.n_sys, dtype=float)
    xx_means = {(i, j): 0.0 for i, j, _ in xx_terms}
    bath_acc = np.zeros((int(n_cycles), cfg.n_bath), dtype=float)

    for k, (init_label, n_sh) in enumerate(plan):
        if n_sh <= 0:
            continue

        weight_init = float(n_sh) / float(shots)

        # -------------------------
        # Pasada Z
        # -------------------------
        qc_Z = get_isa_dynamic_circuit_ising(
            compiled,
            n_cycles=n_cycles,
            init_label=init_label,
            final_basis="Z",
            optimization_level=optimization_level,
            backend_bundle=backend_bundle,
        )

        res_Z = sim.run(
            qc_Z,
            shots=int(n_sh),
            seed_simulator=int(seed_base + 1000 * k + 17 * n_cycles),
        ).result()

        eig_Z, w_Z, bath_Z = parse_counts_final_eigs_D(
            res_Z.get_counts(),
            n_cycles=n_cycles,
            n_sys=cfg.n_sys,
            n_bath=cfg.n_bath,
        )

        z_means += weight_init * np.sum(eig_Z * w_Z[:, None], axis=0)
        bath_acc += weight_init * bath_Z

        # -------------------------
        # Pasada X
        # -------------------------
        qc_X = get_isa_dynamic_circuit_ising(
            compiled,
            n_cycles=n_cycles,
            init_label=init_label,
            final_basis="X",
            optimization_level=optimization_level,
            backend_bundle=backend_bundle,
        )

        res_X = sim.run(
            qc_X,
            shots=int(n_sh),
            seed_simulator=int(seed_base + 1000 * k + 29 * n_cycles),
        ).result()

        eig_X, w_X, _ = parse_counts_final_eigs_D(
            res_X.get_counts(),
            n_cycles=n_cycles,
            n_sys=cfg.n_sys,
            n_bath=cfg.n_bath,
        )

        for i, j, _ in xx_terms:
            xx_means[(i, j)] += weight_init * np.sum(
                eig_X[:, i] * eig_X[:, j] * w_X
            )

    E_Z = 0.0
    for i, coef in z_terms:
        E_Z += coef * z_means[i]

    E_X = 0.0
    for i, j, coef in xx_terms:
        E_X += coef * xx_means[(i, j)]

    E_total = float(E_Z + E_X)

    return {
        "E_Z": float(E_Z),
        "E_X": float(E_X),
        "E_total": E_total,
        "bath_p1_per_cycle": np.mean(bath_acc, axis=1),
        "bath_p1_last": float(np.mean(bath_acc[-1])),
        "z_means": z_means,
        "xx_means": xx_means,
    }


# celdas para limpiar el notebook

def find_mu_for_target_N(n_sites=2, t=1.0, U=4.0, beta=2.0, target_N=None, mu_bounds=(-8, 8), n_grid=161):
    if target_N is None:
        target_N = n_sites

    mus = np.linspace(mu_bounds[0], mu_bounds[1], n_grid)
    rows = []

    for mu in mus:
        m = FermiHubbardJW(n_sites=n_sites, bonds=[(0, 1)], t=t, U=U, mu=mu)
        rho = m.gibbs_state(beta)
        rows.append({
            "mu": mu,
            "N": m.total_number(rho),
            "E": m.energy(rho),
            "D_avg": m.avg_double_occupancy(rho),
        })

    df = pd.DataFrame(rows)
    idx = int(np.argmin(np.abs(df["N"] - target_N)))
    return df, df.iloc[idx].to_dict()

def coupling_symmetry_table(model):
    N_op = model.number_operator()
    Sz_op = model.sz_total_operator()

    families = {
        "paper (Z+Y)/√2": build_coupling_ops_paper(model),
        "hubbard físico": build_coupling_operators(model),
        "hubbard dressed": build_coupling_ops_dressed(model),
        "solo X": build_coupling_ops_pauli(model, "X"),
        "solo Y": build_coupling_ops_pauli(model, "Y"),
        "solo Z": build_coupling_ops_pauli(model, "Z"),
    }

    rows = []
    for family, out in families.items():
        A_ops, labels = out

        comm_N = []
        comm_Sz = []

        for A in A_ops:
            comm_N.append(np.linalg.norm(A @ N_op - N_op @ A, ord="fro"))
            comm_Sz.append(np.linalg.norm(A @ Sz_op - Sz_op @ A, ord="fro"))

        rows.append({
            "familia": family,
            "n_ops": len(A_ops),
            "max ||[A,N]||": np.max(comm_N),
            "max ||[A,Sz]||": np.max(comm_Sz),
            "conserva N": np.max(comm_N) < 1e-10,
            "conserva Sz": np.max(comm_Sz) < 1e-10,
        })

    return pd.DataFrame(rows)

def tune_protocol_full_gibbs(U, t=1.0, beta=2.0):
    """
    Retuning para comparar contra el Gibbs completo del dímero.

    Estrategia:
    - h sigue la escala de carga cuando U crece.
    - delta baja cuando crece la escala energética.
    - theta baja suavemente para mantener el régimen perturbativo.
    - M_T se ajusta para mantener una ventana temporal del filtro.
    """
    U = float(U)
    t = float(t)
    beta = float(beta)

    # Escala principal del baño.
    # Para Gibbs completo, no basta con mirar solo J_spin.
    h = max(2.0 * t, U)

    # Acoplo más pequeño si h crece.
    theta = np.sqrt(0.05 / np.sqrt(beta * h))
    theta = float(np.clip(theta, 0.06, 0.20))

    # Paso temporal controlado por la escala energética dominante.
    H_scale = max(t, 0.5 * U)
    delta = min(0.20, 0.25 / H_scale)

    # Anchura temporal del filtro.
    a = compute_a(h, beta)
    T_filter = 5.0 / a
    M_T = int(np.ceil(T_filter / delta))
    M_T = max(M_T, 12)

    return {
        "h_bath": float(h),
        "beta": float(beta),
        "theta": float(theta),
        "delta": float(delta),
        "M_T": int(M_T),
        "J_spin": float(4.0 * t * t / U),
        "a": float(a),
        "T_filter": float(T_filter),
        "H_scale": float(H_scale),
        "delta_H": float(delta * H_scale),
    }
    
def tune_params_h_diagnostic(
    U,
    h_choice,
    t=1.0,
    beta=2.0,
    theta_min=0.06,
    theta_max=0.20,
):
    """
    Diagnóstico controlado: solo cambia la elección conceptual de h.

    h_choice:
    - full_gibbs: h = max(2t, U)
    - carga:      h = U
    - intermedio: h = 2t
    - spin:       h = J_spin = 4t²/U
    """
    U = float(U)
    t = float(t)
    beta = float(beta)

    J_spin = 4.0 * t * t / U

    if h_choice == "full_gibbs":
        h = max(2.0 * t, U)
    elif h_choice == "carga":
        h = U
    elif h_choice == "intermedio":
        h = 2.0 * t
    elif h_choice == "spin":
        h = J_spin
    else:
        raise ValueError("h_choice debe ser full_gibbs, carga, intermedio o spin")

    h = max(h, 0.30)

    theta = np.sqrt(0.05 / np.sqrt(beta * h))
    theta = float(np.clip(theta, theta_min, theta_max))

    H_scale = max(t, 0.5 * U)
    delta = min(0.20, 0.25 / H_scale)

    a = compute_a(h, beta)
    T_filter = 5.0 / a
    M_T = int(np.ceil(T_filter / delta))
    M_T = max(M_T, 12)

    return {
        "h_choice": h_choice,
        "h": float(h),
        "J_spin": float(J_spin),
        "theta": float(theta),
        "delta": float(delta),
        "M_T": int(M_T),
        "a": float(a),
        "T_filter": float(T_filter),
    }
    
def candidate_params_strong(
    U,
    h_choice,
    theta,
    delta_factor,
    window_factor,
    t=1.0,
    beta=2.0,
    h_floor=0.30,
):
    """
    Construye una familia de parámetros para diagnóstico de régimen fuerte.

    h_choice:
    - full_gibbs: h = max(2t, U)
    - carga:      h = U
    - intermedio: h = 2t
    - spin:       h = 4t²/U

    theta:
    - fuerza de acoplo sistema-baño.

    delta_factor:
    - controla delta = min(0.30, delta_factor / H_scale).

    window_factor:
    - controla la ventana temporal del filtro:
      M_T = ceil((window_factor/a)/delta).
    """
    U = float(U)
    t = float(t)
    beta = float(beta)

    J_spin = 4.0 * t * t / U

    if h_choice == "full_gibbs":
        h = max(2.0 * t, U)
    elif h_choice == "carga":
        h = U
    elif h_choice == "intermedio":
        h = 2.0 * t
    elif h_choice == "spin":
        h = J_spin
    else:
        raise ValueError("h_choice debe ser full_gibbs, carga, intermedio o spin")

    h = max(float(h), h_floor)

    H_scale = max(t, 0.5 * U)
    delta = min(0.30, delta_factor / H_scale)

    a = compute_a(h, beta)
    T_filter = window_factor / a
    M_T = int(np.ceil(T_filter / delta))
    M_T = max(M_T, 8)

    return {
        "h_choice": h_choice,
        "h": float(h),
        "J_spin": float(J_spin),
        "theta": float(theta),
        "delta": float(delta),
        "M_T": int(M_T),
        "a": float(a),
        "T_filter": float(T_filter),
        "delta_factor": float(delta_factor),
        "window_factor": float(window_factor),
    }
    
def params_temperature_sweep(beta, strategy, t=1.0, U=4.0):
    """
    Parámetros para el barrido en temperatura del dímero.

    strategy:
    - benchmark_h1: h=1, referencia del benchmark del dímero.
    - full_gibbs: h=max(2t,U), regla heurística de escala de carga.
    """
    beta = float(beta)
    t = float(t)
    U = float(U)

    if strategy == "benchmark_h1":
        h = 1.0
        theta = 0.35
        delta = 0.30

    elif strategy == "full_gibbs":
        h = max(2.0 * t, U)

        theta = np.sqrt(0.05 / np.sqrt(beta * h))
        theta = float(np.clip(theta, 0.06, 0.20))

        H_scale = max(t, 0.5 * U)
        delta = min(0.20, 0.25 / H_scale)

    else:
        raise ValueError("strategy debe ser 'benchmark_h1' o 'full_gibbs'")

    a = compute_a(h, beta)
    T_filter = 5.0 / a
    M_T = int(np.ceil(T_filter / delta))
    M_T = max(M_T, 8)

    return {
        "strategy": strategy,
        "h": float(h),
        "theta": float(theta),
        "delta": float(delta),
        "M_T": int(M_T),
        "a": float(a),
        "T_filter": float(T_filter),
    }
    

def make_fh_dimer(t=1.0, U=4.0):
    return FermiHubbardJW(n_sites=2, bonds=[(0, 1)], t=t, U=U)


def dominant_bohr_frequency_fh(model, A_ops, tol_gap=1e-9, decimals=8):
    """
    Frecuencia de Bohr dominante ponderada por los operadores de acoplo.
    """
    E, V = model.eigensystem()
    weights = {}

    for A in A_ops:
        Ae = V.conj().T @ A @ V

        for a in range(len(E)):
            for b in range(len(E)):
                omega = abs(E[a] - E[b])

                if omega > tol_gap:
                    key = np.round(omega, decimals)
                    weights[key] = weights.get(key, 0.0) + abs(Ae[a, b])**2

    df = (
        pd.DataFrame(
            [{"omega": float(k), "weight": float(v)} for k, v in weights.items()]
        )
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )

    omega_ref = float(df.loc[0, "omega"])
    return omega_ref, df


def grid_edges(x):
    """
    Bordes de celdas para un grid no uniformemente espaciado.
    """
    x = np.asarray(x, dtype=float)
    edges = np.zeros(len(x) + 1)

    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])

    return edges


def add_harmonic_guides_MT(ax, omega_ref, delta, kmax=5, color="red"):
    """
    Guías $M_T delta omega_ref = k pi$.
    """
    for k in range(1, kmax + 1):
        mt_k = k * np.pi / (delta * omega_ref)

        if k == 1:
            ls = "--"
        elif k == 2:
            ls = ":"
        else:
            ls = "-."

        ax.axhline(
            mt_k,
            color=color,
            ls=ls,
            alpha=0.55,
            lw=1.5,
            label=fr"$M_T\delta\omega_\star={k}\pi$" if k <= 2 else None,
        )


def fixed_point_error_protocol(
    protocol_cls,
    model,
    A_ops,
    labels,
    beta,
    h_bath,
    theta,
    delta,
    M_T,
    energy_norm,
):
    rho_g = model.gibbs_state(beta)
    E_g = model.energy(rho_g)

    if protocol_cls is ThermalProtocolSequential:
        proto = protocol_cls(
            model,
            A_ops=A_ops,
            labels=labels,
            h_bath=h_bath,
            beta=beta,
            theta=theta,
            delta=delta,
            M_T=int(M_T),
            random_order=False,
            random_evolution=False,
        )
    else:
        proto = protocol_cls(
            model,
            A_ops=A_ops,
            labels=labels,
            h_bath=h_bath,
            beta=beta,
            theta=theta,
            delta=delta,
            M_T=int(M_T),
            random_evolution=False,
        )

    rho_fp = proto.fixed_point()
    E_fp = model.energy(rho_fp)

    return {
        "theta": float(theta),
        "M_T": int(M_T),
        "E_fp": float(E_fp),
        "E_g": float(E_g),
        "err_E_density": abs(E_fp - E_g) / energy_norm,
        "Dtr": trace_distance(rho_fp, rho_g),
        "D_avg": model.avg_double_occupancy(rho_fp),
        "D_g": model.avg_double_occupancy(rho_g),
    }


def compute_theta_MT_map(
    protocol_cls,
    model,
    A_ops,
    labels,
    beta,
    h_bath,
    delta,
    thetas,
    MTs,
    energy_norm,
):
    rows = []
    t0 = time.time()

    for MT in MTs:
        print(f"M_T = {MT}")

        for th in thetas:
            rows.append(
                fixed_point_error_protocol(
                    protocol_cls=protocol_cls,
                    model=model,
                    A_ops=A_ops,
                    labels=labels,
                    beta=beta,
                    h_bath=h_bath,
                    theta=th,
                    delta=delta,
                    M_T=MT,
                    energy_norm=energy_norm,
                )
            )

    df = pd.DataFrame(rows)
    print(f"Tiempo total: {(time.time() - t0)/60:.2f} min")

    return df


def plot_theta_MT_heatmap(
    df,
    theta_grid,
    MT_grid,
    title,
    omega_ref,
    delta,
    ylabel_norm="$N_{sites}$",
    kmax=4,
):
    pivot = df.pivot(index="M_T", columns="theta", values="err_E_density")
    Z = pivot.loc[MT_grid, theta_grid].values

    theta_edges = grid_edges(theta_grid)
    MT_edges = grid_edges(MT_grid)

    fig, ax = plt.subplots(figsize=(9, 6))

    vmin = max(np.nanmin(Z), 1e-8)
    vmax = np.nanmax(Z)

    im = ax.pcolormesh(
        theta_edges,
        MT_edges,
        Z,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
        shading="auto",
    )

    add_harmonic_guides_MT(
        ax,
        omega_ref=omega_ref,
        delta=delta,
        kmax=kmax,
        color="red",
    )

    mt_valley = 1.5 * np.pi / (delta * omega_ref)
    ax.axhline(
        mt_valley,
        color="white",
        lw=2,
        ls="-.",
        alpha=0.85,
        label=fr"guía $M_T\delta\omega_\star=1.5\pi$",
    )

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$M_T$")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04)
    cbar.set_label(fr"$|E_{{fp}}-E_\beta|/{ylabel_norm}$")

    plt.tight_layout()
    plt.show()
    
def rho_fp_for_point(protocol_cls, model, A_ops, labels, beta, h_bath, theta, delta, M_T):
    if protocol_cls is ThermalProtocolSequential:
        proto = protocol_cls(
            model,
            A_ops=A_ops,
            labels=labels,
            h_bath=h_bath,
            beta=beta,
            theta=theta,
            delta=delta,
            M_T=int(M_T),
            random_order=False,
            random_evolution=False,
        )
    else:
        proto = protocol_cls(
            model,
            A_ops=A_ops,
            labels=labels,
            h_bath=h_bath,
            beta=beta,
            theta=theta,
            delta=delta,
            M_T=int(M_T),
            random_evolution=False,
        )

    return proto.fixed_point()

# ============================================================
# A.5.1 — Herramientas espectrales para Fermi-Hubbard
# ============================================================



def fh_spectral_data(model, A_ops=None, beta=2.0, min_weight=1e-14):
    """
    Transiciones de Bohr ponderadas por los operadores de acoplo A_mu.

    Para cada par ordenado n -> m, con n != m:

        omega = E_m - E_n
        |omega| = |E_m - E_n|
        peso = p_n * sum_mu |<m|A_mu|n>|^2

    Usamos pares ordenados para conservar el peso térmico del estado inicial.
    Para la visualización agrupamos por la frecuencia positiva |omega|.
    """
    if A_ops is None:
        A_ops, _ = build_coupling_ops_paper(model)

    evals, evecs = model.eigensystem()
    evals = np.real(evals)

    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    beta = float(beta)

    boltz = np.exp(-beta * (evals - evals.min()))
    probs = boltz / boltz.sum()

    A_eig = [
        evecs.conj().T @ A @ evecs
        for A in A_ops
    ]

    rows = []

    for n in range(len(evals)):
        for m in range(len(evals)):
            if m == n:
                continue

            omega = float(evals[m] - evals[n])
            omega_abs = abs(omega)

            if omega_abs < 1e-12:
                continue

            matrix_weight = float(
                sum(abs(Ae[m, n]) ** 2 for Ae in A_eig)
            )

            weight = float(probs[n] * matrix_weight)

            if weight < min_weight:
                continue

            rows.append({
                "n inicial": n,
                "m final": m,
                "E_n": float(evals[n]),
                "E_m": float(evals[m]),
                "omega": omega,
                "|omega|": omega_abs,
                "p_n": float(probs[n]),
                "matrix_weight": matrix_weight,
                "peso": weight,
            })

    spec = pd.DataFrame(rows)

    if len(spec) > 0:
        spec["peso_normalizado"] = spec["peso"] / spec["peso"].sum()
        spec["peso_relativo"] = spec["peso"] / spec["peso"].max()

    return (
        spec
        .sort_values("peso", ascending=False)
        .reset_index(drop=True)
    )


def fh_filter_response(omega_abs, h, beta):
    """
    Envolvente gaussiana del filtro en frecuencia.

        f_h(omega) = exp[-(|omega|-h)^2 / (2a^2)]
        a = sqrt(4h/beta)
    """
    a = compute_a(h, beta)
    omega_abs = np.asarray(omega_abs)

    return np.exp(
        -0.5 * ((omega_abs - h) / a) ** 2
    )


def fh_spectral_capture_table(spec, h_values, beta=2.0):
    """
    Fracción del peso espectral ponderado capturada por cada filtro.
    No es una predicción exacta del error final: es un diagnóstico espectral.
    """
    if len(spec) == 0:
        return pd.DataFrame(columns=["h", "a=sqrt(4h/beta)", "fracción capturada"])

    total_weight = float(spec["peso"].sum())

    rows = []

    for h in h_values:
        a = compute_a(h, beta)

        response = fh_filter_response(
            spec["|omega|"].values,
            h=h,
            beta=beta,
        )

        captured = float(
            np.sum(spec["peso"].values * response) / total_weight
        )

        rows.append({
            "h": float(h),
            "a=sqrt(4h/beta)": float(a),
            "fracción capturada": captured,
        })

    return pd.DataFrame(rows)


def fh_top_spectral_transitions(spec, k=20):
    """
    Muestra las transiciones con mayor peso efectivo.
    """
    cols = [
        "n inicial",
        "m final",
        "E_n",
        "E_m",
        "omega",
        "|omega|",
        "p_n",
        "matrix_weight",
        "peso",
        "peso_normalizado",
    ]

    return (
        spec[cols]
        .sort_values("peso", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )


def fh_plot_spectrum_and_filters(
    model,
    spec,
    beta=2.0,
    h_values=(0.75, 1.0, 2.0, 4.0),
    bins=70,
    w_max=None,
    title_extra="",
):
    """
    Figura espectral con estética tipo notebooks anteriores:

    - Panel izquierdo: niveles de energía.
    - Panel derecho: peso espectral efectivo normalizado y filtros gaussianos.
    - La leyenda de cada filtro incluye la fracción de peso capturada.
    """
    evals, _ = model.eigensystem()
    evals = np.sort(np.real(evals))

    unique_E = np.unique(np.round(evals, 10))

    if w_max is None:
        w_max = max(
            1.05 * spec["|omega|"].max(),
            1.25 * max(h_values),
        )

    spec_plot = spec[spec["|omega|"] <= w_max].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    # --------------------------------------------------------
    # Panel izquierdo: espectro
    # --------------------------------------------------------
    ax = axes[0]

    for E in unique_E:
        ax.hlines(
            E,
            0.0,
            1.0,
            color="tab:blue",
            lw=2,
            alpha=0.85,
        )

    ax.set_xlim(-0.1, 1.15)
    ax.set_xticks([])
    ax.set_ylabel("Energía")
    ax.set_title(
        f"Espectro Fermi-Hubbard 2 sitios "
        f"(t={model.t:g}, U={model.U:g}){title_extra}"
    )
    ax.grid(alpha=0.3)

    # Etiquetas de los primeros niveles distintos.
    n_labels = min(4, len(unique_E))
    for k in range(n_labels):
        ax.text(
            0.02,
            unique_E[k],
            rf"$E_{{{k}}}={unique_E[k]:.3f}$",
            va="center",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.75,
            ),
        )

    # --------------------------------------------------------
    # Panel derecho: densidad espectral efectiva
    # --------------------------------------------------------
    ax = axes[1]

    counts, bin_edges = np.histogram(
        spec_plot["|omega|"].values,
        bins=bins,
        range=(0.0, w_max),
        weights=spec_plot["peso"].values,
    )

    if counts.max() > 0:
        counts_plot = counts / counts.max()
    else:
        counts_plot = counts

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    ax.bar(
        bin_centers,
        counts_plot,
        width=0.90 * bin_width,
        alpha=0.35,
        color="tab:blue",
        label=r"peso espectral $\propto p_n|\langle m|A|n\rangle|^2$",
    )

    omega_grid = np.linspace(0.0, w_max, 800)
    total_weight = float(spec["peso"].sum())

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:brown",
        "tab:pink",
    ]

    for k, h in enumerate(h_values):
        color = colors[k % len(colors)]
        a = compute_a(h, beta)

        filt = fh_filter_response(
            omega_grid,
            h=h,
            beta=beta,
        )

        response_at_transitions = fh_filter_response(
            spec["|omega|"].values,
            h=h,
            beta=beta,
        )

        captured = float(
            np.sum(spec["peso"].values * response_at_transitions) / total_weight
        )

        ax.plot(
            omega_grid,
            filt,
            color=color,
            lw=2.2,
            label=rf"$h={h:g}$, captura $\approx {100*captured:.1f}\%$",
        )

        ax.axvline(
            h,
            color=color,
            ls="--",
            lw=1,
            alpha=0.25,
        )

    ax.set_xlabel(r"$|\omega|=|E_m-E_n|$")
    ax.set_ylabel("peso efectivo / filtro")
    ax.set_title(rf"Densidad espectral efectiva a $\beta={beta:g}$")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def project_density(rho):
    rho = 0.5 * (rho + rho.conj().T)
    return rho / np.real(np.trace(rho))


def system_free_unitary(model, time):
    evals, evecs = model.eigensystem()
    return evecs @ np.diag(np.exp(-1j * time * evals)) @ evecs.conj().T


def fullbath_proto_base(
    model,
    A_ops,
    labels,
    beta=2.0,
    h_bath=1.0,
    theta=0.35,
    delta=0.30,
    M_T=12,
):
    return ThermalProtocolFullBath(
        model,
        A_ops=A_ops,
        labels=labels,
        h_bath=h_bath,
        beta=beta,
        theta=theta,
        delta=delta,
        M_T=M_T,
        random_evolution=False,
    )


def fullbath_Q_variant(proto, use_rewind=False):
    Q = proto.Q_base

    if use_rewind:
        # Duración aproximada de la ventana: número de capas U_S por delta.
        T_window = (2 * proto.M_T + 1) * proto.delta

        # system_free_unitary usa exp(-i H t), así que t negativo implementa exp(+i H T).
        U_rew = system_free_unitary(proto.model, -T_window)
        Q = kron(U_rew, np.eye(proto.dB)) @ Q

    return Q


def fullbath_fixed_point_from_Q(proto, Q):
    dS = proto.dS
    d2 = dS**2
    S = np.zeros((d2, d2), dtype=complex)

    for c in range(dS):
        for d in range(dS):
            rho_cd = np.zeros((dS, dS), dtype=complex)
            rho_cd[c, d] = 1.0

            rho_tot = Q @ kron(rho_cd, proto.phi) @ Q.conj().T
            rho_out = partial_trace_B(rho_tot, proto.dS, proto.dB)

            S[:, c * dS + d] = rho_out.reshape(-1)

    eigvals, eigvecs = np.linalg.eig(S)
    idx = np.argmin(np.abs(eigvals - 1.0))

    rho_fp = eigvecs[:, idx].reshape(dS, dS)
    return project_density(rho_fp)


def run_fullbath_random_stationary(
    proto,
    use_rewind=False,
    n_cycles=700,
    burn=450,
    keep=120,
    seed=123,
):
    rng = np.random.default_rng(seed)

    rho = np.eye(proto.dS, dtype=complex) / proto.dS
    IB = np.eye(proto.dB, dtype=complex)

    Q_base = fullbath_Q_variant(proto, use_rewind=use_rewind)

    kept = []

    for n in range(1, n_cycles + 1):
        m_rand = rng.integers(1, 2 * proto.M_T + 1)
        U_rand = system_free_unitary(proto.model, proto.delta * int(m_rand))
        Q = kron(U_rand, IB) @ Q_base

        rho_tot = Q @ kron(rho, proto.phi) @ Q.conj().T
        rho = partial_trace_B(rho_tot, proto.dS, proto.dB)
        rho = project_density(rho)

        if n > burn and len(kept) < keep:
            kept.append(rho.copy())

    return project_density(np.mean(kept, axis=0))


def make_fh_case(case, U=4.0, t=1.0, mu=0.0):
    if case == "dimer":
        return fhm.FermiHubbardJW(
            n_sites=2,
            bonds=[(0, 1)],
            t=t,
            U=U,
            mu=mu,
        )

    if case == "chain3":
        return fhm.FermiHubbardJW(
            n_sites=3,
            bonds=[(0, 1), (1, 2)],
            t=t,
            U=U,
            mu=mu,
        )

    if case == "plaquette2x2":
        return fhm.FermiHubbardJW(
            n_sites=4,
            bonds=[(0, 1), (2, 3), (0, 2), (1, 3)],
            t=t,
            U=U,
            mu=mu,
        )

    raise ValueError(f"Caso no reconocido: {case}")


def dense_complex_matrix_mb(dim):
    return dim * dim * 16 / 1024**2

# ============================================================
# Helpers para análisis half-filling canónico8,
# ============================================================

def half_filling_N(model):
    """
    Half-filling para Fermi-Hubbard con 2 espines por sitio:
    N = número de sitios.
    """
    return int(model.n_sites)


def number_sector_indices(model, N):
    """
    Índices de la base computacional con N partículas.
    Usa la función ya implementada en FermiHubbardJW.
    """
    return np.array(model.basis_indices_number_sector(int(N)), dtype=int)


def number_sector_projector(model, N):
    """
    Proyector P_N al sector de número total de partículas N.
    """
    idx = number_sector_indices(model, N)
    P = np.zeros((model.dim, model.dim), dtype=complex)
    P[idx, idx] = 1.0
    return P


def maximally_mixed_in_sector(model, N):
    """
    Estado maximally mixed dentro del sector N.
    Es el análogo canónico de I/d, pero restringido a half-filling.
    """
    idx = number_sector_indices(model, N)
    rho = np.zeros((model.dim, model.dim), dtype=complex)
    rho[np.ix_(idx, idx)] = np.eye(len(idx), dtype=complex) / len(idx)
    return rho


def sector_weight(model, rho, N):
    """
    Peso Tr(P_N rho). Sirve para comprobar si un estado permanece en half-filling.
    """
    P = number_sector_projector(model, N)
    return float(np.real(np.trace(P @ rho)))


def project_operator_to_sector(A, model, N):
    """
    Proyecta un operador A al sector N:
        A_N = P_N A P_N.
    Si A rompe N, esta proyección elimina las partes que sacan del sector.
    """
    P = number_sector_projector(model, N)
    return P @ A @ P


def project_ops_to_sector(A_ops, labels, model, N, drop_zero=True, tol=1e-12):
    """
    Proyecta una familia de operadores al sector N.
    """
    A_proj = []
    labels_proj = []

    for A, lab in zip(A_ops, labels):
        Ap = project_operator_to_sector(A, model, N)
        norm = np.linalg.norm(Ap, ord="fro")

        if (not drop_zero) or norm > tol:
            A_proj.append(Ap)
            labels_proj.append(lab + f" | projected N={N}")

    return A_proj, labels_proj


def leakage_from_sector(A, model, N):
    """
    Mide cuánto saca A del sector N:
        ||(I-P) A P||_F / ||A P||_F.

    Si vale 0, A conserva el sector.
    Si vale ~1, A principalmente saca del sector.
    """
    P = number_sector_projector(model, N)
    I = np.eye(model.dim, dtype=complex)

    AP = A @ P
    denom = np.linalg.norm(AP, ord="fro")

    if denom < 1e-14:
        return 0.0

    leak = np.linalg.norm((I - P) @ A @ P, ord="fro") / denom
    return float(leak)


def state_summary_canonic(model, rho, label, rho_ref=None, N=None):
    """
    Resumen compacto de un estado, incluyendo peso en half-filling.
    """
    row = model.summary_row(rho, label=label)

    if N is not None:
        row[f"weight_N={N}"] = sector_weight(model, rho, N)

    if rho_ref is not None:
        row["D_trace_to_ref"] = trace_distance(rho, rho_ref)

    return row

# ============================================================
# Helpers definitivos para half-filling sin normalizar
# ============================================================

def hf_N(model):
    """
    Half-filling en Fermi-Hubbard con dos espines por sitio:
    N = número de sitios.
    """
    return int(model.n_sites)


def sector_indices(model, N):
    """
    Índices de la base computacional con N partículas.
    """
    return np.array(model.basis_indices_number_sector(int(N)), dtype=int)


def sector_projector(model, N):
    """
    Proyector P_N al sector de número total N.
    """
    idx = sector_indices(model, N)
    P = np.zeros((model.dim, model.dim), dtype=complex)
    P[idx, idx] = 1.0
    return P


def maximally_mixed_sector(model, N):
    """
    Estado maximally mixed dentro del sector N.
    """
    idx = sector_indices(model, N)
    rho = np.zeros((model.dim, model.dim), dtype=complex)
    rho[np.ix_(idx, idx)] = np.eye(len(idx), dtype=complex) / len(idx)
    return rho


def sector_weight(model, rho, N):
    """
    Peso del estado en el sector N:
    
    Tr(P_N rho).
    """
    P = sector_projector(model, N)
    return float(np.real(np.trace(P @ rho)))


def project_family_no_norm(A_ops, labels, model, N, tol=1e-12):
    """
    Proyecta A_mu -> P_N A_mu P_N sin normalizar.
    
    Elimina los operadores que se anulan al proyectar.
    """
    P = sector_projector(model, N)

    A_out = []
    labels_out = []
    rows = []

    for A, lab in zip(A_ops, labels):
        Ap = P @ A @ P
        Ap = 0.5 * (Ap + Ap.conj().T)

        norm = np.linalg.norm(Ap, ord="fro")
        is_zero = norm < tol
        is_diag = np.max(np.abs(Ap - np.diag(np.diag(Ap)))) < 1e-10

        rows.append({
            "label": lab,
            "norm_projected": norm,
            "zero_after_projection": is_zero,
            "diagonal_after_projection": is_diag,
        })

        if not is_zero:
            A_out.append(Ap)
            labels_out.append(lab)

    return A_out, labels_out, pd.DataFrame(rows)


def hf_history(model, out, rho_target, N):
    """
    Recalcula observables de la trayectoria respecto al target canónico.
    """
    D = np.array([
        trace_distance(rho, rho_target)
        for rho in out["states"]
    ])

    E = np.array([
        model.energy(rho)
        for rho in out["states"]
    ])

    Ddbl = np.array([
        model.avg_double_occupancy(rho)
        for rho in out["states"]
    ])

    Czz = np.array([
        model.spin_correlation(rho, 0, 1)
        for rho in out["states"]
    ])

    W = np.array([
        sector_weight(model, rho, N)
        for rho in out["states"]
    ])

    return D, E, Ddbl, Czz, W


def run_hf_protocol_no_norm(
    model,
    A_raw,
    labs_raw,
    *,
    beta,
    h_bath,
    theta,
    delta,
    M_T,
    n_cycles,
    record_every,
    seed,
):
    """
    Ejecuta el protocolo half-filling con operadores proyectados sin normalizar.
    """
    N = hf_N(model)

    rho_target = model.gibbs_state(beta, N_sector=N)
    rho0 = maximally_mixed_sector(model, N)

    A_use, labs_use, df_projection = project_family_no_norm(
        A_raw,
        labs_raw,
        model,
        N,
    )

    prot = ThermalProtocolSequential(
        model,
        A_ops=A_use,
        labels=labs_use,
        h_bath=h_bath,
        beta=beta,
        theta=theta,
        delta=delta,
        M_T=M_T,
        #random_order=True,
        random_evolution=True,
    )

    out = prot.run(
        n_cycles=n_cycles,
        rho_init=rho0,
        verbose=False,
        record_every=record_every,
        seed=seed,
    )

    D, E, Ddbl, Czz, W = hf_history(model, out, rho_target, N)

    out["D_can"] = D
    out["E_hist"] = E
    out["Ddbl_hist"] = Ddbl
    out["Czz_hist"] = Czz
    out["W_hist"] = W
    out["rho_target"] = rho_target
    out["rho0"] = rho0
    out["N_hf"] = N
    out["df_projection"] = df_projection
    out["A_labels_used"] = labs_use

    return out

def select_best_h_half_filling(
    model,
    beta=2.0,
    h_grid=(0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0),
    theta=0.18,
    delta=0.15,
    M_T=24,
    n_cycles=600,
    record_every=50,
    seed=7,
):
    N_hf = hf_N(model)
    rho_target = model.gibbs_state(beta, N_sector=N_hf)

    A_raw, labs_raw = fhm.build_coupling_operators(model)

    rows = []
    best_out = None
    best_h = None
    best_D = np.inf

    for h in h_grid:
        out = run_hf_protocol_no_norm(
            model,
            A_raw,
            labs_raw,
            beta=beta,
            h_bath=float(h),
            theta=theta,
            delta=delta,
            M_T=M_T,
            n_cycles=n_cycles,
            record_every=record_every,
            seed=seed,
        )

        D_final = out["D_can"][-1]

        rows.append({
            "h": h,
            "D_final": D_final,
            "E_final": out["E_hist"][-1],
            "Ddbl_final": out["Ddbl_hist"][-1],
            "Czz_final": out["Czz_hist"][-1],
        })

        if D_final < best_D:
            best_D = D_final
            best_h = float(h)
            best_out = out

    df_scan = pd.DataFrame(rows).sort_values("D_final")
    return best_h, best_out, df_scan

def sparse_op_to_df(op):
    rows = []
    for label, coeff in zip(op.paulis.to_labels(), op.coeffs):
        rows.append({
            "pauli": label,
            "coeff_real": float(np.real(coeff)),
            "coeff_imag": float(np.imag(coeff)),
        })
    return pd.DataFrame(rows).sort_values("pauli").reset_index(drop=True)

def build_qiskit_nature_fh_line(n_sites=2, t=1.0, U=4.0):
    try:
        from qiskit_nature.second_q.hamiltonians.lattices import LineLattice, BoundaryCondition
        from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
        from qiskit_nature.second_q.mappers import JordanWignerMapper
    except Exception as exc:
        print("Qiskit Nature no disponible en este entorno.")
        print("Error:", repr(exc))
        return None

    lattice = LineLattice(
        num_nodes=n_sites,
        boundary_condition=BoundaryCondition.OPEN,
    )

    # Convención usual: hopping -t.
    lattice = lattice.uniform_parameters(
        uniform_interaction=-t,
        uniform_onsite_potential=0.0,
    )

    hubbard = FermiHubbardModel(
        lattice,
        onsite_interaction=U,
    )

    ferm_op = hubbard.second_q_op()
    qubit_op = JordanWignerMapper().map(ferm_op).simplify()

    return qubit_op



def resource_table(protocols, cycles=(1, 2, 4), transpiled_values=(False, True)):
    rows = []
    for proto in protocols:
        for ncyc in cycles:
            for transpiled in transpiled_values:
                r = proto.resources(
                    n_cycles=ncyc,
                    final_pauli={q: "Z" for q in range(proto.n_sys)},
                    transpiled=transpiled,
                )
                r["ISA"] = bool(transpiled)
                rows.append(r)
    return pd.DataFrame(rows)


q_full = ThermalProtocolFullBathQiskit(qk_model, backend_bundle=bundle, **qk_params)
q_seq = ThermalProtocolSequentialQiskit(qk_model, backend_bundle=bundle, **qk_params)
