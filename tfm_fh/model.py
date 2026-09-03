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
from scipy.linalg import expm, eigh
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import warnings, time
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