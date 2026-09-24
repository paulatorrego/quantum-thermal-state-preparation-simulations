# tfm_shared/qiskit_impl.py

from __future__ import annotations

from typing import Optional, Any

import numpy as np
import pandas as pd

from .core import (
    expect,
    gibbs_state,
    fidelity_dm,
    trace_distance_dm,
    project_to_physical_dm,
    ptrace_bath,
    ptrace_system,
    cycle_unitary_matrix,
    gaussian_filter_values,
    compile_protocol,
)

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import DensityMatrix, Operator
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    HAVE_QISKIT = True

except Exception:
    HAVE_QISKIT = False


def _require_qiskit():
    if not HAVE_QISKIT:
        raise RuntimeError("Qiskit/Aer no está disponible en este entorno.")


# ============================================================
# FUNCIONES QISKIT PARA EL ESTUDIO DE CIRCUITOS - buenas
# ============================================================

def append_measure_in_basis(qc, qubit, cbit, basis="Z"):
    basis = basis.upper()
    if basis == "X":
        qc.h(qubit)
    elif basis == "Y":
        qc.sdg(qubit)
        qc.h(qubit)
    elif basis != "Z":
        raise ValueError("basis debe ser 'X', 'Y' o 'Z'")
    qc.measure(qubit, cbit)


def append_one_cycle_to_circuit(
    qc,
    compiled,
    q_sys,
    q_bath,
    c_bath=None,
    measure_bath=True,
    bath_measure_basis="Z",
):
    _require_qiskit()

    n_total = len(q_sys) + len(q_bath)
    U = cycle_unitary_matrix(compiled, mr=0)
    gate = UnitaryGate(U, label="U_cycle")
    qc.append(gate, list(q_sys) + list(q_bath))

    if measure_bath and c_bath is not None:
        for qb, cb in zip(q_bath, c_bath):
            append_measure_in_basis(qc, qb, cb, basis=bath_measure_basis)
        for qb in q_bath:
            qc.reset(qb)


def build_protocol_circuit(
    compiled,
    n_cycles,
    measure_bath=True,
    bath_measure_basis="Z",
):
    _require_qiskit()

    cfg = compiled["cfg"]
    q_sys = QuantumRegister(cfg.n_sys, "s")
    q_bath = QuantumRegister(cfg.n_bath, "b")
    c_bath = ClassicalRegister(cfg.n_bath * n_cycles, "cb") if measure_bath else None

    if c_bath is None:
        qc = QuantumCircuit(q_sys, q_bath)
    else:
        qc = QuantumCircuit(q_sys, q_bath, c_bath)

    for r in range(n_cycles):
        c_slice = None
        if measure_bath:
            start = r * cfg.n_bath
            stop = (r + 1) * cfg.n_bath
            c_slice = [c_bath[i] for i in range(start, stop)]

        append_one_cycle_to_circuit(
            qc,
            compiled,
            q_sys,
            q_bath,
            c_bath=c_slice,
            measure_bath=measure_bath,
            bath_measure_basis=bath_measure_basis,
        )

    return qc

def circuit_stats(compiled):
    """
    Analiza cómo escalan los recursos del circuito con MT y δ.
    No ejecuta nada: solo cuenta gates.
    """
    cfg = compiled["cfg"]
    n_sys, n_bath = cfg.n_sys, cfg.n_bath
    n_layers = 2 * cfg.MT + 1

    # Caso A: 1 UnitaryGate
    stats_A = {"depth": 3, "gates": 3, "2q_gates": 1}  # unitary + meas + reset

    # Caso B (single spin): por capa = 2 Rz + 4 Rx + 2 CX + 1 Rz = 9
    # (capas con f_τ ≈ 0 no tienen la parte CX, solo 2 Rz)
    _, fvals, _ = gaussian_filter_values(cfg)
    n_active = int(np.sum(np.abs(fvals) > 1e-12))
    stats_B = {
        "n_layers": n_layers,
        "n_active_layers": n_active,
        "rz": n_layers * 2 + n_active,         # 2 por capa (US+UB) + 1 en activas
        "rx": n_active * 4,
        "cx": n_active * 2,
        "total_gates": n_layers * 2 + n_active * 7,
    }
    return stats_A, stats_B


def run_qiskit_density_matrix(compiled, n_cycles, rho0_sys=None):
    """
    Simula el protocolo REALMENTE con el backend density_matrix de Aer.
    No es numpy: pasa por el simulador de Qiskit.

    Devuelve energías, distancias traza, fidelidades, estados del baño.
    """
    cfg = compiled["cfg"]
    n_sys, n_bath = cfg.n_sys, cfg.n_bath
    n_total = n_sys + n_bath
    dim_S = 2**n_sys

    Hs_mat = compiled["Hs_mat"]
    rho_gibbs = gibbs_state(Hs_mat, cfg.beta)
    E_gibbs = float(np.real(np.trace(rho_gibbs @ Hs_mat)))

    if rho0_sys is None:
        rho0_sys = np.eye(dim_S, dtype=complex) / dim_S

    Q = cycle_unitary_matrix(compiled, mr=0)
    U_gate = UnitaryGate(Q, label="U_cycle")
    phi_B = compiled["bath0"]

    backend = AerSimulator(method="density_matrix")

    rho_sys = rho0_sys.copy()
    energies, dists, fids, bath_p1 = [], [], [], []

    for cycle in range(n_cycles):
        # Preparar estado total
        rho_total = np.kron(rho_sys, phi_B)

        # Circuito: init + U_cycle + save
        qc = QuantumCircuit(n_total, n_bath)
        qc.set_density_matrix(DensityMatrix(rho_total))
        qc.append(U_gate, range(n_total))
        qc.save_density_matrix(label="rho_out")

        # Ejecutar en el simulador Qiskit
        result = backend.run(transpile(qc, backend)).result()
        rho_total_out = np.array(result.data()["rho_out"])

        # Extraer estado del baño ANTES del reset
        rho_bath = ptrace_system(rho_total_out, n_sys, n_bath)
        bath_p1.append(1 - float(np.real(rho_bath[0, 0])))

        # Extraer estado del sistema (= traza parcial sobre baño)
        rho_sys = ptrace_bath(rho_total_out, n_sys, n_bath)

        # Métricas
        E = float(np.real(np.trace(rho_sys @ Hs_mat)))
        D = trace_distance_dm(rho_sys, rho_gibbs)
        F = fidelity_dm(rho_sys, rho_gibbs)
        energies.append(E); dists.append(D); fids.append(F)

    return {
        "energies": np.array(energies), "trace_dists": np.array(dists),
        "fidelities": np.array(fids), "bath_p1": np.array(bath_p1),
        "rho_final": rho_sys, "rho_gibbs": rho_gibbs, "E_gibbs": E_gibbs,
    }


def run_qiskit_shots(compiled, n_cycles, shots=4096, rho0_sys=None):
    """
    Simula con shots reales y extrae P(|1>_bath) por ciclo.
    """
    cfg = compiled["cfg"]
    n_sys, n_bath = cfg.n_sys, cfg.n_bath
    n_total = n_sys + n_bath
    dim_S = 2**n_sys

    if rho0_sys is None:
        rho0_sys = np.eye(dim_S, dtype=complex) / dim_S

    Q = cycle_unitary_matrix(compiled, mr=0)
    U_gate = UnitaryGate(Q, label="U_cycle")
    phi_B = compiled["bath0"]

    qc = QuantumCircuit(n_total)
    rho_total = np.kron(rho0_sys, phi_B)
    qc.set_density_matrix(DensityMatrix(rho_total))

    cregs = []
    for c in range(n_cycles):
        cr = ClassicalRegister(n_bath, f"c{c}")
        qc.add_register(cr)
        cregs.append(cr)

    for c in range(n_cycles):
        qc.append(U_gate, range(n_total))
        for b in range(n_bath):
            qc.measure(n_sys + b, cregs[c][b])
        for b in range(n_bath):
            qc.reset(n_sys + b)

    backend = AerSimulator(method="density_matrix")
    qc_t = transpile(qc, backend, optimization_level=0)
    result = backend.run(qc_t, shots=shots).result()
    counts = result.get_counts()

    bath_p1_per_cycle = np.zeros(n_cycles, dtype=float)

    for bitstring, count in counts.items():
        # Qiskit da algo tipo: "c{n-1} ... c1 c0"
        parts = bitstring.split()

        # fallback por si no hubiera espacios
        if len(parts) != n_cycles:
            bits = bitstring.replace(" ", "")
            parts = [bits[i * n_bath:(i + 1) * n_bath] for i in range(n_cycles)][::-1]

        for c in range(n_cycles):
            reg_bits = parts[n_cycles - 1 - c]   # c0 está al final
            if "1" in reg_bits:
                bath_p1_per_cycle[c] += count

    bath_p1_per_cycle /= shots

    return {
        "counts": counts,
        "bath_p1": bath_p1_per_cycle,
        "shots": shots,
        "n_cycles": n_cycles,
    }


def run_qiskit_noisy(compiled, n_cycles, noise_model, rho0_sys=None):
    cfg = compiled["cfg"]
    n_sys, n_bath = cfg.n_sys, cfg.n_bath
    n_total = n_sys + n_bath
    dim_S = 2**n_sys
    Hs_mat = compiled["Hs_mat"]
    rho_gibbs = gibbs_state(Hs_mat, cfg.beta)

    if rho0_sys is None:
        rho0_sys = np.eye(dim_S, dtype=complex) / dim_S

    Q = cycle_unitary_matrix(compiled, mr=0)
    U_gate = UnitaryGate(Q, label="U_cycle")
    phi_B = compiled["bath0"]

    basis_gates = noise_model.basis_gates if hasattr(noise_model, "basis_gates") else None
    backend = AerSimulator(method="density_matrix", noise_model=noise_model)

    # --- CLAVE: descomponer U_gate a basis gates UNA sola vez, fuera del loop.
    # De esta forma el noise_model se aplica sobre las puertas físicas (cx, u3…),
    # y las directivas Aer (set_density_matrix, save_density_matrix) no pasan
    # por el transpilador con basis_gates restringidas.
    qc_u = QuantumCircuit(n_total)
    qc_u.append(U_gate, range(n_total))
    qc_u_decomposed = transpile(qc_u, basis_gates=basis_gates, optimization_level=0)

    rho_sys = rho0_sys.copy()
    energies, dists = [], []

    for cycle in range(n_cycles):
        rho_total = np.kron(rho_sys, phi_B)

        qc = QuantumCircuit(n_total)
        qc.set_density_matrix(DensityMatrix(rho_total))
        qc.compose(qc_u_decomposed, inplace=True)       # ← puertas físicas, ya descompuestas
        qc.save_density_matrix(label="rho_out")

        # Ejecutar directamente. Aer maneja sus directivas nativas; no hace falta
        # re-transpilar aquí, que es lo que antes rompía.
        result = backend.run(qc).result()
        rho_total_out = np.array(result.data()["rho_out"])
        rho_sys = ptrace_bath(rho_total_out, n_sys, n_bath)

        energies.append(float(np.real(np.trace(rho_sys @ Hs_mat))))
        dists.append(trace_distance_dm(rho_sys, rho_gibbs))

    return {
        "energies": np.array(energies),
        "trace_dists": np.array(dists),
        "rho_final": rho_sys,
    }

def make_depol_noise(p1q=0.001, p2q=0.01):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1q, 1),
                                    ['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'id'])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), ['cx'])
    return nm

print("Funciones Qiskit cargadas.")


def transpiled_circuit_stats(compiled, n_cycles=1, measure_bath=True, basis_gates=None):
    qc = build_protocol_circuit(
        compiled,
        n_cycles=n_cycles,
        measure_bath=measure_bath,
        bath_measure_basis="Z",
    )

    backend = AerSimulator()
    qc_t = transpile(
        qc,
        backend=backend,
        basis_gates=basis_gates,
        optimization_level=0,
    )

    ops = dict(qc_t.count_ops())
    return {
        "depth": qc_t.depth(),
        "size": qc_t.size(),
        "width": qc_t.width(),
        "cx": ops.get("cx", 0),
        "measure": ops.get("measure", 0),
        "reset": ops.get("reset", 0),
        "rz": ops.get("rz", 0),
        "sx": ops.get("sx", 0),
        "x": ops.get("x", 0),
        "ops": ops,
    }


# ============================================================
# B.0 — utilidades locales para single spin
# ============================================================

NATIVE_BASIS = ["rz", "sx", "x", "cx", "measure", "reset"]


def make_compiled_single_spin(params):
    """
    Helper single-spin. Import lazy para evitar dependencia circular:
    tfm_single_spin.model importa tfm_shared.qiskit_impl.
    """
    from tfm_single_spin.model import build_single_spin_cfg

    cfg = build_single_spin_cfg(**params)
    compiled = compile_protocol(cfg)
    return cfg, compiled


def exact_and_dm_single_spin(params, rho0):
    """
    Comparación exacta NumPy vs simulación density_matrix de Qiskit.
    """
    from tfm_single_spin.model import run_single_spin_case

    cfg, compiled = make_compiled_single_spin(params)

    out_exact = run_single_spin_case(
        params,
        rho0=rho0,
        compute_fp=False,
    )

    out_dm = run_qiskit_density_matrix(
        compiled,
        n_cycles=params["n_cycles"],
        rho0_sys=rho0,
    )

    d_exact = np.array(
        [
            trace_distance_dm(r, out_exact["rho_g"])
            for r in out_exact["res"]["rhos"][1:]
        ],
        dtype=float,
    )

    return cfg, compiled, out_exact, out_dm, d_exact


def resource_scan_single_spin(
    base_params,
    sweep_name,
    values,
    *,
    n_cycles_for_circuit=4,
    optimization_level=0,
):
    rows = []
    backend = AerSimulator()

    for val in values:
        p = base_params.copy()
        p[sweep_name] = val

        _, compiled = make_compiled_single_spin(p)

        n_cycles_circ = (
            p["n_cycles"]
            if n_cycles_for_circuit is None
            else int(n_cycles_for_circuit)
        )

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

