#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_definitive_N3_mixed_v2.py

Estudio autocontenido DESDE CERO del protocolo térmico Lloyd--Abanin
para N=3, modelo Ising mixed.

No lee .npz ni .csv antiguos. No importa módulos del proyecto.
Solo usa numpy, scipy, pandas, matplotlib, pathlib + librería estándar.

Qué construye desde cero:
  - Sistema N=3 Ising transverse-field:
        Hs = -J sum_<i,j> X_i X_j - g sum_i Z_i
    con frontera periódica por defecto.
  - Baño de NB=N qubits:
        Hb = -h/2 sum_mu Z_mu
  - Acoplo mixed:
        V = sum_mu A_mu \otimes Y_mu^B,
        A_mu = (Z_mu + Y_mu)/sqrt(2)
  - Filtro gaussiano normalizado en [-T,T]:
        f(t) propto exp[-a^2 t^2/2], a=sqrt(4h/beta), integral f dt = 1.
  - Canal exacto finito:
        rho -> D_lambda( Tr_B[ U_theta rho\otimes|0><0| U_theta^dag ] )
    donde U_theta es la evolución time-ordered bajo
        H_T(t) = Hs\otimes I + I\otimes Hb + theta f(t) V.
  - Punto fijo del canal por iteración física y, además, por diagonalización/solución lineal como diagnósticos.
  - H_fix_corr = dev[-1/beta log(rho_fix) - Hs], C_num=H_fix_corr/theta^2.
  - Mean force reducido sin modulación:
        rho_mf_stored := Tr_B Gibbs(Hs+Hb+theta V)
    que aquí se llama "stored" porque es el benchmark histórico, pero NO se lee
    de ningún archivo: se calcula desde cero con f=1, tal como pidió Paula.
  - Mean force instant_fmax y f2avg_Hmf.
  - Lamb shift GLS, GDB y DeltaG mediante integrales discretizadas de segundo orden
    en el picture de interacción.
  - C_paper_formula en base de energía.
  - Comparaciones off-diagonal en base de energía.
  - Descomposición de Pauli en base computacional.

Ejecución típica:
    cd ~/TFM_Ising_MPS/exact_lambshift
    python run_from_scratch_N3_mixed.py

Ejecución más fina/lenta:
    python run_from_scratch_N3_mixed.py --n-time 301 --T-factor 5

Outputs:
    analysis_from_scratch_N3_mixed/
"""

from pathlib import Path
import argparse
import json
import sys
import time
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm, eigh, eig


# ============================================================
# Utilidades numéricas básicas
# ============================================================

EPS_LOG = 1e-14
EPS_DEN = 1e-14


def cplx(A):
    return np.array(A, dtype=np.complex128)


def dagger(A):
    return np.asarray(A).conj().T


def hermitize(A):
    A = cplx(A)
    return 0.5 * (A + dagger(A))


def fro_norm(A):
    A = np.asarray(A, dtype=complex)
    return float(np.linalg.norm(A.reshape(-1)))


def dev(A):
    """Parte desviadora: A - Tr(A)/d I."""
    A = cplx(A)
    d = A.shape[0]
    return A - np.trace(A) / d * np.eye(d, dtype=np.complex128)


def normalize_density(rho, clip=True):
    """Hermitiza, normaliza traza y opcionalmente proyecta autovalores negativos pequeños."""
    rho = hermitize(rho)
    tr = np.trace(rho)
    if abs(tr) < EPS_DEN:
        raise ValueError("density matrix con traza casi cero")
    rho = rho / tr
    rho = hermitize(rho)
    if clip:
        vals, vecs = eigh(rho)
        vals = np.real(vals)
        vals = np.clip(vals, 0.0, None)
        s = vals.sum()
        if s < EPS_DEN:
            vals = np.ones_like(vals) / len(vals)
        else:
            vals = vals / s
        rho = (vecs * vals) @ dagger(vecs)
        rho = hermitize(rho)
        rho = rho / np.trace(rho)
    return rho


def logm_density(rho, eps=EPS_LOG):
    """Logaritmo hermítico estable de una matriz densidad."""
    rho = normalize_density(rho, clip=True)
    vals, vecs = eigh(rho)
    vals = np.clip(np.real(vals), eps, None)
    L = (vecs * np.log(vals)) @ dagger(vecs)
    return hermitize(L)


def rho_beta(H, beta):
    """Gibbs: exp(-beta H)/Tr exp(-beta H), estable por shift de energía."""
    H = hermitize(H)
    vals, vecs = eigh(H)
    x = -beta * vals
    x = x - np.max(x)
    w = np.exp(x)
    w = w / np.sum(w)
    rho = (vecs * w) @ dagger(vecs)
    return normalize_density(rho, clip=True)


def Hcorr_from_state(rho, Hs, beta):
    """Hcorr = dev[-1/beta log(rho) - Hs]."""
    return hermitize(dev((-1.0 / beta) * logm_density(rho) - Hs))


def trace_distance(rho, sigma):
    """D(rho,sigma)=1/2 ||rho-sigma||_1 para matrices hermíticas."""
    A = hermitize(rho - sigma)
    vals = np.linalg.eigvalsh(A)
    return float(0.5 * np.sum(np.abs(vals)))


def matrix_sqrt_psd(rho):
    vals, vecs = eigh(hermitize(rho))
    vals = np.clip(np.real(vals), 0.0, None)
    return (vecs * np.sqrt(vals)) @ dagger(vecs)


def fidelity(rho, sigma):
    """F(rho,sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
    sr = matrix_sqrt_psd(rho)
    X = hermitize(sr @ sigma @ sr)
    vals = np.linalg.eigvalsh(X)
    vals = np.clip(np.real(vals), 0.0, None)
    return float(np.sum(np.sqrt(vals)) ** 2)


# ============================================================
# Operadores de Pauli y tensor products
# ============================================================

I2 = np.eye(2, dtype=np.complex128)
X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULI_SINGLE = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}


def kron_all(mats):
    out = np.array([[1.0 + 0.0j]])
    for M in mats:
        out = np.kron(out, M)
    return out


def local_op(single, site, N):
    mats = [I2] * N
    mats[site] = single
    return kron_all(mats)


def pauli_string_op(label):
    return kron_all([PAULI_SINGLE[ch] for ch in label])


# ============================================================
# Modelo N=3 Ising mixed construido desde cero
# ============================================================


def build_system_hamiltonian(N, J, g, periodic=True):
    """Hs = -J sum X_i X_j - g sum Z_i."""
    d = 2 ** N
    H = np.zeros((d, d), dtype=np.complex128)
    Xs = [local_op(X2, i, N) for i in range(N)]
    Zs = [local_op(Z2, i, N) for i in range(N)]
    for i in range(N):
        H += -g * Zs[i]
    pairs = [(i, i + 1) for i in range(N - 1)]
    if periodic and N > 2:
        pairs.append((N - 1, 0))
    for i, j in pairs:
        H += -J * (Xs[i] @ Xs[j])
    return hermitize(H), pairs


def build_bath_hamiltonian(NB, h):
    """Hb = -h/2 sum Z_mu."""
    dB = 2 ** NB
    Hb = np.zeros((dB, dB), dtype=np.complex128)
    for mu in range(NB):
        Hb += -(h / 2.0) * local_op(Z2, mu, NB)
    return hermitize(Hb)


def build_mixed_couplings_system(N):
    """A_mu = (Z_mu + Y_mu)/sqrt(2), el modo 'mixed'."""
    As = []
    for mu in range(N):
        A = (local_op(Z2, mu, N) + local_op(Y2, mu, N)) / np.sqrt(2.0)
        As.append(hermitize(A))
    return As


def build_total_operators(N, J, g, h, periodic=True):
    """Construye Hs, Hb, V, Hfree con orden tensorial sistema primero, baño segundo."""
    NB = N
    Hs, pairs = build_system_hamiltonian(N=N, J=J, g=g, periodic=periodic)
    Hb = build_bath_hamiltonian(NB=NB, h=h)
    As = build_mixed_couplings_system(N)
    dS = 2 ** N
    dB = 2 ** NB
    IB = np.eye(dB, dtype=np.complex128)
    IS = np.eye(dS, dtype=np.complex128)
    Hfree = np.kron(Hs, IB) + np.kron(IS, Hb)
    V = np.zeros_like(Hfree)
    for mu, A_mu in enumerate(As):
        Yb = local_op(Y2, mu, NB)
        V += np.kron(A_mu, Yb)
    return {
        "N": N,
        "NB": NB,
        "dS": dS,
        "dB": dB,
        "Hs": hermitize(Hs),
        "Hb": hermitize(Hb),
        "Hfree": hermitize(Hfree),
        "V": hermitize(V),
        "As": As,
        "pairs": pairs,
    }


# ============================================================
# Filtro gaussiano y evolución time-ordered
# ============================================================


def gaussian_filter_grid(beta, h, n_time=201, T_factor=4.0):
    """
    Filtro f(t)=C exp[-a^2 t^2/2], a=sqrt(4h/beta), en [-T,T].
    Se normaliza numéricamente para que sum_j f_j dt = 1.
    """
    a = np.sqrt(4.0 * h / beta)
    T = T_factor / a
    if n_time % 2 == 0:
        n_time += 1
    times = np.linspace(-T, T, n_time)
    dt = float(times[1] - times[0])
    f = np.exp(-0.5 * (a * times) ** 2)
    norm = np.sum(f) * dt
    f = f / norm
    return times, f, dt, T, a


def time_ordered_unitary(Hfree, V, theta, times, fvals):
    """
    U ≈ prod_j exp[-i dt (Hfree + theta f(t_j) V)].
    Se usa regla de puntos de malla. Para aumentar rigor, subir --n-time.
    """
    n = len(times)
    dt = float(times[1] - times[0])
    dim = Hfree.shape[0]
    U = np.eye(dim, dtype=np.complex128)
    for fj in fvals:
        Ht = Hfree + theta * fj * V
        U_step = expm(-1j * dt * Ht)
        U = U_step @ U
    return U


# ============================================================
# Canal exacto con reset y randomización
# ============================================================


def kraus_from_total_unitary(Utot, dS, dB):
    """K_b = <b_B| U |0_B>, orden tensorial S primero, B segundo."""
    Ut = Utot.reshape(dS, dB, dS, dB)
    kraus = []
    init_b = 0
    for b in range(dB):
        Kb = Ut[:, b, :, init_b]
        kraus.append(Kb)
    return kraus


def super_from_kraus(kraus):
    """Superoperador column-stacking: vec(K rho K^dag)=(K.conj() kron K) vec(rho)."""
    d = kraus[0].shape[0]
    S = np.zeros((d * d, d * d), dtype=np.complex128)
    for K in kraus:
        S += np.kron(K.conj(), K)
    return S


def apply_dephasing_randomization(rho, E, Ue, lam, T):
    """
    Randomización Lloyd--Abanin: en base de energía,
    rho_ab -> rho_ab / (1 - i (E_a-E_b) lambda T).
    Para lambda=0 es identidad.
    """
    if abs(lam) < EPS_DEN:
        return rho
    rhoE = dagger(Ue) @ rho @ Ue
    d = len(E)
    out = np.zeros_like(rhoE)
    for a in range(d):
        for b in range(d):
            omega = E[a] - E[b]
            out[a, b] = rhoE[a, b] / (1.0 - 1j * omega * lam * T)
    return Ue @ out @ dagger(Ue)


def randomization_super(E, Ue, lam, T):
    """
    Superoperador de la randomización D_lambda en convención vec_F.

    IMPORTANTE: no se rellena la matriz base mediante
        B.reshape(-1, order="F")[j] = 1
    porque NumPy puede devolver una copia al pedir orden Fortran sobre una
    matriz C-contigua. Ese bug hacía que S_rand fuese cero y destruía la traza.

    Para lambda=0, D_lambda es exactamente la identidad.
    """
    d = len(E)
    if abs(lam) < EPS_DEN:
        return np.eye(d * d, dtype=np.complex128)
    S = np.zeros((d * d, d * d), dtype=np.complex128)
    for j in range(d * d):
        v = np.zeros(d * d, dtype=np.complex128)
        v[j] = 1.0
        B = v.reshape((d, d), order="F")
        DB = apply_dephasing_randomization(B, E, Ue, lam, T)
        S[:, j] = DB.reshape(-1, order="F")
    return S


def apply_super_to_rho(S, rho):
    """Aplica superoperador column-stacking a una matriz densidad."""
    d = rho.shape[0]
    v = S @ rho.reshape(-1, order="F")
    return v.reshape((d, d), order="F")


def trace_row_for_vec(d):
    """Fila que implementa Tr(rho) sobre vec_F(rho)."""
    row = np.zeros(d * d, dtype=np.complex128)
    for i in range(d):
        row[i + i * d] = 1.0
    return row


def fixed_point_from_super_eig(S, d):
    """
    Candidato algebraico: autovector de S con autovalor más cercano a 1.
    OJO: si el canal tiene degeneraciones o casi-degeneraciones, este candidato
    puede no coincidir con el estado físico alcanzado desde una condición inicial.
    """
    vals, vecs = eig(S)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    rho_raw = vecs[:, idx].reshape((d, d), order="F")
    rho = normalize_density(rho_raw, clip=True)
    residual = fro_norm(S @ rho.reshape(-1, order="F") - rho.reshape(-1, order="F"))

    # gap espectral efectivo: distancia del segundo autovalor más cercano a 1.
    order = np.argsort(np.abs(vals - 1.0))
    eig1 = vals[order[0]]
    eig2 = vals[order[1]] if len(order) > 1 else np.nan + 0j
    gap_abs = float(1.0 - abs(eig2)) if np.isfinite(eig2) else np.nan
    n_close_1e8 = int(np.sum(np.abs(vals - 1.0) < 1e-8))
    n_close_1e6 = int(np.sum(np.abs(vals - 1.0) < 1e-6))
    return rho, eig1, residual, eig2, gap_abs, n_close_1e8, n_close_1e6


def fixed_point_from_super_lstsq(S, d):
    """
    Solución lineal de (S-I) vec(rho)=0 con Tr(rho)=1.
    Es útil como diagnóstico algebraico; si el fixed point no es único, devuelve
    una solución de mínimos cuadrados, no necesariamente la rama física.
    """
    A = S - np.eye(d * d, dtype=np.complex128)
    b = np.zeros(d * d, dtype=np.complex128)
    A2 = A.copy()
    b2 = b.copy()
    A2[-1, :] = trace_row_for_vec(d)
    b2[-1] = 1.0
    v, *_ = np.linalg.lstsq(A2, b2, rcond=None)
    rho = normalize_density(v.reshape((d, d), order="F"), clip=True)
    residual = fro_norm(S @ rho.reshape(-1, order="F") - rho.reshape(-1, order="F"))
    return rho, residual


def iterate_channel_to_fixed(S, rho0, max_cycles=200000, tol=1e-13, check_every=25):
    """
    Termalización explícita por ciclos:
        rho_{n+1}=E(rho_n)
    Devuelve el estado alcanzado desde rho0. Esto es el fixed point físico
    relevante si el canal no es primitivo o tiene subespacios casi estacionarios.
    """
    rho = normalize_density(rho0, clip=True)
    last_delta = np.inf
    converged = False
    n_done = 0
    for n in range(1, max_cycles + 1):
        rho_next = apply_super_to_rho(S, rho)
        # No proyectamos agresivamente en cada paso: solo hermitizamos/traza.
        rho_next = hermitize(rho_next)
        tr = np.trace(rho_next)
        if abs(tr) > EPS_DEN:
            rho_next = rho_next / tr
        if n % check_every == 0 or n == 1:
            last_delta = trace_distance(normalize_density(rho_next, clip=True),
                                        normalize_density(rho, clip=True))
            if last_delta < tol:
                converged = True
                rho = rho_next
                n_done = n
                break
        rho = rho_next
        n_done = n
    rho = normalize_density(rho, clip=True)
    residual = fro_norm(S @ rho.reshape(-1, order="F") - rho.reshape(-1, order="F"))
    return rho, {
        "iter_cycles": int(n_done),
        "iter_converged": bool(converged),
        "iter_last_delta_trace": float(last_delta),
        "iter_residual_F": float(residual),
    }


def exact_channel_fixed_point(model, theta, lam, times, fvals, T, evals_Hs, Ue_Hs,
                              rho_beta_target=None, fixed_method="iterate",
                              max_cycles=200000, tol_cycle=1e-13):
    """
    Construye el superoperador exacto de un ciclo y obtiene rho_fix.

    Se calculan tres diagnósticos:
      - eig: autovector de S con eigenvalue más cercano a 1.
      - lstsq: solución algebraica con Tr(rho)=1.
      - iter: estado obtenido al iterar el canal desde I/d.

    Por defecto, el rho_fix usado en las tablas es el iterado desde I/d, porque
    representa la termalización real del protocolo desde el estado inicial físico.
    """
    Utot = time_ordered_unitary(model["Hfree"], model["V"], theta, times, fvals)
    kraus = kraus_from_total_unitary(Utot, model["dS"], model["dB"])
    S_reset = super_from_kraus(kraus)
    S_rand = randomization_super(evals_Hs, Ue_Hs, lam, T)
    S = S_rand @ S_reset
    d = model["dS"]

    # Diagnósticos de preservación de traza.
    # kraus_TP_error_F comprueba solo el reset. super_TP_error_F comprueba
    # el ciclo completo, incluida la randomización. Ambos deben ser pequeños.
    tp_error = fro_norm(sum(dagger(K) @ K for K in kraus) - np.eye(d, dtype=np.complex128))
    trace_row = trace_row_for_vec(d)
    super_tp_error = float(np.linalg.norm(trace_row @ S - trace_row))
    S_rand_identity_error = (
        fro_norm(S_rand - np.eye(d * d, dtype=np.complex128)) if abs(lam) < EPS_DEN else np.nan
    )

    rho_eig, eigval, eig_residual, eig2, gap_abs, nclose8, nclose6 = fixed_point_from_super_eig(S, d)
    rho_lstsq, lstsq_residual = fixed_point_from_super_lstsq(S, d)

    rho_mixed = np.eye(d, dtype=np.complex128) / d
    rho_iter_mixed, diag_mixed = iterate_channel_to_fixed(
        S, rho_mixed, max_cycles=max_cycles, tol=tol_cycle
    )

    if rho_beta_target is not None:
        rho_iter_beta, diag_beta = iterate_channel_to_fixed(
            S, rho_beta_target, max_cycles=max_cycles, tol=tol_cycle
        )
    else:
        rho_iter_beta, diag_beta = rho_iter_mixed.copy(), diag_mixed.copy()

    D_iter_mixed_vs_beta_init = trace_distance(rho_iter_mixed, rho_iter_beta)
    D_iter_mixed_vs_eig = trace_distance(rho_iter_mixed, rho_eig)
    D_iter_mixed_vs_lstsq = trace_distance(rho_iter_mixed, rho_lstsq)

    if fixed_method == "eig":
        rho_fix = rho_eig
        method_used = "eig_nearest_one"
    elif fixed_method == "lstsq":
        rho_fix = rho_lstsq
        method_used = "linear_lstsq_trace_constraint"
    else:
        rho_fix = rho_iter_mixed
        method_used = "iterate_from_maximally_mixed"

    residual = fro_norm(S @ rho_fix.reshape(-1, order="F") - rho_fix.reshape(-1, order="F"))
    diagnostics = {
        "fixed_method_used": method_used,
        "kraus_TP_error_F": float(tp_error),
        "super_TP_error_F": float(super_tp_error),
        "S_rand_identity_error_F_if_lambda0": float(S_rand_identity_error) if np.isfinite(S_rand_identity_error) else np.nan,
        "eigval1_real": float(np.real(eigval)),
        "eigval1_imag": float(np.imag(eigval)),
        "eig_residual_F": float(eig_residual),
        "eigval2_real": float(np.real(eig2)),
        "eigval2_imag": float(np.imag(eig2)),
        "spectral_gap_abs_1_minus_abs_eig2": float(gap_abs),
        "n_eigvals_close_to_1_tol_1e8": int(nclose8),
        "n_eigvals_close_to_1_tol_1e6": int(nclose6),
        "lstsq_residual_F": float(lstsq_residual),
        "D_iter_mixed_vs_iter_beta": float(D_iter_mixed_vs_beta_init),
        "D_iter_mixed_vs_eig": float(D_iter_mixed_vs_eig),
        "D_iter_mixed_vs_lstsq": float(D_iter_mixed_vs_lstsq),
        "iter_mixed_cycles": int(diag_mixed["iter_cycles"]),
        "iter_mixed_converged": bool(diag_mixed["iter_converged"]),
        "iter_mixed_last_delta_trace": float(diag_mixed["iter_last_delta_trace"]),
        "iter_mixed_residual_F": float(diag_mixed["iter_residual_F"]),
        "iter_beta_cycles": int(diag_beta["iter_cycles"]),
        "iter_beta_converged": bool(diag_beta["iter_converged"]),
        "iter_beta_last_delta_trace": float(diag_beta["iter_last_delta_trace"]),
        "iter_beta_residual_F": float(diag_beta["iter_residual_F"]),
    }
    return rho_fix, eigval, residual, S, diagnostics, rho_eig, rho_lstsq, rho_iter_beta

# ============================================================
# Partial trace y mean force reducido
# ============================================================


def ptrace_bath(rho_tot, dS, dB):
    """Tr_B con orden S primero, B segundo."""
    R = rho_tot.reshape(dS, dB, dS, dB)
    rhoS = np.einsum("ibjb->ij", R)
    return normalize_density(rhoS, clip=True)


def rho_eff_from_total_gibbs(Htot, beta, dS, dB):
    rhoT = rho_beta(Htot, beta)
    return ptrace_bath(rhoT, dS, dB)


def mean_force_variants(model, beta, theta, times, fvals):
    """
    Tres variantes principales:
      1. stored_rho_mf: sin modulación, Hfree + theta V.
      2. instant_fmax: Hfree + theta fmax V.
      3. f2avg_Hmf: promedio de H_mf(t) ponderado por f(t)^2.
    """
    Hs = model["Hs"]
    Hfree = model["Hfree"]
    V = model["V"]
    dS = model["dS"]
    dB = model["dB"]

    out = {}

    # 1. Benchmark histórico definido aquí DESDE CERO: f=1, sin modulación.
    rho_unmod = rho_eff_from_total_gibbs(Hfree + theta * V, beta, dS, dB)
    Hmf_unmod = Hcorr_from_state(rho_unmod, Hs, beta)
    out["stored_rho_mf"] = {
        "rho_eff": rho_unmod,
        "Hmf": Hmf_unmod,
        "source": "from_scratch_unmodulated_f_eq_1",
    }

    # 2. Instante con |f| máximo.
    idx = int(np.argmax(np.abs(fvals)))
    fmax = float(fvals[idx])
    rho_fmax = rho_eff_from_total_gibbs(Hfree + theta * fmax * V, beta, dS, dB)
    Hmf_fmax = Hcorr_from_state(rho_fmax, Hs, beta)
    out["instant_fmax"] = {
        "rho_eff": rho_fmax,
        "Hmf": Hmf_fmax,
        "source": f"from_scratch_instant_fmax={fmax:.12g}",
    }

    # 3. Promedio de Hmf(t) ponderado por f(t)^2.
    weights = np.abs(fvals) ** 2
    weights = weights / np.sum(weights)
    Havg = np.zeros_like(Hs, dtype=np.complex128)
    rho_avg = np.zeros_like(Hs, dtype=np.complex128)
    for w, fj in zip(weights, fvals):
        rho_t = rho_eff_from_total_gibbs(Hfree + theta * float(fj) * V, beta, dS, dB)
        Hmf_t = Hcorr_from_state(rho_t, Hs, beta)
        Havg += w * Hmf_t
        rho_avg += w * rho_t
    Havg = hermitize(dev(Havg))
    rho_from_Havg = rho_beta(Hs + Havg, beta)
    rho_avg = normalize_density(rho_avg, clip=True)
    out["f2avg_Hmf"] = {
        "rho_eff": rho_from_Havg,
        "Hmf": Havg,
        "source": "from_scratch_f2_weighted_average_of_Hmf_t",
    }
    out["f2avg_rho_diagnostic"] = {
        "rho_eff": rho_avg,
        "Hmf": Hcorr_from_state(rho_avg, Hs, beta),
        "source": "diagnostic_f2_weighted_average_of_rho_eff_t",
    }
    return out


# ============================================================
# GLS, GDB, DeltaG y Cpaper_formula
# ============================================================


def interaction_picture_A_grid(A, evals, Ue, times):
    """A_I(t)=exp(iHs t) A exp(-iHs t)."""
    AE = dagger(Ue) @ A @ Ue
    d = len(evals)
    grid = []
    for t in times:
        phase = np.exp(1j * (evals[:, None] - evals[None, :]) * t)
        AtE = phase * AE
        grid.append(hermitize(Ue @ AtE @ dagger(Ue)))
    return grid


def compute_GLS_GDB_DeltaG(model, beta, h, times, fvals, evals, Ue):
    """
    Calcula operadores de segundo orden del protocolo en el picture de interacción.

    L_mu = int dt f(t) exp(i h t) A_mu(t)
    K    = 1/2 sum_mu L_mu^dag L_mu
    M_mu = int_{t>s} dt ds f(t)f(s) exp[i h (s-t)] A_mu(t) A_mu(s)
    GLS  = sum_mu (M_mu - M_mu^dag)/(2i)

    GDB en base de energía:
        GDB_ab = -i tanh[ beta (E_b-E_a)/4 ] K_ab.
    """
    Hs = model["Hs"]
    d = Hs.shape[0]
    dt = float(times[1] - times[0])
    Ls = []
    K = np.zeros((d, d), dtype=np.complex128)
    GLS = np.zeros((d, d), dtype=np.complex128)

    for A in model["As"]:
        Agrid = interaction_picture_A_grid(A, evals, Ue, times)

        L = np.zeros((d, d), dtype=np.complex128)
        for t, f, At in zip(times, fvals, Agrid):
            L += dt * float(f) * np.exp(1j * h * t) * At
        Ls.append(L)
        K += 0.5 * dagger(L) @ L

        # M usando suma acumulada para no hacer doble loop explícito.
        M = np.zeros((d, d), dtype=np.complex128)
        S_pre = np.zeros((d, d), dtype=np.complex128)
        for t, f, At in zip(times, fvals, Agrid):
            # contribución t_j > t_k ya acumulada en S_pre
            M += dt * float(f) * np.exp(-1j * h * t) * (At @ S_pre)
            S_pre += dt * float(f) * np.exp(1j * h * t) * At
        GLS += (M - dagger(M)) / (2j)

    K = hermitize(K)
    GLS = hermitize(GLS)

    KE = dagger(Ue) @ K @ Ue
    GDBE = np.zeros_like(KE)
    d = len(evals)
    for a in range(d):
        for b in range(d):
            omega_paper = evals[b] - evals[a]
            GDBE[a, b] = -1j * np.tanh(beta * omega_paper / 4.0) * KE[a, b]
    GDB = hermitize(Ue @ GDBE @ dagger(Ue))
    DeltaG = hermitize(GLS - GDB)
    return {
        "Ls": Ls,
        "K": K,
        "GLS": GLS,
        "GDB": GDB,
        "DeltaG": DeltaG,
    }


def Cpaper_formula_from_DeltaG(DeltaG, evals, Ue, lam, T):
    """
    Fórmula pedida por Paula, en base de energía:

    Para a != b, omega_ab = E_a - E_b,
        C_ab = [ i omega_ab exp(i omega_ab T)
                 / (1 - i lambda omega_ab T - exp(2 i omega_ab T)) ] DeltaG_ab
    diagonal cero.
    """
    DE = dagger(Ue) @ DeltaG @ Ue
    d = len(evals)
    CE = np.zeros_like(DE)
    for a in range(d):
        for b in range(d):
            if a == b:
                CE[a, b] = 0.0
            else:
                omega = evals[a] - evals[b]

                # Caso degenerado omega -> 0:
                # límite de i omega exp(i omega T) /
                # (1 - i lambda omega T - exp(2 i omega T))
                # = -1 / ((lambda + 2) T)
                if abs(omega) < 1e-12:
                    pref = -1.0 / ((lam + 2.0) * T)
                    CE[a, b] = pref * DE[a, b]
                else:
                    denom = 1.0 - 1j * lam * omega * T - np.exp(2j * omega * T)
                    if abs(denom) < 1e-12:
                        CE[a, b] = np.nan + 1j * np.nan
                    else:
                        pref = 1j * omega * np.exp(1j * omega * T) / denom
                        CE[a, b] = pref * DE[a, b]
    C = Ue @ CE @ dagger(Ue)
    return hermitize(dev(C))


# ============================================================
# Comparaciones off-diagonal en base de energía
# ============================================================


def to_energy_basis(O, Ue):
    return dagger(Ue) @ O @ Ue


def offdiag_energy(O, Ue):
    OE = to_energy_basis(O, Ue)
    return OE - np.diag(np.diag(OE))


def offdiag_metrics(A, B, Ue):
    """
    s(A,B), alpha*(A,B), r(A,B), todo off-diagonal en base de energía.
    alpha minimiza ||A_off - alpha B_off||_F.
    """
    Aoff = offdiag_energy(A, Ue)
    Boff = offdiag_energy(B, Ue)
    nA = np.linalg.norm(Aoff, ord="fro")
    nB = np.linalg.norm(Boff, ord="fro")
    if nA < EPS_DEN or nB < EPS_DEN:
        return np.nan, np.nan, np.nan
    inner = np.trace(dagger(Aoff) @ Boff)
    cos = float(np.real(inner) / (nA * nB))
    alpha = float(np.real(np.trace(dagger(Boff) @ Aoff)) / (nB ** 2))
    resid = float(np.linalg.norm(Aoff - alpha * Boff, ord="fro") / nA)
    return cos, alpha, resid


# ============================================================
# Pauli decomposition en base computacional
# ============================================================


def pauli_decomposition(O, N):
    O = cplx(O)
    d = 2 ** N
    rows = []
    for label_tuple in product("IXYZ", repeat=N):
        label = "".join(label_tuple)
        P = pauli_string_op(label)
        coeff = np.trace(dagger(P) @ O) / d
        weight = sum(ch != "I" for ch in label)
        rows.append({
            "pauli": label,
            "coeff_real": float(np.real(coeff)),
            "coeff_imag": float(np.imag(coeff)),
            "abs_coeff": float(abs(coeff)),
            "weight": int(weight),
        })
    df = pd.DataFrame(rows).sort_values("abs_coeff", ascending=False)
    return df


def save_pauli_outputs(outdir, operators_by_case, N):
    pdir = outdir / "pauli_operators"
    pdir.mkdir(parents=True, exist_ok=True)
    combined_top = []
    for case_key, ops in operators_by_case.items():
        dataset, lam, theta2 = case_key
        for opname, O in ops.items():
            safe_op = opname.replace("^", "").replace(" ", "_").replace("/", "_")
            fname = pdir / f"pauli_{safe_op}_{dataset}_lambda{lam:g}_theta2{theta2:.6g}.csv"
            df = pauli_decomposition(O, N)
            df.to_csv(fname, index=False)
            top = df.head(15).copy()
            top.insert(0, "operator", opname)
            top.insert(0, "theta2", theta2)
            top.insert(0, "lambda", lam)
            top.insert(0, "dataset", dataset)
            combined_top.append(top)

            # Plot de los 15 términos dominantes por operador.
            fig, ax = plt.subplots(figsize=(7, 3.5))
            tplot = df.head(15).iloc[::-1]
            ax.barh(tplot["pauli"], tplot["abs_coeff"])
            ax.set_xlabel("|c_P|")
            ax.set_title(f"Top 15 Pauli: {opname}\n{dataset}, lambda={lam:g}, theta^2={theta2:.4g}", fontsize=9)
            fig.tight_layout()
            fig.savefig(pdir / f"top15_{safe_op}_{dataset}_lambda{lam:g}_theta2{theta2:.6g}.png", dpi=180)
            plt.close(fig)
    if combined_top:
        pd.concat(combined_top, ignore_index=True).to_csv(pdir / "pauli_top15_combined.csv", index=False)


# ============================================================
# Plots resumen de seis paneles
# ============================================================


def plot_summary(df, outdir, dataset, lam):
    sub = df[(df["dataset"] == dataset) & (np.isclose(df["lambda"], lam))].copy()
    if sub.empty:
        return
    sub = sub.sort_values("theta2")
    variants = ["stored_rho_mf", "instant_fmax", "f2avg_Hmf"]
    variant_labels = {
        "stored_rho_mf": "stored=f=1",
        "instant_fmax": "instant fmax",
        "f2avg_Hmf": "f2avg Hmf",
    }

    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    ax = axs.ravel()

    # Panel 1: log difference.
    for v in variants:
        s = sub[sub["variant"] == v]
        if not s.empty:
            ax[0].plot(s["theta2"], s["norm_logeff_minus_logfix_dev_F"], marker="o", label=variant_labels[v])
    ax[0].set_title("1. ||dev(log rho_eff - log rho_fix)||_F", fontsize=10)
    ax[0].set_xlabel(r"$\theta^2$")
    ax[0].set_ylabel("norma")
    ax[0].legend(fontsize=7)

    # Panel 2: direct state distances.
    for v in variants:
        s = sub[sub["variant"] == v]
        if not s.empty:
            ax[1].plot(s["theta2"], s["D_fix_eff"], marker="o", label=variant_labels[v])
    first = sub[sub["variant"] == variants[0]]
    if not first.empty:
        ax[1].plot(first["theta2"], first["D_fix_beta"], marker="x", linestyle="--", label="D(fix,beta)")
    ax[1].set_title("2. Distancias directas", fontsize=10)
    ax[1].set_xlabel(r"$\theta^2$")
    ax[1].set_ylabel("trace distance")
    ax[1].legend(fontsize=7)

    # Panel 3: normas.
    base = first.drop_duplicates("theta2") if not first.empty else sub.drop_duplicates("theta2")
    norm_cols = [
        ("norm_Hfixcorr_F", "Hfixcorr"),
        ("norm_theta2_Cpaper_F", "theta2 Cpaper"),
        ("norm_theta2_Cpaper_formula_F", "theta2 Cpaper_formula"),
        ("norm_theta2_GLS_F", "theta2 GLS"),
        ("norm_theta2_DeltaG_F", "theta2 DeltaG"),
    ]
    for col, lab in norm_cols:
        if col in base:
            ax[2].plot(base["theta2"], base[col], marker="o", label=lab)
    for v in variants:
        s = sub[sub["variant"] == v]
        if not s.empty:
            ax[2].plot(s["theta2"], s["norm_Hmf_F"], marker=".", linestyle="--", label=f"Hmf {variant_labels[v]}")
    ax[2].set_title("3. Normas: escala, no estructura", fontsize=10)
    ax[2].set_xlabel(r"$\theta^2$")
    ax[2].set_ylabel("Frobenius")
    ax[2].legend(fontsize=6)

    # Panel 4: similitudes off-diagonal.
    sim_invariant = [
        ("abs_cos_Cnum_Cpaper_offE", "Cpaper"),
        ("abs_cos_Cnum_Cpaper_formula_offE", "Cpaper_formula"),
        ("abs_cos_Cnum_GLS_offE", "GLS"),
        ("abs_cos_Cnum_DeltaG_offE", "DeltaG"),
    ]
    for col, lab in sim_invariant:
        if col in base:
            ax[3].plot(base["theta2"], base[col], marker="o", label=lab)
    for v in variants:
        s = sub[sub["variant"] == v]
        if not s.empty:
            ax[3].plot(s["theta2"], s["abs_cos_Cnum_Cmf_offE"], marker=".", linestyle="--", label=f"Cmf {variant_labels[v]}")
    ax[3].set_title("4. |s| off-diagonal en base de energía", fontsize=10)
    ax[3].set_xlabel(r"$\theta^2$")
    ax[3].set_ylim(-0.05, 1.05)
    ax[3].legend(fontsize=6)

    # Panel 5: residuales tras ajuste de escala.
    resid_invariant = [
        ("resid_Cnum_vs_Cpaper_offE", "Cpaper"),
        ("resid_Cnum_vs_Cpaper_formula_offE", "Cpaper_formula"),
        ("resid_Cnum_vs_GLS_offE", "GLS"),
        ("resid_Cnum_vs_DeltaG_offE", "DeltaG"),
    ]
    for col, lab in resid_invariant:
        if col in base:
            ax[4].plot(base["theta2"], base[col], marker="o", label=lab)
    for v in variants:
        s = sub[sub["variant"] == v]
        if not s.empty:
            ax[4].plot(s["theta2"], s["resid_Cnum_vs_Cmf_offE"], marker=".", linestyle="--", label=f"Cmf {variant_labels[v]}")
    ax[4].set_title("5. Residual relativo tras escala óptima", fontsize=10)
    ax[4].set_xlabel(r"$\theta^2$")
    ax[4].set_ylabel("menor es mejor")
    ax[4].legend(fontsize=6)

    # Panel 6: state candidates.
    if not base.empty:
        ax[5].plot(base["theta2"], base["D_fix_beta"], marker="x", linestyle="--", label="rho_beta")
        ax[5].plot(base["theta2"], base["D_fix_Gibbs_Hs_plus_theta2_Cpaper"], marker="o", label="Hs+theta2 Cpaper")
        ax[5].plot(base["theta2"], base["D_fix_Gibbs_Hs_plus_theta2_Cpaper_formula"], marker="o", label="Hs+theta2 Cpaper_formula")
    for v in variants:
        s = sub[sub["variant"] == v]
        if not s.empty:
            ax[5].plot(s["theta2"], s["D_fix_Gibbs_Hs_plus_Hmf"], marker=".", linestyle="--", label=f"Hs+Hmf {variant_labels[v]}")
    ax[5].set_title("6. D(rho_fix, Gibbs candidato)", fontsize=10)
    ax[5].set_xlabel(r"$\theta^2$")
    ax[5].set_ylabel("trace distance")
    ax[5].legend(fontsize=6)

    for a in ax:
        a.grid(True, alpha=0.25)
    fig.suptitle(f"From scratch N=3 Ising mixed: {dataset}, lambda={lam:g}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = outdir / "plots" / f"summary_{dataset}_lambda{lam:g}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)


# ============================================================
# Resumen textual automático
# ============================================================


def automatic_summary(df):
    print("\n" + "=" * 80)
    print("RESUMEN AUTOMÁTICO DESDE CERO, usando solo theta^2 <= 0.05")
    print("Criterio: no concluir por normas; mirar |cos| y residual off-diagonal.")
    print("=" * 80)
    if df.empty:
        print("No hay filas válidas.")
        return
    small = df[df["theta2"] <= 0.05].copy()
    if small.empty:
        print("No hay filas con theta^2 <= 0.05.")
        return

    candidates_invariant = [
        ("Cpaper", "abs_cos_Cnum_Cpaper_offE", "resid_Cnum_vs_Cpaper_offE"),
        ("Cpaper_formula", "abs_cos_Cnum_Cpaper_formula_offE", "resid_Cnum_vs_Cpaper_formula_offE"),
        ("GLS", "abs_cos_Cnum_GLS_offE", "resid_Cnum_vs_GLS_offE"),
        ("DeltaG", "abs_cos_Cnum_DeltaG_offE", "resid_Cnum_vs_DeltaG_offE"),
    ]
    variants = ["stored_rho_mf", "instant_fmax", "f2avg_Hmf"]

    for (dataset, lam), sub in small.groupby(["dataset", "lambda"]):
        print(f"\nCaso: {dataset}, lambda={lam:g}")
        base = sub[sub["variant"] == "stored_rho_mf"].copy()
        if base.empty:
            base = sub.drop_duplicates("theta2")
        scores = []
        for name, ccol, rcol in candidates_invariant:
            cmean = float(np.nanmean(base[ccol])) if ccol in base else np.nan
            rmean = float(np.nanmean(base[rcol])) if rcol in base else np.nan
            scores.append((name, cmean, rmean))
            print(f"  {name:16s}: mean |cos|={cmean: .4f}, mean residual={rmean: .4f}")
        for v in variants:
            sv = sub[sub["variant"] == v]
            cmean = float(np.nanmean(sv["abs_cos_Cnum_Cmf_offE"])) if not sv.empty else np.nan
            rmean = float(np.nanmean(sv["resid_Cnum_vs_Cmf_offE"])) if not sv.empty else np.nan
            scores.append((f"Cmf::{v}", cmean, rmean))
            print(f"  Cmf::{v:12s}: mean |cos|={cmean: .4f}, mean residual={rmean: .4f}")
        valid_cos = [x for x in scores if np.isfinite(x[1])]
        valid_res = [x for x in scores if np.isfinite(x[2])]
        if valid_cos:
            best_cos = max(valid_cos, key=lambda x: x[1])
            print(f"  Mayor similitud media: {best_cos[0]} con |cos|={best_cos[1]:.4f}")
        if valid_res:
            best_res = min(valid_res, key=lambda x: x[2])
            print(f"  Menor residual medio: {best_res[0]} con residual={best_res[2]:.4f}")
        built = sub["microscopic_status"].dropna().unique().tolist()
        print(f"  rho_eff(t): construido rigurosamente desde Hs,Hb,V,f(t). status={built}")


# ============================================================
# Main
# ============================================================


def parse_theta2_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def save_npz(path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="From-scratch N=3 Ising mixed Lloyd--Abanin study")
    parser.add_argument("--N", type=int, default=3, help="Número de qubits del sistema. Default: 3")
    parser.add_argument("--J", type=float, default=1.0, help="Acoplo Ising XX. Default: 1")
    parser.add_argument("--g", type=float, default=1.0, help="Campo transversal Z. Default: 1")
    parser.add_argument("--beta", type=float, default=1.0, help="Inversa de temperatura. Default: 1")
    parser.add_argument("--h", type=float, default=None, help="Energía del baño. Default: max(2g,4J)")
    parser.add_argument("--open-boundary", action="store_true", help="Usar cadena abierta. Default: periódica")
    parser.add_argument("--n-time", type=int, default=161, help="Puntos temporales del filtro/evolución. Default: 161")
    parser.add_argument("--T-factor", type=float, default=4.0, help="T=T_factor/a. Default: 4")
    parser.add_argument("--theta2-general", type=str, default="0.005,0.01,0.02,0.04,0.05",
                        help="Lista theta^2 para mixed_general")
    parser.add_argument("--theta2-wide", type=str, default="0.005,0.01,0.02,0.04,0.08,0.12,0.2,0.3,0.5",
                        help="Lista theta^2 para mixed_theta_wide. Default llega hasta 0.5")
    parser.add_argument("--lambdas", type=str, default="0,2", help="Lista lambdas. Default: 0,2")
    parser.add_argument("--out", type=str, default="analysis_definitive_N3_mixed", help="Carpeta de output")
    parser.add_argument("--include-f2avg-rho", action="store_true", help="Incluir diagnóstico f2avg_rho en CSV/plots secundarios")
    parser.add_argument("--fixed-method", choices=["iterate", "eig", "lstsq"], default="iterate",
                        help="Cómo elegir rho_fix. Default: iterate desde I/d, que representa termalización por ciclos.")
    parser.add_argument("--max-cycles", type=int, default=200000,
                        help="Máximo de ciclos para iterar el canal si --fixed-method iterate. Default: 200000")
    parser.add_argument("--tol-cycle", type=float, default=1e-13,
                        help="Tolerancia de convergencia en trace distance entre checks. Default: 1e-13")
    args = parser.parse_args()

    if args.N != 3:
        print("[AVISO] Este script está pensado para N=3; se permite otro N, pero Pauli/tiempos pueden crecer.")
    h = float(args.h) if args.h is not None else max(2.0 * args.g, 4.0 * args.J)
    periodic = not args.open_boundary

    outdir = Path(args.out).resolve()
    for sub in ["plots", "tables", "matrices", "logs", "pauli_operators"]:
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    cfg = {
        "mode": "from_scratch_no_old_npz_no_old_csv",
        "N": args.N,
        "NB": args.N,
        "J": args.J,
        "g": args.g,
        "beta": args.beta,
        "h": h,
        "periodic": periodic,
        "n_time": args.n_time,
        "T_factor": args.T_factor,
        "theta2_general": parse_theta2_list(args.theta2_general),
        "theta2_wide": parse_theta2_list(args.theta2_wide),
        "lambdas": parse_theta2_list(args.lambdas),
        "definition_Hs": "-J sum_<ij> X_i X_j - g sum_i Z_i",
        "definition_Hb": "-h/2 sum_mu Z_mu",
        "definition_mixed_A_mu": "(Z_mu + Y_mu)/sqrt(2)",
        "definition_V": "sum_mu A_mu tensor Y_mu^B",
        "definition_stored_rho_mf": "Tr_B Gibbs(Hfree + theta V), f=1, no modulation, generated from scratch",
        "packages": "numpy scipy pandas matplotlib pathlib only, plus stdlib",
        "fixed_method": args.fixed_method,
        "max_cycles": args.max_cycles,
        "tol_cycle": args.tol_cycle,
    }
    with open(outdir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Logs explícitos para evitar confusión con análisis antiguos.
    pd.DataFrame([{
        "message": "No se han leído CSV antiguos ni NPZ antiguos. Este estudio construye Hs,Hb,V,f(t), canal y rho_eff desde cero.",
        "root": str(Path.cwd().resolve()),
    }]).to_csv(outdir / "logs" / "no_old_inputs_used.csv", index=False)
    pd.DataFrame(columns=["file", "reason"]).to_csv(outdir / "logs" / "skipped.csv", index=False)

    t0 = time.time()
    print("Construyendo modelo N=3 Ising mixed desde cero...")
    print(f"  Hs = -J sum XX - g sum Z, J={args.J}, g={args.g}, periodic={periodic}")
    print(f"  Hb = -h/2 sum Z, h={h}")
    print(f"  beta={args.beta}")
    model = build_total_operators(args.N, args.J, args.g, h, periodic=periodic)
    Hs = model["Hs"]
    dS = model["dS"]
    evals, Ue = eigh(Hs)
    rhoB = rho_beta(Hs, args.beta)
    times, fvals, dt, T, a = gaussian_filter_grid(args.beta, h, n_time=args.n_time, T_factor=args.T_factor)
    print(f"  filtro: a={a:.8g}, T={T:.8g}, n_time={len(times)}, dt={dt:.4g}, integral≈{np.sum(fvals)*dt:.12g}")

    # Status de construcción: todo construido desde cero.
    build_status = pd.DataFrame([{
        "file": "from_scratch",
        "dataset": "all",
        "found_Hb": True,
        "found_V": True,
        "found_Hfree": True,
        "found_f_values": True,
        "found_t_values": True,
        "could_build_rhoeff_t": True,
        "microscopic_origin": "constructed_from_explicit_N3_Ising_mixed_defaults",
        "filter_origin": "constructed_gaussian_from_beta_h_normalized_on_grid",
        "warnings": "none",
    }])
    build_status.to_csv(outdir / "tables" / "build_status.csv", index=False)

    # Guardar matrices base.
    save_npz(outdir / "matrices" / "base_model_N3_mixed.npz",
             Hs=Hs, Hb=model["Hb"], Hfree=model["Hfree"], V=model["V"],
             evals_Hs=evals, Ue_Hs=Ue, rho_beta=rhoB,
             times=times, f_values=fvals, beta=args.beta, h=h, J=args.J, g=args.g, T=T, a=a)

    # Operadores perturbativos del protocolo independientes de theta.
    print("Calculando GLS, GDB, DeltaG desde integrales de segundo orden...")
    pert = compute_GLS_GDB_DeltaG(model, args.beta, h, times, fvals, evals, Ue)
    save_npz(outdir / "matrices" / "perturbative_GLS_GDB_DeltaG.npz",
             K=pert["K"], GLS=pert["GLS"], GDB=pert["GDB"], DeltaG=pert["DeltaG"])

    datasets = {
        "mixed_general": parse_theta2_list(args.theta2_general),
        "mixed_theta_wide": parse_theta2_list(args.theta2_wide),
    }
    lambdas = parse_theta2_list(args.lambdas)

    rows = []
    consistency_rows = []
    basis_rows = []
    fixed_cache = {}
    mf_cache = {}
    pauli_cases = {}

    # Calcular todos los casos.
    for dataset, theta2_values in datasets.items():
        for lam in lambdas:
            Cpaper_formula = Cpaper_formula_from_DeltaG(pert["DeltaG"], evals, Ue, lam, T)
            # Desde cero identificamos Cpaper con la fórmula del paper.
            Cpaper = Cpaper_formula.copy()
            for theta2 in theta2_values:
                theta = float(np.sqrt(theta2))
                print(f"\nCaso {dataset}, lambda={lam:g}, theta^2={theta2:g}...")

                # Mean force cacheado por theta: no depende de lambda ni de dataset.
                if theta not in mf_cache:
                    print("  mean force: stored=f=1, instant_fmax, f2avg_Hmf")
                    mf_cache[theta] = mean_force_variants(model, args.beta, theta, times, fvals)
                mfvars = mf_cache[theta]

                # Canal exacto cacheado por theta/lambda.
                key_fp = (theta, lam)
                if key_fp not in fixed_cache:
                    print("  canal exacto y punto fijo...")
                    rho_fix, eigval, fp_residual, S, fp_diag, rho_eig, rho_lstsq, rho_iter_beta = exact_channel_fixed_point(
                        model, theta, lam, times, fvals, T, evals, Ue,
                        rho_beta_target=rhoB,
                        fixed_method=args.fixed_method,
                        max_cycles=args.max_cycles,
                        tol_cycle=args.tol_cycle,
                    )
                    Hfixcorr = Hcorr_from_state(rho_fix, Hs, args.beta)
                    Cnum = Hfixcorr / theta2
                    fixed_cache[key_fp] = {
                        "rho_fix": rho_fix,
                        "rho_eig": rho_eig,
                        "rho_lstsq": rho_lstsq,
                        "rho_iter_beta": rho_iter_beta,
                        "eigval": eigval,
                        "fp_residual": fp_residual,
                        "fp_diag": fp_diag,
                        "Hfixcorr": Hfixcorr,
                        "Cnum": Cnum,
                    }
                    save_npz(outdir / "matrices" / f"case_lambda{lam:g}_theta2{theta2:.6g}.npz",
                             rho_fix=rho_fix, rho_eig=rho_eig, rho_lstsq=rho_lstsq, rho_iter_beta=rho_iter_beta,
                             Hfixcorr=Hfixcorr, Cnum=Cnum,
                             Cpaper=Cpaper, Cpaper_formula=Cpaper_formula,
                             GLS=pert["GLS"], GDB=pert["GDB"], DeltaG=pert["DeltaG"],
                             theta=theta, theta2=theta2, lam=lam, beta=args.beta, T=T,
                             **{k: np.array(v) for k, v in fp_diag.items()})
                fp = fixed_cache[key_fp]
                rho_fix = fp["rho_fix"]
                Hfixcorr = fp["Hfixcorr"]
                Cnum = fp["Cnum"]

                # Estados candidatos independientes de variante.
                rho_gibbs_Cpaper = rho_beta(Hs + theta2 * Cpaper, args.beta) if np.all(np.isfinite(Cpaper)) else np.full_like(Hs, np.nan)
                rho_gibbs_Cpaper_formula = rho_beta(Hs + theta2 * Cpaper_formula, args.beta) if np.all(np.isfinite(Cpaper_formula)) else np.full_like(Hs, np.nan)

                # Métricas Cnum contra objetos invariantes.
                cos_cp, alpha_cp, resid_cp = offdiag_metrics(Cnum, Cpaper, Ue) if np.all(np.isfinite(Cpaper)) else (np.nan, np.nan, np.nan)
                cos_cpf, alpha_cpf, resid_cpf = offdiag_metrics(Cnum, Cpaper_formula, Ue) if np.all(np.isfinite(Cpaper_formula)) else (np.nan, np.nan, np.nan)
                cos_gls, alpha_gls, resid_gls = offdiag_metrics(Cnum, pert["GLS"], Ue)
                cos_dg, alpha_dg, resid_dg = offdiag_metrics(Cnum, pert["DeltaG"], Ue)

                # Detección de base: desde cero todo está en computacional.
                basis_rows.append({
                    "dataset": dataset,
                    "lambda": lam,
                    "theta2": theta2,
                    "basis_status": "all_operators_constructed_in_computational_basis; offdiag_metrics_transform_to_energy_basis",
                })

                variants_to_use = ["stored_rho_mf", "instant_fmax", "f2avg_Hmf"]
                if args.include_f2avg_rho:
                    variants_to_use.append("f2avg_rho_diagnostic")

                for variant in variants_to_use:
                    rho_eff = mfvars[variant]["rho_eff"]
                    Hmf = mfvars[variant]["Hmf"]
                    Cmf = Hmf / theta2
                    rho_gibbs_Hmf = rho_beta(Hs + Hmf, args.beta)

                    cos_mf, alpha_mf, resid_mf = offdiag_metrics(Cnum, Cmf, Ue)
                    logeff = logm_density(rho_eff)
                    logfix = logm_density(rho_fix)

                    row = {
                        "file": "from_scratch_generated",
                        "dataset": dataset,
                        "theta": theta,
                        "theta2": theta2,
                        "lambda": lam,
                        "beta": args.beta,
                        "variant": variant,
                        "source": mfvars[variant]["source"],
                        "microscopic_status": "constructed_rigorously_from_Hs_Hb_V",
                        "filter_status": "constructed_gaussian_f_t_on_grid",
                        "T_status": f"T={T:.12g}; a={a:.12g}; n_time={len(times)}",
                        "fixed_point_eigval_real": float(np.real(fp["eigval"])),
                        "fixed_point_eigval_imag": float(np.imag(fp["eigval"])),
                        "fixed_point_residual_F": float(fp["fp_residual"]),
                        **fp["fp_diag"],

                        "D_fix_beta": trace_distance(rho_fix, rhoB),
                        "D_fix_eff": trace_distance(rho_fix, rho_eff),
                        "D_eff_beta": trace_distance(rho_eff, rhoB),
                        "D_fix_Gibbs_Hs_plus_Hmf": trace_distance(rho_fix, rho_gibbs_Hmf),
                        "D_fix_Gibbs_Hs_plus_theta2_Cpaper": trace_distance(rho_fix, rho_gibbs_Cpaper) if np.all(np.isfinite(rho_gibbs_Cpaper)) else np.nan,
                        "D_fix_Gibbs_Hs_plus_theta2_Cpaper_formula": trace_distance(rho_fix, rho_gibbs_Cpaper_formula) if np.all(np.isfinite(rho_gibbs_Cpaper_formula)) else np.nan,

                        "norm_logeff_minus_logfix_dev_F": fro_norm(dev(logeff - logfix)),
                        "norm_logeff_minus_logfix_raw_F": fro_norm(logeff - logfix),

                        "norm_Hfixcorr_F": fro_norm(Hfixcorr),
                        "norm_Hmf_F": fro_norm(Hmf),
                        "norm_dev_Hfixcorr_minus_Hmf_F": fro_norm(dev(Hfixcorr - Hmf)),

                        "norm_theta2_Cpaper_F": fro_norm(theta2 * Cpaper) if np.all(np.isfinite(Cpaper)) else np.nan,
                        "norm_theta2_Cpaper_formula_F": fro_norm(theta2 * Cpaper_formula) if np.all(np.isfinite(Cpaper_formula)) else np.nan,
                        "norm_theta2_GLS_F": fro_norm(theta2 * pert["GLS"]),
                        "norm_theta2_DeltaG_F": fro_norm(theta2 * pert["DeltaG"]),

                        "cos_Cnum_Cpaper_offE": cos_cp,
                        "cos_Cnum_Cpaper_formula_offE": cos_cpf,
                        "cos_Cnum_Cmf_offE": cos_mf,
                        "cos_Cnum_GLS_offE": cos_gls,
                        "cos_Cnum_DeltaG_offE": cos_dg,

                        "abs_cos_Cnum_Cpaper_offE": abs(cos_cp) if np.isfinite(cos_cp) else np.nan,
                        "abs_cos_Cnum_Cpaper_formula_offE": abs(cos_cpf) if np.isfinite(cos_cpf) else np.nan,
                        "abs_cos_Cnum_Cmf_offE": abs(cos_mf) if np.isfinite(cos_mf) else np.nan,
                        "abs_cos_Cnum_GLS_offE": abs(cos_gls) if np.isfinite(cos_gls) else np.nan,
                        "abs_cos_Cnum_DeltaG_offE": abs(cos_dg) if np.isfinite(cos_dg) else np.nan,

                        "alpha_Cnum_vs_Cpaper_offE": alpha_cp,
                        "alpha_Cnum_vs_Cpaper_formula_offE": alpha_cpf,
                        "alpha_Cnum_vs_Cmf_offE": alpha_mf,
                        "alpha_Cnum_vs_GLS_offE": alpha_gls,
                        "alpha_Cnum_vs_DeltaG_offE": alpha_dg,

                        "resid_Cnum_vs_Cpaper_offE": resid_cp,
                        "resid_Cnum_vs_Cpaper_formula_offE": resid_cpf,
                        "resid_Cnum_vs_Cmf_offE": resid_mf,
                        "resid_Cnum_vs_GLS_offE": resid_gls,
                        "resid_Cnum_vs_DeltaG_offE": resid_dg,
                    }
                    rows.append(row)

                    if variant == "stored_rho_mf":
                        consistency_rows.append({
                            "file": "from_scratch_generated",
                            "dataset": dataset,
                            "lambda": lam,
                            "theta": theta,
                            "theta2": theta2,
                            "variant": variant,
                            "definition": "rho_mf = Tr_B Gibbs(Hfree + theta V), f=1, no modulation",
                            "D_rho_mf_vs_Gibbs_Hs_plus_Hmf_from_rho": trace_distance(rho_eff, rho_gibbs_Hmf),
                            "norm_Hmf_saved_F": np.nan,
                            "norm_Hmf_from_rho_F": fro_norm(Hmf),
                            "norm_dev_Hmf_saved_minus_Hmf_from_rho_F": np.nan,
                            "relative_difference": np.nan,
                            "note": "No Hmf viejo guardado: Hmf se calcula directamente desde rho_mf generado desde cero.",
                        })

                # Pauli para theta representativo cercano a theta^2=0.04.
                # Se rellena al final el theta más cercano por caso; aquí guardamos candidatos.

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "tables" / "compact_results.csv", index=False)
    pd.DataFrame(consistency_rows).to_csv(outdir / "tables" / "stored_mf_consistency.csv", index=False)
    pd.DataFrame(basis_rows).to_csv(outdir / "logs" / "basis_status.csv", index=False)

    # Comparación explícita de puntos fijos lambda=0 vs lambda=2.
    lambda_comp_rows = []
    if 0.0 in lambdas and 2.0 in lambdas:
        for dataset, theta2_values in datasets.items():
            for theta2 in theta2_values:
                theta = float(np.sqrt(theta2))
                key0 = (theta, 0.0)
                key2 = (theta, 2.0)
                if key0 in fixed_cache and key2 in fixed_cache:
                    lambda_comp_rows.append({
                        "dataset": dataset,
                        "theta": theta,
                        "theta2": theta2,
                        "D_rho_fix_lambda0_lambda2": trace_distance(fixed_cache[key0]["rho_fix"], fixed_cache[key2]["rho_fix"]),
                        "D_rho_eig_lambda0_lambda2": trace_distance(fixed_cache[key0]["rho_eig"], fixed_cache[key2]["rho_eig"]),
                        "D_rho_iter_beta_lambda0_lambda2": trace_distance(fixed_cache[key0]["rho_iter_beta"], fixed_cache[key2]["rho_iter_beta"]),
                        "D_fix0_beta": trace_distance(fixed_cache[key0]["rho_fix"], rhoB),
                        "D_fix2_beta": trace_distance(fixed_cache[key2]["rho_fix"], rhoB),
                        "method_lambda0": fixed_cache[key0]["fp_diag"].get("fixed_method_used", ""),
                        "method_lambda2": fixed_cache[key2]["fp_diag"].get("fixed_method_used", ""),
                    })
    lambda_comp_df = pd.DataFrame(lambda_comp_rows)
    lambda_comp_df.to_csv(outdir / "tables" / "fixedpoint_lambda_comparison.csv", index=False)
    if not lambda_comp_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for dataset, sd in lambda_comp_df.groupby("dataset"):
            ax.plot(sd["theta2"], sd["D_rho_fix_lambda0_lambda2"], marker="o", label=dataset)
        ax.set_xlabel(r"$\theta^2$")
        ax.set_ylabel(r"$D(\rho_{fix}^{\lambda=0},\rho_{fix}^{\lambda=2})$")
        ax.set_title("Diferencia entre punto fijo sin/randomizado")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / "plots" / "fixedpoint_lambda0_vs_lambda2.png", dpi=200)
        plt.close(fig)

    # Plots resumen.
    print("\nGenerando plots resumen...")
    for dataset in datasets.keys():
        for lam in lambdas:
            plot_summary(df, outdir, dataset, lam)

    # Pauli decomposition: elegir theta2 más cercano a 0.04 por dataset/lambda.
    print("Generando descomposiciones de Pauli...")
    for dataset, theta2_values in datasets.items():
        theta2_rep = min(theta2_values, key=lambda x: abs(x - 0.04))
        theta_rep = float(np.sqrt(theta2_rep))
        mfvars = mf_cache[theta_rep]
        for lam in lambdas:
            fp = fixed_cache[(theta_rep, lam)]
            Cpaper_formula = Cpaper_formula_from_DeltaG(pert["DeltaG"], evals, Ue, lam, T)
            ops = {
                "H_fix_corr": fp["Hfixcorr"],
                "H_mf_stored": mfvars["stored_rho_mf"]["Hmf"],
                "H_mf_fmax": mfvars["instant_fmax"]["Hmf"],
                "H_mf_f2avg": mfvars["f2avg_Hmf"]["Hmf"],
                "theta2_Cpaper": theta2_rep * Cpaper_formula,
                "theta2_Cpaper_formula": theta2_rep * Cpaper_formula,
                "theta2_GLS": theta2_rep * pert["GLS"],
                "theta2_DeltaG": theta2_rep * pert["DeltaG"],
            }
            pauli_cases[(dataset, lam, theta2_rep)] = ops
    save_pauli_outputs(outdir, pauli_cases, args.N)

    automatic_summary(df)

    elapsed = time.time() - t0
    print("\nOutputs escritos en:")
    print(f"  {outdir}")
    print("Tablas principales:")
    print(f"  {outdir / 'tables' / 'compact_results.csv'}")
    print(f"  {outdir / 'tables' / 'stored_mf_consistency.csv'}")
    print(f"  {outdir / 'tables' / 'build_status.csv'}")
    print("Plots resumen:")
    for dataset in datasets.keys():
        for lam in lambdas:
            print(f"  {outdir / 'plots' / f'summary_{dataset}_lambda{lam:g}.png'}")
    print(f"Tiempo total: {elapsed:.1f} s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido por usuario.")
        sys.exit(130)
