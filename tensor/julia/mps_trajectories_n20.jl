# ising_mps_mixed_safe_eta_gold_squeeze.jl
# Stronger compression/squeeze diagnostics variant derived from gold_compress.
# Experimental compression/diagnostics variant for N=20 probes.
# Adds optional RESET_CUTOFF/MAXDIM, intra-cycle compression, reset-site compression,
# post-reset/post-randomization compression, CHECK_BOND_EVERY, EARLY_STOP_ON_SAT.
# Safe default runner for mixed-field Ising MPS trajectories with exact cycle ETA.
# Derived from ising_mps_trajectories_paper_optimized.jl
# MPS quantum trajectories for the Lloyd-Abanin modulated-coupling protocol
# and Hahn-style diagnostics: randomization/resonances/theta^2/error/convergence.
#
# Dependencies in Project.toml:
#   ITensors, ITensorMPS, CSV, DataFrames, ProgressMeter, HDF5(optional)
# No CairoMakie: plotting is done from Python notebooks reading CSV files.
#
# Usage examples:
#   julia --project=. ising_mps_trajectories_paper_optimized.jl
#   MODE=scan julia --project=. ising_mps_trajectories_paper_optimized.jl
#   MODE=convergence julia --project=. ising_mps_trajectories_paper_optimized.jl
#   MODE=theta julia --project=. ising_mps_trajectories_paper_optimized.jl
#   MODE=memory julia --project=. ising_mps_trajectories_paper_optimized.jl
#   MODE=variance julia --project=. ising_mps_trajectories_paper_optimized.jl
#
# Important convention:
# - One MPS site is a composite site S_i \otimes B_i with local dimension d=4.
# - This implements one bath qubit per system site. This is the natural scalable
#   1D version of the resettable-bath collision model.

"""
MODOS DISPONIBLES:
quick        = una simulación temporal concreta
scan         = resonancias vs reset_time y lambda_rand
convergence  = error numérico: delta, maxdim, ntraj
theta        = escalado del error con theta^2
memory       = memoria del estado inicial / mezcla
variance     = varianza entre realizaciones aleatorias tipo Hahn
gibbs        = valor Gibbs exacto para N pequeño

MODELOS DISPONIBLES (los 2 son 1D):
MODEL=tfim
H = -JXX Σ X_i X_{i+1} - GZ Σ Z_i

MODEL=mixed_ising
H = JZZ Σ Z_i Z_{i+1} + GX Σ X_i + HZ Σ Z_i 
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using Random
using Statistics
using CSV
using DataFrames
using ProgressMeter
using Base.Threads

# Avoid nested oversubscription when many trajectories run in parallel.
# You can override this externally if needed.
try
    BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", "1")))
catch
end

# ============================================================
# 0. ENV PARSING HELPERS
# ============================================================

getenv_str(name::String, default::String) = get(ENV, name, default)
getenv_int(name::String, default::Int) = parse(Int, get(ENV, name, string(default)))
getenv_float(name::String, default::Float64) = parse(Float64, get(ENV, name, string(default)))

function getenv_float_list(name::String, default::Vector{Float64})
    s = get(ENV, name, "")
    isempty(strip(s)) && return default
    return [parse(Float64, strip(x)) for x in split(s, ",") if !isempty(strip(x))]
end

function getenv_int_list(name::String, default::Vector{Int})
    s = get(ENV, name, "")
    isempty(strip(s)) && return default
    return [parse(Int, strip(x)) for x in split(s, ",") if !isempty(strip(x))]
end

# ============================================================
# 1. PARAMETERS
# ============================================================

Base.@kwdef struct Params
    # System size
    N::Int = 8

    # Model. Options:
    #   "tfim"        H = -Jxx Σ X_i X_{i+1} - gz Σ Z_i
    #   "mixed_ising" H =  Jzz Σ Z_i Z_{i+1} + gx Σ X_i + hz Σ Z_i
    model::String = "mixed_ising"

    # TFIM parameters
    Jxx::Float64 = 1.0
    gz::Float64 = 1.5

    # Mixed-field Ising parameters, close to Hahn paper Eq. (50)
    Jzz::Float64 = 1.0
    gx::Float64 = 0.9045
    hz::Float64 = 0.809

    # Target inverse temperature and bath scale
    beta::Float64 = 1.0
    # bath_h::Float64 = 2.0
    # bath_h < 0 significa: elegir automáticamente h según el modelo.
    # Para TFIM usamos h = max(2g, 4J), como en las simulaciones Ising del paper.
    bath_h::Float64 = -1.0

    # System-bath coupling strength θ in Lloyd-Abanin notation.
    # Hahn paper calls the analogous coupling J; the predicted fixed point error is O(θ^2).
    # theta::Float64 = 0.10
    # theta < 0 significa: elegir automáticamente theta^2 = 0.05/sqrt(beta*h),
    # que es la escala usada en las simulaciones Ising del paper.
    theta::Float64 = -1.0

    # Digital protocol parameters
    #delta::Float64 = 0.05       # Trotter angle δ
    # δ = π/40, como en las simulaciones digitales Ising del paper.
    delta::Float64 = π / 40    

    # reset_time::Float64 = 1.0   # T = M δ, sequence τ=-M,...,M
    # reset_time < 0 significa: elegir automáticamente T = 3/a.
    reset_time::Float64 = -1.0    


    # If reset_time < 0, choose T = T_over_a/a. Paper-like Ising default: 3; use 8-10 for truncation checks/free-fermion-style tests.
    T_over_a::Float64 = 3.0

    lambda_rand::Float64 = 0.0  # λ in p(MR) ∝ exp[-MR/(λM)] ; 0 = unrandomized

    # Filter width. If filter_a <= 0, use a = sqrt(4*bath_h/beta), Lloyd-Abanin Eq. (10).
    filter_a::Float64 = -1.0

    # Cooling operator A_i in V(τ)=θ f(τ) Σ_i A_i ⊗ Y_Bi.
    # Options: "Y", "Z", "X", "ZplusY", "XplusZ".
    Aop::String = "ZplusY"

    # MPS trajectory settings
    ncycles::Int = 60
    ntraj::Int = 20
    seed::Int = 1234
    init::String = "random_product" # "all0", "neel", "random_product", "allplus_not_supported"

    # Observable policy:
    #   "fast" = only observables needed for energy + magnetization + bond dimension
    #   "all"  = E, MZ, MX, XX, ZZ, bath excitation. Slower.
    obs_level::String = "fast"

    # MPS accuracy. These defaults are meant for exploratory scans.
    # For final plots use e.g. MAXDIM=64 CUTOFF=1e-8 or 1e-9.
    maxdim::Int = 32
    cutoff::Float64 = 1e-6

    # Steady-state window for final estimates
    steady_frac::Float64 = 0.5

    # Output
    outdir::String = "results"
    out_prefix::String = "abin_mps"

end

function params_from_env(; prefix="")
    return Params(
        N = getenv_int(prefix * "N", 8),
        model = getenv_str(prefix * "MODEL", "mixed_ising"),
        Jxx = getenv_float(prefix * "JXX", 1.0),
        gz = getenv_float(prefix * "GZ", 1.5),
        Jzz = getenv_float(prefix * "JZZ", 1.0),
        gx = getenv_float(prefix * "GX", 0.9045),
        hz = getenv_float(prefix * "HZ", 0.809),
        beta = getenv_float(prefix * "BETA", 1.0),
        bath_h = getenv_float(prefix * "BATH_H", -1.0),
        theta = getenv_float(prefix * "THETA", -1.0),
        delta = getenv_float(prefix * "DELTA", π / 40),
        reset_time = getenv_float(prefix * "RESET_TIME", -1.0),
        lambda_rand = getenv_float(prefix * "LAMBDA_RAND", 0.0),
        filter_a = getenv_float(prefix * "FILTER_A", -1.0),
        Aop = getenv_str(prefix * "AOP", "ZplusY"),
        ncycles = getenv_int(prefix * "NCYCLES", 60),
        ntraj = getenv_int(prefix * "NTRAJ", 20),
        seed = getenv_int(prefix * "SEED", 1234),
        init = getenv_str(prefix * "INIT", "random_product"),
        obs_level = getenv_str(prefix * "OBS_LEVEL", "fast"),
        maxdim = getenv_int(prefix * "MAXDIM", 32),
        cutoff = getenv_float(prefix * "CUTOFF", 1e-6),
        steady_frac = getenv_float(prefix * "STEADY_FRAC", 0.7), # asi el 'valor final' se estima usando el ultimo 30% de los ciclos, no media simulacion entera (0.5)
        outdir = getenv_str(prefix * "OUTDIR", "results"),
        out_prefix = getenv_str(prefix * "OUT_PREFIX", "abin_mps"),
        T_over_a = getenv_float(prefix * "T_OVER_A", 3.0),
       )
end

# ============================================================
# 2. LOCAL MATRICES FOR COMPOSITE SITE S_i ⊗ B_i
# ============================================================

const I2 = ComplexF64[1 0; 0 1]
const X2 = ComplexF64[0 1; 1 0]
const Y2 = ComplexF64[0 -im; im 0]
const Z2 = ComplexF64[1 0; 0 -1]

# Local basis convention for each composite site:
#   1 = |S0,B0>
#   2 = |S0,B1>
#   3 = |S1,B0>
#   4 = |S1,B1>

const XS = kron(X2, I2)
const YS = kron(Y2, I2)
const ZS = kron(Z2, I2)
const IS = kron(I2, I2)

const XB = kron(I2, X2)
const YB = kron(I2, Y2)
const ZB = kron(I2, Z2)

# Non-unitary reset Kraus operators on the bath qubit.
# RB0: |B=0> -> |B=0>, RB1: |B=1> -> |B=0>.
const RB0 = kron(I2, ComplexF64[1 0; 0 0])
const RB1 = kron(I2, ComplexF64[0 1; 0 0])

# For diagnostics before reset
const PB1 = kron(I2, ComplexF64[0 0; 0 1])

# System-only Pauli matrices, dense ED utilities
const SX = X2
const SY = Y2
const SZ = Z2
const SI = I2

# ============================================================
# 3. ITENSOR UTILITIES
# ============================================================

one_site_op(mat::AbstractMatrix, s) = ITensor(mat, prime(s), dag(s))

function A_matrix(p::Params)
    if p.Aop == "Y"
        return YS
    elseif p.Aop == "Z"
        return ZS
    elseif p.Aop == "X"
        return XS
    elseif p.Aop == "ZplusY"
        return (ZS + YS) / sqrt(2)
    elseif p.Aop == "XplusZ"
        return (XS + ZS) / sqrt(2)
    else
        error("Unknown Aop=$(p.Aop). Use Y, Z, X, ZplusY, XplusZ.")
    end
end

function system_local_matrix(p::Params)
    if p.model == "tfim"
        return -p.gz * ZS
    elseif p.model == "mixed_ising"
        return p.gx * XS + p.hz * ZS
    else
        error("Unknown model=$(p.model).")
    end
end

function system_bond_terms(p::Params)
    if p.model == "tfim"
        return [(p.Jxx, XS, XS, -1.0)]  # coeff = sign * Jxx
    elseif p.model == "mixed_ising"
        return [(p.Jzz, ZS, ZS, +1.0)]
    else
        error("Unknown model=$(p.model).")
    end
end

function make_system_gates(sites, p::Params, dt::Float64; include_bath::Bool=false)
    # Second-order TEBD for exp[-i dt (H_S + optional H_B)].
    # include_bath=true is used inside the reset-cycle layers U0=U_B U_S.
    # Since H_B commutes with all system terms, folding it into the local
    # half steps is equivalent to a separate U_B sweep but saves one full pass.
    N = p.N
    gates = ITensor[]

    Hloc = system_local_matrix(p)
    if include_bath
        h = effective_bath_h(p)
        Hloc = Hloc + (-(h / 2) * ZB)
    end

    # Local half-step
    for i in 1:N
        push!(gates, exp(-1im * (dt / 2) * one_site_op(Hloc, sites[i])))
    end

    # Even half-step
    for i in 1:2:(N - 1)
        for (J, O1, O2, sgn) in system_bond_terms(p)
            hbond = (sgn * J) * one_site_op(O1, sites[i]) * one_site_op(O2, sites[i + 1])
            push!(gates, exp(-1im * (dt / 2) * hbond))
        end
    end

    # Odd full-step
    for i in 2:2:(N - 1)
        for (J, O1, O2, sgn) in system_bond_terms(p)
            hbond = (sgn * J) * one_site_op(O1, sites[i]) * one_site_op(O2, sites[i + 1])
            push!(gates, exp(-1im * dt * hbond))
        end
    end

    # Even half-step again
    for i in 1:2:(N - 1)
        for (J, O1, O2, sgn) in system_bond_terms(p)
            hbond = (sgn * J) * one_site_op(O1, sites[i]) * one_site_op(O2, sites[i + 1])
            push!(gates, exp(-1im * (dt / 2) * hbond))
        end
    end

    # Local half-step again
    for i in 1:N
        push!(gates, exp(-1im * (dt / 2) * one_site_op(Hloc, sites[i])))
    end

    return gates
end

function make_bath_gates(sites, p::Params, dt::Float64)
    # H_B = -h/2 Σ Z_B
    gates = ITensor[]
    h = effective_bath_h(p)
    HB = -(h / 2) * ZB
    for i in 1:p.N
        push!(gates, exp(-1im * dt * one_site_op(HB, sites[i])))
    end
    return gates
end

function make_coupling_gates(sites, p::Params, fτ::Float64, dt::Float64)
    # U_SB(τ) = exp[-i δ θ f(τ) Σ_i A_i ⊗ Y_Bi]
    gates = ITensor[]
    Vloc = A_matrix(p) * YB
    θ = effective_theta(p)
    α = -1im * dt * θ * fτ
    for i in 1:p.N
        push!(gates, exp(α * one_site_op(Vloc, sites[i])))
    end
    return gates
end

function apply_gates(psi::MPS, gates, p::Params; normalize_state::Bool=true)
    psi = apply(gates, psi; cutoff=p.cutoff, maxdim=p.maxdim)
    normalize_state && normalize!(psi)
    return psi
end

# ============================================================
# 3b. OPTIONAL EXTRA COMPRESSION / SATURATION DIAGNOSTICS
# ============================================================
# These knobs do not change the target protocol. They only control how
# aggressively the MPS representation is truncated after the standard gate
# applications. Useful N=20 probe variables:
#   RESET_CUTOFF=<float>          cutoff used only during bath-reset Kraus gates
#   EXTRA_COMPRESS_EVERY=<int>    0 disables; 1 compresses after every cycle
#   EXTRA_COMPRESS_CUTOFF=<float> cutoff for the extra compression pass
#   EXTRA_COMPRESS_MAXDIM=<int>   maxdim for the extra compression pass
#   CHECK_BOND_EVERY=<int>        compute max link dimension every k cycles
#   EARLY_STOP_ON_SAT=1           stop a trajectory once it stays saturated
#   SAT_FRAC=0.95                 saturation threshold fraction of MAXDIM
#   SAT_PATIENCE=<int>            consecutive checks before early stop

function compress_mps_extra(psi::MPS, p::Params; cutoff::Float64=p.cutoff,
                            maxdim::Int=p.maxdim, normalize_state::Bool=true)
    # ITensorMPS.truncate! is a global MPS recompression/canonical truncation.
    # If an older ITensors version lacks it or errors, keep running with a warning.
    try
        truncate!(psi; cutoff=cutoff, maxdim=maxdim)
    catch err
        if getenv_int("WARN_TRUNCATE_FALLBACK", 1) == 1
            @warn "Extra truncate! failed; continuing without extra compression" exception=(err, catch_backtrace())
        end
    end
    normalize_state && normalize!(psi)
    return psi
end

function maybe_extra_compress(psi::MPS, p::Params, cycle::Int)
    every = getenv_int("EXTRA_COMPRESS_EVERY", 0)
    if every <= 0 || (cycle % every != 0)
        return psi
    end
    ccut = getenv_float("EXTRA_COMPRESS_CUTOFF", p.cutoff)
    cdim = getenv_int("EXTRA_COMPRESS_MAXDIM", p.maxdim)
    return compress_mps_extra(psi, p; cutoff=ccut, maxdim=cdim, normalize_state=true)
end

function maybe_intra_layer_compress(psi::MPS, p::Params, layer::Int)
    every = getenv_int("INTRA_COMPRESS_EVERY_LAYER", 0)
    if every <= 0 || (layer % every != 0)
        return psi
    end
    ccut = getenv_float("INTRA_COMPRESS_CUTOFF", p.cutoff)
    cdim = getenv_int("INTRA_COMPRESS_MAXDIM", p.maxdim)
    return compress_mps_extra(psi, p; cutoff=ccut, maxdim=cdim, normalize_state=true)
end

function maybe_post_randomize_compress(psi::MPS, p::Params)
    if getenv_int("POST_RANDOMIZE_COMPRESS", 0) != 1
        return psi
    end
    ccut = getenv_float("POST_RANDOMIZE_CUTOFF", getenv_float("EXTRA_COMPRESS_CUTOFF", p.cutoff))
    cdim = getenv_int("POST_RANDOMIZE_MAXDIM", getenv_int("EXTRA_COMPRESS_MAXDIM", p.maxdim))
    return compress_mps_extra(psi, p; cutoff=ccut, maxdim=cdim, normalize_state=true)
end

function maybe_post_reset_compress(psi::MPS, p::Params)
    if getenv_int("POST_RESET_COMPRESS", 0) != 1
        return psi
    end
    ccut = getenv_float("POST_RESET_CUTOFF", getenv_float("RESET_CUTOFF", p.cutoff))
    cdim = getenv_int("POST_RESET_MAXDIM", getenv_int("RESET_MAXDIM", p.maxdim))
    return compress_mps_extra(psi, p; cutoff=ccut, maxdim=cdim, normalize_state=true)
end

function maybe_reset_site_compress(psi::MPS, p::Params, site_number::Int)
    every = getenv_int("RESET_SITE_COMPRESS_EVERY", 0)
    if every <= 0 || (site_number % every != 0)
        return psi
    end
    ccut = getenv_float("RESET_SITE_COMPRESS_CUTOFF", getenv_float("RESET_CUTOFF", p.cutoff))
    cdim = getenv_int("RESET_SITE_COMPRESS_MAXDIM", getenv_int("RESET_MAXDIM", p.maxdim))
    return compress_mps_extra(psi, p; cutoff=ccut, maxdim=cdim, normalize_state=true)
end

function max_bond_dim(psi::MPS)
    ds = linkdims(psi)
    isempty(ds) && return 1.0
    return Float64(maximum(ds))
end

# ============================================================
# 4. FILTER AND RANDOMIZATION
# ============================================================

function filter_width_a(p::Params)
    h = effective_bath_h(p)
    return p.filter_a > 0 ? p.filter_a : sqrt(4 * h / p.beta)
end

function protocol_M(p::Params)
    T = effective_reset_time(p)
    return max(1, round(Int, T / p.delta))
end

function filter_values(p::Params)
    M = protocol_M(p)
    a = filter_width_a(p)
    taus = collect(-M:M)
    raw = exp.(-0.5 .* (a .* p.delta .* taus).^2)
    # Lloyd-Abanin normalization: δ Σ |f(τ)| = 1.
    normfac = p.delta * sum(abs.(raw))
    return taus, raw ./ normfac
end

function sample_random_MR(p::Params, M::Int, rng::AbstractRNG)
    if p.lambda_rand <= 0
        return 0
    end
    # Exponential distribution with mean λM, rounded to integer depth.
    return max(0, round(Int, randexp(rng) * p.lambda_rand * M))
end

# ============================================================
# 5. INITIAL STATES
# ============================================================

function initial_mps(sites, p::Params, rng::AbstractRNG)
    if p.init == "all0"
        return MPS(sites, fill(1, p.N)) # |S0,B0>
    elseif p.init == "neel"
        states = [isodd(i) ? 1 : 3 for i in 1:p.N] # |S0,B0>, |S1,B0>, ...
        return MPS(sites, states)
    elseif p.init == "random_product"
        states = [rand(rng, Bool) ? 1 : 3 for _ in 1:p.N] # system random, bath 0
        return MPS(sites, states)
    else
        error("Unsupported init=$(p.init). Use all0, neel, random_product.")
    end
end

# ============================================================
# 6. MEASUREMENT + BATH RESET
# ============================================================

function bath_excitation_prob!(psi::MPS, sites, i::Int)
    # Probability that the bath qubit on composite site i is |1>.
    # This is much cheaper than constructing both post-measurement branches.
    orthogonalize!(psi, i)
    wf = psi[i]
    P = one_site_op(PB1, sites[i])
    p1 = real(scalar(dag(prime(wf, "Site")) * P * wf))
    return clamp(p1, 0.0, 1.0)
end

function measure_reset_one_bath_site(psi::MPS, sites, i::Int, rng::AbstractRNG, p::Params)
    p1 = bath_excitation_prob!(psi, sites, i)
    p0 = 1.0 - p1

    if rand(rng) < p0
        R = one_site_op(RB0, sites[i])
        reset_cutoff = getenv_float("RESET_CUTOFF", p.cutoff)
        psi = apply([R], psi; cutoff=reset_cutoff, maxdim=getenv_int("RESET_MAXDIM", p.maxdim))
        nrm = norm(psi)
        nrm < 1e-14 && error("Nearly zero reset probability on bath site $i, branch 0")
        normalize!(psi)
        return psi, 0, p0
    else
        R = one_site_op(RB1, sites[i])
        reset_cutoff = getenv_float("RESET_CUTOFF", p.cutoff)
        psi = apply([R], psi; cutoff=reset_cutoff, maxdim=getenv_int("RESET_MAXDIM", p.maxdim))
        nrm = norm(psi)
        nrm < 1e-14 && error("Nearly zero reset probability on bath site $i, branch 1")
        normalize!(psi)
        return psi, 1, p1
    end
end

function measure_reset_all_baths(psi::MPS, sites, rng::AbstractRNG, p::Params)
    outcomes = zeros(Int, p.N)
    probs = zeros(Float64, p.N)
    for i in 1:p.N
        psi, b, prob = measure_reset_one_bath_site(psi, sites, i, rng, p)
        outcomes[i] = b
        probs[i] = prob
        psi = maybe_reset_site_compress(psi, p, i)
    end
    normalize!(psi)
    psi = maybe_post_reset_compress(psi, p)
    return psi, outcomes, probs
end

# ============================================================
# 7. OBSERVABLES WITH MPS
# ============================================================

function expect1!(psi::MPS, sites, i::Int, Omat::AbstractMatrix)
    orthogonalize!(psi, i)
    wf = psi[i]
    O = one_site_op(Omat, sites[i])
    val = scalar(dag(prime(wf, "Site")) * O * wf)
    return real(val)
end

function expect2!(psi::MPS, sites, i::Int, Omat1::AbstractMatrix, Omat2::AbstractMatrix)
    orthogonalize!(psi, i)
    wf = psi[i] * psi[i + 1]
    O = one_site_op(Omat1, sites[i]) * one_site_op(Omat2, sites[i + 1])
    val = scalar(dag(prime(wf, "Site")) * O * wf)
    return real(val)
end

function energy_per_site!(psi::MPS, sites, p::Params)
    N = p.N
    e = 0.0
    if p.model == "tfim"
        for i in 1:N
            e += -p.gz * expect1!(psi, sites, i, ZS)
        end
        for i in 1:(N - 1)
            e += -p.Jxx * expect2!(psi, sites, i, XS, XS)
        end
    elseif p.model == "mixed_ising"
        for i in 1:N
            e += p.gx * expect1!(psi, sites, i, XS) + p.hz * expect1!(psi, sites, i, ZS)
        end
        for i in 1:(N - 1)
            e += p.Jzz * expect2!(psi, sites, i, ZS, ZS)
        end
    end
    return e / N
end

function obs_summary_all!(psi::MPS, sites, p::Params)
    N = p.N
    mz = mean(expect1!(psi, sites, i, ZS) for i in 1:N)
    mx = mean(expect1!(psi, sites, i, XS) for i in 1:N)
    bath_exc = mean(expect1!(psi, sites, i, PB1) for i in 1:N)
    xx = N > 1 ? mean(expect2!(psi, sites, i, XS, XS) for i in 1:(N - 1)) : 0.0
    zz = N > 1 ? mean(expect2!(psi, sites, i, ZS, ZS) for i in 1:(N - 1)) : 0.0
    E = p.model == "tfim" ? (-p.gz*mz - p.Jxx*xx*(N-1)/N) : (p.gx*mx + p.hz*mz + p.Jzz*zz*(N-1)/N)
    return (E=E, MZ=mz, MX=mx, XX=xx, ZZ=zz, BATH=bath_exc, BOND=max_bond_dim(psi))
end

function obs_summary_fast!(psi::MPS, sites, p::Params)
    # Fewer orthogonalize!/expectation calls than the original obs_summary!.
    # This is the important mode for scans. Non-essential observables are NaN.
    N = p.N
    if p.model == "tfim"
        mz = mean(expect1!(psi, sites, i, ZS) for i in 1:N)
        xx = N > 1 ? mean(expect2!(psi, sites, i, XS, XS) for i in 1:(N - 1)) : 0.0
        E = -p.gz*mz - p.Jxx*xx*(N-1)/N
        return (E=E, MZ=mz, MX=NaN, XX=xx, ZZ=NaN, BATH=NaN, BOND=max_bond_dim(psi))
    elseif p.model == "mixed_ising"
        mz = mean(expect1!(psi, sites, i, ZS) for i in 1:N)
        mx = mean(expect1!(psi, sites, i, XS) for i in 1:N)
        zz = N > 1 ? mean(expect2!(psi, sites, i, ZS, ZS) for i in 1:(N - 1)) : 0.0
        E = p.gx*mx + p.hz*mz + p.Jzz*zz*(N-1)/N
        return (E=E, MZ=mz, MX=mx, XX=NaN, ZZ=zz, BATH=NaN, BOND=max_bond_dim(psi))
    else
        error("Unknown model=$(p.model).")
    end
end

function obs_summary!(psi::MPS, sites, p::Params)
    if lowercase(p.obs_level) == "all"
        return obs_summary_all!(psi, sites, p)
    elseif lowercase(p.obs_level) == "fast"
        return obs_summary_fast!(psi, sites, p)
    else
        error("Unknown OBS_LEVEL=$(p.obs_level). Use fast or all.")
    end
end

# ============================================================
# 8. CORE LLOYD-ABANIN RESET CYCLE
# ============================================================

function precompute_cycle_data(sites, p::Params)
    taus, fvals = filter_values(p)

    # Free layer U0 = U_B U_S folded into a single TEBD pass.
    # Randomization R must be system-only, so we keep a separate U_S.
    U0 = make_system_gates(sites, p, p.delta; include_bath=true)
    US = make_system_gates(sites, p, p.delta; include_bath=false)

    # Coupling gates depend on f(τ), so precompute all.
    USBs = [make_coupling_gates(sites, p, fτ, p.delta) for fτ in fvals]

    # One complete U(τ)=U_SB(τ) U_B U_S layer as a single gate vector.
    # apply(gates, psi) applies them in vector order.
    Utaus = [vcat(U0, USBs[k]) for k in eachindex(fvals)]

    return (taus=taus, fvals=fvals, U0=U0, US=US, USBs=USBs, Utaus=Utaus)
end

function apply_randomization!(psi::MPS, p::Params, rng::AbstractRNG, cache)
    M = protocol_M(p)
    MR = sample_random_MR(p, M, rng)
    if MR == 0
        return psi, MR
    end

    # Normalize only every few randomization steps. This is safe for unitary
    # evolution and avoids a costly norm computation after every TEBD pass.
    norm_every = getenv_int("RAND_NORM_EVERY", 8)
    for m in 1:MR
        psi = apply_gates(psi, cache.US, p; normalize_state=false)
        if norm_every > 0 && (m % norm_every == 0)
            normalize!(psi)
        end
    end
    normalize!(psi)
    return psi, MR
end

function apply_one_reset_cycle(psi::MPS, sites, p::Params, rng::AbstractRNG, cache)
    # Implements Q = R U(M)...U(0)...U(-M), with U(τ)=U_SB(τ) U_B U_S.
    # U_B has been folded into U0 to remove one extra sweep per τ.
    for k in eachindex(cache.taus)
        psi = apply_gates(psi, cache.Utaus[k], p; normalize_state=true)
        psi = maybe_intra_layer_compress(psi, p, k)
    end

    # Randomization unitary R = U_S^MR, λ=0 disables it.
    psi, MR = apply_randomization!(psi, p, rng, cache)
    psi = maybe_post_randomize_compress(psi, p)

    # Measure/reset bath qubits.
    psi, outcomes, _ = measure_reset_all_baths(psi, sites, rng, p)
    return psi, outcomes, MR
end

# ============================================================
# 9. TRAJECTORIES AND ONLINE STATISTICS
# ============================================================

function run_one_trajectory(p::Params; traj_seed::Int=p.seed, save_series::Bool=true)
    rng = MersenneTwister(traj_seed)
    sites = siteinds("Qudit", p.N; dim=4)
    psi = initial_mps(sites, p, rng)
    cache = precompute_cycle_data(sites, p)

    E = zeros(Float64, p.ncycles)
    MZ = zeros(Float64, p.ncycles)
    MX = zeros(Float64, p.ncycles)
    XX = zeros(Float64, p.ncycles)
    ZZ = zeros(Float64, p.ncycles)
    BATH = zeros(Float64, p.ncycles)
    BOND = zeros(Float64, p.ncycles)
    MR = zeros(Int, p.ncycles)
    OUTCOME_MEAN = zeros(Float64, p.ncycles)

    for c in 1:p.ncycles
        psi, outcomes, mr = apply_one_reset_cycle(psi, sites, p, rng, cache)
        psi = maybe_extra_compress(psi, p, c)
        ob = obs_summary!(psi, sites, p)
        E[c] = ob.E
        MZ[c] = ob.MZ
        MX[c] = ob.MX
        XX[c] = ob.XX
        ZZ[c] = ob.ZZ
        BATH[c] = ob.BATH
        BOND[c] = ob.BOND
        MR[c] = mr
        OUTCOME_MEAN[c] = mean(outcomes)
    end

    return (E=E, MZ=MZ, MX=MX, XX=XX, ZZ=ZZ, BATH=BATH, BOND=BOND, MR=MR, OUTCOME_MEAN=OUTCOME_MEAN)
end

function online_update!(meanv, M2, x, r)
    if r == 1
        meanv .= x
        M2 .= 0.0
    else
        delta = x .- meanv
        meanv .+= delta ./ r
        delta2 = x .- meanv
        M2 .+= delta .* delta2
    end
end

stderr_from_M2(M2, R) = R <= 1 ? zeros(length(M2)) : sqrt.(M2 ./ (R * (R - 1)))

function run_many_trajectories(p::Params)
    # Parallelize at the safest/granular level: independent quantum trajectories.
    # Use: JULIA_NUM_THREADS=8 MODE=scan julia --project=. ising_mps_trajectories_optimized.jl
    outs = Vector{Any}(undef, p.ntraj)
    nt = nthreads()
    println("Running $(p.ntraj) trajectories with JULIA_NUM_THREADS=$nt, OBS_LEVEL=$(p.obs_level), MAXDIM=$(p.maxdim), CUTOFF=$(p.cutoff)")

    if nt > 1 && p.ntraj > 1
        Threads.@threads for r in 1:p.ntraj
            outs[r] = run_one_trajectory(p; traj_seed=p.seed + 100_000*r)
        end
    else
        @showprogress for r in 1:p.ntraj
            outs[r] = run_one_trajectory(p; traj_seed=p.seed + 100_000*r)
        end
    end

    meanE = zeros(Float64, p.ncycles); M2E = zeros(Float64, p.ncycles)
    meanMZ = zeros(Float64, p.ncycles); M2MZ = zeros(Float64, p.ncycles)
    meanMX = zeros(Float64, p.ncycles); M2MX = zeros(Float64, p.ncycles)
    meanXX = zeros(Float64, p.ncycles); M2XX = zeros(Float64, p.ncycles)
    meanZZ = zeros(Float64, p.ncycles); M2ZZ = zeros(Float64, p.ncycles)
    meanBATH = zeros(Float64, p.ncycles); M2BATH = zeros(Float64, p.ncycles)
    meanBOND = zeros(Float64, p.ncycles); M2BOND = zeros(Float64, p.ncycles)
    meanMR = zeros(Float64, p.ncycles); M2MR = zeros(Float64, p.ncycles)
    meanOUT = zeros(Float64, p.ncycles); M2OUT = zeros(Float64, p.ncycles)

    for r in 1:p.ntraj
        out = outs[r]
        online_update!(meanE, M2E, out.E, r)
        online_update!(meanMZ, M2MZ, out.MZ, r)
        online_update!(meanMX, M2MX, out.MX, r)
        online_update!(meanXX, M2XX, out.XX, r)
        online_update!(meanZZ, M2ZZ, out.ZZ, r)
        online_update!(meanBATH, M2BATH, out.BATH, r)
        online_update!(meanBOND, M2BOND, out.BOND, r)
        online_update!(meanMR, M2MR, Float64.(out.MR), r)
        online_update!(meanOUT, M2OUT, out.OUTCOME_MEAN, r)
    end

    return (
        E_mean=meanE, E_stderr=stderr_from_M2(M2E, p.ntraj),
        MZ_mean=meanMZ, MZ_stderr=stderr_from_M2(M2MZ, p.ntraj),
        MX_mean=meanMX, MX_stderr=stderr_from_M2(M2MX, p.ntraj),
        XX_mean=meanXX, XX_stderr=stderr_from_M2(M2XX, p.ntraj),
        ZZ_mean=meanZZ, ZZ_stderr=stderr_from_M2(M2ZZ, p.ntraj),
        BATH_mean=meanBATH, BATH_stderr=stderr_from_M2(M2BATH, p.ntraj),
        BOND_mean=meanBOND, BOND_stderr=stderr_from_M2(M2BOND, p.ntraj),
        MR_mean=meanMR, MR_stderr=stderr_from_M2(M2MR, p.ntraj),
        OUTCOME_mean=meanOUT, OUTCOME_stderr=stderr_from_M2(M2OUT, p.ntraj),
    )
end

function steady_slice(p::Params)
    start = max(1, floor(Int, p.steady_frac * p.ncycles))
    return start:p.ncycles
end


function drift_last_window(y, p::Params)
    sl = collect(steady_slice(p))
    n = length(sl)

    if n < 4
        return NaN
    end

    h = max(1, floor(Int, n / 2))
    first_half = sl[1:h]
    second_half = sl[(end - h + 1):end]

    return abs(mean(y[second_half]) - mean(y[first_half]))
end

function final_summary(p::Params, res)
    sl = steady_slice(p)

    E_val = mean(res.E_mean[sl])
    E_err = sqrt(mean(res.E_stderr[sl].^2))
    E_drift = drift_last_window(res.E_mean, p)

    MZ_val = mean(res.MZ_mean[sl])
    MZ_err = sqrt(mean(res.MZ_stderr[sl].^2))
    MZ_drift = drift_last_window(res.MZ_mean, p)

    MX_val = mean(res.MX_mean[sl])
    MX_err = sqrt(mean(res.MX_stderr[sl].^2))
    MX_drift = drift_last_window(res.MX_mean, p)

    XX_val = mean(res.XX_mean[sl])
    XX_err = sqrt(mean(res.XX_stderr[sl].^2))
    XX_drift = drift_last_window(res.XX_mean, p)

    ZZ_val = mean(res.ZZ_mean[sl])
    ZZ_err = sqrt(mean(res.ZZ_stderr[sl].^2))
    ZZ_drift = drift_last_window(res.ZZ_mean, p)

    stationary_E = E_drift <= max(3 * E_err, 1e-3)

    return (
        E = E_val, E_err = E_err, E_drift = E_drift,
        MZ = MZ_val, MZ_err = MZ_err, MZ_drift = MZ_drift,
        MX = MX_val, MX_err = MX_err, MX_drift = MX_drift,
        XX = XX_val, XX_err = XX_err, XX_drift = XX_drift,
        ZZ = ZZ_val, ZZ_err = ZZ_err, ZZ_drift = ZZ_drift,
        stationary_E = stationary_E,
        BOND = maximum(res.BOND_mean),
        BOND_final = res.BOND_mean[end],
    )
end

# ============================================================
# 10. CSV OUTPUT
# ============================================================

function save_timeseries_csv(p::Params, res; suffix="timeseries")
    mkpath(p.outdir)
    df = DataFrame(
        cycle = 1:p.ncycles,
        E_mean = res.E_mean, E_stderr = res.E_stderr,
        MZ_mean = res.MZ_mean, MZ_stderr = res.MZ_stderr,
        MX_mean = res.MX_mean, MX_stderr = res.MX_stderr,
        XX_mean = res.XX_mean, XX_stderr = res.XX_stderr,
        ZZ_mean = res.ZZ_mean, ZZ_stderr = res.ZZ_stderr,
        BATH_mean = res.BATH_mean, BATH_stderr = res.BATH_stderr,
        BOND_mean = res.BOND_mean, BOND_stderr = res.BOND_stderr,
        MR_mean = res.MR_mean, MR_stderr = res.MR_stderr,
        OUTCOME_mean = res.OUTCOME_mean, OUTCOME_stderr = res.OUTCOME_stderr,
    )
    if p.N <= 10
        target = gibbs_exact_observables(p)

        df.E_gibbs = fill(target.E, p.ncycles)
        df.MZ_gibbs = fill(target.MZ, p.ncycles)
        df.MX_gibbs = fill(target.MX, p.ncycles)
        df.XX_gibbs = fill(target.XX, p.ncycles)
        df.ZZ_gibbs = fill(target.ZZ, p.ncycles)

        df.E_abs_error = abs.(df.E_mean .- target.E)
        df.MZ_abs_error = abs.(df.MZ_mean .- target.MZ)
        df.MX_abs_error = abs.(df.MX_mean .- target.MX)
        df.XX_abs_error = abs.(df.XX_mean .- target.XX)
        df.ZZ_abs_error = abs.(df.ZZ_mean .- target.ZZ)
    end
    θeff = round(effective_theta(p), sigdigits=6)
    heff = round(effective_bath_h(p), sigdigits=6)
    Teff = round(effective_reset_time(p), sigdigits=6)

    fname = joinpath(
        p.outdir,
        "$(p.out_prefix)_$(suffix)_N$(p.N)_theta$(θeff)_h$(heff)_T$(Teff)_lambda$(p.lambda_rand).csv"
    )
    CSV.write(fname, df)
    println("Saved: $fname")
    return fname
end


function as_kwargs(p::Params)
    return (; (name => getfield(p, name) for name in fieldnames(typeof(p)))...)
end

function with_params(p::Params; kwargs...)
    return Params(; merge(as_kwargs(p), (; kwargs...))...)
end


# ============================================================
# EFFECTIVE PAPER-LIKE PARAMETERS
# ============================================================

function effective_bath_h(p::Params)
    if p.bath_h > 0
        return p.bath_h
    end

    if p.model == "tfim"
        # Paper-like choice for Ising: h = max(2g, 4J)
        return max(2 * abs(p.gz), 4 * abs(p.Jxx))
    elseif p.model == "mixed_ising"
        # Conservative local energy scale for mixed-field Ising 1D.
        return max(2 * abs(p.gx), 2 * abs(p.hz), 4 * abs(p.Jzz))
    else
        error("Unknown model=$(p.model).")
    end
end

function effective_theta(p::Params)
    if p.theta > 0
        return p.theta
    end

    h = effective_bath_h(p)

    # Paper-like scaling:
    # theta^2 = 0.05 / sqrt(beta*h)
    return sqrt(0.05 / sqrt(p.beta * h))
end

function effective_reset_time(p::Params)
    if p.reset_time > 0
        return p.reset_time
    end

    # T = T_over_a/a. Default T_over_a=3 follows the modest Ising runs in the paper.
    return p.T_over_a / filter_width_a(p)
end

function print_effective_params(p::Params)
    println("Effective parameters:")
    println("  bath_h     = ", effective_bath_h(p))
    println("  theta      = ", effective_theta(p))
    println("  reset_time = ", effective_reset_time(p))
    println("  T_over_a   = ", p.T_over_a)
    println("  delta      = ", p.delta)
    println("  M          = ", protocol_M(p))
    println("  a          = ", filter_width_a(p))
    println("  obs_level  = ", p.obs_level)
    println("  threads    = ", nthreads())
end

function param_row(p::Params)
    return (
        N=p.N, model=p.model, Jxx=p.Jxx, gz=p.gz, Jzz=p.Jzz, gx=p.gx, hz=p.hz,
        #beta=p.beta, bath_h=p.bath_h, theta=p.theta, delta=p.delta,
        #reset_time=p.reset_time, lambda_rand=p.lambda_rand,
        beta=p.beta, bath_h=effective_bath_h(p), theta=effective_theta(p), delta=p.delta,
	reset_time=effective_reset_time(p), T_over_a=p.T_over_a, lambda_rand=p.lambda_rand,
	filter_a=filter_width_a(p), M=protocol_M(p), Aop=p.Aop,
        ncycles=p.ncycles, ntraj=p.ntraj, maxdim=p.maxdim, cutoff=p.cutoff,
        init=p.init, obs_level=p.obs_level,
    )
end

# ============================================================
# 11. DENSE ED GIBBS OBSERVABLES FOR SMALL N VALIDATION
# ============================================================

function kronN(mats::Vector{Matrix{ComplexF64}})
    out = mats[1]
    for k in 2:length(mats)
        out = kron(out, mats[k])
    end
    return out
end

function dense_one_site(N::Int, i::Int, O::Matrix{ComplexF64})
    mats = [j == i ? O : SI for j in 1:N]
    return kronN(mats)
end

function dense_two_site(N::Int, i::Int, O1::Matrix{ComplexF64}, O2::Matrix{ComplexF64})
    mats = [j == i ? O1 : (j == i + 1 ? O2 : SI) for j in 1:N]
    return kronN(mats)
end

function dense_Hsys(p::Params)
    N = p.N
    dim = 2^N
    H = zeros(ComplexF64, dim, dim)
    if p.model == "tfim"
        for i in 1:N
            H .+= -p.gz .* dense_one_site(N, i, SZ)
        end
        for i in 1:(N - 1)
            H .+= -p.Jxx .* dense_two_site(N, i, SX, SX)
        end
    elseif p.model == "mixed_ising"
        for i in 1:N
            H .+= p.gx .* dense_one_site(N, i, SX) .+ p.hz .* dense_one_site(N, i, SZ)
        end
        for i in 1:(N - 1)
            H .+= p.Jzz .* dense_two_site(N, i, SZ, SZ)
        end
    end
    return Hermitian(H)
end

function gibbs_exact_observables(p::Params)
    p.N > 10 && error("ED Gibbs is intended for N<=10. Current N=$(p.N)")
    H = dense_Hsys(p)
    F = eigen(H)
    evals = real(F.values)
    e0 = minimum(evals)
    w = exp.(-p.beta .* (evals .- e0))
    w ./= sum(w)
    E = sum(w .* evals) / p.N

    # For simple observables, transform operators to eigenbasis and sum diagonal weights.
    V = F.vectors
    function thermal_expect(O)
        Oe = V' * O * V
        return real(sum(w .* real(diag(Oe))))
    end

    N = p.N
    MZ = mean(thermal_expect(dense_one_site(N, i, SZ)) for i in 1:N)
    MX = mean(thermal_expect(dense_one_site(N, i, SX)) for i in 1:N)
    XX = N > 1 ? mean(thermal_expect(dense_two_site(N, i, SX, SX)) for i in 1:(N-1)) : 0.0
    ZZ = N > 1 ? mean(thermal_expect(dense_two_site(N, i, SZ, SZ)) for i in 1:(N-1)) : 0.0
    return (E=E, MZ=MZ, MX=MX, XX=XX, ZZ=ZZ)
end

function getenv_optional_float(name::String)
    s = get(ENV, name, "")
    isempty(strip(s)) && return NaN
    return parse(Float64, s)
end

function reference_observables(p::Params)
    use_ed = getenv_str("USE_ED_GIBBS", "1")

    if use_ed == "1" && p.N <= 10
        return gibbs_exact_observables(p)
    end

    return (
        E  = getenv_optional_float("REF_E"),
        MZ = getenv_optional_float("REF_MZ"),
        MX = getenv_optional_float("REF_MX"),
        XX = getenv_optional_float("REF_XX"),
        ZZ = getenv_optional_float("REF_ZZ"),
    )
end

abs_or_nan(x, y) = isnan(y) ? NaN : abs(x - y)
# ============================================================
# 12. MODES
# ============================================================

function pushrow!(df::DataFrame, row)
    if ncol(df) == 0
        append!(df, DataFrame([row]); cols=:union)
    else
        push!(df, row; cols=:union)
    end
    return df
end

# ============================================================
# MODE=steady
# Late-time sampling estimator with blocked error bars.
#
# Crucial correction versus the previous file:
#   samples from the same trajectory are autocorrelated, so the reported SEM is
#   computed across trajectory means, not by treating every late-time sample as
#   independent. This is the estimator you should use for paper plots.
# ============================================================

mutable struct OnlineScalar
    n::Int
    mean::Float64
    M2::Float64
end
OnlineScalar() = OnlineScalar(0, 0.0, 0.0)

function update!(st::OnlineScalar, x::Real)
    xf = Float64(x)
    if !isfinite(xf)
        return st
    end
    st.n += 1
    if st.n == 1
        st.mean = xf
        st.M2 = 0.0
    else
        δ = xf - st.mean
        st.mean += δ / st.n
        st.M2 += δ * (xf - st.mean)
    end
    return st
end

mean_or_nan(st::OnlineScalar) = st.n == 0 ? NaN : st.mean
std_or_nan(st::OnlineScalar) = st.n <= 1 ? NaN : sqrt(st.M2 / (st.n - 1))
stderr_or_nan(st::OnlineScalar) = st.n <= 1 ? NaN : sqrt(st.M2 / (st.n * (st.n - 1)))

function sample_start_default(p::Params)
    return max(1, floor(Int, p.steady_frac * p.ncycles))
end

function drift_samples(v::Vector{Float64})
    w = [x for x in v if isfinite(x)]
    n = length(w)
    n < 4 && return NaN
    h = max(1, floor(Int, n / 2))
    return abs(mean(w[(end - h + 1):end]) - mean(w[1:h]))
end

# Signed late-time drift over the sampled steady-state window.
# This keeps the sign, so averaging it over trajectories gives the true
# drift of the trajectory-averaged signal over the same two windows.
function drift_signed_samples(v::Vector{Float64})
    w = [x for x in v if isfinite(x)]
    n = length(w)
    n < 4 && return NaN
    h = max(1, floor(Int, n / 2))
    return mean(w[(end - h + 1):end]) - mean(w[1:h])
end

function run_one_trajectory_steady(p::Params; traj_seed::Int=p.seed, traj_id::Int=1,
                                   sample_start::Int=sample_start_default(p),
                                   sample_stride::Int=10,
                                   store_samples::Bool=false,
                                   progress=nothing)
    rng = MersenneTwister(traj_seed)
    sites = siteinds("Qudit", p.N; dim=4)
    psi = initial_mps(sites, p, rng)
    cache = precompute_cycle_data(sites, p)

    stE = OnlineScalar(); stMZ = OnlineScalar(); stMX = OnlineScalar()
    stXX = OnlineScalar(); stZZ = OnlineScalar(); stBATH = OnlineScalar()
    stBOND = OnlineScalar(); stMR = OnlineScalar(); stOUT = OnlineScalar()

    E_samples = Float64[]
    MZ_samples = Float64[]
    MX_samples = Float64[]
    ZZ_samples = Float64[]
    sample_rows = DataFrame()
    maxBOND = 0.0
    bond_peak_all = 0.0
    completed_cycles = 0
    early_stopped = false
    sat_count = 0
    check_bond_every = getenv_int("CHECK_BOND_EVERY", 1)
    early_stop_on_sat = getenv_int("EARLY_STOP_ON_SAT", 0) == 1
    sat_frac = getenv_float("SAT_FRAC", 0.95)
    sat_patience = getenv_int("SAT_PATIENCE", 1)

    for c in 1:p.ncycles
        psi, outcomes, mr = apply_one_reset_cycle(psi, sites, p, rng, cache)
        psi = maybe_extra_compress(psi, p, c)
        completed_cycles = c

        currentBOND = NaN
        if check_bond_every > 0 && (c % check_bond_every == 0)
            currentBOND = max_bond_dim(psi)
            bond_peak_all = max(bond_peak_all, currentBOND)
            maxBOND = max(maxBOND, currentBOND)
            if currentBOND >= sat_frac * p.maxdim
                sat_count += 1
            else
                sat_count = 0
            end
        end

        if progress !== nothing
            next!(progress; showvalues=[(:traj, traj_id), (:cycle, c), (:N, p.N), (:maxdim, p.maxdim), (:bond, currentBOND)])
        end

        if c >= sample_start && ((c - sample_start) % sample_stride == 0)
            ob = obs_summary!(psi, sites, p)
            outcome_mean = mean(outcomes)

            update!(stE, ob.E); update!(stMZ, ob.MZ); update!(stMX, ob.MX)
            update!(stXX, ob.XX); update!(stZZ, ob.ZZ); update!(stBATH, ob.BATH)
            update!(stBOND, ob.BOND); update!(stMR, mr); update!(stOUT, outcome_mean)

            push!(E_samples, ob.E)
            push!(MZ_samples, ob.MZ)
            push!(MX_samples, ob.MX)
            push!(ZZ_samples, ob.ZZ)
            maxBOND = max(maxBOND, Float64(ob.BOND))
            bond_peak_all = max(bond_peak_all, Float64(ob.BOND))

            if store_samples
                pushrow!(sample_rows, (
                    traj = traj_id, cycle = c,
                    E = ob.E, MZ = ob.MZ, MX = ob.MX, XX = ob.XX, ZZ = ob.ZZ,
                    BATH = ob.BATH, BOND = ob.BOND, BOND_peak_all = bond_peak_all,
                    MR = mr, OUTCOME_mean = outcome_mean,
                ))
            end
        end

        if early_stop_on_sat && sat_count >= sat_patience
            early_stopped = true
            @warn "EARLY_STOP_ON_SAT triggered" traj=traj_id cycle=c currentBOND=currentBOND maxdim=p.maxdim sat_frac=sat_frac sat_patience=sat_patience
            break
        end
    end

    return (
        traj = traj_id,
        n_samples = stE.n,
        E = mean_or_nan(stE), E_sample_stderr = stderr_or_nan(stE),
        E_drift = drift_samples(E_samples), E_signed_drift = drift_signed_samples(E_samples),
        MZ = mean_or_nan(stMZ), MZ_sample_stderr = stderr_or_nan(stMZ),
        MZ_drift = drift_samples(MZ_samples), MZ_signed_drift = drift_signed_samples(MZ_samples),
        MX = mean_or_nan(stMX), MX_drift = drift_samples(MX_samples), MX_signed_drift = drift_signed_samples(MX_samples),
        XX = mean_or_nan(stXX),
        ZZ = mean_or_nan(stZZ), ZZ_drift = drift_samples(ZZ_samples), ZZ_signed_drift = drift_signed_samples(ZZ_samples),
        BATH = mean_or_nan(stBATH), BOND = mean_or_nan(stBOND), BOND_max = maxBOND,
        BOND_peak_all = max(bond_peak_all, maxBOND),
        completed_cycles = completed_cycles,
        early_stopped = early_stopped,
        MR = mean_or_nan(stMR), OUTCOME = mean_or_nan(stOUT),
        sample_rows = sample_rows,
    )
end

function aggregate_steady_outputs(p::Params, outs; sample_start::Int, sample_stride::Int)
    gE = OnlineScalar(); gMZ = OnlineScalar(); gMX = OnlineScalar()
    gXX = OnlineScalar(); gZZ = OnlineScalar(); gBATH = OnlineScalar()
    gBOND = OnlineScalar(); gMR = OnlineScalar(); gOUT = OnlineScalar()
    gEdrift = OnlineScalar(); gMZdrift = OnlineScalar(); gMXdrift = OnlineScalar(); gZZdrift = OnlineScalar()
    gEdrift_signed = OnlineScalar(); gMZdrift_signed = OnlineScalar(); gMXdrift_signed = OnlineScalar(); gZZdrift_signed = OnlineScalar()
    gEsterr_sample = OnlineScalar(); gMZsterr_sample = OnlineScalar()

    total_samples = 0
    maxBOND = 0.0
    maxBOND_all = 0.0
    completed_cycles_vals = Float64[]
    early_stop_count = 0
    for out in outs
        total_samples += out.n_samples
        push!(completed_cycles_vals, Float64(out.completed_cycles))
        early_stop_count += out.early_stopped ? 1 : 0
        update!(gE, out.E); update!(gMZ, out.MZ); update!(gMX, out.MX)
        update!(gXX, out.XX); update!(gZZ, out.ZZ); update!(gBATH, out.BATH)
        update!(gBOND, out.BOND); update!(gMR, out.MR); update!(gOUT, out.OUTCOME)
        update!(gEdrift, out.E_drift); update!(gMZdrift, out.MZ_drift)
        update!(gMXdrift, out.MX_drift); update!(gZZdrift, out.ZZ_drift)
        update!(gEdrift_signed, out.E_signed_drift); update!(gMZdrift_signed, out.MZ_signed_drift)
        update!(gMXdrift_signed, out.MX_signed_drift); update!(gZZdrift_signed, out.ZZ_signed_drift)
        update!(gEsterr_sample, out.E_sample_stderr); update!(gMZsterr_sample, out.MZ_sample_stderr)
        maxBOND = max(maxBOND, out.BOND_max)
        maxBOND_all = max(maxBOND_all, out.BOND_peak_all)
    end

    target = reference_observables(p)
    Emean = mean_or_nan(gE); MZmean = mean_or_nan(gMZ); MXmean = mean_or_nan(gMX)
    XXmean = mean_or_nan(gXX); ZZmean = mean_or_nan(gZZ)
    Esterr = stderr_or_nan(gE)
    Edrift = mean_or_nan(gEdrift)
    Edrift_signed = mean_or_nan(gEdrift_signed)
    Edrift_aggregate = isfinite(Edrift_signed) ? abs(Edrift_signed) : NaN

    return merge(
        param_row(p),
        (
            mode = "steady_blocked",
            sample_start = sample_start,
            sample_stride = sample_stride,
            n_late_samples_total = total_samples,
            n_trajectory_blocks = gE.n,
            samples_per_traj_mean = total_samples / max(1, p.ntraj),
            completed_cycles_mean = isempty(completed_cycles_vals) ? NaN : mean(completed_cycles_vals),
            completed_cycles_min = isempty(completed_cycles_vals) ? NaN : minimum(completed_cycles_vals),
            completed_cycles_max = isempty(completed_cycles_vals) ? NaN : maximum(completed_cycles_vals),
            n_early_stopped = early_stop_count,
            early_stopped_any = early_stop_count > 0,

            E_mean = Emean,
            E_stderr = Esterr,
            E_sample_stderr_mean = mean_or_nan(gEsterr_sample),
            # Conservative: mean over trajectories of |late-window drift|.
            E_mean_abs_drift = Edrift,
            # Aggregate: |mean over trajectories of signed late-window drift|.
            # This is the drift of the trajectory-averaged signal over the same sampled windows.
            E_signed_drift_mean = Edrift_signed,
            E_aggregate_abs_drift = Edrift_aggregate,
            stationary_E = isfinite(Edrift) && isfinite(Esterr) ? (Edrift <= max(3 * Esterr, 1e-3)) : false,
            stationary_E_conservative = isfinite(Edrift) && isfinite(Esterr) ? (Edrift <= max(3 * Esterr, 1e-3)) : false,
            stationary_E_aggregate = isfinite(Edrift_aggregate) && isfinite(Esterr) ? (Edrift_aggregate <= max(3 * Esterr, 1e-3)) : false,

            MZ_mean = MZmean,
            MZ_stderr = stderr_or_nan(gMZ),
            MZ_sample_stderr_mean = mean_or_nan(gMZsterr_sample),
            MZ_mean_abs_drift = mean_or_nan(gMZdrift),
            MZ_signed_drift_mean = mean_or_nan(gMZdrift_signed),
            MZ_aggregate_abs_drift = abs_or_nan(mean_or_nan(gMZdrift_signed), 0.0),

            MX_mean = MXmean,
            MX_stderr = stderr_or_nan(gMX),
            MX_mean_abs_drift = mean_or_nan(gMXdrift),
            MX_signed_drift_mean = mean_or_nan(gMXdrift_signed),
            MX_aggregate_abs_drift = abs_or_nan(mean_or_nan(gMXdrift_signed), 0.0),
            XX_mean = XXmean,
            XX_stderr = stderr_or_nan(gXX),
            ZZ_mean = ZZmean,
            ZZ_stderr = stderr_or_nan(gZZ),
            ZZ_mean_abs_drift = mean_or_nan(gZZdrift),
            ZZ_signed_drift_mean = mean_or_nan(gZZdrift_signed),
            ZZ_aggregate_abs_drift = abs_or_nan(mean_or_nan(gZZdrift_signed), 0.0),
            BATH_mean = mean_or_nan(gBATH),
            BATH_stderr = stderr_or_nan(gBATH),
            BOND_mean = mean_or_nan(gBOND),
            BOND_stderr = stderr_or_nan(gBOND),
            BOND_max = maxBOND,
            BOND_peak_all = max(maxBOND, maxBOND_all),
            check_bond_every = getenv_int("CHECK_BOND_EVERY", 1),
            reset_cutoff = getenv_float("RESET_CUTOFF", p.cutoff),
            reset_maxdim = getenv_int("RESET_MAXDIM", p.maxdim),
            reset_site_compress_every = getenv_int("RESET_SITE_COMPRESS_EVERY", 0),
            reset_site_compress_cutoff = getenv_float("RESET_SITE_COMPRESS_CUTOFF", getenv_float("RESET_CUTOFF", p.cutoff)),
            reset_site_compress_maxdim = getenv_int("RESET_SITE_COMPRESS_MAXDIM", getenv_int("RESET_MAXDIM", p.maxdim)),
            post_reset_compress = getenv_int("POST_RESET_COMPRESS", 0) == 1,
            post_reset_cutoff = getenv_float("POST_RESET_CUTOFF", getenv_float("RESET_CUTOFF", p.cutoff)),
            post_reset_maxdim = getenv_int("POST_RESET_MAXDIM", getenv_int("RESET_MAXDIM", p.maxdim)),
            post_randomize_compress = getenv_int("POST_RANDOMIZE_COMPRESS", 0) == 1,
            post_randomize_cutoff = getenv_float("POST_RANDOMIZE_CUTOFF", getenv_float("EXTRA_COMPRESS_CUTOFF", p.cutoff)),
            post_randomize_maxdim = getenv_int("POST_RANDOMIZE_MAXDIM", getenv_int("EXTRA_COMPRESS_MAXDIM", p.maxdim)),
            intra_compress_every_layer = getenv_int("INTRA_COMPRESS_EVERY_LAYER", 0),
            intra_compress_cutoff = getenv_float("INTRA_COMPRESS_CUTOFF", p.cutoff),
            intra_compress_maxdim = getenv_int("INTRA_COMPRESS_MAXDIM", p.maxdim),
            extra_compress_every = getenv_int("EXTRA_COMPRESS_EVERY", 0),
            extra_compress_cutoff = getenv_float("EXTRA_COMPRESS_CUTOFF", p.cutoff),
            extra_compress_maxdim = getenv_int("EXTRA_COMPRESS_MAXDIM", p.maxdim),
            early_stop_on_sat = getenv_int("EARLY_STOP_ON_SAT", 0) == 1,
            sat_frac = getenv_float("SAT_FRAC", 0.95),
            sat_patience = getenv_int("SAT_PATIENCE", 1),
            MR_mean = mean_or_nan(gMR),
            OUTCOME_mean = mean_or_nan(gOUT),

            E_gibbs = target.E,
            MZ_gibbs = target.MZ,
            MX_gibbs = target.MX,
            XX_gibbs = target.XX,
            ZZ_gibbs = target.ZZ,

            E_abs_error = abs_or_nan(Emean, target.E),
            MZ_abs_error = abs_or_nan(MZmean, target.MZ),
            MX_abs_error = abs_or_nan(MXmean, target.MX),
            XX_abs_error = abs_or_nan(XXmean, target.XX),
            ZZ_abs_error = abs_or_nan(ZZmean, target.ZZ),

            saturated = max(maxBOND, maxBOND_all) >= getenv_float("SAT_FRAC", 0.95) * p.maxdim,
        )
    )
end

function run_steady_estimator(p::Params; sample_start::Int=sample_start_default(p),
                              sample_stride::Int=getenv_int("SAMPLE_STRIDE", 10),
                              store_samples::Bool=false)
    if sample_start < 1 || sample_start > p.ncycles
        error("SAMPLE_START must satisfy 1 <= SAMPLE_START <= NCYCLES.")
    end
    sample_stride < 1 && error("SAMPLE_STRIDE must be >= 1.")

    println("Running steady estimator: N=$(p.N), ntraj=$(p.ntraj), ncycles=$(p.ncycles), sample_start=$sample_start, stride=$sample_stride")
    print_effective_params(p)

    outs = Vector{Any}(undef, p.ntraj)
    progress_cycles = getenv_int("PROGRESS_CYCLES", 1) == 1
    threaded = (nthreads() > 1 && p.ntraj > 1 && !store_samples && !progress_cycles)

    t0 = time()
    if threaded
        println("Threaded mode: progress is per trajectory only. Set PROGRESS_CYCLES=1 for exact cycle ETA.")
        prog = Progress(p.ntraj; desc="trajectories", dt=1.0)
        lk = ReentrantLock()
        Threads.@threads for r in 1:p.ntraj
            outs[r] = run_one_trajectory_steady(
                p; traj_seed=p.seed + 100_000*r, traj_id=r,
                sample_start=sample_start, sample_stride=sample_stride,
                store_samples=false
            )
            lock(lk); try next!(prog; showvalues=[(:done_traj, r), (:N, p.N), (:maxdim, p.maxdim)]) finally unlock(lk) end
        end
        finish!(prog)
    else
        prog = Progress(p.ntraj * p.ncycles; desc="traj×cycle", dt=1.0)
        for r in 1:p.ntraj
            outs[r] = run_one_trajectory_steady(
                p; traj_seed=p.seed + 100_000*r, traj_id=r,
                sample_start=sample_start, sample_stride=sample_stride,
                store_samples=store_samples,
                progress=prog
            )
        end
        finish!(prog)
    end
    elapsed = time() - t0

    row = aggregate_steady_outputs(p, outs; sample_start=sample_start, sample_stride=sample_stride)
    row = merge(row, (
        walltime_sec = elapsed,
        sec_per_traj_cycle = elapsed / max(1, p.ntraj * p.ncycles),
        progress_cycles = progress_cycles,
        julia_threads = nthreads(),
    ))
    return row, outs
end

function save_steady_outputs(p::Params, row, outs; suffix="steady", store_samples::Bool=false)
    mkpath(p.outdir)
    summary = DataFrame([row])
    summary_file = joinpath(p.outdir, "$(p.out_prefix)_$(suffix)_summary.csv")
    CSV.write(summary_file, summary)
    println("Saved summary: $summary_file")

    if store_samples
        sample_rows = DataFrame()
        for out in outs
            if nrow(out.sample_rows) > 0
                append!(sample_rows, out.sample_rows; cols=:union)
            end
        end
        samples_file = joinpath(p.outdir, "$(p.out_prefix)_$(suffix)_samples.csv")
        CSV.write(samples_file, sample_rows)
        println("Saved samples: $samples_file")
    end
    return summary_file
end

function mode_steady()
    p = params_from_env()
    sample_start = getenv_int("SAMPLE_START", sample_start_default(p))
    sample_stride = getenv_int("SAMPLE_STRIDE", 10)
    store_samples = getenv_int("STORE_SAMPLES", 0) == 1

    println("Running STEADY with blocked trajectory error bars")
    println(p)
    row, outs = run_steady_estimator(p; sample_start=sample_start, sample_stride=sample_stride, store_samples=store_samples)
    save_steady_outputs(p, row, outs; suffix="steady", store_samples=store_samples)
    println("\nSTEADY SUMMARY")
    println(row)
end

function mode_quick()
    p = params_from_env()
    println("Running QUICK with parameters:")
    println(p)
    print_effective_params(p)
    res = run_many_trajectories(p)
    save_timeseries_csv(p, res; suffix="quick")
    fs = final_summary(p, res)
    println("\nFINAL / STEADY-WINDOW SUMMARY")
    println(fs)
    if p.N <= 10
        println("\nExact Gibbs target for small-N validation:")
        println(gibbs_exact_observables(p))
    end
end

function mode_scan()
    base = params_from_env()
    reset_times = getenv_float_list("RESET_TIMES", collect(0.5:0.25:3.0))
    lambdas = getenv_float_list("LAMBDAS", [0.0, 1.0, 2.0])
    sample_stride = getenv_int("SAMPLE_STRIDE", 10)
    sample_start_env = get(ENV, "SAMPLE_START", "")

    rows = DataFrame()
    mkpath(base.outdir)
    out = joinpath(base.outdir, "$(base.out_prefix)_scan.csv")

    for λ in lambdas
        for T in reset_times
            p = with_params(base; reset_time=T, lambda_rand=λ,
                       out_prefix="$(base.out_prefix)_scan_T$(round(T,digits=4))_lam$(λ)")
            sample_start = isempty(strip(sample_start_env)) ? sample_start_default(p) : parse(Int, sample_start_env)
            println("\n=== SCAN reset_time=$T lambda=$λ ===")
            row, _ = run_steady_estimator(p; sample_start=sample_start, sample_stride=sample_stride, store_samples=false)
            row = merge(row, (
                reset_time_input = T,
                lambda_input = λ,
                x_tfim = T * p.gz / π,
                scan_score = isfinite(row.E_abs_error) ? row.E_abs_error + row.E_mean_abs_drift : NaN,
            ))
            pushrow!(rows, row)
            CSV.write(out, rows)
            println("Partial scan saved: $out")
        end
    end
end

function mode_convergence()
    base = params_from_env()
    deltas = getenv_float_list("DELTAS", [base.delta, base.delta/2])
    maxdims = getenv_int_list("MAXDIMS", [base.maxdim, 2*base.maxdim])
    cutoffs = getenv_float_list("CUTOFFS", [base.cutoff])
    ntrajs = getenv_int_list("NTRAJS", [base.ntraj, max(base.ntraj*2, base.ntraj+1)])
    sample_stride = getenv_int("SAMPLE_STRIDE", 10)

    rows = DataFrame()
    mkpath(base.outdir)
    out = joinpath(base.outdir, "$(base.out_prefix)_convergence.csv")

    for δ in deltas, χ in maxdims, co in cutoffs, R in ntrajs
        p = with_params(base; delta=δ, maxdim=χ, cutoff=co, ntraj=R,
                   out_prefix="$(base.out_prefix)_conv_dt$(round(δ,digits=5))_chi$(χ)_co$(co)_R$(R)")
        println("\n=== CONVERGENCE delta=$δ maxdim=$χ cutoff=$co ntraj=$R ===")
        row, _ = run_steady_estimator(p; sample_start=sample_start_default(p), sample_stride=sample_stride, store_samples=false)
        pushrow!(rows, row)
        CSV.write(out, rows)
        println("Partial convergence saved: $out")
    end
end

function mode_theta_scaling()
    base = params_from_env()

    thetas = getenv_float_list("THETAS", [0.05, 0.075, 0.10, 0.15, 0.20, 0.25])
    any(thetas .<= 0) && error("All THETAS must be positive.")

    cycle_mode = getenv_str("THETA_CYCLE_MODE", "scaled")
    theta_ref = getenv_float("THETA_REF", maximum(thetas))
    cycle_power = getenv_float("THETA_CYCLE_POWER", 2.0)
    min_ncycles = getenv_int("MIN_NCYCLES", base.ncycles)
    max_ncycles = getenv_int("MAX_NCYCLES", 20000)
    sample_stride = getenv_int("SAMPLE_STRIDE", 10)
    store_samples = getenv_int("STORE_SAMPLES", getenv_int("SAVE_SAMPLES", 0)) == 1

    println("Theta cycle policy:")
    println("  THETA_CYCLE_MODE  = ", cycle_mode)
    println("  THETA_REF         = ", theta_ref)
    println("  THETA_CYCLE_POWER = ", cycle_power)
    println("  MIN_NCYCLES       = ", min_ncycles)
    println("  MAX_NCYCLES       = ", max_ncycles)
    println("  STORE_SAMPLES     = ", store_samples)

    rows = DataFrame()
    mkpath(base.outdir)
    out = joinpath(base.outdir, "$(base.out_prefix)_theta_scaling.csv")

    for θ in thetas
        ncy = base.ncycles
        if cycle_mode == "scaled"
            ncy = ceil(Int, base.ncycles * (theta_ref / θ)^cycle_power)
            ncy = clamp(ncy, min_ncycles, max_ncycles)
        elseif cycle_mode == "fixed"
            ncy = clamp(base.ncycles, min_ncycles, max_ncycles)
        else
            error("Unknown THETA_CYCLE_MODE=$cycle_mode. Use 'scaled' or 'fixed'.")
        end

        p = with_params(base; theta=θ, ncycles=ncy,
                        out_prefix="$(base.out_prefix)_theta$(θ)_ncy$(ncy)")
        println("\n=== THETA SCALING theta=$θ theta²=$(θ^2) ncycles=$ncy ===")
        row, outs = run_steady_estimator(p; sample_start=sample_start_default(p), sample_stride=sample_stride, store_samples=store_samples)
        row = merge(row, (
            theta2 = θ^2,
            theta_input = θ,
            ncycles_used = ncy,
            theta_cycle_mode = cycle_mode,
            theta_ref = theta_ref,
            theta_cycle_power = cycle_power,
            min_ncycles = min_ncycles,
            max_ncycles = max_ncycles,
        ))
        pushrow!(rows, row)
        CSV.write(out, rows)
        if store_samples
            save_steady_outputs(p, row, outs; suffix="theta_samples", store_samples=true)
        end
        println("Partial theta scaling saved: $out")
        println("E_abs_error = ", row.E_abs_error)
        println("E_drift_conservative = ", row.E_mean_abs_drift)
        println("E_drift_aggregate    = ", row.E_aggregate_abs_drift)
        println("BOND_mean = ", row.BOND_mean)
        println("BOND_max = ", row.BOND_max)
        println("BOND_peak_all = ", row.BOND_peak_all)
        println("saturated = ", row.saturated)
        println("early_stopped_any = ", row.early_stopped_any, "  n_early_stopped = ", row.n_early_stopped)
        println("stationary_E_conservative = ", row.stationary_E_conservative)
        println("stationary_E_aggregate    = ", row.stationary_E_aggregate)
    end
end

function mode_memory()
    base = params_from_env()
    pA = with_params(base; init="all0", out_prefix="$(base.out_prefix)_mem_all0")
    pB = with_params(base; init="neel", out_prefix="$(base.out_prefix)_mem_neel")
    println("\n=== MEMORY: init all0 ===")
    resA = run_many_trajectories(pA)
    println("\n=== MEMORY: init neel ===")
    resB = run_many_trajectories(pB)

    mkpath(base.outdir)
    df = DataFrame(
        cycle = 1:base.ncycles,
        E_all0 = resA.E_mean, E_neel = resB.E_mean, D_E = abs.(resA.E_mean .- resB.E_mean),
        MZ_all0 = resA.MZ_mean, MZ_neel = resB.MZ_mean, D_MZ = abs.(resA.MZ_mean .- resB.MZ_mean),
        MX_all0 = resA.MX_mean, MX_neel = resB.MX_mean, D_MX = abs.(resA.MX_mean .- resB.MX_mean),
        XX_all0 = resA.XX_mean, XX_neel = resB.XX_mean, D_XX = abs.(resA.XX_mean .- resB.XX_mean),
        ZZ_all0 = resA.ZZ_mean, ZZ_neel = resB.ZZ_mean, D_ZZ = abs.(resA.ZZ_mean .- resB.ZZ_mean),
        BOND_all0 = resA.BOND_mean, BOND_neel = resB.BOND_mean,
    )
    out = joinpath(base.outdir, "$(base.out_prefix)_memory.csv")
    CSV.write(out, df)
    println("Saved memory data: $out")
end

function mode_variance()
    # Hahn Fig. 5-like: many randomized channel realizations, keep each sequence.
    base = params_from_env()
    nseq = getenv_int("NSEQ", 50)
    rows = DataFrame()
    mkpath(base.outdir)
    out = joinpath(base.outdir, "$(base.out_prefix)_variance_sequences.csv")

    @showprogress for s in 1:nseq
        p = with_params(base; ntraj=1, seed=base.seed + 9999*s)
        traj = run_one_trajectory(p; traj_seed=p.seed)
        for c in 1:p.ncycles
            pushrow!(rows, (
                sequence=s, cycle=c,
                E=traj.E[c], MZ=traj.MZ[c], MX=traj.MX[c], XX=traj.XX[c], ZZ=traj.ZZ[c],
                BOND=traj.BOND[c], MR=traj.MR[c], bath_outcome_mean=traj.OUTCOME_MEAN[c],
                theta = effective_theta(p),
		lambda_rand = p.lambda_rand,
		reset_time = effective_reset_time(p),
		bath_h = effective_bath_h(p),
            ))
        end
        CSV.write(out, rows) # partial save sequence by sequence
    end
    println("Saved variance data: $out")
end

function mode_gibbs_target()
    p = params_from_env()
    target = gibbs_exact_observables(p)
    println("Exact Gibbs observables for N=$(p.N), beta=$(p.beta), model=$(p.model):")
    println(target)
end

# ============================================================
# 13. MAIN
# ============================================================

function main()
    mode = get(ENV, "MODE", "steady")
    println("MODE = $mode")
    if mode == "quick"
        mode_quick()
    elseif mode == "scan"
        mode_scan()
    elseif mode == "convergence"
        mode_convergence()
    elseif mode == "theta"
        mode_theta_scaling()
    elseif mode == "memory"
        mode_memory()
    elseif mode == "variance"
        mode_variance()
    elseif mode == "gibbs"
        mode_gibbs_target()
    elseif mode == "steady"
        mode_steady()
    else
        error("Unknown MODE=$mode. Use quick, scan, convergence, theta, memory, variance, gibbs, steady.")
    end
end

main()
