# ising_mps_trajectories.jl
# MPS quantum trajectories for the Lloyd-Abanin modulated-coupling protocol
# and Hahn-style diagnostics: randomization/resonances/theta^2/error/convergence.
#
# Dependencies in Project.toml:
#   ITensors, ITensorMPS, CSV, DataFrames, ProgressMeter, HDF5(optional)
# No CairoMakie: plotting is done from Python notebooks reading CSV files.
#
# Usage examples:
#   julia --project=. ising_mps_traj_v2.jl
#   MODE=scan julia --project=. ising_mps_traj_v2.jl
#   MODE=convergence julia --project=. ising_mps_traj_v2.jl
#   MODE=theta julia --project=. ising_mps_traj_v2.jl
#   MODE=memory julia --project=. ising_mps_traj_v2.jl
#   MODE=variance julia --project=. ising_mps_traj_v2.jl
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
    model::String = "tfim"

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


    # If reset_time < 0, choose T = T_over_a/a. Quick: 3; production: 8-10.
    T_over_a::Float64 = 8.0

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

    # MPS accuracy
    maxdim::Int = 64
    cutoff::Float64 = 1e-9

    # Steady-state window for final estimates
    steady_frac::Float64 = 0.5

    # Output
    outdir::String = "results"
    out_prefix::String = "abin_mps"

end

function params_from_env(; prefix="")
    return Params(
        N = getenv_int(prefix * "N", 8),
        model = getenv_str(prefix * "MODEL", "tfim"),
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
        maxdim = getenv_int(prefix * "MAXDIM", 64),
        cutoff = getenv_float(prefix * "CUTOFF", 1e-9),
        steady_frac = getenv_float(prefix * "STEADY_FRAC", 0.7), # asi el 'valor final' se estima usando el ultimo 30% de los ciclos, no media simulacion entera (0.5)
        outdir = getenv_str(prefix * "OUTDIR", "results"),
        out_prefix = getenv_str(prefix * "OUT_PREFIX", "abin_mps"),
        T_over_a = getenv_float(prefix * "T_OVER_A", 8.0),
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

function make_system_gates(sites, p::Params, dt::Float64)
    # Second-order TEBD for e^{-i dt H_S}.
    # For the Lloyd-Abanin digital protocol this is the system unitary U_S.
    N = p.N
    gates = ITensor[]

    Hloc = system_local_matrix(p)

    # Local half-step
    for i in 1:N
        push!(gates, exp(-1im * (dt / 2) * one_site_op(Hloc, sites[i])))
    end

    # Even half-step
    for i in 1:2:(N - 1)
        for (J, O1, O2, sgn) in system_bond_terms(p)
            h = (sgn * J) * one_site_op(O1, sites[i]) * one_site_op(O2, sites[i + 1])
            push!(gates, exp(-1im * (dt / 2) * h))
        end
    end

    # Odd full-step
    for i in 2:2:(N - 1)
        for (J, O1, O2, sgn) in system_bond_terms(p)
            h = (sgn * J) * one_site_op(O1, sites[i]) * one_site_op(O2, sites[i + 1])
            push!(gates, exp(-1im * dt * h))
        end
    end

    # Even half-step again
    for i in 1:2:(N - 1)
        for (J, O1, O2, sgn) in system_bond_terms(p)
            h = (sgn * J) * one_site_op(O1, sites[i]) * one_site_op(O2, sites[i + 1])
            push!(gates, exp(-1im * (dt / 2) * h))
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
    for i in 1:p.N
	θ = effective_theta(p)
	push!(gates, exp(-1im * dt * θ * fτ * one_site_op(Vloc, sites[i])))
        #push!(gates, exp(-1im * dt * p.theta * fτ * one_site_op(Vloc, sites[i])))
    end
    return gates
end

function apply_gates(psi::MPS, gates, p::Params)
    psi = apply(gates, psi; cutoff=p.cutoff, maxdim=p.maxdim)
    normalize!(psi)
    return psi
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

function measure_reset_one_bath_site(psi::MPS, sites, i::Int, rng::AbstractRNG, p::Params)
    R0 = one_site_op(RB0, sites[i])
    R1 = one_site_op(RB1, sites[i])

    psi0 = apply([R0], psi; cutoff=p.cutoff, maxdim=p.maxdim)
    psi1 = apply([R1], psi; cutoff=p.cutoff, maxdim=p.maxdim)

    p0 = norm(psi0)^2
    p1 = norm(psi1)^2
    ptot = p0 + p1
    ptot < 1e-14 && error("Nearly zero reset probability at bath site $i")

    q0 = p0 / ptot
    if rand(rng) < q0
        normalize!(psi0)
        return psi0, 0, q0
    else
        normalize!(psi1)
        return psi1, 1, 1 - q0
    end
end

function measure_reset_all_baths(psi::MPS, sites, rng::AbstractRNG, p::Params)
    outcomes = zeros(Int, p.N)
    probs = zeros(Float64, p.N)
    for i in 1:p.N
        psi, b, prob = measure_reset_one_bath_site(psi, sites, i, rng, p)
        outcomes[i] = b
        probs[i] = prob
    end
    normalize!(psi)
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

function obs_summary!(psi::MPS, sites, p::Params)
    N = p.N
    mz = mean(expect1!(psi, sites, i, ZS) for i in 1:N)
    mx = mean(expect1!(psi, sites, i, XS) for i in 1:N)
    bath_exc = mean(expect1!(psi, sites, i, PB1) for i in 1:N)
    xx = N > 1 ? mean(expect2!(psi, sites, i, XS, XS) for i in 1:(N - 1)) : 0.0
    zz = N > 1 ? mean(expect2!(psi, sites, i, ZS, ZS) for i in 1:(N - 1)) : 0.0
    return (
        E = energy_per_site!(psi, sites, p),
        MZ = mz,
        MX = mx,
        XX = xx,
        ZZ = zz,
        BATH = bath_exc,
        BOND = maximum(linkdims(psi)),
    )
end

# ============================================================
# 8. CORE LLOYD-ABANIN RESET CYCLE
# ============================================================

function precompute_cycle_data(sites, p::Params)
    taus, fvals = filter_values(p)
    US = make_system_gates(sites, p, p.delta)
    UB = make_bath_gates(sites, p, p.delta)
    # Coupling gates depend on f(τ), so precompute all.
    USBs = [make_coupling_gates(sites, p, fτ, p.delta) for fτ in fvals]
    return (taus=taus, fvals=fvals, US=US, UB=UB, USBs=USBs)
end

function apply_one_reset_cycle(psi::MPS, sites, p::Params, rng::AbstractRNG, cache)
    # Implements Q = R U(M)...U(0)...U(-M), with U(τ)=U_SB(τ) U_B U_S.
    # Since operators act on a state from right to left, here we apply U_S, then U_B, then U_SB.
    for k in eachindex(cache.taus)
        psi = apply_gates(psi, cache.US, p)
        psi = apply_gates(psi, cache.UB, p)
        psi = apply_gates(psi, cache.USBs[k], p)
    end

    # Randomization unitary R = U_S^MR, λ=0 disables it.
    M = protocol_M(p)
    MR = sample_random_MR(p, M, rng)
    for _ in 1:MR
        psi = apply_gates(psi, cache.US, p)
    end

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
    meanE = zeros(Float64, p.ncycles); M2E = zeros(Float64, p.ncycles)
    meanMZ = zeros(Float64, p.ncycles); M2MZ = zeros(Float64, p.ncycles)
    meanMX = zeros(Float64, p.ncycles); M2MX = zeros(Float64, p.ncycles)
    meanXX = zeros(Float64, p.ncycles); M2XX = zeros(Float64, p.ncycles)
    meanZZ = zeros(Float64, p.ncycles); M2ZZ = zeros(Float64, p.ncycles)
    meanBATH = zeros(Float64, p.ncycles); M2BATH = zeros(Float64, p.ncycles)
    meanBOND = zeros(Float64, p.ncycles); M2BOND = zeros(Float64, p.ncycles)
    meanMR = zeros(Float64, p.ncycles); M2MR = zeros(Float64, p.ncycles)
    meanOUT = zeros(Float64, p.ncycles); M2OUT = zeros(Float64, p.ncycles)

    @showprogress for r in 1:p.ntraj
        out = run_one_trajectory(p; traj_seed=p.seed + 100_000*r)
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

    # T = 3/a as practical default, NO, lo cambiamos a 8 creo
    return p.T_over_a / filter_width_a(p)
    # return 3.0 / filter_width_a(p)
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
        init=p.init,
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
# Late-time sampling estimator.
#
# Difference with MODE=quick:
# - quick measures observables at every reset cycle.
# - steady evolves through burn-in and only measures every SAMPLE_STRIDE
#   cycles after SAMPLE_START.
#
# This is closer to the paper-style late-time sampling strategy:
# many measurements sampled from the late-time trajectory.
# ============================================================

function welford_scalar(mean::Float64, M2::Float64, x::Real, k::Int)
    xf = Float64(x)
    if k == 1
        return xf, 0.0
    else
        δ = xf - mean
        mean_new = mean + δ / k
        M2_new = M2 + δ * (xf - mean_new)
        return mean_new, M2_new
    end
end

stderr_scalar(M2::Float64, K::Int) = K <= 1 ? NaN : sqrt(M2 / (K * (K - 1)))

function mode_steady()
    p = params_from_env()

    sample_start_default = max(1, floor(Int, p.steady_frac * p.ncycles))
    sample_start = getenv_int("SAMPLE_START", sample_start_default)
    sample_stride = getenv_int("SAMPLE_STRIDE", 10)
    store_samples = getenv_int("STORE_SAMPLES", 0) == 1

    if sample_start < 1 || sample_start > p.ncycles
        error("SAMPLE_START must satisfy 1 <= SAMPLE_START <= NCYCLES.")
    end
    if sample_stride < 1
        error("SAMPLE_STRIDE must be >= 1.")
    end

    println("Running STEADY late-time sampling with parameters:")
    println(p)
    println("Late-time sampling:")
    println("  SAMPLE_START  = ", sample_start)
    println("  SAMPLE_STRIDE = ", sample_stride)
    println("  STORE_SAMPLES = ", store_samples)
    if isdefined(Main, :print_effective_params)
        print_effective_params(p)
    end

    mkpath(p.outdir)

    # Online statistics over all late-time samples.
    K = 0

    meanE = 0.0; M2E = 0.0
    meanMZ = 0.0; M2MZ = 0.0
    meanMX = 0.0; M2MX = 0.0
    meanXX = 0.0; M2XX = 0.0
    meanZZ = 0.0; M2ZZ = 0.0
    meanBATH = 0.0; M2BATH = 0.0
    meanBOND = 0.0; M2BOND = 0.0
    meanMR = 0.0; M2MR = 0.0
    meanOUT = 0.0; M2OUT = 0.0

    maxBOND = 0.0

    sample_rows = DataFrame()

    @showprogress for r in 1:p.ntraj
        rng = MersenneTwister(p.seed + 100_000 * r)
        sites = siteinds("Qudit", p.N; dim=4)
        psi = initial_mps(sites, p, rng)
        cache = precompute_cycle_data(sites, p)

        for c in 1:p.ncycles
            psi, outcomes, mr = apply_one_reset_cycle(psi, sites, p, rng, cache)

            if c >= sample_start && ((c - sample_start) % sample_stride == 0)
                ob = obs_summary!(psi, sites, p)
                outcome_mean = mean(outcomes)

                K += 1

                meanE, M2E = welford_scalar(meanE, M2E, ob.E, K)
                meanMZ, M2MZ = welford_scalar(meanMZ, M2MZ, ob.MZ, K)
                meanMX, M2MX = welford_scalar(meanMX, M2MX, ob.MX, K)
                meanXX, M2XX = welford_scalar(meanXX, M2XX, ob.XX, K)
                meanZZ, M2ZZ = welford_scalar(meanZZ, M2ZZ, ob.ZZ, K)
                meanBATH, M2BATH = welford_scalar(meanBATH, M2BATH, ob.BATH, K)
                meanBOND, M2BOND = welford_scalar(meanBOND, M2BOND, ob.BOND, K)
                meanMR, M2MR = welford_scalar(meanMR, M2MR, mr, K)
                meanOUT, M2OUT = welford_scalar(meanOUT, M2OUT, outcome_mean, K)

                maxBOND = max(maxBOND, Float64(ob.BOND))

                if store_samples
                    pushrow!(sample_rows, (
                        traj = r,
                        cycle = c,
                        E = ob.E,
                        MZ = ob.MZ,
                        MX = ob.MX,
                        XX = ob.XX,
                        ZZ = ob.ZZ,
                        BATH = ob.BATH,
                        BOND = ob.BOND,
                        MR = mr,
                        OUTCOME_mean = outcome_mean,
                    ))
                end
            end
        end
    end

    if K == 0
        error("No late-time samples collected. Check SAMPLE_START, SAMPLE_STRIDE, NCYCLES.")
    end

    target = p.N <= 10 ? gibbs_exact_observables(p) : nothing

    row = merge(
        param_row(p),
        (
            mode = "steady",
            sample_start = sample_start,
            sample_stride = sample_stride,
            n_late_samples = K,
            samples_per_traj = K / p.ntraj,

            E_mean = meanE,
            E_stderr = stderr_scalar(M2E, K),
            MZ_mean = meanMZ,
            MZ_stderr = stderr_scalar(M2MZ, K),
            MX_mean = meanMX,
            MX_stderr = stderr_scalar(M2MX, K),
            XX_mean = meanXX,
            XX_stderr = stderr_scalar(M2XX, K),
            ZZ_mean = meanZZ,
            ZZ_stderr = stderr_scalar(M2ZZ, K),
            BATH_mean = meanBATH,
            BATH_stderr = stderr_scalar(M2BATH, K),
            BOND_mean = meanBOND,
            BOND_stderr = stderr_scalar(M2BOND, K),
            BOND_max = maxBOND,
            MR_mean = meanMR,
            MR_stderr = stderr_scalar(M2MR, K),
            OUTCOME_mean = meanOUT,
            OUTCOME_stderr = stderr_scalar(M2OUT, K),

            E_gibbs = target === nothing ? NaN : target.E,
            MZ_gibbs = target === nothing ? NaN : target.MZ,
            MX_gibbs = target === nothing ? NaN : target.MX,
            XX_gibbs = target === nothing ? NaN : target.XX,
            ZZ_gibbs = target === nothing ? NaN : target.ZZ,

            E_abs_error = target === nothing ? NaN : abs(meanE - target.E),
            MZ_abs_error = target === nothing ? NaN : abs(meanMZ - target.MZ),
            MX_abs_error = target === nothing ? NaN : abs(meanMX - target.MX),
            XX_abs_error = target === nothing ? NaN : abs(meanXX - target.XX),
            ZZ_abs_error = target === nothing ? NaN : abs(meanZZ - target.ZZ),
        )
    )

    summary = DataFrame([row])
    summary_file = joinpath(p.outdir, "$(p.out_prefix)_steady_summary.csv")
    CSV.write(summary_file, summary)
    println("Saved steady summary: $summary_file")

    if store_samples
        samples_file = joinpath(p.outdir, "$(p.out_prefix)_steady_samples.csv")
        CSV.write(samples_file, sample_rows)
        println("Saved steady samples: $samples_file")
    end

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
    lambdas = getenv_float_list("LAMBDAS", [0.0, 1.0])
    rows = DataFrame()
    target = reference_observables(base)
    #target = base.N <= 10 ? gibbs_exact_observables(base) : nothing
    mkpath(base.outdir)
    out = joinpath(base.outdir, "$(base.out_prefix)_scan.csv")

    for λ in lambdas
        for T in reset_times
            p = with_params(base; reset_time=T, lambda_rand=λ,
                       out_prefix="$(base.out_prefix)_scan_T$(round(T,digits=3))_lam$(λ)")
            println("\n=== SCAN reset_time=$T lambda=$λ ===")
            res = run_many_trajectories(p)
            fs = final_summary(p, res)
            row = merge(param_row(p), (
                E_final=fs.E, E_stderr=fs.E_err,                 
		E_gibbs = target === nothing ? NaN : target.E,
                E_abs_error = target === nothing ? NaN : abs(fs.E - target.E),
                E_drift = fs.E_drift,
                stationary_E = fs.stationary_E, MZ_final=fs.MZ, MZ_stderr=fs.MZ_err,
                MX_final=fs.MX, XX_final=fs.XX, ZZ_final=fs.ZZ,
                BOND_max=fs.BOND, BOND_final=fs.BOND_final,
                x_tfim = T * p.gz / π,
            ))
            pushrow!(rows, row)
            CSV.write(out, rows) # partial save after every completed point
            println("Partial scan saved: $out")
        end
    end
end

function mode_convergence()
    base = params_from_env()
    deltas = getenv_float_list("DELTAS", [base.delta, base.delta/2])
    maxdims = getenv_int_list("MAXDIMS", [base.maxdim, 2*base.maxdim])
    ntrajs = getenv_int_list("NTRAJS", [base.ntraj, max(base.ntraj*2, base.ntraj+1)])

    rows = DataFrame()
    #target = base.N <= 10 ? gibbs_exact_observables(base) : nothing
    target = reference_observables(base)
    mkpath(base.outdir)
    out = joinpath(base.outdir, "$(base.out_prefix)_convergence.csv")

    for δ in deltas, χ in maxdims, R in ntrajs
        p = with_params(base; delta=δ, maxdim=χ, ntraj=R,
                   out_prefix="$(base.out_prefix)_conv_dt$(δ)_chi$(χ)_R$(R)")
        println("\n=== CONVERGENCE delta=$δ maxdim=$χ ntraj=$R ===")
        res = run_many_trajectories(p)
        fs = final_summary(p, res)
        row = merge(param_row(p), (
            E_final=fs.E, E_stderr=fs.E_err, 
	    E_gibbs = target === nothing ? NaN : target.E,
            E_abs_error = target === nothing ? NaN : abs(fs.E - target.E),
            E_drift=fs.E_drift,
            stationary_E=fs.stationary_E,	
	    MZ_final=fs.MZ, MX_final=fs.MX,
            XX_final=fs.XX, ZZ_final=fs.ZZ, BOND_max=fs.BOND, BOND_final=fs.BOND_final,
            saturated = fs.BOND >= 0.95χ,
        ))
        pushrow!(rows, row)
        CSV.write(out, rows)
        println("Partial convergence saved: $out")
    end
end

function mode_theta_scaling()
    base = params_from_env()

    thetas = getenv_float_list(
        "THETAS",
        [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    )

    if any(thetas .<= 0)
        error("All THETAS must be positive.")
    end

    if base.N > 10
        println("Warning: exact Gibbs comparison skipped for N > 10.")
    end

    #target = base.N <= 10 ? gibbs_exact_observables(base) : nothing
    target = reference_observables(base)

    # ------------------------------------------------------------
    # Important:
    # The physical mixing time scales roughly as 1/theta^2.
    # If we use the same number of cycles for all theta values,
    # small theta points may not have reached the fixed point.
    #
    # THETA_CYCLE_MODE:
    #   "scaled" -> ncycles(theta) = base.ncycles * (theta_ref/theta)^2
    #   "fixed"  -> use base.ncycles for all theta values
    # ------------------------------------------------------------

    cycle_mode = getenv_str("THETA_CYCLE_MODE", "scaled")
    theta_ref = getenv_float("THETA_REF", maximum(thetas))

    # Exponente del escalado de ciclos.
    # Por defecto 2 porque t_mix ~ 1/theta^2 en régimen débil.
    cycle_power = getenv_float("THETA_CYCLE_POWER", 2.0)

    # Evita que theta grande use absurdamente pocos ciclos.
    min_ncycles = getenv_int("MIN_NCYCLES", base.ncycles)

    # Evita que theta pequeño explote el tiempo de cálculo.
    max_ncycles = getenv_int("MAX_NCYCLES", 20000)
    
    println("Theta cycle policy:")
    println("  THETA_CYCLE_MODE  = ", cycle_mode)
    println("  THETA_REF         = ", theta_ref)
    println("  THETA_CYCLE_POWER = ", cycle_power)
    println("  MIN_NCYCLES       = ", min_ncycles)
    println("  MAX_NCYCLES       = ", max_ncycles)

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

        p = with_params(
            base;
            theta = θ,
            ncycles = ncy,
            out_prefix = "$(base.out_prefix)_theta$(θ)_ncy$(ncy)"
        )

        println()
        println("=== THETA SCALING theta=$θ theta²=$(θ^2) ncycles=$ncy ===")

        res = run_many_trajectories(p)
        fs = final_summary(p, res)

        E_drift = (:E_drift in keys(fs)) ? fs.E_drift : NaN
        MZ_drift = (:MZ_drift in keys(fs)) ? fs.MZ_drift : NaN
        MX_drift = (:MX_drift in keys(fs)) ? fs.MX_drift : NaN
        XX_drift = (:XX_drift in keys(fs)) ? fs.XX_drift : NaN
        ZZ_drift = (:ZZ_drift in keys(fs)) ? fs.ZZ_drift : NaN
        stationary_E = (:stationary_E in keys(fs)) ? fs.stationary_E : false

        row = merge(
            param_row(p),
            (
                theta2 = θ^2,
                theta_input = θ,
                ncycles_used = ncy,
                theta_cycle_mode = cycle_mode,
		theta_ref = theta_ref,
		theta_cycle_power = cycle_power,
		min_ncycles = min_ncycles,
		max_ncycles = max_ncycles,

                E_final = fs.E,
                E_stderr = fs.E_err,
                E_drift = E_drift,
                stationary_E = stationary_E,

                MZ_final = fs.MZ,
                MZ_stderr = fs.MZ_err,
                MZ_drift = MZ_drift,

                MX_final = fs.MX,
                MX_stderr = fs.MX_err,
                MX_drift = MX_drift,

                XX_final = fs.XX,
                XX_stderr = fs.XX_err,
                XX_drift = XX_drift,

                ZZ_final = fs.ZZ,
                ZZ_stderr = fs.ZZ_err,
                ZZ_drift = ZZ_drift,

                BOND_max = fs.BOND,
                BOND_final = fs.BOND_final,

                E_gibbs = target === nothing ? NaN : target.E,
                MZ_gibbs = target === nothing ? NaN : target.MZ,
                MX_gibbs = target === nothing ? NaN : target.MX,
                XX_gibbs = target === nothing ? NaN : target.XX,
                ZZ_gibbs = target === nothing ? NaN : target.ZZ,

                E_abs_error = target === nothing ? NaN : abs(fs.E - target.E),
                MZ_abs_error = target === nothing ? NaN : abs(fs.MZ - target.MZ),
                MX_abs_error = target === nothing ? NaN : abs(fs.MX - target.MX),
                XX_abs_error = target === nothing ? NaN : abs(fs.XX - target.XX),
                ZZ_abs_error = target === nothing ? NaN : abs(fs.ZZ - target.ZZ),
            )
        )

        pushrow!(rows, row)
        CSV.write(out, rows)

        println("Partial theta scaling saved: $out")
        println("E_abs_error = ", row.E_abs_error)
        println("E_drift     = ", row.E_drift)
        println("stationary_E = ", row.stationary_E)
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
    mode = get(ENV, "MODE", "quick")
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
