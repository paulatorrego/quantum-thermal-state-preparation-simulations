# n20_mpo_composite_density_channel.jl
# Deterministic averaged-channel simulation of the Lloyd-Abanin protocol
# using a vectorized density MPO over composite local sites S_i ⊗ B_i.
#
# Purpose for N=20 rescue tests:
#   - Same explicit bath memory within each reset cycle as the trajectory code.
#   - No stochastic unraveling: applies the unconditional reset channel
#       rho -> sum_b R_b rho R_b^†
#     on every bath site after each cycle.
#   - If this has much lower BOND than pure trajectories, the problem is the
#     trajectory unraveling, not the mixed-state channel.
#
# Dependencies: ITensors, ITensorMPS, CSV, DataFrames, ProgressMeter.
# Run example:
#   MODE=theta N=20 THETAS=0.45 NCYCLES=30 MAXDIM=384 CUTOFF=1e-8 \
#   OUTDIR=results_N20 OUT_PREFIX=N20_mpo_composite_theta045 \
#   julia --project=.. n20_mpo_composite_density_channel.jl

using ITensors
using ITensorMPS
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using ProgressMeter

try
    BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", "1")))
catch
end

# -------------------------
# ENV helpers
# -------------------------
getenv_str(name::String, default::String) = get(ENV, name, default)
getenv_int(name::String, default::Int) = parse(Int, get(ENV, name, string(default)))
getenv_float(name::String, default::Float64) = parse(Float64, get(ENV, name, string(default)))
function getenv_float_list(name::String, default::Vector{Float64})
    s = get(ENV, name, "")
    isempty(strip(s)) && return default
    return [parse(Float64, strip(x)) for x in split(s, ",") if !isempty(strip(x))]
end

Base.@kwdef struct Params
    N::Int = 20
    model::String = "mixed_ising"
    Jzz::Float64 = 1.0
    gx::Float64 = 0.9045
    hz::Float64 = 0.809
    Jxx::Float64 = 1.0
    gz::Float64 = 1.5
    beta::Float64 = 1.0
    bath_h::Float64 = 4.0
    theta::Float64 = 0.45
    delta::Float64 = π / 40
    reset_time::Float64 = 1.0
    T_over_a::Float64 = 3.0
    lambda_rand::Float64 = 0.0
    filter_a::Float64 = -1.0
    Aop::String = "ZplusY"
    ncycles::Int = 30
    maxdim::Int = 384
    cutoff::Float64 = 1e-8
    steady_frac::Float64 = 0.3
    sample_start::Int = -1
    sample_stride::Int = 1
    init::String = "maxmix_system_bath0"
    outdir::String = "results_N20"
    out_prefix::String = "N20_mpo_composite"
    ref_E::Float64 = NaN
    ref_MZ::Float64 = NaN
    ref_MX::Float64 = NaN
    ref_ZZ::Float64 = NaN
end

function params_from_env()
    return Params(
        N=getenv_int("N", 20),
        model=getenv_str("MODEL", "mixed_ising"),
        Jzz=getenv_float("JZZ", 1.0), gx=getenv_float("GX", 0.9045), hz=getenv_float("HZ", 0.809),
        Jxx=getenv_float("JXX", 1.0), gz=getenv_float("GZ", 1.5),
        beta=getenv_float("BETA", 1.0), bath_h=getenv_float("BATH_H", 4.0),
        theta=getenv_float("THETA", 0.45), delta=getenv_float("DELTA", π/40),
        reset_time=getenv_float("RESET_TIME", 1.0), T_over_a=getenv_float("T_OVER_A", 3.0),
        lambda_rand=getenv_float("LAMBDA_RAND", 0.0), filter_a=getenv_float("FILTER_A", -1.0),
        Aop=getenv_str("AOP", "ZplusY"),
        ncycles=getenv_int("NCYCLES", 30), maxdim=getenv_int("MAXDIM", 384),
        cutoff=getenv_float("CUTOFF", 1e-8), steady_frac=getenv_float("STEADY_FRAC", 0.3),
        sample_start=getenv_int("SAMPLE_START", -1), sample_stride=getenv_int("SAMPLE_STRIDE", 1),
        init=getenv_str("INIT", "maxmix_system_bath0"),
        outdir=getenv_str("OUTDIR", "results_N20"), out_prefix=getenv_str("OUT_PREFIX", "N20_mpo_composite"),
        ref_E=getenv_float("REF_E", NaN), ref_MZ=getenv_float("REF_MZ", NaN),
        ref_MX=getenv_float("REF_MX", NaN), ref_ZZ=getenv_float("REF_ZZ", NaN),
    )
end

# -------------------------
# Matrices
# -------------------------
const I2 = ComplexF64[1 0; 0 1]
const X2 = ComplexF64[0 1; 1 0]
const Y2 = ComplexF64[0 -im; im 0]
const Z2 = ComplexF64[1 0; 0 -1]

# Composite pure basis per site: |S,B>, dcomp=4
const XS = kron(X2, I2)
const YS = kron(Y2, I2)
const ZS = kron(Z2, I2)
const IS = kron(I2, I2)
const XB = kron(I2, X2)
const YB = kron(I2, Y2)
const ZB = kron(I2, Z2)
const IB = IS
const RB0 = kron(I2, ComplexF64[1 0; 0 0])
const RB1 = kron(I2, ComplexF64[0 1; 0 0])

# density local dimension = dcomp^2 = 16
const DCOMP = 4
const DDENS = 16

function A_matrix(p::Params)
    p.Aop == "Y" && return YS
    p.Aop == "Z" && return ZS
    p.Aop == "X" && return XS
    p.Aop == "ZplusY" && return (ZS + YS) / sqrt(2)
    p.Aop == "XplusZ" && return (XS + ZS) / sqrt(2)
    error("Unknown AOP=$(p.Aop)")
end

function hlocal_comp(p::Params; include_bath::Bool)
    Hs = p.model == "mixed_ising" ? (p.gx * XS + p.hz * ZS) : (-p.gz * ZS)
    if include_bath
        Hs = Hs + (-(p.bath_h/2) * ZB)
    end
    return Hs
end

function bond_terms_comp(p::Params)
    if p.model == "mixed_ising"
        return [(p.Jzz, ZS, ZS, +1.0)]
    elseif p.model == "tfim"
        return [(p.Jxx, XS, XS, -1.0)]
    else
        error("Unknown model=$(p.model)")
    end
end

filter_width_a(p::Params) = p.filter_a > 0 ? p.filter_a : sqrt(4 * p.bath_h / p.beta)
protocol_M(p::Params) = max(1, round(Int, p.reset_time / p.delta))
function filter_values(p::Params)
    M = protocol_M(p)
    a = filter_width_a(p)
    taus = collect(-M:M)
    fvals = exp.(-0.5 .* (a .* (taus .* p.delta)).^2)
    return taus, fvals
end

# -------------------------
# Vectorization utilities
# local density basis index: |ket><bra|, ket fastest
# -------------------------
den_index(ket::Int, bra::Int, d::Int) = ket + d * (bra - 1)
pair_pure_index(ai::Int, aj::Int, d::Int) = ai + d * (aj - 1)
pair_den_index(local_i::Int, local_j::Int, D::Int) = local_i + D * (local_j - 1)

function one_site_super_from_kraus(kraus::Vector{Matrix{ComplexF64}}; d::Int=DCOMP)
    D = d^2
    S = zeros(ComplexF64, D, D)
    for K in kraus
        # vec(K rho K†) = kron(conj(K), K) vec(rho), column-major
        S .+= kron(conj(K), K)
    end
    return S
end
one_site_super_unitary(U::Matrix{ComplexF64}; d::Int=DCOMP) = one_site_super_from_kraus([U]; d=d)

function pair_super_unitary(U::Matrix{ComplexF64}; d::Int=DCOMP)
    # U acts on two pure sites, dimension d^2, basis (site i fastest, site j slowest).
    Dloc = d^2
    Dden = Dloc^2 # local density pair dimension in global vectorization
    Dsite = d^2   # density dimension of one site
    S = zeros(ComplexF64, Dsite^2, Dsite^2)
    for ki in 1:d, kj in 1:d, bi in 1:d, bj in 1:d
        col_i = den_index(ki, bi, d)
        col_j = den_index(kj, bj, d)
        col = pair_den_index(col_i, col_j, Dsite)
        ket_in = pair_pure_index(ki, kj, d)
        bra_in = pair_pure_index(bi, bj, d)
        for ko_i in 1:d, ko_j in 1:d, bo_i in 1:d, bo_j in 1:d
            row_i = den_index(ko_i, bo_i, d)
            row_j = den_index(ko_j, bo_j, d)
            row = pair_den_index(row_i, row_j, Dsite)
            ket_out = pair_pure_index(ko_i, ko_j, d)
            bra_out = pair_pure_index(bo_i, bo_j, d)
            S[row, col] += U[ket_out, ket_in] * conj(U[bra_out, bra_in])
        end
    end
    return S
end

function one_site_gate(S::Matrix{ComplexF64}, s)
    T = ITensor(prime(s), dag(s))
    D = dim(s)
    @assert size(S) == (D, D)
    for a in 1:D, b in 1:D
        T[prime(s)=>a, dag(s)=>b] = S[a,b]
    end
    return T
end

function two_site_gate(S::Matrix{ComplexF64}, si, sj)
    D = dim(si)
    @assert dim(sj) == D
    @assert size(S) == (D^2, D^2)
    T = ITensor(prime(si), prime(sj), dag(si), dag(sj))
    for ai_out in 1:D, aj_out in 1:D, ai_in in 1:D, aj_in in 1:D
        row = pair_den_index(ai_out, aj_out, D)
        col = pair_den_index(ai_in, aj_in, D)
        val = S[row, col]
        if val != 0
            T[prime(si)=>ai_out, prime(sj)=>aj_out, dag(si)=>ai_in, dag(sj)=>aj_in] = val
        end
    end
    return T
end

function product_mps_from_vectors(sites, vecs::Vector{Vector{ComplexF64}})
    N = length(sites)
    tensors = Vector{ITensor}(undef, N)
    for i in 1:N
        T = ITensor(sites[i])
        for a in 1:length(vecs[i])
            T[sites[i]=>a] = vecs[i][a]
        end
        tensors[i] = T
    end
    return MPS(tensors)
end

function vec_density(rho::Matrix{ComplexF64})
    d = size(rho,1)
    v = zeros(ComplexF64, d^2)
    for k in 1:d, b in 1:d
        v[den_index(k,b,d)] = rho[k,b]
    end
    return v
end

function trace_vec(d::Int)
    v = zeros(ComplexF64, d^2)
    for a in 1:d
        v[den_index(a,a,d)] = 1
    end
    return v
end

function obs_vec(O::Matrix{ComplexF64})
    d = size(O,1)
    v = zeros(ComplexF64, d^2)
    # Tr(O rho) = sum_{ket,bra} O[bra,ket] rho[ket,bra]
    for ket in 1:d, bra in 1:d
        v[den_index(ket,bra,d)] = O[bra,ket]
    end
    return v
end

function product_bra(sites, local_vecs)
    return product_mps_from_vectors(sites, [ComplexF64.(conj(v)) for v in local_vecs])
end

function trace_rho(rho::MPS, sites)
    tv = trace_vec(DCOMP)
    bra = product_bra(sites, fill(tv, length(sites)))
    return real(inner(bra, rho))
end

function linear_expect(rho::MPS, sites, local_ops::Dict{Int,Matrix{ComplexF64}})
    tv = trace_vec(DCOMP)
    vecs = [copy(tv) for _ in 1:length(sites)]
    for (i,O) in local_ops
        vecs[i] = obs_vec(O)
    end
    bra = product_bra(sites, vecs)
    tr = trace_rho(rho, sites)
    return real(inner(bra, rho)) / tr
end

function observables(rho::MPS, sites, p::Params)
    N = p.N
    mx = mean(linear_expect(rho, sites, Dict(i=>XS)) for i in 1:N)
    mz = mean(linear_expect(rho, sites, Dict(i=>ZS)) for i in 1:N)
    zz = mean(linear_expect(rho, sites, Dict(i=>ZS, i+1=>ZS)) for i in 1:N-1)
    E = p.gx*mx + p.hz*mz + p.Jzz*zz*(N-1)/N
    return (E=E, MZ=mz, MX=mx, ZZ=zz, BOND=maximum(linkdims(rho)))
end

function rescale_trace!(rho::MPS, sites)
    tr = trace_rho(rho, sites)
    if !(isfinite(tr)) || abs(tr) < 1e-14
        error("Invalid trace after channel: $tr")
    end
    rho[1] *= (1/tr)
    return rho
end

function apply_gates(rho::MPS, gates, p::Params)
    rho = apply(gates, rho; cutoff=p.cutoff, maxdim=p.maxdim)
    return rho
end

function make_system_gates(sites, p::Params, dt::Float64; include_bath::Bool=false)
    gates = ITensor[]
    Hloc = hlocal_comp(p; include_bath=include_bath)
    Uloc = exp(-1im * (dt/2) * Hloc)
    Sloc = one_site_super_unitary(Uloc; d=DCOMP)
    for i in 1:p.N
        push!(gates, one_site_gate(Sloc, sites[i]))
    end
    for i in 1:2:p.N-1
        for (J,O1,O2,sgn) in bond_terms_comp(p)
            H = (sgn*J) * kron(O1, O2)
            U = exp(-1im * (dt/2) * H)
            push!(gates, two_site_gate(pair_super_unitary(U; d=DCOMP), sites[i], sites[i+1]))
        end
    end
    for i in 2:2:p.N-1
        for (J,O1,O2,sgn) in bond_terms_comp(p)
            H = (sgn*J) * kron(O1, O2)
            U = exp(-1im * dt * H)
            push!(gates, two_site_gate(pair_super_unitary(U; d=DCOMP), sites[i], sites[i+1]))
        end
    end
    for i in 1:2:p.N-1
        for (J,O1,O2,sgn) in bond_terms_comp(p)
            H = (sgn*J) * kron(O1, O2)
            U = exp(-1im * (dt/2) * H)
            push!(gates, two_site_gate(pair_super_unitary(U; d=DCOMP), sites[i], sites[i+1]))
        end
    end
    for i in 1:p.N
        push!(gates, one_site_gate(Sloc, sites[i]))
    end
    return gates
end

function make_coupling_gates(sites, p::Params, fτ::Float64, dt::Float64)
    U = exp(-1im * dt * p.theta * fτ * (A_matrix(p) * YB))
    S = one_site_super_unitary(U; d=DCOMP)
    return [one_site_gate(S, sites[i]) for i in 1:p.N]
end

function make_reset_gates(sites, p::Params)
    S = one_site_super_from_kraus([RB0, RB1]; d=DCOMP)
    return [one_site_gate(S, sites[i]) for i in 1:p.N]
end

function precompute(sites, p::Params)
    taus, fvals = filter_values(p)
    U0 = make_system_gates(sites, p, p.delta; include_bath=true)
    USBs = [make_coupling_gates(sites, p, f, p.delta) for f in fvals]
    Utaus = [vcat(U0, USBs[k]) for k in eachindex(fvals)]
    reset = make_reset_gates(sites, p)
    return (taus=taus, fvals=fvals, Utaus=Utaus, reset=reset)
end

function initial_density_mps(sites, p::Params)
    rhoS = I2 / 2
    rhoB0 = ComplexF64[1 0; 0 0]
    rho = kron(rhoS, rhoB0)
    v = vec_density(rho)
    return product_mps_from_vectors(sites, [v for _ in 1:p.N])
end

function run_channel(p::Params)
    if p.lambda_rand != 0
        @warn "This deterministic composite-density script currently assumes LAMBDA_RAND=0. Ignoring lambda_rand=$(p.lambda_rand)."
    end
    mkpath(p.outdir)
    sites = siteinds("Qudit", p.N; dim=DDENS)
    rho = initial_density_mps(sites, p)
    cache = precompute(sites, p)
    rescale_trace!(rho, sites)

    sample_start = p.sample_start > 0 ? p.sample_start : max(1, floor(Int, (1-p.steady_frac)*p.ncycles) + 1)
    rows = NamedTuple[]
    BOND_peak = maximum(linkdims(rho))

    println("MODE=mpo_composite_density")
    println("N=$(p.N), theta=$(p.theta), reset_time=$(p.reset_time), M=$(protocol_M(p)), maxdim=$(p.maxdim), cutoff=$(p.cutoff)")
    prog = Progress(p.ncycles; desc="mpo_composite cycles")
    for c in 1:p.ncycles
        for gates in cache.Utaus
            rho = apply_gates(rho, gates, p)
        end
        rho = apply_gates(rho, cache.reset, p)
        rescale_trace!(rho, sites)
        BOND_peak = max(BOND_peak, maximum(linkdims(rho)))
        if c >= sample_start && ((c - sample_start) % p.sample_stride == 0)
            ob = observables(rho, sites, p)
            push!(rows, (cycle=c, E=ob.E, MZ=ob.MZ, MX=ob.MX, ZZ=ob.ZZ, BOND=ob.BOND, trace=trace_rho(rho, sites)))
        end
        next!(prog)
    end
    df_samp = DataFrame(rows)
    CSV.write(joinpath(p.outdir, "$(p.out_prefix)_samples.csv"), df_samp)

    E_mean = mean(df_samp.E); MZ_mean=mean(df_samp.MZ); MX_mean=mean(df_samp.MX); ZZ_mean=mean(df_samp.ZZ)
    E_drift = nrow(df_samp) >= 2 ? abs(df_samp.E[end] - df_samp.E[1]) : NaN
    out = DataFrame((
        N=p.N, model=p.model, representation="composite_density_mpo", theta=p.theta, theta2=p.theta^2,
        beta=p.beta, bath_h=p.bath_h, reset_time=p.reset_time, delta=p.delta, M=protocol_M(p),
        ncycles=p.ncycles, maxdim=p.maxdim, cutoff=p.cutoff, sample_start=sample_start, sample_stride=p.sample_stride,
        E_mean=E_mean, MZ_mean=MZ_mean, MX_mean=MX_mean, ZZ_mean=ZZ_mean,
        E_abs_error=abs(E_mean - p.ref_E), MZ_abs_error=abs(MZ_mean - p.ref_MZ),
        MX_abs_error=abs(MX_mean - p.ref_MX), ZZ_abs_error=abs(ZZ_mean - p.ref_ZZ),
        E_mean_abs_drift=E_drift, BOND_mean=mean(df_samp.BOND), BOND_max=maximum(df_samp.BOND), BOND_peak_all=BOND_peak,
        saturated=(BOND_peak >= 0.95*p.maxdim), n_late_samples_total=nrow(df_samp)
    ))
    outpath = joinpath(p.outdir, "$(p.out_prefix)_theta_scaling.csv")
    CSV.write(outpath, out)
    println("Saved: $outpath")
    println(out)
end

function main()
    mode = getenv_str("MODE", "theta")
    p0 = params_from_env()
    thetas = getenv_float_list("THETAS", [p0.theta])
    for th in thetas
        p = Params(; (field=>getfield(p0, field) for field in fieldnames(Params))..., theta=th,
                   out_prefix=length(thetas)>1 ? "$(p0.out_prefix)_theta$(replace(string(th),'.'=>''))" : p0.out_prefix)
        run_channel(p)
    end
end

main()
