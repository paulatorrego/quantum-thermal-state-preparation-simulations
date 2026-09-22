# tn_mpo_reference_and_filter.jl
# ------------------------------------------------------------
# Minimum viable tensor-network tools for the TFM thermal-state project.
#
# What this file is for:
#   1) thermal_ref: scalable 1D finite-temperature references by purification MPS.
#      Useful for mixed_ising where ED is impossible beyond small N.
#   2) filtered_operator: TN-MPO/Choi construction of filtered local operators
#      K_a ≈ ∫ ds f(s) e^{iHs} A_a e^{-iHs}, following the MPO/TEBD idea of
#      Zhan-Ding-Huhn-Gray-Preskill-Chan-Lin.
#
# What this file is NOT yet:
#   - It does not integrate the full Lindblad equation dρ/dt=L(ρ).
#   - It does not simulate the reset/collision protocol. Keep using
#     ising_mps_trajectories.jl for Hahn/Abanin resonance and randomization tests.
#
# Dependencies:
#   ITensors, ITensorMPS, CSV, DataFrames, ProgressMeter
#
# Examples:
#   MODE=thermal_ref N=40 MODEL=mixed_ising BETA=1.0 JZZ=1.0 GX=0.9045 HZ=0.809 \
#     MAXDIM=256 CUTOFF=1e-10 NTAU=200 OUTDIR=tn_results \
#     julia --project=. tn_mpo_reference_and_filter.jl
#
#   MODE=filtered_operator N=30 MODEL=mixed_ising BETA=1.0 SIGMA=1.0 S_MAX=5.0 DS=0.05 \
#     CENTER_SITE=15 AOP=XplusZ MAXDIM=256 CUTOFF=1e-9 OUTDIR=tn_results \
#     julia --project=. tn_mpo_reference_and_filter.jl

using ITensors
using ITensorMPS
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using ProgressMeter

# -----------------------------
# ENV helpers
# -----------------------------
getenv_str(name::String, default::String) = get(ENV, name, default)
getenv_int(name::String, default::Int) = parse(Int, get(ENV, name, string(default)))
getenv_float(name::String, default::Float64) = parse(Float64, get(ENV, name, string(default)))

function getenv_float_list(name::String, default::Vector{Float64})
    s = get(ENV, name, "")
    isempty(strip(s)) && return default
    return [parse(Float64, strip(x)) for x in split(s, ",") if !isempty(strip(x))]
end

Base.@kwdef struct Params
    N::Int = getenv_int("N", 20)
    model::String = getenv_str("MODEL", "mixed_ising")

    # tfim:        H = -JXX Σ X_i X_{i+1} - GZ Σ Z_i
    # mixed_ising: H =  JZZ Σ Z_i Z_{i+1} + GX Σ X_i + HZ Σ Z_i
    JXX::Float64 = getenv_float("JXX", 1.0)
    GZ::Float64 = getenv_float("GZ", 1.5)
    JZZ::Float64 = getenv_float("JZZ", 1.0)
    GX::Float64 = getenv_float("GX", 0.9045)
    HZ::Float64 = getenv_float("HZ", 0.809)

    beta::Float64 = getenv_float("BETA", 1.0)
    betas::Vector{Float64} = getenv_float_list("BETAS", [getenv_float("BETA", 1.0)])

    # purification imaginary-time evolution
    ntau::Int = getenv_int("NTAU", 120)

    # filtered operator / Choi TEBD parameters
    center_site::Int = getenv_int("CENTER_SITE", max(1, div(getenv_int("N", 20) + 1, 2)))
    Aop::String = getenv_str("AOP", "XplusZ")
    sigma::Float64 = getenv_float("SIGMA", 1.0)    # Hahn convention f(s) width
    smax::Float64 = getenv_float("S_MAX", 5.0)
    ds::Float64 = getenv_float("DS", 0.05)

    maxdim::Int = getenv_int("MAXDIM", 256)
    cutoff::Float64 = getenv_float("CUTOFF", 1e-9)
    outdir::String = getenv_str("OUTDIR", "tn_results")
    out_prefix::String = getenv_str("OUT_PREFIX", "tn")
end

# -----------------------------
# Local matrices
# -----------------------------
const I2 = ComplexF64[1 0; 0 1]
const X2 = ComplexF64[0 1; 1 0]
const Y2 = ComplexF64[0 -im; im 0]
const Z2 = ComplexF64[1 0; 0 -1]

# Composite local Hilbert space convention:
#   thermal_ref:      physical spin ⊗ purification ancilla
#   filtered_operator: ket spin ⊗ bra spin (Choi/vectorized operator)
const XP = kron(X2, I2)
const YP = kron(Y2, I2)
const ZP = kron(Z2, I2)
const IP = kron(I2, I2)

const XR = kron(I2, transpose(X2))
const YR = kron(I2, transpose(Y2))
const ZR = kron(I2, transpose(Z2))
const IR = kron(I2, I2)

one_site_op(mat::AbstractMatrix, s) = ITensor(mat, prime(s), dag(s))

function product_mps_from_local_vectors(sites, vecs::Vector{Vector{ComplexF64}})
    N = length(sites)
    psi = MPS(sites)
    for i in 1:N
        psi[i] = ITensor(vecs[i], sites[i])
    end
    return psi
end

function scaled_copy(psi::MPS, α::Number)
    phi = deepcopy(psi)
    phi[1] = α * phi[1]
    return phi
end

function add_compressed(a::MPS, b::MPS, p::Params)
    return add(a, b; cutoff=p.cutoff, maxdim=p.maxdim)
end

function normalize_safe!(psi::MPS)
    nrm = norm(psi)
    nrm == 0 && error("zero MPS norm")
    normalize!(psi)
    return psi
end

# -----------------------------
# Model terms
# -----------------------------
function model_local_left(p::Params)
    if p.model == "tfim"
        return -p.GZ * ZP
    elseif p.model == "mixed_ising"
        return p.GX * XP + p.HZ * ZP
    else
        error("Unknown MODEL=$(p.model). Use tfim or mixed_ising.")
    end
end

function model_local_right(p::Params)
    if p.model == "tfim"
        return -p.GZ * ZR
    elseif p.model == "mixed_ising"
        return p.GX * XR + p.HZ * ZR
    else
        error("Unknown MODEL=$(p.model). Use tfim or mixed_ising.")
    end
end

function model_bonds_left(p::Params)
    if p.model == "tfim"
        return [(-p.JXX, XP, XP)]
    elseif p.model == "mixed_ising"
        return [(+p.JZZ, ZP, ZP)]
    end
end

function model_bonds_right(p::Params)
    if p.model == "tfim"
        return [(-p.JXX, XR, XR)]
    elseif p.model == "mixed_ising"
        return [(+p.JZZ, ZR, ZR)]
    end
end

# -----------------------------
# Observable measurements on purification MPS
# -----------------------------
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

function thermal_observables!(psi::MPS, sites, p::Params)
    N = p.N
    mz = mean(expect1!(psi, sites, i, ZP) for i in 1:N)
    mx = mean(expect1!(psi, sites, i, XP) for i in 1:N)
    xx = N > 1 ? mean(expect2!(psi, sites, i, XP, XP) for i in 1:(N-1)) : 0.0
    zz = N > 1 ? mean(expect2!(psi, sites, i, ZP, ZP) for i in 1:(N-1)) : 0.0

    e = 0.0
    if p.model == "tfim"
        e += -p.GZ * sum(expect1!(psi, sites, i, ZP) for i in 1:N)
        e += -p.JXX * sum(expect2!(psi, sites, i, XP, XP) for i in 1:(N-1))
    elseif p.model == "mixed_ising"
        e += sum(p.GX * expect1!(psi, sites, i, XP) + p.HZ * expect1!(psi, sites, i, ZP) for i in 1:N)
        e += p.JZZ * sum(expect2!(psi, sites, i, ZP, ZP) for i in 1:(N-1))
    end

    return (E=e/N, MZ=mz, MX=mx, XX=xx, ZZ=zz, BOND=maximum(linkdims(psi)))
end

# -----------------------------
# Purification finite-temperature reference
# -----------------------------
function make_imag_gates(sites, p::Params, dtau::Float64)
    gates = ITensor[]
    Hloc = model_local_left(p)  # physical copy only

    # local half step
    for i in 1:p.N
        push!(gates, exp(-(dtau/2) * one_site_op(Hloc, sites[i])))
    end

    # even half bonds
    for i in 1:2:(p.N-1)
        for (c, O1, O2) in model_bonds_left(p)
            h = c * one_site_op(O1, sites[i]) * one_site_op(O2, sites[i+1])
            push!(gates, exp(-(dtau/2) * h))
        end
    end

    # odd full bonds
    for i in 2:2:(p.N-1)
        for (c, O1, O2) in model_bonds_left(p)
            h = c * one_site_op(O1, sites[i]) * one_site_op(O2, sites[i+1])
            push!(gates, exp(-dtau * h))
        end
    end

    # even half bonds
    for i in 1:2:(p.N-1)
        for (c, O1, O2) in model_bonds_left(p)
            h = c * one_site_op(O1, sites[i]) * one_site_op(O2, sites[i+1])
            push!(gates, exp(-(dtau/2) * h))
        end
    end

    # local half step
    for i in 1:p.N
        push!(gates, exp(-(dtau/2) * one_site_op(Hloc, sites[i])))
    end
    return gates
end

function infinite_temperature_purification(sites, p::Params)
    # local thermofield pair |00> + |11>, normalized.
    v = ComplexF64[1, 0, 0, 1] ./ sqrt(2)
    return product_mps_from_local_vectors(sites, [copy(v) for _ in 1:p.N])
end

function purification_reference(p::Params; beta::Float64=p.beta)
    sites = siteinds("Qudit", p.N; dim=4)
    psi = infinite_temperature_purification(sites, p)
    if beta > 0
        total_tau = beta / 2
        dtau = total_tau / p.ntau
        gates = make_imag_gates(sites, p, dtau)
        @showprogress for _ in 1:p.ntau
            psi = apply(gates, psi; cutoff=p.cutoff, maxdim=p.maxdim)
            normalize_safe!(psi)
        end
    end
    obs = thermal_observables!(psi, sites, p)
    return merge((beta=beta, norm=norm(psi)), obs)
end

function mode_thermal_ref(p::Params)
    mkpath(p.outdir)
    outrows = NamedTuple[]
    for β in p.betas
        println("Computing purification reference β=$β, N=$(p.N), model=$(p.model)")
        obs = purification_reference(p; beta=β)
        push!(outrows, (; N=p.N, model=p.model, JXX=p.JXX, GZ=p.GZ, JZZ=p.JZZ, GX=p.GX, HZ=p.HZ,
                       beta=β, ntau=p.ntau, maxdim=p.maxdim, cutoff=p.cutoff,
                       E=obs.E, MZ=obs.MZ, MX=obs.MX, XX=obs.XX, ZZ=obs.ZZ,
                       BOND=obs.BOND))
    end
    rows = DataFrame(outrows)
    fname = joinpath(p.outdir, "$(p.out_prefix)_thermal_ref_N$(p.N)_$(p.model).csv")
    CSV.write(fname, rows)
    println("Saved: $fname")
end

# -----------------------------
# Choi-MPS filtered operator construction
# -----------------------------
function local_vec_from_matrix(M::AbstractMatrix)
    # Basis convention |ket,bra>: [00, 01, 10, 11].
    # This is row-major flattening. It is internally consistent with the left/right
    # actions XP=kron(X,I), XR=kron(I,X^T) used below.
    return ComplexF64[M[1,1], M[1,2], M[2,1], M[2,2]]
end

function local_A_matrix(name::String)
    if name == "X"
        return X2
    elseif name == "Y"
        return Y2
    elseif name == "Z"
        return Z2
    elseif name == "XplusZ"
        return (X2 + Z2) / sqrt(2)
    elseif name == "ZplusY"
        return (Z2 + Y2) / sqrt(2)
    else
        error("Unknown AOP=$name. Use X, Y, Z, XplusZ, ZplusY.")
    end
end

function local_operator_choi_mps(sites, p::Params)
    idv = local_vec_from_matrix(I2)
    Av = local_vec_from_matrix(local_A_matrix(p.Aop))
    vecs = [copy(idv) for _ in 1:p.N]
    vecs[p.center_site] = Av
    psi = product_mps_from_local_vectors(sites, vecs)
    normalize_safe!(psi)
    return psi
end

function make_choi_heisenberg_gates(sites, p::Params, dt::Float64)
    # Evolves |A>> by A(t+dt)=e^{iHdt} A(t) e^{-iHdt}.
    # Generator on vectorized operator state: +H_left - H_right.
    gates = ITensor[]
    HlocL = model_local_left(p)
    HlocR = model_local_right(p)

    # one-site half step
    for i in 1:p.N
        h = one_site_op(HlocL - HlocR, sites[i])
        push!(gates, exp(+1im * (dt/2) * h))
    end

    # even half bonds
    for i in 1:2:(p.N-1)
        for ((cL,O1L,O2L),(cR,O1R,O2R)) in zip(model_bonds_left(p), model_bonds_right(p))
            hL = cL * one_site_op(O1L, sites[i]) * one_site_op(O2L, sites[i+1])
            hR = cR * one_site_op(O1R, sites[i]) * one_site_op(O2R, sites[i+1])
            push!(gates, exp(+1im * (dt/2) * (hL - hR)))
        end
    end

    # odd full bonds
    for i in 2:2:(p.N-1)
        for ((cL,O1L,O2L),(cR,O1R,O2R)) in zip(model_bonds_left(p), model_bonds_right(p))
            hL = cL * one_site_op(O1L, sites[i]) * one_site_op(O2L, sites[i+1])
            hR = cR * one_site_op(O1R, sites[i]) * one_site_op(O2R, sites[i+1])
            push!(gates, exp(+1im * dt * (hL - hR)))
        end
    end

    # even half bonds again
    for i in 1:2:(p.N-1)
        for ((cL,O1L,O2L),(cR,O1R,O2R)) in zip(model_bonds_left(p), model_bonds_right(p))
            hL = cL * one_site_op(O1L, sites[i]) * one_site_op(O2L, sites[i+1])
            hR = cR * one_site_op(O1R, sites[i]) * one_site_op(O2R, sites[i+1])
            push!(gates, exp(+1im * (dt/2) * (hL - hR)))
        end
    end

    # one-site half step again
    for i in 1:p.N
        h = one_site_op(HlocL - HlocR, sites[i])
        push!(gates, exp(+1im * (dt/2) * h))
    end
    return gates
end

function hahn_filter_weight(s::Float64, p::Params)
    # Hahn/Ong-Parameswaran-Placke-Hahn convention:
    # f(s)=sqrt(2/(πσ²))*exp[-2/σ²*(s - iβ/4)^2]
    σ = p.sigma
    return sqrt(2/(pi*σ^2)) * exp(-(2/σ^2) * (s - 1im*p.beta/4)^2)
end

function evolve_choi!(psi::MPS, gates, p::Params)
    psi = apply(gates, psi; cutoff=p.cutoff, maxdim=p.maxdim)
    # Heisenberg evolution is unitary in exact Choi norm. Normalize to suppress drift;
    # the quadrature weights carry the physical amplitude.
    normalize_safe!(psi)
    return psi
end

function filtered_operator_mps(p::Params)
    sites = siteinds("Qudit", p.N; dim=4)
    A0 = local_operator_choi_mps(sites, p)

    # Build A(s=-smax) from A(0), then sweep forward.
    npre = round(Int, p.smax / p.ds)
    neg_gates = make_choi_heisenberg_gates(sites, p, -p.ds)
    pos_gates = make_choi_heisenberg_gates(sites, p, +p.ds)

    psi = deepcopy(A0)
    @showprogress "pre-evolve to -S_MAX" for _ in 1:npre
        psi = evolve_choi!(psi, neg_gates, p)
    end

    ns = 2*npre + 1
    svals = collect(range(-npre*p.ds, step=p.ds, length=ns))
    K = nothing
    rows = DataFrame(step=Int[], s=Float64[], abs_weight=Float64[], bond=Int[], norm2=Float64[])

    @showprogress "quadrature" for (j,s) in enumerate(svals)
        w = hahn_filter_weight(s, p) * p.ds
        term = scaled_copy(psi, w)
        if K === nothing
            K = term
        else
            K = add_compressed(K, term, p)
        end
        push!(rows, (j, s, abs(w), maximum(linkdims(psi)), real(inner(psi, psi))))
        if j < length(svals)
            psi = evolve_choi!(psi, pos_gates, p)
        end
    end

    K === nothing && error("empty quadrature")
    # K has already been compressed at every addition step.
    return K, rows
end

function mode_filtered_operator(p::Params)
    mkpath(p.outdir)
    println("Constructing filtered Choi-MPS operator")
    println("  N=$(p.N), model=$(p.model), beta=$(p.beta), sigma=$(p.sigma), smax=$(p.smax), ds=$(p.ds)")
    println("  AOP=$(p.Aop), center_site=$(p.center_site), maxdim=$(p.maxdim), cutoff=$(p.cutoff)")
    K, rows = filtered_operator_mps(p)

    diag = DataFrame(
        N=[p.N], model=[p.model], beta=[p.beta], sigma=[p.sigma], smax=[p.smax], ds=[p.ds],
        center_site=[p.center_site], AOP=[p.Aop], maxdim=[p.maxdim], cutoff=[p.cutoff],
        final_bond=[maximum(linkdims(K))], frob_norm=[sqrt(max(real(inner(K,K)),0.0))]
    )

    f1 = joinpath(p.outdir, "$(p.out_prefix)_filtered_operator_diag_N$(p.N)_$(p.model)_site$(p.center_site)_$(p.Aop).csv")
    f2 = joinpath(p.outdir, "$(p.out_prefix)_filtered_operator_quadrature_N$(p.N)_$(p.model)_site$(p.center_site)_$(p.Aop).csv")
    CSV.write(f1, diag)
    CSV.write(f2, rows)
    println("Saved: $f1")
    println("Saved: $f2")
    println("Note: the filtered operator is kept in memory only. For production, add HDF5/JLD2 saving after this point.")
end

function main()
    p = Params()
    mode = getenv_str("MODE", "thermal_ref")
    println("MODE=$mode")
    println("Params: N=$(p.N), MODEL=$(p.model), MAXDIM=$(p.maxdim), CUTOFF=$(p.cutoff)")
    if mode == "thermal_ref"
        mode_thermal_ref(p)
    elseif mode == "filtered_operator"
        mode_filtered_operator(p)
    else
        error("Unknown MODE=$mode. Use thermal_ref or filtered_operator.")
    end
end

main()
