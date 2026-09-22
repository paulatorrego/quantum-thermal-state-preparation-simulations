# n20_mpo_system_effective_channel.jl
# Bath-eliminated system-only density MPO approximation.
#
# It applies:
#   1) system unitary TEBD layer U_S,
#   2) local effective collision channel Phi_i from a fresh bath qubit |0>,
#      traced out immediately after each filter layer.
#
# This is NOT identical to the original protocol because the bath has no memory
# over the full sequence tau=-M,...,M. It is a controlled diagnostic alternative:
# if this scales well, explicit bath memory / trajectory unraveling is the source
# of the bond explosion.

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
    beta::Float64 = 1.0
    bath_h::Float64 = 4.0
    theta::Float64 = 0.45
    delta::Float64 = π / 40
    reset_time::Float64 = 1.0
    filter_a::Float64 = -1.0
    Aop::String = "ZplusY"
    ncycles::Int = 30
    maxdim::Int = 384
    cutoff::Float64 = 1e-8
    steady_frac::Float64 = 0.3
    sample_start::Int = -1
    sample_stride::Int = 1
    outdir::String = "results_N20"
    out_prefix::String = "N20_mpo_system_effective"
    ref_E::Float64 = NaN
    ref_MZ::Float64 = NaN
    ref_MX::Float64 = NaN
    ref_ZZ::Float64 = NaN
end

function params_from_env()
    return Params(
        N=getenv_int("N", 20), model=getenv_str("MODEL", "mixed_ising"),
        Jzz=getenv_float("JZZ", 1.0), gx=getenv_float("GX", 0.9045), hz=getenv_float("HZ", 0.809),
        beta=getenv_float("BETA", 1.0), bath_h=getenv_float("BATH_H", 4.0), theta=getenv_float("THETA", 0.45),
        delta=getenv_float("DELTA", π/40), reset_time=getenv_float("RESET_TIME", 1.0), filter_a=getenv_float("FILTER_A", -1.0),
        Aop=getenv_str("AOP", "ZplusY"), ncycles=getenv_int("NCYCLES", 30), maxdim=getenv_int("MAXDIM", 384),
        cutoff=getenv_float("CUTOFF", 1e-8), steady_frac=getenv_float("STEADY_FRAC", 0.3),
        sample_start=getenv_int("SAMPLE_START", -1), sample_stride=getenv_int("SAMPLE_STRIDE", 1),
        outdir=getenv_str("OUTDIR", "results_N20"), out_prefix=getenv_str("OUT_PREFIX", "N20_mpo_system_effective"),
        ref_E=getenv_float("REF_E", NaN), ref_MZ=getenv_float("REF_MZ", NaN),
        ref_MX=getenv_float("REF_MX", NaN), ref_ZZ=getenv_float("REF_ZZ", NaN)
    )
end

const I2 = ComplexF64[1 0; 0 1]
const X2 = ComplexF64[0 1; 1 0]
const Y2 = ComplexF64[0 -im; im 0]
const Z2 = ComplexF64[1 0; 0 -1]
const DSYS = 2
const DDENS = 4

function A_matrix(p::Params)
    p.Aop == "Y" && return Y2
    p.Aop == "Z" && return Z2
    p.Aop == "X" && return X2
    p.Aop == "ZplusY" && return (Z2 + Y2) / sqrt(2)
    p.Aop == "XplusZ" && return (X2 + Z2) / sqrt(2)
    error("Unknown AOP=$(p.Aop)")
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

den_index(ket::Int, bra::Int, d::Int) = ket + d * (bra - 1)
pair_pure_index(ai::Int, aj::Int, d::Int) = ai + d * (aj - 1)
pair_den_index(local_i::Int, local_j::Int, D::Int) = local_i + D * (local_j - 1)

function one_site_super_from_kraus(kraus::Vector{Matrix{ComplexF64}}; d::Int=DSYS)
    D = d^2
    S = zeros(ComplexF64, D, D)
    for K in kraus
        S .+= kron(conj(K), K)
    end
    return S
end
one_site_super_unitary(U::Matrix{ComplexF64}; d::Int=DSYS) = one_site_super_from_kraus([U]; d=d)

function pair_super_unitary(U::Matrix{ComplexF64}; d::Int=DSYS)
    Dsite = d^2
    S = zeros(ComplexF64, Dsite^2, Dsite^2)
    for ki in 1:d, kj in 1:d, bi in 1:d, bj in 1:d
        col_i = den_index(ki, bi, d); col_j = den_index(kj, bj, d)
        col = pair_den_index(col_i, col_j, Dsite)
        ket_in = pair_pure_index(ki,kj,d); bra_in = pair_pure_index(bi,bj,d)
        for ko_i in 1:d, ko_j in 1:d, bo_i in 1:d, bo_j in 1:d
            row_i = den_index(ko_i, bo_i, d); row_j = den_index(ko_j, bo_j, d)
            row = pair_den_index(row_i, row_j, Dsite)
            ket_out = pair_pure_index(ko_i,ko_j,d); bra_out = pair_pure_index(bo_i,bo_j,d)
            S[row,col] += U[ket_out,ket_in] * conj(U[bra_out,bra_in])
        end
    end
    return S
end

function one_site_gate(S::Matrix{ComplexF64}, s)
    D = dim(s); @assert size(S)==(D,D)
    T = ITensor(prime(s), dag(s))
    for a in 1:D, b in 1:D
        T[prime(s)=>a, dag(s)=>b] = S[a,b]
    end
    return T
end
function two_site_gate(S::Matrix{ComplexF64}, si, sj)
    D = dim(si); @assert size(S)==(D^2,D^2)
    T = ITensor(prime(si), prime(sj), dag(si), dag(sj))
    for ai_out in 1:D, aj_out in 1:D, ai_in in 1:D, aj_in in 1:D
        row = pair_den_index(ai_out, aj_out, D); col = pair_den_index(ai_in, aj_in, D)
        val = S[row,col]
        if val != 0
            T[prime(si)=>ai_out, prime(sj)=>aj_out, dag(si)=>ai_in, dag(sj)=>aj_in] = val
        end
    end
    return T
end

function product_mps_from_vectors(sites, vecs::Vector{Vector{ComplexF64}})
    tensors = Vector{ITensor}(undef, length(sites))
    for i in 1:length(sites)
        T=ITensor(sites[i])
        for a in 1:length(vecs[i])
            T[sites[i]=>a] = vecs[i][a]
        end
        tensors[i]=T
    end
    return MPS(tensors)
end
function vec_density(rho::Matrix{ComplexF64})
    d=size(rho,1); v=zeros(ComplexF64,d^2)
    for k in 1:d, b in 1:d
        v[den_index(k,b,d)] = rho[k,b]
    end
    return v
end
function trace_vec(d::Int)
    v=zeros(ComplexF64,d^2); for a in 1:d; v[den_index(a,a,d)] = 1; end; return v
end
function obs_vec(O::Matrix{ComplexF64})
    d=size(O,1); v=zeros(ComplexF64,d^2)
    for ket in 1:d, bra in 1:d
        v[den_index(ket,bra,d)] = O[bra,ket]
    end
    return v
end
function product_bra(sites, local_vecs)
    return product_mps_from_vectors(sites, [ComplexF64.(conj(v)) for v in local_vecs])
end
function trace_rho(rho::MPS, sites)
    bra = product_bra(sites, fill(trace_vec(DSYS), length(sites)))
    return real(inner(bra, rho))
end
function linear_expect(rho::MPS, sites, local_ops::Dict{Int,Matrix{ComplexF64}})
    vecs=[trace_vec(DSYS) for _ in 1:length(sites)]
    for (i,O) in local_ops
        vecs[i]=obs_vec(O)
    end
    bra=product_bra(sites, vecs)
    return real(inner(bra,rho)) / trace_rho(rho,sites)
end
function observables(rho::MPS, sites, p::Params)
    N=p.N
    mx=mean(linear_expect(rho,sites,Dict(i=>X2)) for i in 1:N)
    mz=mean(linear_expect(rho,sites,Dict(i=>Z2)) for i in 1:N)
    zz=mean(linear_expect(rho,sites,Dict(i=>Z2,i+1=>Z2)) for i in 1:N-1)
    E=p.gx*mx + p.hz*mz + p.Jzz*zz*(N-1)/N
    return (E=E,MZ=mz,MX=mx,ZZ=zz,BOND=maximum(linkdims(rho)))
end
function rescale_trace!(rho::MPS, sites)
    tr=trace_rho(rho,sites)
    if !(isfinite(tr)) || abs(tr)<1e-14; error("Invalid trace $tr"); end
    rho[1] *= (1/tr)
    return rho
end

function system_local_h(p::Params)
    p.model == "mixed_ising" && return p.gx*X2 + p.hz*Z2
    error("Only mixed_ising implemented in this diagnostic script")
end
function make_system_gates(sites,p::Params,dt::Float64)
    gates=ITensor[]
    Uloc=exp(-1im*(dt/2)*system_local_h(p)); Sloc=one_site_super_unitary(Uloc)
    for i in 1:p.N; push!(gates, one_site_gate(Sloc, sites[i])); end
    for i in 1:2:p.N-1
        U=exp(-1im*(dt/2)*(p.Jzz*kron(Z2,Z2)))
        push!(gates,two_site_gate(pair_super_unitary(U),sites[i],sites[i+1]))
    end
    for i in 2:2:p.N-1
        U=exp(-1im*dt*(p.Jzz*kron(Z2,Z2)))
        push!(gates,two_site_gate(pair_super_unitary(U),sites[i],sites[i+1]))
    end
    for i in 1:2:p.N-1
        U=exp(-1im*(dt/2)*(p.Jzz*kron(Z2,Z2)))
        push!(gates,two_site_gate(pair_super_unitary(U),sites[i],sites[i+1]))
    end
    for i in 1:p.N; push!(gates, one_site_gate(Sloc, sites[i])); end
    return gates
end

function effective_collision_super(p::Params, fτ::Float64)
    # Bath starts in |0>, U=exp[-i dt theta f A⊗Y_B], bath traced/reset immediately.
    A=A_matrix(p)
    U=exp(-1im*p.delta*p.theta*fτ*kron(A,Y2)) # system-bath pure dimension 4, basis S fastest, B slowest
    # K_b = <b|U|0> acting on system.
    K = Matrix{ComplexF64}[]
    for bout in 1:2
        Kb=zeros(ComplexF64,2,2)
        for so in 1:2, si in 1:2
            row = so + 2*(bout-1)
            col = si + 2*(1-1)
            Kb[so,si]=U[row,col]
        end
        push!(K,Kb)
    end
    return one_site_super_from_kraus(K; d=DSYS)
end

function initial_density_mps(sites,p::Params)
    rhoS=I2/2
    v=vec_density(rhoS)
    return product_mps_from_vectors(sites,[v for _ in 1:p.N])
end

function run_channel(p::Params)
    mkpath(p.outdir)
    sites=siteinds("Qudit",p.N; dim=DDENS)
    rho=initial_density_mps(sites,p)
    rescale_trace!(rho,sites)
    taus,fvals=filter_values(p)
    U0=make_system_gates(sites,p,p.delta)
    Phis=[[one_site_gate(effective_collision_super(p,f),sites[i]) for i in 1:p.N] for f in fvals]
    sample_start=p.sample_start>0 ? p.sample_start : max(1,floor(Int,(1-p.steady_frac)*p.ncycles)+1)
    rows=NamedTuple[]; BOND_peak=maximum(linkdims(rho))
    println("MODE=mpo_system_effective_channel")
    println("N=$(p.N), theta=$(p.theta), reset_time=$(p.reset_time), M=$(protocol_M(p)), maxdim=$(p.maxdim), cutoff=$(p.cutoff)")
    prog=Progress(p.ncycles; desc="mpo_system cycles")
    for c in 1:p.ncycles
        for k in eachindex(fvals)
            rho=apply(vcat(U0,Phis[k]),rho; cutoff=p.cutoff,maxdim=p.maxdim)
        end
        rescale_trace!(rho,sites)
        BOND_peak=max(BOND_peak,maximum(linkdims(rho)))
        if c>=sample_start && ((c-sample_start)%p.sample_stride==0)
            ob=observables(rho,sites,p)
            push!(rows,(cycle=c,E=ob.E,MZ=ob.MZ,MX=ob.MX,ZZ=ob.ZZ,BOND=ob.BOND,trace=trace_rho(rho,sites)))
        end
        next!(prog)
    end
    df=DataFrame(rows)
    CSV.write(joinpath(p.outdir,"$(p.out_prefix)_samples.csv"),df)
    E=mean(df.E); MZ=mean(df.MZ); MX=mean(df.MX); ZZ=mean(df.ZZ)
    drift=nrow(df)>=2 ? abs(df.E[end]-df.E[1]) : NaN
    out=DataFrame((N=p.N,model=p.model,representation="system_effective_channel_mpo",theta=p.theta,theta2=p.theta^2,
        beta=p.beta,bath_h=p.bath_h,reset_time=p.reset_time,delta=p.delta,M=protocol_M(p),ncycles=p.ncycles,
        maxdim=p.maxdim,cutoff=p.cutoff,sample_start=sample_start,sample_stride=p.sample_stride,
        E_mean=E,MZ_mean=MZ,MX_mean=MX,ZZ_mean=ZZ,
        E_abs_error=abs(E-p.ref_E),MZ_abs_error=abs(MZ-p.ref_MZ),MX_abs_error=abs(MX-p.ref_MX),ZZ_abs_error=abs(ZZ-p.ref_ZZ),
        E_mean_abs_drift=drift,BOND_mean=mean(df.BOND),BOND_max=maximum(df.BOND),BOND_peak_all=BOND_peak,
        saturated=(BOND_peak>=0.95*p.maxdim),n_late_samples_total=nrow(df)))
    outpath=joinpath(p.outdir,"$(p.out_prefix)_theta_scaling.csv")
    CSV.write(outpath,out)
    println("Saved: $outpath")
    println(out)
end

function main()
    p0=params_from_env()
    thetas=getenv_float_list("THETAS",[p0.theta])
    for th in thetas
        p=Params(; (field=>getfield(p0,field) for field in fieldnames(Params))..., theta=th,
                 out_prefix=length(thetas)>1 ? "$(p0.out_prefix)_theta$(replace(string(th),'.'=>''))" : p0.out_prefix)
        run_channel(p)
    end
end
main()
