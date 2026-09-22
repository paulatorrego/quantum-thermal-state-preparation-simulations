# Tensor-network simulations

This directory contains the compact, reproducible tensor-network part of the quantum thermal-state preparation project. The exact calculations for the different models remain in `../exact/`.

## What is included

- `julia/mps_trajectories_n20.jl`: main production implementation of the Lloyd–Abanin repeated-interaction protocol using MPS quantum trajectories for the mixed-field Ising chain. It includes the bath reset/unravelling, randomization, MPS compression and convergence diagnostics used in the N=20 study.
- `julia/thermal_reference_and_filtered_mpo.jl`: finite-temperature purification reference and filtered-MPO/Choi utilities.
- `julia/effective_channel_mpo.jl`: system-only effective-channel approximation used for fast parameter screening.
- `julia/composite_density_mpo.jl`: deterministic explicit-bath density-MPO validation of short protocol runs.
- `julia/rhoA_fidelity_scaling.jl`: local reduced-density-matrix fidelity/trace-distance study versus system size.

The four first Julia files correspond to the complementary TN roles used in the project: production trajectories, thermal reference, effective-channel screening, and deterministic channel validation.

## Figures

The figures are the selected outputs needed to document the main numerical story:

1. N=20 theta² scaling of observables.
2. N=20 energy error versus theta².
3. N=20 thermalization/full-trace observables.
4. N=20 bond-dimension evolution.
5. N=20 relaxation-time diagnostic.
6. Bond-dimension scaling with N.
7. Local rho_A fidelity versus N.
8. Local rho_A thermalization-time scaling.

## Results

Only compact summary CSVs are included. Raw trajectory samples, logs, checkpoints, duplicated exports and intermediate archives are intentionally omitted.

## Reproducibility

The archived figures/results do **not** need to be regenerated merely to publish the repository. After copying this directory into the repository, do a cheap environment check:

```bash
cd tensor
julia --project=. -e 'using Pkg; Pkg.instantiate(); using ITensors, ITensorMPS, CSV, DataFrames, ProgressMeter; println("Tensor environment OK")'
```

Do not launch the expensive N=20 campaigns just for the GitHub upload.

To regenerate analyses later, the Python scripts in `analysis/` are the selected post-processing entry points. They expect the corresponding exported CSV structure from the original campaigns; the archived summary CSVs and figures are kept as the publication/reproducibility snapshot.

## Main N=20 protocol

The production implementation uses a composite MPS site `S_i ⊗ B_i` with local dimension 4, representing one system qubit and one resettable bath qubit per site. The bath is evolved, measured/reset, and the resulting pure-state trajectories are averaged over independent realizations.

The N=20 campaign uses the mixed-field Ising Hamiltonian and the parameter set documented in the original analysis package. The supplied results are archived outputs of those runs; no new numerical claim is introduced by this repository organization.
