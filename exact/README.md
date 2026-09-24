# Exact simulations

This directory contains the finite-size, explicit-density-matrix implementation of the quantum thermal-state preparation protocol studied in the TFM.

The protocol follows:

> J. Lloyd and D. A. Abanin, *Quantum Thermal State Preparation for Near-Term Quantum Processors*, Physical Review X **16**, 031053 (2026).  
> https://arxiv.org/abs/2506.21318

The exact implementation evolves a system together with a resettable auxiliary bath as a finite-dimensional quantum channel and compares its dynamics and stationary state with the exact Gibbs state.

## 1. What is implemented?

The exact calculations provide the controlled finite-size reference for the project. They include:

- single-spin systems;
- non-interacting spin systems;
- one-dimensional and two-dimensional Ising models;
- Fermi-Hubbard systems after Jordan-Wigner mapping;
- convergence from different initial states;
- fixed-point calculations;
- trace distance, fidelity and energy diagnostics;
- perturbative scaling with the system-bath coupling `theta`;
- filter-time and bath-scale scans;
- resonances and randomization;
- mixing-time studies;
- selected Qiskit ideal/noisy validation and resource studies.

The same physical protocol is therefore tested across increasingly non-trivial Hamiltonians before the tensor-network extension.

## 2. Numerical workflow

```text
Hamiltonian H_S
     │
     ├── exact Gibbs state rho_beta
     │
     └── repeated system-bath channel
              │
              ├── prepare/reset bath
              ├── joint evolution
              ├── optional randomization
              └── repeat
                    │
                    ▼
              rho_n / rho_fixed
                    │
                    ├── trace distance
                    ├── fidelity
                    ├── energy
                    └── model-specific observables
```

The exact calculations make it possible to distinguish protocol bias from numerical tensor-network approximations later.

## 3. Repository structure

```text
exact/
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 00_global_overview.ipynb
│   ├── 01_single_spin.ipynb
│   ├── 02_noninteracting.ipynb
│   ├── 03_ising.ipynb
│   └── 04_fermi_hubbard.ipynb
│
├── src/
│   ├── tfm_shared/
│   │   ├── core.py
│   │   ├── plots.py
│   │   └── qiskit_impl.py
│   ├── tfm_single_spin/
│   │   └── model.py
│   ├── tfm_nonint/
│   │   └── model.py
│   ├── tfm_ising/
│   │   └── model.py
│   └── tfm_fh/
│       └── model.py
│
├── results/
│   ├── mixing_beta_1d4.csv
│   └── thermalization_fan_light.csv
│
├── figures/
│   ├── single_spin/
│   ├── noninteracting/
│   ├── ising/
│   └── fermi_hubbard/
│
└── lamb_shift/
    ├── run_lamb_shift_n3.py
    ├── make_lamb_shift_plots.py
    ├── results/
    └── figures/
```

## 4. Models

### Single spin

The single-spin notebook is the smallest controlled test of the protocol. It is used to study:

- fixed-point convergence;
- dependence on `theta` and filter duration;
- mixing rates and spectral gap;
- the basic perturbative regime.

### Non-interacting model

The non-interacting model provides a bridge between a single spin and genuinely correlated many-body systems.

It is particularly useful for studying:

- the expected `theta^2` bias;
- filter resonances;
- bath geometry;
- the effect of using fewer auxiliary bath qubits than system qubits.

### Ising model

The Ising notebook is the main interacting spin benchmark.

It studies both one-dimensional and two-dimensional geometries and includes:

- convergence to Gibbs;
- `theta^2` scaling;
- mixing-time scaling;
- bath geometry;
- spectral matching between the bath/filter and system Bohr frequencies;
- randomization and resonance effects;
- selected Qiskit validation.

### Fermi-Hubbard model

The Fermi-Hubbard implementation extends the protocol to a fermionic many-body Hamiltonian mapped to qubits with Jordan-Wigner.

It studies:

- thermalization of the Hubbard dimer and small systems;
- energy, double occupancy and spin correlations;
- dependence on `U/t`, temperature and bath scale `h`;
- perturbative `theta^2` scaling;
- finite-size scaling;
- randomization and filter resonances;
- a half-filling/canonical variant.

## 5. Exact versus tensor-network calculations

The exact and tensor-network directories implement the same physical thermal-state-preparation problem with different numerical representations:

```text
exact/
explicit density matrices
       │
       │ finite-size validation
       ▼
tensor/
MPS / MPO representations
       │
       ▼
larger one-dimensional systems
```

The exact results therefore serve as the reference calculations for the tensor-network simulations.

## 6. Running the notebooks

The notebooks add `exact/src/` to `sys.path` automatically.

From the `exact/` directory:

```bash
pip install -r requirements.txt
jupyter notebook
```

Then open the desired notebook.

The notebooks contain the final simulation cells and their stored outputs. The reusable numerical functions are kept under `src/`.

## 7. Supplementary Lamb-shift analysis

`lamb_shift/` is intentionally a small supplementary section, not a second full simulation framework.

It contains a self-contained `N=3` mixed-field Ising calculation used to analyse the weak-coupling effective Hamiltonian, mean-force and Lamb-shift corrections.

The analysis compares:

- the exact fixed point;
- effective-Hamiltonian constructions;
- perturbative quantities at order `theta^2`;
- the effect of the randomization parameter `lambda`.

The main exact implementation remains in `notebooks/`. Only the compact, final Lamb-shift analysis is retained here.

## 8. What is intentionally excluded

The GitHub version does not include:

- `.git/`;
- Python bytecode and `__pycache__/`;
- execution logs;
- temporary/debug scripts;
- duplicated notebooks;
- checkpoints;
- intermediate sweep directories;
- local machine configuration;
- duplicated paper PDFs.

The selected figures are final outputs extracted from the notebooks, while the notebooks themselves retain the detailed calculations and stored outputs.
