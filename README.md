# Quantum Thermal State Preparation — Original TFM Implementation

Numerical implementation developed as part of my Master's Thesis in Quantum Computing at the Universidad Autónoma de Madrid and the Instituto de Física Teórica.

The project studies the preparation of quantum thermal (Gibbs) states of many-body systems using a repeated-interaction protocol with a resettable auxiliary bath, following the approach introduced by Lloyd and Abanin.

The repository contains two complementary numerical approaches:

* `exact/` — exact finite-size simulations, used to implement and validate the protocol for small quantum systems.
* `tensor/` — tensor-network simulations, developed to extend the study to larger one-dimensional systems beyond the practical limits of exact diagonalization.

---

## 1. Physical problem

For a quantum system described by a Hamiltonian

$$
H_S,
$$

the thermal state at inverse temperature \(\beta = 1/(k_B T)\) is the Gibbs state

$$
\rho_\beta =
\frac{e^{-\beta H_S}}
{\mathrm{Tr}(e^{-\beta H_S})}.
$$

Preparing this state efficiently is a central problem in quantum simulation. For a generic many-body Hamiltonian, directly constructing the Gibbs state requires manipulating a Hilbert space whose dimension grows exponentially with the number of degrees of freedom.

The goal of this project is to numerically study a dissipative state-preparation protocol in which the system is repeatedly coupled to a small auxiliary bath. The bath is reset after each interaction, allowing the same ancilla degrees of freedom to be reused.

---

# 2. Lloyd–Abanin thermal-state preparation protocol

The implementation is based on:

> J. Lloyd and D. A. Abanin,
> **“Quantum Thermal State Preparation for Near-Term Quantum Processors,”**
> *Physical Review X* **16**, 031053 (2026).
> arXiv:2506.21318.

[Paper on arXiv](https://arxiv.org/abs/2506.21318) · [Physical Review X](https://doi.org/10.1103/cbrd-ssnm)

The protocol combines three main ingredients:

1. a set of resettable auxiliary bath qubits,
2. a time-dependent system–bath interaction,
3. a randomization step that suppresses unwanted coherences.

The system is not cooled by coupling it permanently to a large macroscopic reservoir. Instead, a small bath is repeatedly initialized, coupled to the system, and reset.

---

## 3. Repeated-interaction map

Let \(\rho_n\) denote the system density matrix after the \(n\)-th cycle.

Each cycle implements a quantum channel

$$
\rho_{n+1} = \mathcal{E}(\rho_n).
$$

A cycle consists schematically of

```text
       System
          │
          │
   ┌──────▼──────┐
   │ prepare bath│
   │     |0⟩     │
   └──────┬──────┘
          │
          ▼
   ┌─────────────────┐
   │ joint evolution │
   │  H_S + H_B +    │
   │  θ f(t) A ⊗ R   │
   └────────┬────────┘
            │
            ▼
      reset bath
            │
            ▼
        repeat
```

The auxiliary bath is initialized in a fixed state, evolved jointly with the system, and then reset before the next cycle.

This repeated reset is what allows a finite set of auxiliary qubits to mimic an effectively dissipative environment.

---

## 4. System–bath Hamiltonian

During the interaction stage, the total Hamiltonian has the form

$$
H(t)=
H_S + H_B + \theta V(t),
$$

where

$$
V(t)=
f(t)\sum_\mu A_\mu\otimes R_\mu.
$$

Here:

* \(H_S\) is the Hamiltonian of the physical system,
* \(H_B\) is the auxiliary-bath Hamiltonian,
* \(A_\mu\) are local operators acting on the system,
* \(R_\mu\) are operators acting on the bath,
* \(\theta\) controls the system–bath coupling strength,
* \(f(t)\) is a time-dependent modulation/filter function.

A key ingredient of the protocol is therefore that the coupling is **not simply switched on at a constant strength**. Its time dependence is engineered to favor the energy-changing processes required for thermalization.

The protocol only requires local system–bath couplings and does not require the coupling operators themselves to change with time.

---

## 5. Engineered filtering and detailed balance

The time-dependent modulation \(f(t)\) acts as a frequency filter.

The central idea is to make transitions involving different system energy differences occur with approximately the relative weights required by thermal equilibrium.

In the weak-coupling regime, the resulting quantum channel approximately satisfies quantum detailed-balance relations. Consequently, its fixed point approaches the Gibbs state

$$
\rho_\beta \propto e^{-\beta H_S}.
$$

The Lloyd–Abanin analysis predicts that, for sufficiently weak system–bath coupling,

$$
\|\hat{\rho}-\rho_\beta\|_1=
O(\theta^2),
$$

where \(\hat{\rho}\) is the steady state of the implemented protocol.

This perturbative scaling is one of the central numerical signatures studied in this project.

The protocol parameters include the inverse temperature \(\beta\), coupling strength \(\theta\), bath energy scale \(h\), Trotter step \(\delta\), reset time \(T\), and randomization parameter \(\lambda\).

---

## 6. Randomization

In addition to the dissipative system–bath interaction, the protocol includes a short randomization step.

This step consists of a short Hamiltonian evolution for a random time and is designed to suppress unwanted coherences in the energy eigenbasis of \(H_S\).

The randomization does not change the overall resource scaling of the protocol, but helps avoid coherent effects and resonances that can prevent accurate thermalization.

---

# 7. Exact numerical approach

The `exact/` directory contains the original finite-size implementation of the protocol.

For sufficiently small systems, the full system density matrix can be represented explicitly. This makes it possible to simulate the protocol without tensor-network approximations and to calculate the Gibbs state directly from

$$
\rho_\beta =
\frac{e^{-\beta H_S}}
{\mathrm{Tr}(e^{-\beta H_S})}.
$$

The exact simulations therefore provide two important things:

1. a direct implementation of the thermalization protocol;
2. a reference against which the tensor-network calculations can be validated.

The exact implementation was applied to several Hamiltonian models, including:

* single-spin systems,
* non-interacting systems,
* one-dimensional Ising systems,
* Fermi–Hubbard systems.

For each model, the protocol can be compared against the exact Gibbs state through quantities such as energy, magnetization, correlation functions and state-level distances where feasible.

---

## 8. Why an exact approach is not sufficient

For \(N\) qubits, the Hilbert-space dimension is

$$
d = 2^N.
$$

A density matrix therefore contains \(4^N\) complex matrix elements.

Consequently, explicitly storing and evolving the density matrix rapidly becomes impractical as the system size increases.

This motivates the second part of the project:

```text
exact simulations
       │
       │ validate protocol
       ▼
tensor-network simulations
       │
       │ exploit low-entanglement structure
       ▼
larger one-dimensional systems
```

---

# 9. Tensor-network approach

The `tensor/` directory extends the study using tensor-network representations.

The main physical model studied in this part is the one-dimensional mixed-field Ising model, for which the protocol is simulated at system sizes beyond the practical range of exact density-matrix calculations.

Tensor networks replace exponentially large many-body objects by networks of lower-dimensional tensors. Their computational efficiency is controlled primarily by the bond dimension.

The tensor-network calculations therefore make it possible to study not only whether the protocol thermalizes, but also **how the computational cost scales with system size and entanglement**.

---

# 10. MPS quantum trajectories

The main tensor-network method used in this project is a **quantum-trajectory representation using Matrix Product States (MPS)**.

Instead of evolving the full density matrix of the system and bath, individual pure-state trajectories are evolved as MPS.

Schematically,

```text
              Exact density matrix
                     ρ
                     │
             exponentially large
                     │
                     ▼
              ┌─────────────┐
              │             │
              │  trajectories│
              │             │
              └──────┬──────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      |ψ₁⟩         |ψ₂⟩         |ψ₃⟩       ...
        │            │            │
       MPS          MPS          MPS
        │            │            │
        └────────────┼────────────┘
                     ▼
             ensemble averages
                     │
                     ▼
                 observables
```

For each trajectory:

1. the system and auxiliary bath are initialized;
2. they undergo the joint system–bath evolution;
3. the bath is measured/reset;
4. the resulting pure state is compressed back into an MPS;
5. the process is repeated for many cycles.

Physical observables are obtained by averaging over independent trajectories.

This avoids explicitly storing a \(2^N\times2^N\) density matrix and instead exploits the low-entanglement structure of the evolving wavefunctions.

---

# 11. Why MPS is useful here

The main advantage is that the computational cost is governed by the MPS bond dimension rather than directly by the full Hilbert-space dimension.

For an MPS,

```text
physical dimension:        d
bond dimension:             χ
system size:                N
```

the storage and contraction costs can remain manageable when the required \(\chi\) stays moderate.

The bond dimension is therefore an important diagnostic in this project.

If the bond dimension remains small or saturates, the simulation remains computationally tractable.

If it grows rapidly, the tensor-network representation becomes increasingly expensive.

The simulations therefore track quantities such as:

* maximum bond dimension,
* bond-dimension growth with cycle,
* dependence on truncation parameters,
* dependence on system size,
* computational time.

---

# 12. Tensor-network reference state

The tensor-network implementation also constructs a Gibbs-state reference using a thermal purification / MPS representation.

This provides the target thermal state against which the larger-system simulations are compared.

Thus, for the N=20 calculations, the conceptual structure is

```text
             Gibbs reference
             thermal MPS
                  │
                  │
                  ▼
        ┌──────────────────┐
        │ Lloyd–Abanin     │
        │ protocol         │
        │                  │
        │ MPS trajectories │
        └────────┬─────────┘
                 │
                 ▼
          prepared state
                 │
                 ▼
       observables / errors
```

The reference calculation is not itself a simulation of the dissipative protocol; it provides the target thermal state.

---

# 13. Additional tensor-network formulations

Several tensor-network representations were explored during the project.

### Effective system-only channel

An effective channel can be constructed directly on the system degrees of freedom.

This representation is useful for rapidly screening protocol parameters such as:

* \(\theta\),
* reset time,
* bath parameters,
* randomization,
* coupling operators.

It is computationally cheaper than explicitly evolving the full system–bath state, but it is used primarily as an auxiliary numerical method rather than as the main large-scale simulation.

### Composite density MPO

A second formulation explicitly retains the system and bath while representing the density operator as an MPO/MPDO-like object.

This provides a deterministic description of the averaged channel and is useful for validating the trajectory formulation on smaller systems.

However, density-operator tensor networks generally carry a larger tensor-network cost than pure-state MPS trajectories.

---

# 14. Numerical questions studied with tensor networks

The tensor-network simulations are designed to answer several questions that cannot be addressed efficiently with exact diagonalization alone.

### Thermalization

How quickly do observables converge towards their Gibbs-state values?

### Coupling-strength scaling

Does the final thermalization error decrease approximately as

$$
O(\theta^2)
$$

in the weak-coupling regime predicted by Lloyd and Abanin?

### Bond-dimension growth

How does the tensor-network complexity evolve during the protocol?

### Trajectory convergence

How does the statistical error decrease as the number of trajectories is increased?

### System-size scaling

How do thermalization accuracy, bond dimension and computational cost change with \(N\)?

### Local thermalization

Can reduced density matrices of subsystems approach the corresponding reduced Gibbs states even when the full many-body density matrix is inaccessible?

---

# 15. Relationship between the two approaches

The exact and tensor-network parts of the repository are complementary rather than independent projects.

```text
                         Quantum thermal
                         state preparation
                                │
                                ▼
                    Lloyd–Abanin protocol
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
              EXACT                         TENSOR
                 │                             │
        small systems                    larger 1D systems
                 │                             │
        full density matrix              MPS / MPO
                 │                             │
                 │                     reduced representation
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                    comparison with Gibbs state
```

The exact calculations provide a controlled reference for small systems.

The tensor-network calculations then extend the same physical protocol to larger one-dimensional systems while monitoring the additional numerical approximations introduced by tensor-network truncation and Monte Carlo sampling.

---

# 16. Numerical errors and convergence

The tensor-network results involve several distinct sources of error:

* finite system–bath coupling \(\theta\),
* incomplete convergence with the number of protocol cycles,
* Trotter discretization,
* MPS truncation,
* finite bond dimension,
* finite number of trajectories.

The numerical analysis therefore separates physical and numerical convergence whenever possible.

In particular, the project studies convergence with respect to:

* \(\theta\),
* number of cycles,
* MPS bond dimension / truncation cutoff,
* number of trajectories,
* system size.

This distinction is important because agreement with the Gibbs state can otherwise be limited by either the physical approximation of the protocol or by the numerical approximation used to simulate it.

---

# 17. Repository structure

```text
quantum-thermal-state-preparation-simulations/
│
├── exact/
│   ├── single_spin/
│   ├── noninteracting/
│   ├── ising/
│   └── fermi_hubbard/
│
├── tensor/
│   ├── julia/
│   ├── analysis/
│   ├── results/
│   ├── figures/
│   └── validation/
│
├── README.md
└── ...
```

### `exact/`

Original finite-size simulations based on explicit state-vector and density-matrix representations.

### `tensor/`

Tensor-network extension of the project, with emphasis on MPS quantum trajectories for larger one-dimensional systems.

### `tensor/julia/`

Julia implementations of the tensor-network simulations and thermal-reference calculations.

### `tensor/analysis/`

Python scripts used to process simulation outputs and generate the final figures.

### `tensor/results/`

Selected numerical results required to reproduce the reported plots without storing the full collection of intermediate simulation data.

### `tensor/figures/`

Final figures illustrating thermalization, coupling-strength scaling, bond-dimension behaviour and system-size scaling.

---

# 18. Main software

The simulations were developed using:

* Python
* Julia
* ITensors / ITensorMPS
* NumPy
* SciPy
* Matplotlib
* Pandas

The exact simulations additionally use standard numerical linear-algebra tools for finite-size Hamiltonian and density-matrix calculations.

---

# 19. Thesis

This repository contains the numerical implementation developed for my Master's Thesis in Quantum Computing at the

**Universidad Autónoma de Madrid**
**Instituto de Física Teórica (UAM-CSIC)**

The thesis focuses on quantum thermal-state preparation using engineered dissipation and repeated system–bath interactions, with tensor-network simulations used to investigate larger one-dimensional systems.

---

# 20. References

### Main protocol

J. Lloyd and D. A. Abanin,
**Quantum Thermal State Preparation for Near-Term Quantum Processors**,
*Physical Review X* **16**, 031053 (2026).
[arXiv:2506.21318](https://arxiv.org/abs/2506.21318)
[DOI: 10.1103/cbrd-ssnm](https://doi.org/10.1103/cbrd-ssnm)

### Tensor networks

G. Vidal,
**Efficient Simulation of One-Dimensional Quantum Many-Body Systems**,
*Physical Review Letters* **93**, 040502 (2004).

F. Verstraete, M. M. Wolf and J. I. Cirac,
**Quantum computation and quantum-state engineering driven by dissipation**,
*Nature Physics* **5**, 633–636 (2009).

F. Verstraete, J. J. García-Ripoll and J. I. Cirac,
**Matrix Product Density Operators: Simulation of Finite-Temperature and Dissipative Systems**,
*Physical Review Letters* **93**, 207204 (2004).

U. Schollwöck,
**The density-matrix renormalization group in the age of matrix product states**,
*Annals of Physics* **326**, 96–192 (2011).

---

## 21. Status

This repository contains the original TFM implementation together with the tensor-network extension developed for larger one-dimensional simulations.

The `exact/` and `tensor/` directories should be viewed as two levels of the same numerical study:

> **exact finite-size validation → tensor-network scaling to larger systems.**
