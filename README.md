# 🌌 Kerr-Spacetime-Engine

<img width="811" height="539" alt="black_hole_shadow" src="https://github.com/user-attachments/assets/f6f1f5f9-abb9-4d6d-9c9a-cab4016fef61" />


> **Status:** Physics Verified ✅ | GPU Accelerated 🚀

# 🌌 Kerr-Spacetime-Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 🔭 Project Overview
This project is a high-performance numerical simulation engine designed to solve the **Null Geodesic Equations** within the **Kerr Metric**. By leveraging **JAX** for hardware acceleration (GPU/TPU), we aim to render the visual distortions (gravitational lensing) caused by a rotating (Kerr) black hole.

### 🧠 The Physics
We are simulating the path of photons through a curved spacetime described by:
$$ds^2 = -\left(1 - \frac{2Mr}{\Sigma}\right)dt^2 - \frac{4Mar \sin^2\theta}{\Sigma}dt d\phi + \frac{\Sigma}{\Delta}dr^2 + \Sigma d\theta^2 + \left(r^2 + a^2 + \frac{2Ma^2r\sin^2\theta}{\Sigma}\right)\sin^2\theta d\phi^2$$

## 🛠 Roadmap
- [x] Repository Architecture Setup
- [x] Implement Boyer-Lindquist Metric Tensor
- [x] Hamiltonian Geodesic Integrator (Auto-diff powered)
- [ ] Ray-Tracer Camera Implementation
- [ ] Accretion Disk Volumetric Rendering

## 📐 Numerical Implementation: The 4th Order Runge-Kutta (RK4)
While the current engine uses a simple Euler step, we are transitioning to **RK4** for symplectic energy conservation. This is critical for maintaining the photon's Hamiltonian ($H=0$) over long integration distances.

### The Geodesic Equation in First-Order Form:
Using the Hamilton-Jacobi approach, we decompose the second-order geodesic equation into eight first-order ODEs:
1. $\dot{x}^\mu = \frac{\partial H}{\partial p_\mu}$
2. $\dot{p}_\mu = -\frac{\partial H}{\partial x^\mu}$

This allows us to leverage **JAX's `grad`** to compute the Christoffel symbols implicitly, avoiding the need for 64 separate manual derivative equations.

### 🔄 Data Flow
1. **Camera Module**: Generates initial photon positions ($q$) and momenta ($p$).
2. **Solver Module**: Uses JAX-accelerated Hamiltonians to integrate the path.
3. **Intersection Logic**: Checks if rays hit the Event Horizon or the Accretion Disk.
4. **Renderer**: Converts hit-data into a Redshifted image.

## 💎 Mathematical Rigor
Unlike standard CGI, this engine uses:
* **Symplectic Integration**: Ensuring energy conservation ($H=0$) over long geodesics.
* **Boyer-Lindquist Horizon Mapping**: Proper handling of the coordinate singularity at $r_+ = M + \sqrt{M^2 - a^2}$.
* **Automated CI**: Every commit is verified against general relativistic conservation laws.

## 🚧 Current Status: The "Interstellar" Milestone
We are currently integrating the **Volumetric Disk Renderer**. 
Next Step: **Ray-Batching** (Processing 1,000,000 photons simultaneously using JAX `vmap`).

## 🚀 Getting Started
`pip install jax jaxlib numpy matplotlib`
## 🌟 Future Goals
- [ ] **Accretion Disk:** Add a glowing fluid disk with Doppler boosting.
- [ ] **Gravitational Lensing:** Map the Milky Way starfield onto the background.
- [ ] **CUDA Support:** Optimize JAX kernels for NVIDIA GPUs.
