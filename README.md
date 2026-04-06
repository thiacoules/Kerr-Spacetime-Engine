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
- [ ] Implement Boyer-Lindquist Metric Tensor
- [ ] Hamiltonian Geodesic Integrator (RK4)
- [ ] Differentiable Ray-Tracing Engine
- [ ] Accretion Disk Volumetric Rendering

## 🚀 Getting Started
`pip install jax jaxlib numpy matplotlib`
