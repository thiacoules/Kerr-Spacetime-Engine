import jax.numpy as jnp
from jax import jit

@jit
def get_carter_constant(state, a, M=1.0):
    t, r, theta, phi, p_t, p_r, p_theta, p_phi = state
    
    # Carter Constant (Q) for the Kerr Metric
    # This is the 'hidden' constant of motion
    sigma = r**2 + a**2 * jnp.cos(theta)**2
    
    # Q = p_theta^2 + cos^2(theta) * (a^2 * (1 - p_t^2) + p_phi^2 / sin^2(theta))
    # For a photon, p_t is usually -1 (Energy = 1)
    q_val = p_theta**2 + jnp.cos(theta)**2 * (a**2 * (1.0 - p_t**2) + (p_phi**2 / jnp.sin(theta)**2))
    return q_val
