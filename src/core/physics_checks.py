import jax.numpy as jnp
from jax import jit

@jit
def get_carter_constant(state, a, M=1.0):
    """
    Calculates the Carter Constant (Q). 
    In a perfect simulation, this value remains constant along the geodesic.
    """
    t, r, theta, phi, p_t, p_r, p_theta, p_phi = state
    
    # Carter's Constant formula for Kerr Metric
    # Q = p_theta^2 + cos^2(theta) * (a^2 * (1 - p_t^2) + p_phi^2 / sin^2(theta))
    term1 = p_theta**2
    term2 = (jnp.cos(theta)**2) * (a**2 * (1 - p_t**2) + (p_phi**2 / jnp.sin(theta)**2))
    return term1 + term2
