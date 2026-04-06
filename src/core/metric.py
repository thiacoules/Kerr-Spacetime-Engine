import jax.numpy as jnp
from jax import jit

@jit
def kerr_metric(r, theta, a, M=1.0):
    """
    Computes components of the Kerr metric in Boyer-Lindquist coordinates.
    a: Spin parameter (0 <= a < M)
    """
    sigma = r**2 + (a**2 * jnp.cos(theta)**2)
    delta = r**2 - (2 * M * r) + a**2
    
    # Components
    g_tt = -(1 - (2 * M * r) / sigma)
    g_rr = sigma / delta
    g_thth = sigma
    
    return {"g_tt": g_tt, "g_rr": g_rr, "sigma": sigma, "delta": delta}
