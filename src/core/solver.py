import jax.numpy as jnp
from jax import grad, jit

@jit
def hamiltonian(phase_space, a, M=1.0):
    """
    The Hamiltonian for a photon in Kerr spacetime.
    phase_space: [t, r, theta, phi, p_t, p_r, p_theta, p_phi]
    For photons, H must always equal 0.
    """
    q = phase_space[:4] # Coordinates
    p = phase_space[4:] # Momenta
    
    r, theta = q[1], q[2]
    
    # Calculate metric helpers
    sigma = r**2 + a**2 * jnp.cos(theta)**2
    delta = r**2 - 2*M*r + a**2
    
    # Inverse Metric Components (Required for Hamiltonian)
    # These equations describe how the 'momentum' translates to motion
    g_inv_tt = -( (r**2 + a**2)**2 - delta * a**2 * jnp.sin(theta)**2 ) / (sigma * delta)
    g_inv_tphi = -( 2 * M * r * a ) / (sigma * delta)
    g_inv_rr = delta / sigma
    g_inv_thth = 1 / sigma
    g_inv_phiphi = (delta - a**2 * jnp.sin(theta)**2) / (sigma * delta * jnp.sin(theta)**2)

    # H = 0.5 * g^uv * p_u * p_v
    h = 0.5 * (g_inv_tt * p[0]**2 + 
               2 * g_inv_tphi * p[0] * p[3] + 
               g_inv_rr * p[1]**2 + 
               g_inv_thth * p[2]**2 + 
               g_inv_phiphi * p[3]**2)
    return h

@jit
def geodesic_step(phase_space, dt, a):
    """
    RK4 Integration: Takes 4 smaller 'test steps' to find the best path.
    """
    def get_derivs(s):
        grads = grad(hamiltonian, argnums=0)(s, a).ravel()
        return jnp.concatenate([grads[4:], -grads[:4]])

    # RK4 Formula
    k1 = get_derivs(phase_space)
    k2 = get_derivs(phase_space + 0.5 * dt * k1)
    k3 = get_derivs(phase_space + 0.5 * dt * k2)
    k4 = get_derivs(phase_space + dt * k3)

    return phase_space + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
