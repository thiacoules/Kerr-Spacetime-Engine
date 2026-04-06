import jax.numpy as jnp
from jax import jit, vmap

@jit
def generate_rays(resolution, distance, fov=60):
    """
    Creates the initial momentum vectors for photons hitting a camera sensor.
    resolution: (width, height)
    distance: How far the observer is from the black hole (e.g., 50.0)
    """
    x = jnp.linspace(-1, 1, resolution[0])
    y = jnp.linspace(-1, 1, resolution[1])
    X, Y = jnp.meshgrid(x, y)
    
    # Observer position in Boyer-Lindquist (t, r, theta, phi)
    # We place the observer on the equatorial plane
    pos = jnp.array([0.0, distance, jnp.pi/2, 0.0])
    
    # Initial momenta for each pixel
    # We assume the camera is pointing directly at the center (the black hole)
    # These p_values define the 'direction' of each ray
    p_r = -jnp.ones_like(X) # Moving toward the center
    p_theta = Y * (fov / 180.0)
    p_phi = X * (fov / 180.0)
    p_t = -1.0 # Energy of the photon
    
    return pos, jnp.stack([jnp.full_like(X, p_t), p_r, p_theta, p_phi], axis=-1)
