import jax.numpy as jnp
from jax import jit

@jit
def check_disk_collision(old_pos, new_pos, inner_edge=6.0, outer_edge=20.0):
    """
    Detects if a photon crossed the equatorial plane (theta = pi/2) 
    within the bounds of the accretion disk.
    """
    # Check if the ray crossed theta = pi/2 (90 degrees)
    # This happens if (theta - pi/2) changes sign between steps
    crossed_plane = (old_pos[2] - jnp.pi/2) * (new_pos[2] - jnp.pi/2) < 0
    
    r = new_pos[1]
    is_within_bounds = (r > inner_edge) & (r < outer_edge)
    
    return crossed_plane & is_within_bounds

@jit
def get_redshift(r, a, M=1.0):
    """
    Calculates the Gravitational + Doppler redshift.
    This makes the disk look 'warped' in color.
    """
    # Simplified redshift factor for the 'famous' look
    # Real physics involves the four-velocity of the disk gas
    z_grav = 1.0 / jnp.sqrt(1 - 3*M/r) # Gravitational redshift
    return z_grav
