import jax.numpy as jnp
from jax import vmap, jit
from src.core.solver import geodesic_step
from src.optics.camera import generate_rays
from src.optics.disk import check_disk_collision

@jit
def trace_batch(initial_state, a, steps=500, dt=0.1):
    """
    Traces a batch of photons through Kerr spacetime.
    Returns the final coordinates and a 'hit' mask.
    """
    def body_fn(carry, _):
        state, active_mask = carry
        new_state = geodesic_step(state, dt, a)
        
        # Logic: Stop if r < Event Horizon (approx 2M) or r > Escape Distance
        r = new_state[1]
        still_active = (r > 2.05) & (r < 100.0) & active_mask
        
        return (new_state, still_active), None

    # Use JAX scan for a fast, compiled loop
    (final_state, _), _ = jax.lax.scan(body_fn, (initial_state, True), jnp.arange(steps))
    return final_state

# The "Magic" function: Vectorize the tracer over millions of pixels
render_pixels = vmap(trace_batch, in_axes=(0, None))
