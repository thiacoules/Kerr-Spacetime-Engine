import jax.numpy as jnp
from jax import vmap, jit
import matplotlib.pyplot as plt
from src.core.solver import geodesic_step

# 1. Setup the Camera
RESOLUTION = 100  # Start small (100x100) to test speed
view_grid = jnp.linspace(-0.3, 0.3, RESOLUTION)
X, Y = jnp.meshgrid(view_grid, view_grid)

@jit
def trace_photon(x_pixel, y_pixel):
    """
    Traces a single pixel's path. 
    Returns 0 if it falls in, 1 if it escapes.
    """
    a = 0.9  # Black Hole Spin
    # Initial State: [t, r, theta, phi, pt, pr, ptheta, pphi]
    # We place the camera at r=20
    state = jnp.array([0.0, 20.0, jnp.pi/2, 0.0, -1.0, -1.0, y_pixel, x_pixel])
    
    current_state = state
    for _ in range(500):
        current_state = geodesic_step(current_state, 0.1, a)
    
    # If r < 2.0 (Horizon), it's black.
    return jnp.where(current_state[1] < 2.1, 0.0, 1.0)

# 2. Vectorize the function for the whole grid
render_view = vmap(vmap(trace_photon, in_axes=(0, None)), in_axes=(None, 0))

print("🔭 Starting Render...")
image = render_view(view_grid, view_grid)
print("✅ Render Complete!")

# 3. Save the result
plt.imshow(image, cmap='magma')
plt.title("Kerr Black Hole Shadow (Raw Geodesics)")
plt.axis('off')
plt.savefig("black_hole_shadow.png")
plt.show()
