import jax.numpy as jnp
from jax import jit, vmap, lax
import matplotlib.pyplot as plt
from src.core.solver import geodesic_step

# --- Black Hole Parameters ---
M = 1.0     # Mass
a = 0.99    # High spin (Gargantua style)
R_camera = 50.0 

# --- Resolution ---
RESOLUTION = 150 
view_grid = jnp.linspace(-0.25, 0.25, RESOLUTION)

def trace_photon(x_pixel, y_pixel):
    dt = 0.2
    # Initial State: [t, r, theta, phi, pt, pr, ptheta, pphi]
    # Tilt the camera slightly (pi/2.05) to see the disk's top/bottom lensing
    init_state = jnp.array([0.0, R_camera, jnp.pi/2.05, 0.0, -1.0, -1.0, y_pixel, x_pixel])

    # We use lax.scan instead of a Python for-loop
    # 'carry' holds the variables that change every step
    def step_fn(carry, _):
        state, current_color, active = carry
        
        # Calculate the next position
        new_state = geodesic_step(state, dt, a)
        
        r = new_state[1]
        theta = new_state[2]
        old_theta = state[2]

        # 1. Logic: Did we hit the Event Horizon?
        # horizon r_plus = M + sqrt(M^2 - a^2)
        horizon = M + jnp.sqrt(M**2 - a**2)
        is_swallowed = r < (horizon + 0.05)
        
        # 2. Logic: Did we cross the Accretion Disk?
        # Crossing theta = pi/2 means (old_theta - pi/2) and (new_theta - pi/2) have different signs
        crossed_plane = (old_theta - jnp.pi/2) * (theta - jnp.pi/2) < 0
        is_in_disk_zone = (r > 6.0) & (r < 15.0) & crossed_plane
        
        # 3. Coloring: 
        # If we hit the disk and are still 'active', calculate brightness based on r
        # This creates a glow that fades as it gets further out
        hit_brightness = 4.0 / (r / 5.0)
        new_color = jnp.where(is_in_disk_zone & active, hit_brightness, current_color)
        
        # Stop updating if we hit the disk or the horizon
        still_active = active & (~is_swallowed) & (~is_in_disk_zone)
        
        return (new_state, new_color, still_active), None

    # Run the loop for 1200 steps
    (final_state, final_color, _), _ = lax.scan(step_fn, (init_state, 0.0, True), jnp.arange(1200))
    
    return final_color

# Vectorize across the grid
render_view = vmap(vmap(trace_photon, in_axes=(0, None)), in_axes=(None, 0))

print("🔭 JAX kernels compiled. Starting Relativistic Ray-Trace...")
image = render_view(view_grid, view_grid)
print("✅ High-fidelity render complete!")

# --- Visual Styling ---
plt.figure(figsize=(10, 10), facecolor='black')
# 'afmhot' or 'magma' gives that deep interstellar orange
plt.imshow(image, cmap='afmhot', origin='lower', vmin=0.0, vmax=2.5)
plt.title("Kerr Spacetime Lensing (Gargantua Milestone)", color='white', size=15)
plt.axis('off')
plt.savefig("gargantua_render.png", bbox_inches='tight', pad_inches=0)
plt.show()
