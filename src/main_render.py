import jax.numpy as jnp
from jax import jit, vmap, lax
import matplotlib.pyplot as plt
from src.core.solver import geodesic_step

# --- Black Hole Parameters ---
M = 1.0     # Mass
a = 0.99    # High spin (Gargantua style)
R_camera = 50.0 

# --- Wider Camera Setup ---
RESOLUTION = 200 # Higher resolution to see the structure
# We need a wider field of view to capture the whole disk
view_grid = jnp.linspace(-0.6, 0.6, RESOLUTION)

def trace_photon(x_pixel, y_pixel):
    dt = 0.4 # Slightly larger steps to bridge the gap from R=50
    
    # 1. Setup 'Aimed' Momentum
    # Instead of firing straight, we tilt the momentum based on pixel position
    # This creates a 'Wide Angle' lens effect
    p_t = -1.0
    p_r = -1.0 # Moving toward the hole
    p_theta = y_pixel * 1.5 # Vertical 'tilt'
    p_phi = x_pixel * 1.5   # Horizontal 'tilt'
    
    init_state = jnp.array([0.0, R_camera, jnp.pi/2.05, 0.0, p_t, p_r, p_theta, p_phi])

    def step_fn(carry, _):
        state, current_color, active = carry
        
        # RK4 Step (Make sure your solver.py is using the RK4 we wrote!)
        new_state = geodesic_step(state, dt, a)
        
        r = new_state[1]
        theta = new_state[2]
        old_theta = state[2]

        # Horizon Check (Approx 2.0 for a=0.99)
        is_swallowed = r < 2.1
        
        # Disk Check (Standard 6M to 20M)
        crossed_plane = (old_theta - jnp.pi/2) * (theta - jnp.pi/2) < 0
        is_in_disk_zone = (r > 6.0) & (r < 20.0) & crossed_plane
        
        # Glow logic: The closer to the hole, the hotter the disk
        glow = 5.0 / jnp.sqrt(r)
        new_color = jnp.where(is_in_disk_zone & active, glow, current_color)
        
        # Stop if hit disk, horizon, or escaped to r=200
        still_active = active & (~is_swallowed) & (~is_in_disk_zone) & (r < 200.0)
        
        return (new_state, new_color, still_active), None

    # 1500 steps is plenty for R=50 to R=2
    (final_state, final_color, _), _ = lax.scan(step_fn, (init_state, 0.0, True), jnp.arange(1500))
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
