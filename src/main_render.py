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
    dt = 0.5 # Larger steps for faster travel from R=50
    # The 'p_r = -1.0' fires it inward. 
    # The 'y_pixel' and 'x_pixel' act as the angle of the lens.
    init_state = jnp.array([
        0.0,            # t
        R_camera,       # r (50.0)
        jnp.pi/2.05,    # theta (slightly tilted)
        0.0,            # phi
        -1.0,           # p_t (Energy)
        -1.0,           # p_r (Initial velocity INWARD)
        y_pixel * 0.1,  # p_theta (Small angular tilt)
        x_pixel * 0.1   # p_phi (Small angular tilt)
    ])

    def step_fn(carry, _):
        state, current_color, active = carry
        new_state = geodesic_step(state, dt, a)
        
        r = new_state[1]
        theta = new_state[2]
        old_theta = state[2]

        # Horizon Check
        horizon = M + jnp.sqrt(M**2 - a**2)
        is_swallowed = r < (horizon + 0.1)
        
        # Disk Check (Widened to 4.0 - 25.0 for more visibility)
        crossed_plane = (old_theta - jnp.pi/2) * (theta - jnp.pi/2) < 0
        is_in_disk_zone = (r > 4.0) & (r < 25.0) & crossed_plane
        
        # Glow logic: Brightness based on proximity
        hit_brightness = 2.0 / jnp.sqrt(r) 
        new_color = jnp.where(is_in_disk_zone & active, hit_brightness, current_color)
        
        still_active = active & (~is_swallowed) & (~is_in_disk_zone)
        # If photon goes too far away (escapes), stop it
        still_active = still_active & (r < 150.0)
        
        return (new_state, new_color, still_active), None

    # Increase steps to 2000 to ensure light reaches the disk from far away
    (final_state, final_color, _), _ = lax.scan(step_fn, (init_state, 0.0, True), jnp.arange(2000))
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
