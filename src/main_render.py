import jax.numpy as jnp
from jax import jit, vmap, lax
import matplotlib.pyplot as plt
from src.core.solver import geodesic_step

# --- Black Hole Parameters ---
M = 1.0
a = 0.95    # Slightly lower spin makes the shadow easier to find initially
R_camera = 40.0 

# --- Resolution (Keep it at 200 for a crisp image) ---
RESOLUTION = 200
# We need a very specific grid range to 'frame' the black hole
view_grid = jnp.linspace(-0.2, 0.2, RESOLUTION)

def trace_photon(x_pixel, y_pixel):
    dt = 0.25
    
    # 1. THE PHYSICS FIX: p_r MUST be negative to move toward the hole
    # But x_pixel and y_pixel must be VERY SMALL to stay focused on the center.
    # At R=100, the black hole is tiny. We need a 'telescope' zoom.
    
    # Angle scale: 0.05 makes the FOV very narrow (High Zoom)
    scale = 0.06 
    
    p_t = -1.0
    p_r = -1.0
    p_theta = y_pixel * scale 
    p_phi = x_pixel * scale 
    
    # Position: Camera at R=100, theta slightly above equator
    init_state = jnp.array([0.0, 100.0, jnp.pi/2 - 0.05, 0.0, p_t, p_r, p_theta, p_phi])

    def step_fn(carry, _):
        state, color, active = carry
        
        # Adaptive step: Get slower as we get closer
        r_current = state[1]
        local_dt = jnp.where(r_current < 15.0, dt * 0.1, dt)
        
        new_state = geodesic_step(state, local_dt, a)
        
        r, theta, old_theta = new_state[1], new_state[2], state[2]

        # Horizon logic
        horizon = M + jnp.sqrt(M**2 - a**2)
        is_swallowed = r < (horizon + 0.05)
        
        # Disk logic: r between 6 and 25
        crossed_plane = (old_theta - jnp.pi/2) * (theta - jnp.pi/2) < 0
        is_in_disk = (r > 6.0) & (r < 25.0) & crossed_plane
        
        # Glow logic
        brightness = jnp.where(is_in_disk, 1.0 / jnp.sqrt(r), 0.0)
        
        # IMPORTANT: Color -1.0 is our 'Shadow' flag
        new_color = jnp.where(active & is_in_disk, brightness, color)
        final_color = jnp.where(active & is_swallowed, -1.0, new_color)
        
        still_active = active & (~is_swallowed) & (~is_in_disk) & (r < 150.0)
        
        return (new_state, final_color, still_active), None

    # 3000 steps to allow for the 'Slow Motion' near the hole
    (final_state, res_color, _), _ = lax.scan(step_fn, (init_state, 0.0, True), jnp.arange(3000))
    return res_color

# Vectorization
render_view = vmap(vmap(trace_photon, in_axes=(0, None)), in_axes=(None, 0))

print("🔭 Compiling Kerr-Geometry Kernels...")
image = render_view(view_grid, view_grid)
print("✅ Rendering Finished.")

# --- Professional Post-Processing ---
plt.figure(figsize=(10, 10), facecolor='black')
# Use 'inferno' or 'gist_heat' for that movie look
# We clip the shadow (values of -1) to be pure black
plt.imshow(image, cmap='hot', origin='lower', vmin=0.0, vmax=0.5)
plt.axis('off')
plt.title("Relativistic Ray-Trace: Gargantua v1.0", color='white', fontsize=18)
plt.savefig("gargantua_v1.png", bbox_inches='tight', pad_inches=0)
plt.show()
