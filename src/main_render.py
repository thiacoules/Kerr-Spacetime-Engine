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
    dt = 0.25 # Precision step
    
    # 1. INITIAL MOMENTUM (The 'Aura' Fix)
    # We fire p_r = -1.0 to go toward the hole.
    # We use x and y pixels as SMALL angular corrections.
    init_state = jnp.array([
        0.0,            # t
        R_camera,       # r
        jnp.pi/2.05,    # theta (tilted view)
        0.0,            # phi
        -1.0,           # p_t (Energy)
        -1.0,           # p_r (Inward)
        y_pixel * 3.5,  # p_theta (VERTICAL TILT - lowered from 10.0)
        x_pixel * 3.5   # p_phi (HORIZONTAL TILT - lowered from 10.0)
    ])

    def step_fn(carry, _):
        state, color, active = carry
        new_state = geodesic_step(state, dt, a)
        
        r, theta, old_theta = new_state[1], new_state[2], state[2]

        # 1. SHADOW LOGIC
        # If r gets too close to the horizon, it's black forever.
        horizon = M + jnp.sqrt(M**2 - a**2)
        is_swallowed = r < (horizon + 0.1)
        
        # 2. DISK LOGIC (The 'Rings')
        # Crossing the equatorial plane
        crossed_plane = (old_theta - jnp.pi/2) * (theta - jnp.pi/2) < 0
        # The disk is a ring from r=6 to r=15
        is_in_disk = (r > 6.0) & (r < 15.0) & crossed_plane
        
        # 3. COLORING (The 'Interstellar' Palette)
        # Glow is brighter at the inner edge (r=6)
        brightness = jnp.where(is_in_disk, 1.0 / (r - 5.5), 0.0)
        new_color = jnp.where(active & is_in_disk, brightness, color)
        
        # If it hits the horizon, set color to -1.0 (Special flag for Black)
        final_color = jnp.where(is_swallowed & active, -1.0, new_color)
        
        still_active = active & (~is_swallowed) & (~is_in_disk) & (r < 100.0)
        
        return (new_state, final_color, still_active), None

    (final_state, final_color, _), _ = lax.scan(step_fn, (init_state, 0.0, True), jnp.arange(1500))
    return final_color

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
