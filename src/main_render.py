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
    dt = 0.2
    # Place camera far away (R=100) and slightly above the plane
    # We use a very small tilt (pi/2 - 0.05) to get the Interstellar 'look'
    init_state = jnp.array([
        0.0, 100.0, jnp.pi/2 - 0.05, 0.0, 
        -1.0, -1.0, y_pixel * 15.0, x_pixel * 15.0
    ])

    def step_fn(carry, _):
        state, color, active = carry
        
        # --- THE FIX: Adaptive Step ---
        # As r gets smaller, gravity gets stronger. We slow down dt to keep it smooth.
        r_current = state[1]
        local_dt = jnp.where(r_current < 10.0, dt * 0.2, dt)
        
        new_state = geodesic_step(state, local_dt, a)
        
        r, theta, old_theta = new_state[1], new_state[2], state[2]

        # 1. EVENT HORIZON (The Shadow)
        horizon = M + jnp.sqrt(M**2 - a**2)
        is_swallowed = r < (horizon + 0.02)
        
        # 2. ACCRETION DISK (The Rings)
        crossed_plane = (old_theta - jnp.pi/2) * (theta - jnp.pi/2) < 0
        # Gargantua's disk is vast: 6M to 30M
        is_in_disk = (r > 6.0) & (r < 30.0) & crossed_plane
        
        # Doppler Shift + Distance Falloff
        # (r**-1.5 is the standard brightness falloff for accretion disks)
        brightness = jnp.where(is_in_disk, 10.0 * (r**-1.5), 0.0)
        
        # Update color only if active
        new_color = jnp.where(active & is_in_disk, brightness, color)
        
        # Set to -1 (Black) if swallowed
        final_color = jnp.where(is_swallowed & active, -1.0, new_color)
        
        # Kill ray if hit disk, horizon, or escaped
        still_active = active & (~is_swallowed) & (~is_in_disk) & (r < 150.0) & (r > horizon)
        
        return (new_state, final_color, still_active), None

    # We need MORE steps (2500) because the camera is now at R=100
    (final_state, final_color, _), _ = lax.scan(step_fn, (init_state, 0.0, True), jnp.arange(2500))
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
