import jax.numpy as jnp
from jax import vmap, jit
import matplotlib.pyplot as plt
from src.core.solver import geodesic_step

# 1. Setup the Camera
RESOLUTION = 100  # Start small (100x100) to test speed
view_grid = jnp.linspace(-0.3, 0.3, RESOLUTION)
X, Y = jnp.meshgrid(view_grid, view_grid)

import jax.numpy as jnp
from jax import jit, vmap
import matplotlib.pyplot as plt
from src.core.solver import geodesic_step

# --- Black Hole Parameters ---
M = 1.0     # Mass
a = 0.99    # High spin for a massive shadow distortion
R_camera = 100.0 # Standard Interstellar view (far out)

# 1. Setup the Camera (A 'closer' look at the center)
RESOLUTION = 120 # Start 120x120 for test
view_grid = jnp.linspace(-0.35, 0.35, RESOLUTION) # Zoom in
X, Y = jnp.meshgrid(view_grid, view_grid)

@jit
def trace_photon(x_pixel, y_pixel):
    """
    Traces a photon. Colors it based on what it hits.
    0.0 = Black (Event Horizon)
    1.0+ = Orange (Accretion Disk)
    0.5 = Blue (Background Lensed Stars)
    """
    dt = 0.1 # Integration Step
    
    # State: [t, r, theta, phi, pt, pr, ptheta, pphi]
    # Place camera at theta = pi/2.1 (slightly above the plane for classic tilt)
    state = jnp.array([0.0, R_camera, jnp.pi/2.1, 0.0, -1.0, -1.0, y_pixel, x_pixel])
    
    current_state = state
    disk_color = 0.0 # Default value

    for _ in range(1500): # Longer trace for better resolution
        old_theta = current_state[2]
        current_state = geodesic_step(current_state, dt, a)
        new_theta = current_state[2]
        r = current_state[1]

        # --- EVENT HORIZON LOGIC ---
        if r < 1.1: # Kerr horizon is near M + sqrt(M^2 - a^2)
            return 0.0 # Pure black

        # --- ACCRETION DISK INTERSECTION LOGIC ---
        # Did the ray cross the equatorial plane (theta = pi/2)?
        # (This is like checking if 'z' changed from positive to negative)
        if (old_theta - jnp.pi/2.0) * (new_theta - jnp.pi/2.0) < 0:
            # Is it within the disk bounds (e.g., 6M to 15M)?
            if (r > 6.0) and (r < 15.0):
                # Apply a Redshift factor: Gas moving TOWARD you is brighter.
                # Use phi momentum (p_phi) and spin (a) for Doppler effect.
                p_phi = current_state[7]
                doppler = 1.0 + (a * r / p_phi) # Basic Doppler approximation
                # Map to orange (1.0) and increase brightness based on Doppler
                # Ensure it never gets TOO dim
                disk_color = jnp.maximum(0.2, (1.0 / doppler)**3) * 3.0 # Basic 'Glow'
                break # Once hit, don't continue that ray

    # --- COLOR MAP LOGIC ---
    # Convert 'disk_color' to a value that will pop in our colormap
    return disk_color

# 2. Vectorize and compiling the function
render_view = vmap(vmap(trace_photon, in_axes=(0, None)), in_axes=(None, 0))

print("🔭 Initializing Kerr-Flow GPU Kernels (JAX)...")
print("🔥 Starting production-grade ray-trace for Gargantua milestone...")
image = render_view(view_grid, view_grid)
print("✅ High-fidelity render complete!")

# 3. Save and Display with professional aesthetics
plt.figure(figsize=(10,10))
# Create a 'Custom' colormap for the classic orange/black Interstellar look
custom_cmap = plt.cm.get_cmap('magma').copy()
custom_cmap.set_under('black') # Make 0.0 truly black
# Apply custom color limits (vmin, vmax) to pop the disk and keep the background dark
plt.imshow(image, cmap=custom_cmap, vmin=0.1, vmax=5.0) 
plt.title("Gargantua Simulation (Kerr Spacetime Lensing)", color='white', size=16)
plt.axis('off')
plt.gcf().set_facecolor('black') # Make the entire window black
plt.savefig("gargantua_first_light.png", bbox_inches='tight', pad_inches=0)
plt.show()
