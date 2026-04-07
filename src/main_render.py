import jax.numpy as jnp
from jax import jit, vmap, lax
import matplotlib.pyplot as plt
from src.core.solver import geodesic_step

# --- Black Hole Parameters ---
M = 1.0
a = 0.99 # Let's go full Interstellar spin!
R_camera = 20.0 # Moved closer to get a better view without losing accuracy

# --- Resolution ---
RESOLUTION = 300 
# The screen coordinates (Impact Parameters). -15 to 15 beautifully frames the hole.
view_grid = jnp.linspace(-15.0, 15.0, RESOLUTION)

@jit
def trace_photon(x_pixel, y_pixel):
    dt = 0.1 # Small, smooth steps
    
    r0 = R_camera
    theta0 = jnp.pi/2 - 0.15 # 8.5 degree tilt to see "over" the disk
    
    # 1. CAMERA SCREEN SETUP
    p_t = -1.0
    p_theta = y_pixel
    p_phi = x_pixel
    
    # 2. THE PHYSICS FIX: Enforcing H=0 (Null Geodesic)
    # We must calculate p_r exactly based on the inverse metric
    sigma = r0**2 + a**2 * jnp.cos(theta0)**2
    delta = r0**2 - 2*M*r0 + a**2
    
    g_inv_tt = -( (r0**2 + a**2)**2 - delta * a**2 * jnp.sin(theta0)**2 ) / (sigma * delta)
    g_inv_tphi = -( 2 * M * r0 * a ) / (sigma * delta)
    g_inv_rr = delta / sigma
    g_inv_thth = 1 / sigma
    g_inv_phiphi = (delta - a**2 * jnp.sin(theta0)**2) / (sigma * delta * jnp.sin(theta0)**2)
    
    # Solve for p_r to ensure the photon acts like light
    C = g_inv_tt * p_t**2 + 2 * g_inv_tphi * p_t * p_phi + g_inv_thth * p_theta**2 + g_inv_phiphi * p_phi**2
    # Use jnp.maximum to prevent math errors during compilation
    p_r = -jnp.sqrt(jnp.maximum(-C / g_inv_rr, 0.0001))
    
    init_state = jnp.array([0.0, r0, theta0, 0.0, p_t, p_r, p_theta, p_phi])

    def step_fn(carry, _):
        state, color, active = carry
        
        new_state = geodesic_step(state, dt, a)
        r, theta = new_state[1], new_state[2]

        # 1. SHADOW: Event Horizon Check
        horizon = M + jnp.sqrt(M**2 - a**2)
        is_swallowed = r < (horizon + 0.1)
        
        # 2. DISK: Volumetric "Thick" Disk (r between 4 and 15)
        # Instead of crossing a plane, we check if it's "inside" the glowing gas
        is_in_disk = (r > 4.0) & (r < 15.0) & (jnp.abs(jnp.cos(theta)) < 0.05)
        
        # 3. GLOW: Brighter near the center
        brightness = jnp.where(is_in_disk, 1.5 / jnp.sqrt(r), 0.0)
        
        # Apply color if we hit the disk
        new_color = jnp.where(active & is_in_disk, brightness, color)
        
        # Turn it pure black if it falls into the black hole
        final_color = jnp.where(active & is_swallowed, -1.0, new_color)
        
        # Stop tracing if swallowed, hit the disk, or escaped to the background
        still_active = active & (~is_swallowed) & (~is_in_disk) & (r < R_camera + 5.0)
        
        return (new_state, final_color, still_active), None

    # Trace for 1000 steps
    (final_state, res_color, _), _ = lax.scan(step_fn, (init_state, 0.0, True), jnp.arange(1000))
    return res_color

# Vectorization (Pushing all pixels to the GPU/CPU at once)
render_view = vmap(vmap(trace_photon, in_axes=(0, None)), in_axes=(None, 0))

print("🔭 Calibrating H=0 Null Geodesics...")
print("🔥 Rendering Gargantua Matrix...")
image = render_view(view_grid, view_grid)
print("✅ Rendering Finished.")

# --- Professional Post-Processing ---
plt.figure(figsize=(10, 10), facecolor='black')
# We use .T (Transpose) to ensure X and Y axes are correct
# vmin=0.0 ensures both the background (0) and shadow (-1) are pitch black
plt.imshow(image.T, cmap='inferno', origin='lower', vmin=0.0, vmax=0.7)
plt.axis('off')
plt.title("Gargantua: H=0 Null Geodesic Engine", color='white', fontsize=16)
plt.savefig("gargantua_master.png", bbox_inches='tight', pad_inches=0, dpi=150)
plt.show()
