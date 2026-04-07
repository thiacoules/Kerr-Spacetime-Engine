import jax.numpy as jnp
from jax import jit, vmap, lax
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from src.core.solver import geodesic_step

# --- Black Hole Parameters ---
M = 1.0
a = 0.99 
R_camera = 25.0 # Perfect cinematic distance
RESOLUTION = 400 

# --- FIXED CAMERA LENS ---
# -1.2 to 1.2 removes the "amoeba" distortion and acts like a 50mm camera lens
view_grid = jnp.linspace(-1.2, 1.2, RESOLUTION)

@jit
def trace_photon(x_pixel, y_pixel):
    dt = 0.2
    r0 = R_camera
    theta0 = jnp.pi/2 - 0.1 # 5.7 degree tilt for the classic Interstellar view
    
    # 1. FIXED SCREEN SETUP (Horizontal & Vertical aligned properly)
    p_t = -1.0
    p_theta = -y_pixel # Controls up/down
    p_phi = x_pixel    # Controls left/right
    
    # 2. Null Geodesic Enforcement (H=0)
    sigma = r0**2 + a**2 * jnp.cos(theta0)**2
    delta = r0**2 - 2*M*r0 + a**2
    g_inv_tt = -( (r0**2 + a**2)**2 - delta * a**2 * jnp.sin(theta0)**2 ) / (sigma * delta)
    g_inv_tphi = -( 2 * M * r0 * a ) / (sigma * delta)
    g_inv_rr = delta / sigma
    g_inv_thth = 1 / sigma
    g_inv_phiphi = (delta - a**2 * jnp.sin(theta0)**2) / (sigma * delta * jnp.sin(theta0)**2)
    
    C = g_inv_tt * p_t**2 + 2 * g_inv_tphi * p_t * p_phi + g_inv_thth * p_theta**2 + g_inv_phiphi * p_phi**2
    p_r = -jnp.sqrt(jnp.maximum(-C / g_inv_rr, 0.00001))
    
    init_state = jnp.array([0.0, r0, theta0, 0.0, p_t, p_r, p_theta, p_phi])

    def step_fn(carry, _):
        state, color, active = carry
        
        # Adaptive step: slow down near the hole so we don't clip through the thin disk
        r_current = state[1]
        local_dt = jnp.where(r_current < 8.0, dt * 0.2, dt)
        
        new_state = geodesic_step(state, local_dt, a)
        r, theta, phi, old_theta = new_state[1], new_state[2], new_state[3], state[2]

        # 1. SHADOW LOGIC
        horizon = M + jnp.sqrt(M**2 - a**2)
        is_swallowed = r < (horizon + 0.05)
        
        # 2. RAZOR-SHARP DISK LOGIC
        # Checking if it crossed the equator is much sharper than a thick volume
        crossed_plane = (old_theta - jnp.pi/2) * (theta - jnp.pi/2) < 0
        is_in_disk = (r > 5.0) & (r < 20.0) & crossed_plane
        
        # 3. DOPPLER BEAMING
        # Gas moving toward us gets intensely bright!
        doppler_factor = 1.0 + 0.85 * jnp.sin(phi)
        base_glow = 1.5 / jnp.sqrt(r)
        cinematic_brightness = base_glow * (doppler_factor ** 3)
        
        brightness = jnp.where(is_in_disk, cinematic_brightness, 0.0)
        
        new_color = jnp.where(active & is_in_disk, brightness, color)
        final_color = jnp.where(active & is_swallowed, -1.0, new_color)
        
        still_active = active & (~is_swallowed) & (~is_in_disk) & (r < R_camera + 2.0)
        
        return (new_state, final_color, still_active), None

    # Trace for 1500 steps to capture the long looping paths
    (final_state, res_color, _), _ = lax.scan(step_fn, (init_state, 0.0, True), jnp.arange(1500))
    return res_color

# Vectorization mapping
render_view = vmap(vmap(trace_photon, in_axes=(0, None)), in_axes=(None, 0))

print("🔭 Initializing H=0 Geodesics...")
print("🔥 Rendering High-Res Cinematic Frame...")
# We feed the grid in X, Y order
raw_image = render_view(view_grid, view_grid)
print("✅ Applying Optical Bloom...")

# --- 🎬 CINEMATIC POST-PROCESSING ---
# 1. Ensure the image is right-side up. 
# Removing the .T and using origin='lower' usually fixes the matrix rotation
clean_image = jnp.maximum(raw_image.T, 0.0)

# 2. Camera Lens Bloom
bloom_light = gaussian_filter(clean_image, sigma=2.0)
bloom_heavy = gaussian_filter(clean_image, sigma=6.0)
final_cinematic_image = clean_image + (0.7 * bloom_light) + (0.3 * bloom_heavy)

# 3. Custom Hollywood Colormap
colors = [(0, 'black'), (0.15, '#2b0000'), (0.4, '#aa2200'), (0.7, '#ffaa00'), (1, 'white')]
cmap_name = 'gargantua'
cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

# --- DISPLAY ---
plt.figure(figsize=(12, 10), facecolor='black')
plt.imshow(final_cinematic_image, cmap=cm, origin='lower', vmin=0.0, vmax=2.0)
plt.axis('off')
plt.title("Gargantua: Relativistic Ray-Tracer", color='white', fontsize=16, alpha=0.7)
plt.tight_layout()
plt.savefig("gargantua_cinematic.png", facecolor='black', bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()
