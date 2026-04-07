import jax.numpy as jnp
from jax import jit, vmap, lax
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter  # For the Cinematic Glow
from src.core.solver import geodesic_step

# --- Black Hole Parameters ---
M = 1.0
a = 0.99 
R_camera = 20.0 
RESOLUTION = 400  # High Res for the final render
view_grid = jnp.linspace(-15.0, 15.0, RESOLUTION)

@jit
def trace_photon(x_pixel, y_pixel):
    dt = 0.1 
    r0 = R_camera
    theta0 = jnp.pi/2 - 0.15 
    
    # 1. CAMERA SCREEN SETUP
    p_t = -1.0
    p_theta = y_pixel
    p_phi = x_pixel
    
    # Enforcing H=0 (Null Geodesic)
    sigma = r0**2 + a**2 * jnp.cos(theta0)**2
    delta = r0**2 - 2*M*r0 + a**2
    g_inv_tt = -( (r0**2 + a**2)**2 - delta * a**2 * jnp.sin(theta0)**2 ) / (sigma * delta)
    g_inv_tphi = -( 2 * M * r0 * a ) / (sigma * delta)
    g_inv_rr = delta / sigma
    g_inv_thth = 1 / sigma
    g_inv_phiphi = (delta - a**2 * jnp.sin(theta0)**2) / (sigma * delta * jnp.sin(theta0)**2)
    
    C = g_inv_tt * p_t**2 + 2 * g_inv_tphi * p_t * p_phi + g_inv_thth * p_theta**2 + g_inv_phiphi * p_phi**2
    p_r = -jnp.sqrt(jnp.maximum(-C / g_inv_rr, 0.0001))
    
    # Note: We now track 'phi' properly (index 3) to calculate Doppler shift
    init_state = jnp.array([0.0, r0, theta0, 0.0, p_t, p_r, p_theta, p_phi])

    def step_fn(carry, _):
        state, color, active = carry
        new_state = geodesic_step(state, dt, a)
        r, theta, phi = new_state[1], new_state[2], new_state[3]

        # 1. SHADOW LOGIC
        horizon = M + jnp.sqrt(M**2 - a**2)
        is_swallowed = r < (horizon + 0.1)
        
        # 2. DISK LOGIC
        is_in_disk = (r > 4.0) & (r < 15.0) & (jnp.abs(jnp.cos(theta)) < 0.05)
        
        # 3. 🎬 CINEMATIC PHYSICS: DOPPLER BEAMING
        # Gas on one side is moving toward us (phi angle determines this)
        # Intensity scales with the cube of the Doppler factor!
        doppler_factor = 1.0 + 0.8 * jnp.sin(phi)
        base_glow = 1.5 / jnp.sqrt(r)
        cinematic_brightness = base_glow * (doppler_factor ** 3)
        
        brightness = jnp.where(is_in_disk, cinematic_brightness, 0.0)
        
        new_color = jnp.where(active & is_in_disk, brightness, color)
        final_color = jnp.where(active & is_swallowed, -1.0, new_color)
        
        still_active = active & (~is_swallowed) & (~is_in_disk) & (r < R_camera + 5.0)
        
        return (new_state, final_color, still_active), None

    (final_state, res_color, _), _ = lax.scan(step_fn, (init_state, 0.0, True), jnp.arange(1000))
    return res_color

render_view = vmap(vmap(trace_photon, in_axes=(0, None)), in_axes=(None, 0))

print("🔭 Calculating Relativistic Beaming...")
print("🔥 Rendering High-Fidelity Matrix...")
raw_image = render_view(view_grid, view_grid)
print("✅ Math Complete. Applying Optical Effects...")

# --- 🎬 CINEMATIC POST-PROCESSING ---
# 1. Remove the '-1' shadow flags and make them pure 0 (black)
clean_image = jnp.maximum(raw_image.T, 0.0)

# 2. Optical Bloom: Simulate the camera lens capturing intense light
bloom_light = gaussian_filter(clean_image, sigma=3.0)
bloom_heavy = gaussian_filter(clean_image, sigma=8.0)
final_cinematic_image = clean_image + (0.6 * bloom_light) + (0.3 * bloom_heavy)

# 3. Custom Hollywood Colormap (Deep red to bright white-yellow)
colors = [(0, 'black'), (0.2, '#3b0000'), (0.5, '#cc3300'), (0.8, '#ffcc00'), (1, 'white')]
cmap_name = 'interstellar'
cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

# --- DISPLAY ---
plt.figure(figsize=(12, 12), facecolor='black')
# We increase vmax slightly to let the brightest parts hit 'pure white'
plt.imshow(final_cinematic_image, cmap=cm, origin='lower', vmin=0.0, vmax=2.5)
plt.axis('off')
plt.tight_layout()
plt.savefig("gargantua_cinematic.png", facecolor='black', bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()
