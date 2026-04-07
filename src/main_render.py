import jax.numpy as jnp
from jax import jit, vmap, lax
import matplotlib.pyplot as plt
from src.core.solver import geodesic_step

# --- Black Hole Parameters ---
M = 1.0     # Mass
a = 0.99    # High spin (Gargantua style)
R_camera = 50.0 

# --- Gargantua Perspective Setup ---
RESOLUTION = 250  # Better resolution for the 'rings'
# We use a wider grid to see the whole horizon + disk
view_grid = jnp.linspace(-0.15, 0.15, RESOLUTION) 

def trace_photon(x_pixel, y_pixel):
    dt = 0.3
    
    # Place camera at r=80, tilted 5 degrees off the equator (pi/2.1)
    # The initial momentum (p_r, p_theta, p_phi) is what 'aims' the lens
    init_state = jnp.array([
        0.0,            # t
        80.0,           # r (Distance)
        jnp.pi/2.1,     # theta (Tilt)
        0.0,            # phi
        -1.0,           # p_t
        -1.0,           # p_r (Fire toward the center)
        y_pixel * 10.0, # p_theta (Vertical spread)
        x_pixel * 10.0  # p_phi (Horizontal spread)
    ])

    def step_fn(carry, _):
        state, color, active = carry
        new_state = geodesic_step(state, dt, a)
        
        r, theta, old_theta = new_state[1], new_state[2], state[2]

        # 1. Shadow Logic: Did we hit the horizon?
        horizon = M + jnp.sqrt(M**2 - a**2)
        is_swallowed = r < (horizon + 0.05)
        
        # 2. Disk Logic: Crossing the 'Glow' plane (theta = pi/2)
        crossed_plane = (old_theta - jnp.pi/2) * (theta - jnp.pi/2) < 0
        # The disk should exist between 6.0 and 25.0 radius
        is_in_disk = (r > 6.0) & (r < 25.0) & crossed_plane
        
        # 3. Coloring: 
        # Inside the disk, brightness falls off as 1/r^2
        brightness = jnp.where(is_in_disk, 15.0 / (r**1.5), 0.0)
        new_color = jnp.where(active & is_in_disk, brightness, color)
        
        # Stop if we hit the hole or disk
        still_active = active & (~is_swallowed) & (~is_in_disk)
        
        return (new_state, new_color, still_active), None

    # Run for 2000 steps to ensure the light has time to travel and curve
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
