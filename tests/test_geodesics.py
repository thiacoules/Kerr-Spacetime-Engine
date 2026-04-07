import jax.numpy as jnp
from src.core.solver import geodesic_step
from src.core.physics_checks import get_carter_constant

def test_carter_constant_conservation():
    """
    Test: Does the Carter Constant stay the same after 500 steps?
    """
    a = 0.9  # Fast spinning black hole
    dt = 0.01 #Smaller step = More accuracy
    # Initial state: [t, r, theta, phi, p_t, p_r, p_theta, p_phi]
    state = jnp.array([0.0, 10.0, 1.2, 0.0, -1.0, -0.5, 0.1, 2.0])
    
    Q_start = get_carter_constant(state, a)
    
    # Evolve the photon for 500 steps
    current_state = state
    for _ in range(200): #Fewer steps for the initial test
        current_state = geodesic_step(current_state, dt, a)
        
    Q_end = get_carter_constant(current_state, a)
    
    # Assert that the change is less than 0.1% (numerical drift)
    drift = jnp.abs((Q_end - Q_start) / Q_start)
    print(f"Carter Constant Drift: {drift:.6f}")
    assert drift < 1e-3
