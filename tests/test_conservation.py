import jnp
from src.core.solver import hamiltonian, geodesic_step

def test_energy_conservation():
    # Setup a random ray near a black hole with spin a=0.5
    a = 0.5
    state = jnp.array([0.0, 10.0, 1.5, 0.0, -1.0, -1.0, 0.1, 0.5])
    
    h_initial = hamiltonian(state, a)
    
    # Take 100 steps through spacetime
    current_state = state
    for _ in range(100):
        current_state = geodesic_step(current_state, 0.01, a)
        
    h_final = hamiltonian(current_state, a)
    
    # If the difference is tiny (e.g. 1e-6), the physics is "Gold Standard"
    print(f"Energy Drift: {jnp.abs(h_initial - h_final)}")
    assert jnp.abs(h_initial - h_final) < 1e-5
