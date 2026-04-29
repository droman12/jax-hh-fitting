import jax.numpy as jnp
"""
Classical Hodgkin-Huxley single-compartment neuron model.

Units:
    V        : mV (absolute membrane potential)
    t        : ms
    I        : µA/cm²
    g (cond) : mS/cm²
    C_m      : µF/cm²

Convention follows Dayan & Abbott (2001). At V_rest = -65 mV, steady-state
gating variables are approximately m_inf ≈ 0.05, h_inf ≈ 0.60, n_inf ≈ 0.32.
"""

# Default biophysical parameters
DEFAULT_PARAMS = {
    'C_m':  1.0,
    'g_Na': 120.0,
    'g_K':  36.0,
    'g_L':  0.3,
    'E_Na': 50.0,
    'E_K':  -77.0,
    'E_L':  -54.387,
}

V_REST = -65.0  # mV

def alpha_m(V):
    x = V + 40.0
    return jnp.where(
        jnp.abs(x) < 1e-6,
        1.0,
        0.1 * x / (1.0 - jnp.exp(-x / 10.0)),
    )

def beta_m(V):
    return 4.0 * jnp.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    return 0.07 * jnp.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    return 1.0 / (1.0 + jnp.exp(-(V + 35.0) / 10.0))

def alpha_n(V):
    x = V + 55.0
    return jnp.where(
        jnp.abs(x) < 1e-6,
        0.1,
        0.01 * x / (1.0 - jnp.exp(-x / 10.0)),
    )

def beta_n(V):
    return 0.125 * jnp.exp(-(V + 65.0) / 80.0)

# Steady-state and initial condition

def steady_state(V):
    """Steady-state values [m_inf, h_inf, n_inf] at voltage V."""
    m_inf = alpha_m(V) / (alpha_m(V) + beta_m(V))
    h_inf = alpha_h(V) / (alpha_h(V) + beta_h(V))
    n_inf = alpha_n(V) / (alpha_n(V) + beta_n(V))
    return jnp.array([m_inf, h_inf, n_inf])


def initial_state(V_rest=V_REST):
    """Initial state vector [V, m, h, n] with gating at steady state."""
    m0, h0, n0 = steady_state(V_rest)
    return jnp.array([V_rest, m0, h0, n0])


# Stimulus

def I_inj(t):
    """Square pulse: 10 µA/cm² between t=10 ms and t=40 ms."""
    return jnp.where((t >= 10.0) & (t <= 70.0), 10.0, 0.0)


# Dynamics

def hodgkin_huxley(t, y, params):
    """
    Hodgkin-Huxley dynamics: dy/dt = f(t, y; params).

    Args:
        t: time, ms (scalar).
        y: state [V, m, h, n].
        params: dict with keys 'C_m', 'g_Na', 'g_K', 'g_L', 'E_Na', 'E_K', 'E_L'.

    Returns:
        dy/dt as a length-4 array.
    """
    V, m, h, n = y

    I_Na = params['g_Na'] * (m ** 3) * h * (V - params['E_Na'])
    I_K  = params['g_K']  * (n ** 4)     * (V - params['E_K'])
    I_L  = params['g_L']                 * (V - params['E_L'])

    dVdt = (I_inj(t) - I_Na - I_K - I_L) / params['C_m']
    dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n

    return jnp.array([dVdt, dmdt, dhdt, dndt])


