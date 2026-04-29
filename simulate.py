"""Simulate the Hodgkin-Huxley model and save a noisy synthetic trace."""
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController

from hh_model import (
    initial_state,
    hodgkin_huxley,
    I_inj,
    DEFAULT_PARAMS,
)


def simulate(params, t0=0.0, t1=100.0, n_save=10000, rtol=1e-6, atol=1e-6):
    """Integrate the HH model with given parameters."""
    y0 = initial_state()
    ts = jnp.linspace(t0, t1, n_save)

    sol = diffeqsolve(
        ODETerm(hodgkin_huxley),
        Tsit5(),
        t0=t0,
        t1=t1,
        dt0=0.01,
        y0=y0,
        args=params,
        saveat=SaveAt(ts=ts),
        stepsize_controller=PIDController(rtol=rtol, atol=atol),
        max_steps=100_000,
    )
    return sol


def main():
    sol = simulate(DEFAULT_PARAMS)

    t = sol.ts
    V, m, h, n = sol.ys.T
    I = I_inj(t)

    # add observation noise to V
    key = jrandom.PRNGKey(0)
    noise_std = 2.0
    V_noisy = V + noise_std * jrandom.normal(key, V.shape)

    # save synthetic data
    np.savez(
        "data/synthetic_trace.npz",
        t=np.array(t),
        V=np.array(V),
        V_noisy=np.array(V_noisy),
        m=np.array(m),
        h=np.array(h),
        n=np.array(n),
        I=np.array(I),
        true_params=DEFAULT_PARAMS,
        noise_std=noise_std,
    )

    # plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, V_noisy, color="lightgray", label="noisy observation")
    axes[0].plot(t, V, color="C0", label="clean trace")
    axes[0].set_ylabel("V (mV)")
    axes[0].set_title("Hodgkin-Huxley simulation")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, m, label="m: Na activation")
    axes[1].plot(t, h, label="h: Na inactivation")
    axes[1].plot(t, n, label="n: K activation")
    axes[1].set_ylabel("Gating variables")
    axes[1].legend(loc="upper right")

    axes[2].plot(t, I)
    axes[2].set_ylabel("I (µA/cm²)")
    axes[2].set_xlabel("Time (ms)")

    plt.tight_layout()
    plt.savefig("figures/voltage_trace.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()