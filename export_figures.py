import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from src.gg_nuts import run_gg_nuts, run_nuts


def samples_to_idata(samples, var_names):
    data = {name: samples[None, :, i] for i, name in enumerate(var_names)}
    return az.convert_to_inference_data(data)


def log_prob_rosenbrock(theta):
    x, y = theta[0], theta[1]
    return -0.05 * (1 - x) ** 2 - (y - x**2) ** 2


def log_prob_funnel(theta):
    v, x = theta[0], theta[1]
    log_p_v = -0.5 * v**2 / 9
    log_p_x_given_v = -0.5 * x**2 * jnp.exp(-v) - 0.5 * v
    return log_p_v + log_p_x_given_v


def main():
    out_dir = root / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    key = jr.PRNGKey(42)
    n_samples = 2000

    var_names_rb = ["x", "y"]
    print("Running GG-NUTS on Rosenbrock...")
    key, k = jr.split(key)
    samples_rb, acc_rb, info_rb = run_gg_nuts(
        k,
        log_prob_rosenbrock,
        jnp.array([1.0, 1.0]),
        n_samples=n_samples,
        max_depth_stable=5,
        max_depth_explore=8,
        a=2.0,
        n_pilot=300,
    )
    x = jnp.linspace(-2, 3, 100)
    y = jnp.linspace(-1, 5, 100)
    X, Y = jnp.meshgrid(x, y)
    positions = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    log_probs = jax.vmap(log_prob_rosenbrock)(positions).reshape(X.shape)
    plt.figure(figsize=(6, 5))
    plt.contour(X, Y, jnp.exp(log_probs), levels=10, colors="gray", alpha=0.5)
    plt.scatter(samples_rb[::2, 0], samples_rb[::2, 1], alpha=0.3, s=5, c="blue")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(f"GG-NUTS — Rosenbrock (acc={acc_rb:.1%})")
    plt.xlim(-2, 3)
    plt.ylim(-1, 5)
    plt.tight_layout()
    plt.savefig(out_dir / "rosenbrock_samples.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'rosenbrock_samples.png'}")

    idata_rb = samples_to_idata(samples_rb, var_names_rb)
    az.plot_trace(idata_rb, combined=True, figsize=(10, 4))
    plt.tight_layout()
    plt.savefig(out_dir / "rosenbrock_trace.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'rosenbrock_trace.png'}")

    az.plot_autocorr(idata_rb, combined=True, figsize=(10, 3))
    plt.tight_layout()
    plt.savefig(out_dir / "rosenbrock_autocorr.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'rosenbrock_autocorr.png'}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(info_rb["geometry_scores"])
    axes[0].axhline(info_rb["threshold_b"], color="r", linestyle="--", label=f"b={float(info_rb['threshold_b']):.2f}")
    axes[0].set_title("Geometry score")
    axes[0].legend()
    axes[1].plot(info_rb["gate_probabilities"])
    axes[1].set_title("Gate probability w(q)")
    axes[2].hist(info_rb["geometry_scores"], bins=50, alpha=0.7, edgecolor="black")
    axes[2].axvline(info_rb["threshold_b"], color="r", linestyle="--", label="threshold b")
    axes[2].set_title("Score histogram")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rosenbrock_gg_diagnostics.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'rosenbrock_gg_diagnostics.png'}")

    var_names_fn = ["v", "x"]
    print("Running GG-NUTS on Funnel...")
    key, k = jr.split(key)
    samples_fn, acc_fn, info_fn = run_gg_nuts(
        k,
        log_prob_funnel,
        jnp.array([0.0, 0.0]),
        n_samples=n_samples,
        max_depth_stable=5,
        max_depth_explore=8,
        a=2.0,
        n_pilot=300,
    )
    x = jnp.linspace(-6, 6, 100)
    y = jnp.linspace(-10, 10, 100)
    X, Y = jnp.meshgrid(x, y)
    positions = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    log_probs = jax.vmap(log_prob_funnel)(positions).reshape(X.shape)
    plt.figure(figsize=(6, 5))
    plt.contour(X, Y, jnp.exp(log_probs), levels=10, colors="gray", alpha=0.5)
    plt.scatter(samples_fn[::2, 0], samples_fn[::2, 1], alpha=0.3, s=5, c="blue")
    plt.xlabel(r"$v$")
    plt.ylabel(r"$x$")
    plt.title(f"GG-NUTS — Funnel (acc={acc_fn:.1%})")
    plt.xlim(-6, 6)
    plt.ylim(-10, 10)
    plt.tight_layout()
    plt.savefig(out_dir / "funnel_samples.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'funnel_samples.png'}")

    idata_fn = samples_to_idata(samples_fn, var_names_fn)
    az.plot_trace(idata_fn, combined=True, figsize=(10, 4))
    plt.tight_layout()
    plt.savefig(out_dir / "funnel_trace.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'funnel_trace.png'}")

    az.plot_autocorr(idata_fn, combined=True, figsize=(10, 3))
    plt.tight_layout()
    plt.savefig(out_dir / "funnel_autocorr.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'funnel_autocorr.png'}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(info_fn["geometry_scores"])
    axes[0].axhline(info_fn["threshold_b"], color="r", linestyle="--", label=f"b={float(info_fn['threshold_b']):.2f}")
    axes[0].set_title("Geometry score")
    axes[0].legend()
    axes[1].plot(info_fn["gate_probabilities"])
    axes[1].set_title("Gate probability w(q)")
    axes[2].hist(info_fn["geometry_scores"], bins=50, alpha=0.7, edgecolor="black")
    axes[2].axvline(info_fn["threshold_b"], color="r", linestyle="--", label="threshold b")
    axes[2].set_title("Score histogram")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "funnel_gg_diagnostics.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'funnel_gg_diagnostics.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
