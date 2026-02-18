"""GG-NUTS: Geometry-Gated No-U-Turn Sampler

This module implements a novel MCMC sampler that adaptively switches between
two NUTS kernels based on local geometry:
- Stable kernel: shallow trees for stiff regions
- Explore kernel: deeper trees for easy regions

The geometry score s(q) = sqrt(g(q)^T M g(q)) measures local stiffness,
where g(q) = -∇log_prob(q) is the gradient of the potential energy.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import blackjax


def geometry_score(gradient, mass_matrix=None):
    """Compute geometry score from gradient.
    
    s(q) = sqrt(g(q)^T M g(q))
    
    Args:
        gradient: Gradient of potential U(q) = -log_prob(q), shape (D,)
        mass_matrix: Optional mass matrix (diagonal), shape (D,). Defaults to identity.
    
    Returns:
        Scalar geometry score
    """
    if mass_matrix is None:
        mass_matrix = jnp.ones_like(gradient)
    return jnp.sqrt(jnp.sum(mass_matrix * gradient**2))


def gate_probability(score, a=2.0, b=0.0, invert=False):
    """Compute gating probability w(q) = sigmoid(a*(s(q) - b)).
    
    When invert=False (default): high score -> stable (shallow) kernel.
    When invert=True: high score -> explore (deep) kernel.
    
    Args:
        score: Geometry score s(q)
        a: Steepness parameter (higher = sharper transition)
        b: Threshold parameter (median of pilot scores)
        invert: If True, flip gate so high gradient triggers explore kernel
    
    Returns:
        Probability of using stable (shallow) kernel
    """
    if invert:
        return jax.nn.sigmoid(a * (b - score))
    return jax.nn.sigmoid(a * (score - b))


def compute_pilot_threshold(key, log_prob_fn, initial_position, mass_matrix, n_pilot=500, max_depth=8, step_size=None, inverse_mass_matrix=None, percentile=50.0):
    """Run a pilot chain to compute threshold b from score percentile.
    
    IMPORTANT: Must use the SAME mass matrix that will be used for main sampling!
    
    Args:
        key: JAX random key
        log_prob_fn: Log probability function
        initial_position: Starting point, shape (D,)
        mass_matrix: Mass matrix M (not inverse!) for geometry score
        n_pilot: Number of pilot samples
        max_depth: Max tree depth for pilot run
        step_size: Optional step size (for consistency with warmup)
        inverse_mass_matrix: Optional inverse mass matrix for pilot NUTS dynamics
        percentile: Percentile of scores to use as threshold (default 50 = median)
    
    Returns:
        threshold: Geometry score at given percentile from pilot run
        pilot_samples: Samples from pilot run (for diagnostics)
        scores: Geometry scores along pilot chain
    """
    pilot_samples, _, pilot_info = run_nuts(
        key, log_prob_fn, initial_position, n_samples=n_pilot, max_depth=max_depth,
        step_size=step_size, inverse_mass_matrix=inverse_mass_matrix
    )
    
    grad_fn = jax.grad(lambda q: -log_prob_fn(q))
    
    def compute_score_at_point(position):
        gradient = grad_fn(position)
        return geometry_score(gradient, mass_matrix)
    
    scores = jax.vmap(compute_score_at_point)(pilot_samples)
    threshold = jnp.percentile(scores, percentile)
    
    return threshold, pilot_samples, scores


def run_nuts(key, log_prob_fn, initial_position, n_samples, max_depth=10, step_size=None, inverse_mass_matrix=None):
    """Run vanilla NUTS with single fixed max_depth.
    
    Args:
        key: JAX random key
        log_prob_fn: Log probability function
        initial_position: Starting point, shape (D,)
        n_samples: Number of samples to draw
        max_depth: Maximum tree depth
        step_size: Optional step size (uses warmup if None)
        inverse_mass_matrix: Optional inverse mass matrix (uses warmup or identity if None)
    
    Returns:
        samples: Array of shape (n_samples, D)
        acceptance_rate: Mean acceptance probability
        info: Dictionary with additional info (grad_evals, etc.)
    """
    if step_size is None:
        warmup = blackjax.window_adaptation(
            blackjax.nuts, log_prob_fn, max_num_doublings=max_depth
        )
        key_warmup, key_sample = jr.split(key)
        (state, parameters), _ = warmup.run(key_warmup, initial_position, num_steps=1000)
        step_size = parameters["step_size"]
        inverse_mass_matrix = parameters["inverse_mass_matrix"]
    else:
        if inverse_mass_matrix is None:
            inverse_mass_matrix = jnp.ones(initial_position.shape[0])
        state = None
    
    nuts = blackjax.nuts(
        log_prob_fn,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
        max_num_doublings=max_depth,
    )
    
    if state is None:
        state = nuts.init(initial_position)
    
    @jax.jit
    def one_step(state, key):
        state, info = nuts.step(key, state)
        return state, (state.position, info.acceptance_rate, info.num_integration_steps)
    
    key_sample = key if step_size is not None else key_sample
    keys = jr.split(key_sample, n_samples)
    _, (samples, acceptance_rates, num_steps) = jax.lax.scan(one_step, state, keys)
    
    info = {
        "acceptance_rate": acceptance_rates.mean(),
        "num_integration_steps": num_steps,
        "total_grad_evals": num_steps.sum(),
        "step_size": step_size,
        "inverse_mass_matrix": inverse_mass_matrix,
    }
    
    return samples, acceptance_rates.mean(), info


def run_gg_nuts(
    key,
    log_prob_fn,
    initial_position,
    n_samples,
    max_depth_stable=6,
    max_depth_explore=9,
    a=2.0,
    b=None,
    n_pilot=500,
    invert_gate=False,
    step_size_ratio=1.0,
    threshold_percentile=50.0,
):
    """Run GG-NUTS: Geometry-Gated NUTS.
    
    At each iteration:
    1. Compute geometry score s(q) = sqrt(g^T M g)
    2. Compute gate probability w(q) = sigmoid(a*(s - b))
    3. With probability w(q), use stable kernel (shallow trees)
       Otherwise, use explore kernel (deep trees)
    
    Args:
        key: JAX random key
        log_prob_fn: Log probability function
        initial_position: Starting point, shape (D,)
        n_samples: Number of samples to draw
        max_depth_stable: Max tree depth for stable kernel (D_s)
        max_depth_explore: Max tree depth for explore kernel (D_l)
        a: Gate steepness parameter
        b: Gate threshold (computed from pilot if None)
        n_pilot: Number of pilot samples (if b is None)
        invert_gate: If True, high gradient triggers explore kernel
        step_size_ratio: Multiplier for stable kernel step size (< 1 means smaller)
        threshold_percentile: Percentile for pilot threshold (default 50 = median)
    
    Returns:
        samples: Array of shape (n_samples, D)
        acceptance_rate: Mean acceptance probability
        info: Dictionary with diagnostics
    """
    key_warmup, key = jr.split(key)
    warmup = blackjax.window_adaptation(
        blackjax.nuts, log_prob_fn, max_num_doublings=max_depth_explore
    )
    (state, parameters), _ = warmup.run(key_warmup, initial_position, num_steps=1000)
    step_size = parameters["step_size"]
    inverse_mass_matrix = parameters["inverse_mass_matrix"]
    mass_matrix = 1.0 / inverse_mass_matrix
    
    print(f"Warmup complete: step_size={step_size:.4f}")
    
    if b is None:
        key_pilot, key = jr.split(key)
        b, pilot_samples, pilot_scores = compute_pilot_threshold(
            key_pilot, log_prob_fn, initial_position, 
            mass_matrix=mass_matrix,
            n_pilot=n_pilot,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            percentile=threshold_percentile,
        )
        pct_label = f"p{threshold_percentile:.0f}" if threshold_percentile != 50.0 else "median"
        print(f"Pilot threshold b = {b:.4f} ({pct_label} of {n_pilot} pilot scores)")
    else:
        pilot_samples = None
        pilot_scores = None
    
    if invert_gate:
        print("Gate mode: INVERTED (high gradient -> explore kernel)")
    
    key_sample = key
    
    stable_step = step_size * step_size_ratio
    
    nuts_stable = blackjax.nuts(
        log_prob_fn,
        step_size=stable_step,
        inverse_mass_matrix=inverse_mass_matrix,
        max_num_doublings=max_depth_stable,
    )
    
    nuts_explore = blackjax.nuts(
        log_prob_fn,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
        max_num_doublings=max_depth_explore,
    )
    
    current_state = nuts_stable.init(state.position)
    
    grad_fn = jax.grad(lambda q: -log_prob_fn(q))
    
    @jax.jit
    def one_step(state, key):
        key_gate, key_step = jr.split(key)
        
        gradient = grad_fn(state.position)
        score = geometry_score(gradient, mass_matrix)
        
        w = gate_probability(score, a=a, b=b, invert=invert_gate)
        use_stable = jr.bernoulli(key_gate, w)
        
        def run_stable(s, k):
            return nuts_stable.step(k, s)
        
        def run_explore(s, k):
            return nuts_explore.step(k, s)
        
        new_state, info = jax.lax.cond(
            use_stable,
            run_stable,
            run_explore,
            state,
            key_step,
        )
        
        return new_state, (
            new_state.position,
            info.acceptance_rate,
            info.num_integration_steps,
            use_stable,
            score,
            w,
        )
    
    keys = jr.split(key_sample, n_samples)
    _, (samples, acceptance_rates, num_steps, used_stable, scores, gate_probs) = jax.lax.scan(
        one_step, current_state, keys
    )
    
    info = {
        "acceptance_rate": acceptance_rates.mean(),
        "num_integration_steps": num_steps,
        "total_grad_evals": num_steps.sum(),
        "step_size": step_size,
        "stable_step_size": stable_step,
        "inverse_mass_matrix": inverse_mass_matrix,
        "threshold_b": b,
        "steepness_a": a,
        "invert_gate": invert_gate,
        "step_size_ratio": step_size_ratio,
        "threshold_percentile": threshold_percentile,
        "used_stable_fraction": used_stable.mean(),
        "geometry_scores": scores,
        "gate_probabilities": gate_probs,
        "pilot_samples": pilot_samples,
        "pilot_scores": pilot_scores,
    }
    
    return samples, acceptance_rates.mean(), info


def run_fixed_mixture_nuts(
    key,
    log_prob_fn,
    initial_position,
    n_samples,
    max_depth_stable=6,
    max_depth_explore=9,
    p_stable=0.5,
):
    """Run fixed-mixture NUTS (ablation baseline).
    
    Uses stable kernel with constant probability p_stable (no geometry).
    
    Args:
        key: JAX random key
        log_prob_fn: Log probability function
        initial_position: Starting point, shape (D,)
        n_samples: Number of samples to draw
        max_depth_stable: Max tree depth for stable kernel
        max_depth_explore: Max tree depth for explore kernel
        p_stable: Constant probability of using stable kernel
    
    Returns:
        samples: Array of shape (n_samples, D)
        acceptance_rate: Mean acceptance probability
        info: Dictionary with diagnostics
    """
    warmup = blackjax.window_adaptation(
        blackjax.nuts, log_prob_fn, max_num_doublings=max_depth_explore
    )
    key_warmup, key_sample = jr.split(key)
    (state, parameters), _ = warmup.run(key_warmup, initial_position, num_steps=1000)
    step_size = parameters["step_size"]
    inverse_mass_matrix = parameters["inverse_mass_matrix"]
    
    print(f"Warmup complete: step_size={step_size:.4f}")
    
    nuts_stable = blackjax.nuts(
        log_prob_fn,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
        max_num_doublings=max_depth_stable,
    )
    
    nuts_explore = blackjax.nuts(
        log_prob_fn,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
        max_num_doublings=max_depth_explore,
    )
    
    current_state = nuts_stable.init(state.position)
    
    @jax.jit
    def one_step(state, key):
        key_gate, key_step = jr.split(key)
        use_stable = jr.bernoulli(key_gate, p_stable)
        
        def run_stable(s, k):
            return nuts_stable.step(k, s)
        
        def run_explore(s, k):
            return nuts_explore.step(k, s)
        
        new_state, info = jax.lax.cond(
            use_stable,
            run_stable,
            run_explore,
            state,
            key_step,
        )
        
        return new_state, (
            new_state.position,
            info.acceptance_rate,
            info.num_integration_steps,
            use_stable,
        )
    
    keys = jr.split(key_sample, n_samples)
    _, (samples, acceptance_rates, num_steps, used_stable) = jax.lax.scan(
        one_step, current_state, keys
    )
    
    info = {
        "acceptance_rate": acceptance_rates.mean(),
        "num_integration_steps": num_steps,
        "total_grad_evals": num_steps.sum(),
        "step_size": step_size,
        "used_stable_fraction": used_stable.mean(),
    }
    
    return samples, acceptance_rates.mean(), info
