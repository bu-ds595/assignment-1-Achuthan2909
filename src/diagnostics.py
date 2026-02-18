"""Diagnostic utilities for MCMC samplers."""

import jax.numpy as jnp


def ess_per_grad_eval(ess, total_grad_evals):
    """Compute ESS per gradient evaluation.
    
    This is a compute-aware metric for comparing samplers with different
    computational costs (e.g., NUTS with different tree depths).
    
    Args:
        ess: Effective sample size (from ArviZ or similar)
        total_grad_evals: Total number of gradient evaluations
    
    Returns:
        ESS per gradient evaluation
    """
    return ess / total_grad_evals


def ess_per_second(ess, elapsed_time):
    """Compute ESS per second of wall-clock time.
    
    Args:
        ess: Effective sample size
        elapsed_time: Wall-clock time in seconds
    
    Returns:
        ESS per second
    """
    return ess / elapsed_time


def mean_tree_depth(num_integration_steps):
    """Estimate mean tree depth from integration steps.
    
    For NUTS, num_integration_steps = 2^depth - 1 on average.
    
    Args:
        num_integration_steps: Array of integration steps per sample
    
    Returns:
        Mean tree depth
    """
    # Approximate: depth ≈ log2(num_steps + 1)
    return jnp.log2(num_integration_steps + 1).mean()
