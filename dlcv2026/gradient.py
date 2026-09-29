"""Numerical checks for analytic gradients."""

import random

import torch


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """Check random coordinates with centered finite differences.

    Return relative errors, restoring each perturbed coordinate even if the
    objective raises an exception. The input and gradient shapes must match.
    """
    if x.shape != analytic_grad.shape or x.numel() == 0:
        raise ValueError("x and analytic_grad must have the same nonempty shape")
    if h <= 0 or num_checks < 1:
        raise ValueError("h and num_checks must be positive")
    errors = []
    for _ in range(num_checks):
        index = tuple(random.randrange(size) for size in x.shape)
        original = x[index].item()
        try:
            with torch.no_grad():
                x[index] = original + h
                positive = float(f(x))
                x[index] = original - h
                negative = float(f(x))
        finally:
            with torch.no_grad():
                x[index] = original
        numerical = (positive - negative) / (2 * h)
        analytic = analytic_grad[index].item()
        error = abs(numerical - analytic) / max(1e-8, abs(numerical) + abs(analytic))
        errors.append(error)
        print(f"numerical: {numerical:.6f}, analytic: {analytic:.6f}, relative error: {error:.2e}")
    return errors
