"""Multiclass SVM loss, manual gradient, and minibatch SGD."""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

import torch


LossFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float],
    Tuple[torch.Tensor, torch.Tensor],
]


def _check_batch(X: torch.Tensor, y: torch.Tensor) -> None:
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("Expected X=(N,D) and y=(N,) with matching N")
    if X.shape[0] == 0:
        raise ValueError("The minibatch must contain at least one example")
    if not X.is_floating_point() or y.dtype != torch.int64:
        raise ValueError("X must be floating point and y must be int64")
    if X.device != y.device:
        raise ValueError("X and y must be on the same device")
    if torch.any(y < 0):
        raise ValueError("Class labels must be nonnegative")


def _check_svm_inputs(W, X, y, reg) -> None:
    _check_batch(X, y)
    if W.ndim != 2 or X.shape[1] != W.shape[0] or W.shape[1] == 0:
        raise ValueError("Expected W=(D,C) with C > 0")
    if W.dtype != X.dtype or W.device != X.device:
        raise ValueError("W and X must have the same dtype and device")
    if torch.any(y >= W.shape[1]):
        raise ValueError("Class labels must be smaller than C")
    if not math.isfinite(reg) or reg < 0:
        raise ValueError("reg must be finite and nonnegative")


def linear_scores(
    X: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return class scores for a batch of examples."""
    if X.ndim != 2 or W.ndim != 2:
        raise ValueError("X and W must both be rank-2 tensors")
    if X.shape[1] != W.shape[0]:
        raise ValueError("X.shape[1] must equal W.shape[0]")
    if b is not None and b.shape != (W.shape[1],):
        raise ValueError("b must have shape (number_of_classes,)")

    scores = X @ W
    if b is not None:
        scores = scores + b
    return scores


def svm_loss_naive(
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    reg: float = 0.0,
    delta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loop reference: mean hinge loss + reg * sum(W**2)."""
    _check_svm_inputs(W, X, y, reg)
    if not math.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")

    num_train = X.shape[0]
    num_classes = W.shape[1]
    loss = W.new_tensor(0.0)
    dW = torch.zeros_like(W)

    for i in range(num_train):
        scores = linear_scores(X[i : i + 1], W).squeeze(0)
        correct_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_score + delta
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    loss = loss / num_train + reg * torch.sum(W * W)
    dW = dW / num_train + 2 * reg * W
    return loss, dW


def svm_loss_vectorized(
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    reg: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized hinge loss (margin 1) and its manual gradient."""
    _check_svm_inputs(W, X, y, reg)

    num_train = X.shape[0]
    rows = torch.arange(num_train, device=X.device)
    scores = linear_scores(X, W)
    correct_scores = scores[rows, y].unsqueeze(1)
    margins = (scores - correct_scores + 1.0).clamp(min=0)
    margins[rows, y] = 0.0
    loss = margins.sum() / num_train + reg * (W * W).sum()

    # An active incorrect margin contributes +X to its class and -X to y.
    # Strict > 0 matches the reference's subgradient at a zero margin.
    coefficients = (margins > 0).to(dtype=W.dtype)
    coefficients[rows, y] = -coefficients.sum(dim=1)
    dW = X.T @ coefficients / num_train + 2 * reg * W

    return loss, dW


def sample_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a minibatch with replacement from ``X`` and ``y``."""
    _check_batch(X, y)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    indices = torch.randint(
        X.shape[0],
        (batch_size,),
        device=X.device,
        generator=generator,
    )
    return X[indices], y[indices]


def train_linear_classifier(
    loss_func: LossFunction,
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 1e-1,
    reg: float = 1e-3,
    num_iters: int = 400,
    batch_size: int = 64,
    W: torch.Tensor | None = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, List[float]]:
    """Run manual SGD, copying any supplied initial weights."""
    _check_batch(X, y)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if num_iters < 0 or batch_size <= 0:
        raise ValueError("num_iters must be nonnegative and batch_size positive")
    generator = torch.Generator(device=X.device).manual_seed(seed)
    if W is None:
        num_classes = int(y.max().item()) + 1
        W = 1e-3 * torch.randn(
            X.shape[1],
            num_classes,
            dtype=X.dtype,
            device=X.device,
            generator=generator,
        )
    else:
        W = W.detach().clone()
    _check_svm_inputs(W, X, y, reg)

    loss_history: List[float] = []
    with torch.no_grad():
        for _ in range(num_iters):
            X_batch, y_batch = sample_batch(X, y, batch_size, generator)
            loss, gradient = loss_func(W, X_batch, y_batch, reg)
            loss_history.append(loss.item())

            W -= learning_rate * gradient

    return W, loss_history


def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Return the highest-scoring class for each example."""
    return linear_scores(X, W).argmax(dim=1)


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Return classification accuracy in the range [0, 1]."""
    if y_pred.shape != y_true.shape:
        raise ValueError("Prediction and target shapes must match")
    if y_true.numel() == 0:
        raise ValueError("Accuracy requires at least one target")
    return float((y_pred == y_true).to(torch.float64).mean())
