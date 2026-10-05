"""Check SVM equivalence, boundary subgradients, and independent derivatives."""
import unittest

import torch

import linear_classifier as lc


class SVMTests(unittest.TestCase):
    def test_random_equivalence(self):
        for dtype in (torch.float32, torch.float64):
            for seed in range(5):
                g = torch.Generator().manual_seed(seed)
                X = torch.randn(13, 7, generator=g, dtype=dtype)
                W = torch.randn(7, 4, generator=g, dtype=dtype) * 0.3
                y = torch.randint(4, (13,), generator=g)
                for reg in (0.0, 0.03, 1.0):
                    with self.subTest(dtype=dtype, seed=seed, reg=reg):
                        actual = lc.svm_loss_vectorized(W, X, y, reg)
                        expected = lc.svm_loss_naive(W, X, y, reg)
                        for a, e in zip(actual, expected):
                            torch.testing.assert_close(a, e)

    def test_autograd_derivative(self):
        g = torch.Generator().manual_seed(19)
        X = torch.randn(11, 6, generator=g, dtype=torch.float64)
        W = (torch.randn(6, 5, generator=g, dtype=torch.float64) * 0.02).requires_grad_()
        y = torch.randint(5, (11,), generator=g)
        reg = 0.07
        scores = X @ W
        mask = torch.ones_like(scores)
        mask[torch.arange(len(y)), y] = 0
        margins = torch.relu(scores - scores.gather(1, y[:, None]) + 1)
        objective = (margins * mask).sum() / len(y) + reg * W.square().sum()
        expected_gradient, = torch.autograd.grad(objective, W)
        loss, gradient = lc.svm_loss_vectorized(W.detach(), X, y, reg)
        torch.testing.assert_close(loss, objective.detach())
        torch.testing.assert_close(gradient, expected_gradient)

    def test_zero_margin(self):
        X = torch.tensor([[1.0]], dtype=torch.float64)
        W = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        y = torch.tensor([0])
        loss, gradient = lc.svm_loss_vectorized(W, X, y)
        self.assertEqual(loss.item(), 0.0)
        torch.testing.assert_close(gradient, torch.zeros_like(W))

    def test_single_example_and_class(self):
        X = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
        W = torch.tensor([[0.5], [0.3]], dtype=torch.float64)
        y = torch.tensor([0])
        loss, gradient = lc.svm_loss_vectorized(W, X, y, reg=0.1)
        torch.testing.assert_close(loss, 0.1 * W.square().sum())
        torch.testing.assert_close(gradient, 0.2 * W)

    def test_empty_batch_rejected(self):
        with self.assertRaises(ValueError):
            lc.svm_loss_vectorized(torch.zeros(2, 3), torch.empty(0, 2), torch.empty(0, dtype=torch.long))

    def test_training_reproducibility(self):
        from dlcv2026 import make_toy_classification
        data = make_toy_classification(seed=11)
        args = (lc.svm_loss_vectorized, data['X_train'], data['y_train'])
        first = lc.train_linear_classifier(*args, num_iters=100, seed=5)
        second = lc.train_linear_classifier(*args, num_iters=100, seed=5)
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        self.assertEqual(first[1], second[1])
        pred = lc.predict_linear_classifier(first[0], data['X_val'])
        self.assertGreater(lc.accuracy(pred, data['y_val']), 0.85)


if __name__ == '__main__':
    unittest.main()
