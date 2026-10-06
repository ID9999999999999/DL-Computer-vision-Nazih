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
        for loss_fn in (lc.svm_loss_naive, lc.svm_loss_vectorized):
            with self.subTest(function=loss_fn.__name__), self.assertRaises(ValueError):
                loss_fn(torch.zeros(2, 3), torch.empty(0, 2), torch.empty(0, dtype=torch.long))

    def test_invalid_labels_rejected(self):
        X = torch.ones(2, 2)
        W = torch.zeros(2, 3)
        for y in (torch.tensor([-1, 0]), torch.tensor([0, 3]),
                  torch.tensor([False, True]), torch.tensor([0.0, 1.0])):
            for loss_fn in (lc.svm_loss_naive, lc.svm_loss_vectorized):
                with self.subTest(y=y, function=loss_fn.__name__), self.assertRaises(ValueError):
                    loss_fn(W, X, y)

    def test_training_copies_initial_weights(self):
        X = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        y = torch.tensor([0, 1])
        initial = torch.zeros(2, 2, requires_grad=True)
        before = initial.detach().clone()
        trained, history = lc.train_linear_classifier(
            lc.svm_loss_vectorized, X, y, W=initial, num_iters=3
        )
        torch.testing.assert_close(initial.detach(), before, rtol=0, atol=0)
        self.assertFalse(trained.requires_grad)
        self.assertFalse(torch.equal(trained, before))
        self.assertEqual(len(history), 3)

    def test_empty_helpers_rejected(self):
        X, y = torch.empty(0, 2), torch.empty(0, dtype=torch.long)
        for operation in (
            lambda: lc.sample_batch(X, y, 2),
            lambda: lc.train_linear_classifier(lc.svm_loss_vectorized, X, y),
            lambda: lc.accuracy(y, y),
        ):
            with self.assertRaises(ValueError):
                operation()

    def test_invalid_training_parameters(self):
        X, y = torch.ones(2, 2), torch.tensor([0, 1])
        for params in ({'num_iters': -1}, {'batch_size': 0},
                       {'learning_rate': 0}, {'reg': -0.1}, {'reg': float('nan')}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                lc.train_linear_classifier(lc.svm_loss_vectorized, X, y, **params)

    def test_gradient_check_restores_weights_on_failure(self):
        from dlcv2026 import grad_check_sparse
        W = torch.ones(2, 2, dtype=torch.float64)
        before = W.clone()

        def failing_objective(weights):
            raise RuntimeError("Failed objective")

        with self.assertRaisesRegex(RuntimeError, "Failed objective"):
            grad_check_sparse(failing_objective, W, torch.zeros_like(W))
        torch.testing.assert_close(W, before, rtol=0, atol=0)

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
