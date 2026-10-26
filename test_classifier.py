import unittest
import numpy as np
from neural_network_mnist import softmax, cross_entropy_loss, train

class ClassifierTests(unittest.TestCase):
    def test_softmax_stable_and_normalized(self):
        result = softmax(np.array([[10000., 10001.], [-10000., -10001.]]))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result.sum(axis=1), 1)

    def test_learning_reduces_loss_on_separable_data(self):
        x = np.array([[1., 0.], [0., 1.]] * 20)
        labels = np.array([0, 1] * 20)
        expected = np.eye(10)[labels]
        weights, bias = train(x, labels, epochs=100, learning_rate=0.5)
        self.assertLess(cross_entropy_loss(expected, softmax(x @ weights.T + bias)), 0.2)

if __name__ == '__main__':
    unittest.main()
