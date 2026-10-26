"""A seeded, single-layer softmax classifier for MNIST."""
import argparse
import numpy as np

def softmax(x):
    values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(np.clip(y_pred, 1e-9, 1)), axis=1))

def accuracy(y_true, y_pred):
    return np.mean(y_true.argmax(axis=1) == y_pred.argmax(axis=1))

def train(x, labels, epochs=20, learning_rate=0.1, batch_size=64, seed=42):
    rng = np.random.default_rng(seed)
    y = np.eye(10)[labels]
    weights = rng.normal(0, 0.01, (10, x.shape[1]))
    bias = np.zeros(10)
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            indices = order[start:start + batch_size]
            batch = x[indices]
            gradient = (softmax(batch @ weights.T + bias) - y[indices]) / len(indices)
            weights -= learning_rate * (gradient.T @ batch)
            bias -= learning_rate * gradient.sum(axis=0)
    return weights, bias

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    # Loading/training is explicit; importing the math helpers downloads nothing.
    from tensorflow.keras.datasets import mnist
    (x, labels), (x_test, test_labels) = mnist.load_data()
    x = x.reshape(-1, 784).astype(np.float64) / 255
    x_test = x_test.reshape(-1, 784).astype(np.float64) / 255
    order = np.random.default_rng(args.seed).permutation(len(x))
    split = int(len(x) * 0.8)
    training, validation = order[:split], order[split:]
    weights, bias = train(x[training], labels[training], args.epochs, seed=args.seed)
    for name, features, targets in [('validation', x[validation], labels[validation]),
                                     ('test', x_test, test_labels)]:
        expected = np.eye(10)[targets]
        predicted = softmax(features @ weights.T + bias)
        print(f'{name}: loss={cross_entropy_loss(expected, predicted):.4f} '
              f'accuracy={accuracy(expected, predicted):.4f}')

if __name__ == '__main__':
    main()
