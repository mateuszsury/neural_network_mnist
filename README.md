# MNIST Softmax Classifier

An educational single-layer classifier implemented with NumPy gradients.
Despite the historical repository name, this is multiclass logistic regression
with a softmax output, not a deep or hidden-layer neural network.

```sh
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -r requirements.txt
python neural_network_mnist.py --epochs 20 --seed 42
python -m unittest -v
```

Use a Python version supported by your TensorFlow release (Python 3.11 is a
reference environment). TensorFlow only loads MNIST; the training math is NumPy.
Training normalizes pixels to [0, 1], averages batch gradients and includes the
last partial batch. An 80/20 training/validation split and the untouched MNIST
test set produce separately reported cross-entropy and accuracy.

Importing the module or running its unit tests does not download MNIST or train
the full dataset. CI exercises numerical stability and learning on a tiny
synthetic fixture. No full-dataset accuracy is claimed without a recorded run.

Maintainer: Mateusz Sury. Original source: MIT. MNIST retains its original data
provenance and terms.

## Project overview

A seeded NumPy single-layer classifier for MNIST with explicit training and held-out evaluation.

![MNIST Softmax Classifier — This is a single-layer softmax model. The diagram is not a training result or an accuracy claim.](docs/assets/presentation/overview.svg)

This is a single-layer softmax model. The diagram is not a training result or an accuracy claim.
