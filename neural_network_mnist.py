import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf


#załadowanie bazy danych MNIST z wykorzystaniem TensorFlow
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#przetworzenie danych na wektory cech
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

#przetworzenie etykiet do postaci one-hot encoding
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)

#podział danych na zbiór treningowy i testowy
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#inicjalizacja wag i biasów
np.random.seed(42)
weights = np.random.randn(10, 784)
bias = np.zeros(10)

#funkcja aktywacji - softmax
def softmax(x):
    max_val = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_val)
    return exps / np.sum(exps, axis=1, keepdims=True)

#funkcja straty - cross-entropy
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

#funkcja dokładności
def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

#parametry uczenia
learning_rate = 0.001
epochs = 1000
batch_size = 64
num_batches = X_train.shape[0] // batch_size

#trening modelu
for epoch in range(epochs):
    for batch in range(num_batches):
        #przygotowanie batcha danych treningowych
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]

        #propagacja w przód
        logits = np.dot(X_batch, weights.T) + bias
        y_pred = softmax(logits)

        #obliczenie straty i dokładności
        loss = cross_entropy_loss(y_batch, y_pred)
        acc = accuracy(y_batch, y_pred)

        #propagacja wsteczna
        grad_logits = y_pred - y_batch
        grad_weights = np.dot(grad_logits.T, X_batch)
        grad_bias = np.sum(grad_logits, axis=0)

        #aktualizacja wag i biasów
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias

    #ocena modelu na zbiorze walidacyjnym
    val_logits = np.dot(X_val, weights.T) + bias
    val_pred = softmax(val_logits)
    val_loss = cross_entropy_loss(y_val, val_pred)
    val_acc = accuracy(y_val, val_pred)

    #wypisanie metryk
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {acc:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.4f}")
