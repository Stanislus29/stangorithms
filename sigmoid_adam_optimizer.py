import numpy as np

# XOR data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Parameters
np.random.seed(42)
hidden_neurons = 4
learning_rate = 0.05
epochs = 20000

# Initialize weights and biases
W1 = np.random.randn(2, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))
W2 = np.random.randn(hidden_neurons, 1)
b2 = np.zeros((1, 1))

# Adam hyperparameters
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Initialize Adam moments
mW1 = np.zeros_like(W1)
vW1 = np.zeros_like(W1)
mb1 = np.zeros_like(b1)
vb1 = np.zeros_like(b1)
mW2 = np.zeros_like(W2)
vW2 = np.zeros_like(W2)
mb2 = np.zeros_like(b2)
vb2 = np.zeros_like(b2)

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# Loss function
def binary_cross_entropy(y_true, y_pred):
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

losses = []
for t in range(1, epochs + 1):
    # Forward pass
    Z1 = X.dot(W1) + b1
    A1 = sigmoid(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = sigmoid(Z2)

    # Compute loss
    loss = binary_cross_entropy(y, A2)
    losses.append(loss)

    # Backpropagation
    dZ2 = A2 - y
    dW2 = A1.T.dot(dZ2) / X.shape[0]
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = X.T.dot(dZ1) / X.shape[0]
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    # --- Adam update for each parameter ---
    for param, grad, m, v in [
        (W1, dW1, mW1, vW1),
        (b1, db1, mb1, vb1),
        (W2, dW2, mW2, vW2),
        (b2, db2, mb2, vb2)
    ]:
        m[:] = beta1 * m + (1 - beta1) * grad
        v[:] = beta2 * v + (1 - beta2) * (grad ** 2)

        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    if t % 2000 == 0:
        print(f"Epoch {t} | Loss: {loss:.6f}")

# Final predictions
print("\nFinal predictions:\n", np.round(A2, 3))
print("Final loss:", losses[-1])
