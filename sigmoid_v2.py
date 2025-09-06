import numpy as np

# XOR data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Parameters
np.random.seed(7)
hidden_neurons = 4
learning_rate = 0.2
epochs = 50000  # train longer

# Initialize weights and biases
W1 = np.random.randn(2, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))
W2 = np.random.randn(hidden_neurons, 1)
b2 = np.zeros((1, 1))

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# Loss function
def binary_cross_entropy(y_true, y_pred):
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

# Training loop
losses = []
for epoch in range(epochs):
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

    # Update weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Print every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss:.6f}")

# Final predictions
print("\nFinal predictions:\n", np.round(A2, 3))
print("Final loss:", losses[-1])
