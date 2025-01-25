import numpy as np

# Example forward pass
def forward_pass(X, weights):
    return 1 / (1 + np.exp(-np.dot(X, weights)))  # Sigmoid activation

# Example backward pass (backpropagation)
def backward_pass(X, y, y_pred, weights, learning_rate):
    # Compute the gradient of the loss
    error = y_pred - y
    gradient = np.dot(X.T, error) / X.shape[0]
    
    # Update weights
    weights -= learning_rate * gradient
    return weights

# Initialize data and weights
np.random.seed(0)
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = np.random.randint(0, 2, 100)  # Binary labels
weights = np.random.rand(3)  # Weights for 3 features
learning_rate = 0.01

# Training loop
for _ in range(1000):
    y_pred = forward_pass(X, weights)
    weights = backward_pass(X, y, y_pred, weights, learning_rate)

print("Updated Weights:", weights)

# IMRPOVEMENTS :

# Dynamic Learning Rate (Learning Rate Schedule): Replace the fixed learning rate with an adaptive one (e.g., using Adam optimizer or reducing the rate over time).
# Use Vectorized Operations: Ensure all operations are fully vectorized using libraries like NumPy or TensorFlow to minimize computational overhead.
# Gradient Clipping: Avoid exploding gradients by clipping them to a maximum norm.
# Batch Updates: Switch from full-batch updates to mini-batch updates to stabilize the training process.