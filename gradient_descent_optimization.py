def gradient_descent(X, y, weights, learning_rate, iterations):
    for i in range(iterations):
        y_pred = forward_pass(X, weights)
        gradient = np.dot(X.T, (y_pred - y)) / len(y)
        weights -= learning_rate * gradient  # Update weights
    return weights

# Example call
weights = gradient_descent(X, y, weights, learning_rate=0.01, iterations=1000)

#IMPROVEMENTS :

# Momentum: Incorporate a momentum term to accelerate convergence.
# Adam Optimizer: Use Adam, which combines momentum and RMSprop, to dynamically adjust the learning rate.
# Adaptive Mini-Batch Size: Start with smaller batches and increase the size as training progresses.
# Gradient Preconditioning: Normalize input features to ensure gradients aren't disproportionately scaled.