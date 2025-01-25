def dropout_forward(X, dropout_rate):
    mask = np.random.binomial(1, 1 - dropout_rate, size=X.shape)
    return X * mask  # Drop neurons

# Example forward pass with dropout
dropout_rate = 0.5
X_dropped = dropout_forward(X, dropout_rate)

# IMRPOVEMENTS :

# Scaled Dropout: Scale the activations during training to maintain consistent output magnitudes.
# Layer-Specific Dropout: Use different dropout rates for different layers.
# DropConnect: Randomly drop weights instead of neurons to improve generalization.
# Attention Mechanisms: Combine dropout with attention to selectively focus on important features.