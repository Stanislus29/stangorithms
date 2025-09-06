import numpy as np

# Hyperparameters
alpha = 0.1      # learning rate
beta1 = 0.9
beta2 = 0.999
eps = 1e-8

# Initial values
m, v = 0.0, 0.0  # first and second moment estimates
w = 1.0          # our single parameter
gradients = [0.1, 0.2, -0.1, -0.3, 0.05]  # fake gradients

for t, g in enumerate(gradients, start=1):
    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * g
    # Update biased second moment estimate
    v = beta2 * v + (1 - beta2) * (g ** 2)
    
    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    # Parameter update
    w = w - alpha * m_hat / (np.sqrt(v_hat) + eps)
    
    print(f"Step {t}")
    print(f"Gradient: {g}")
    print(f"m (1st moment): {m:.6f}")
    print(f"v (2nd moment): {v:.6f}")
    print(f"m_hat (bias-corrected): {m_hat:.6f}")
    print(f"v_hat (bias-corrected): {v_hat:.6f}")
    print(f"Updated w: {w:.6f}")
    print("-" * 40)