import numpy as np
import matplotlib.pyplot as plt

# Adam parameters
alpha = 0.1        # Learning rate
beta1 = 0.9        # Decay rate for first moment
beta2 = 0.999      # Decay rate for second moment
epsilon = 1e-8     # Small constant to avoid division by zero

# Dummy gradients over steps
grads = [0.1, 0.2, -0.1, -0.3, 0.05]
w = 1.0  # Initial weight

# Moment estimates
m = 0
v = 0

# Tracking history for plotting
m_hist = []
v_hist = []
mhat_hist = []
vhat_hist = []
step_hist = []
w_hist = [w]

# Simulation loop
for t, g in enumerate(grads, start=1):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    step_size = alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    w -= step_size

    # Store values
    m_hist.append(m)
    v_hist.append(v)
    mhat_hist.append(m_hat)
    vhat_hist.append(v_hat)
    step_hist.append(step_size)
    w_hist.append(w)

    # Print like earlier
    print(f"Step {t}")
    print(f"Gradient: {g}")
    print(f"m (1st moment): {m:.6f}")
    print(f"v (2nd moment): {v:.6f}")
    print(f"m_hat (bias-corrected): {m_hat:.6f}")
    print(f"v_hat (bias-corrected): {v_hat:.6f}")
    print(f"Updated w: {w:.6f}")
    print("-" * 40)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

# First & second moments
axs[0].plot(m_hist, label='m_t (momentum)')
axs[0].plot(v_hist, label='v_t (squared gradients)')
axs[0].set_title('Moment Estimates Over Time')
axs[0].legend()
axs[0].grid(True)

# Bias corrected moments
axs[1].plot(mhat_hist, label='m_hat (bias corrected momentum)')
axs[1].plot(vhat_hist, label='v_hat (bias corrected squared grads)')
axs[1].set_title('Bias-Corrected Moments')
axs[1].legend()
axs[1].grid(True)

# Step sizes
axs[2].plot(step_hist, label='Step size applied')
axs[2].set_title('Step Size Per Iteration')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()