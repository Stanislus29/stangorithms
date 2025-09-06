import numpy as np
import matplotlib.pyplot as plt

# Adam parameters
alpha = 0.1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Create noisy gradients: gradually decreasing toward zero
np.random.seed(42)
steps = 100
true_grad_trend = np.linspace(0.3, 0.0, steps)  # slow decrease
noise = np.random.normal(0, 0.05, steps)        # random noise
grads = true_grad_trend + noise

# Initialize
w = 1.0
m = 0
v = 0

# History tracking
m_hist, v_hist = [], []
mhat_hist, vhat_hist = [], []
step_hist, w_hist = [], []

# Simulation
for t, g in enumerate(grads, start=1):
    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * g
    # Update biased second raw moment estimate
    v = beta2 * v + (1 - beta2) * (g ** 2)
    
    # Compute bias-corrected estimates
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    # Compute step size
    step_size = alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    
    # Update parameter
    w -= step_size
    
    # Save history
    m_hist.append(m)
    v_hist.append(v)
    mhat_hist.append(m_hat)
    vhat_hist.append(v_hat)
    step_hist.append(step_size)
    w_hist.append(w)
    
    # Print first 5 steps for inspection
    if t <= 5:
        print(f"Step {t}")
        print(f"Gradient: {g:.6f}")
        print(f"m (1st moment): {m:.6f}")
        print(f"v (2nd moment): {v:.6f}")
        print(f"m_hat (bias-corrected): {m_hat:.6f}")
        print(f"v_hat (bias-corrected): {v_hat:.6f}")
        print(f"Step size: {step_size:.6f}")
        print(f"Updated w: {w:.6f}")
        print("-" * 40)

# Plot
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

# Plot raw moments
axs[0].plot(m_hist, label='m_t (momentum)')
axs[0].plot(v_hist, label='v_t (squared grad avg)')
axs[0].set_title('Raw Moments')
axs[0].legend()
axs[0].grid(True)

# Plot bias corrected moments
axs[1].plot(mhat_hist, label='m_hat')
axs[1].plot(vhat_hist, label='v_hat')
axs[1].set_title('Bias-Corrected Moments')
axs[1].legend()
axs[1].grid(True)

# Step size evolution
axs[2].plot(step_hist, label='Step size')
axs[2].set_title('Step Size Over Time')
axs[2].legend()
axs[2].grid(True)

# Weight updates
axs[3].plot(w_hist, label='Weight value')
axs[3].set_title('Weight Value Over Time')
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()