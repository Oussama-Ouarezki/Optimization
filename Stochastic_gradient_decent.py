import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic linear data
np.random.seed(42)
x = np.linspace(0, 10, 100)
m_true = 2.5  # True slope
c_true = 3.0  # True intercept
y = m_true * x + c_true + np.random.normal(0, 2, size=len(x))  # Add noise

# Fix the intercept
c_fixed = c_true  # Use the true intercept (or any other fixed value)

# Define the loss function (Mean Squared Error)
def mse_loss(m, c, x, y):
    y_pred = m * x + c
    return np.mean((y - y_pred) ** 2)

# Generate a range of m values
m_values = np.linspace(0, 5, 100)

# Compute the loss for each m
loss_values = [mse_loss(m, c_fixed, x, y) for m in m_values]

# Plot the loss function
plt.figure(figsize=(8, 5))
plt.plot(m_values, loss_values, label="Loss Function", color="blue", linewidth=2)
plt.axvline(m_true, color="red", linestyle="--", label=f"True Slope (m = {m_true})")
plt.xlabel("Slope (m)")
plt.ylabel("Loss (MSE)")
plt.title(f"Loss Function for Fixed Intercept c = {c_fixed}")
plt.legend()
plt.grid()
plt.show()