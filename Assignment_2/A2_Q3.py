import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define parameters
np.random.seed(44)
sigma_x = 0.25  # Standard deviation for prior on x
sigma_y = 0.25  # Standard deviation for prior on y
sigma_r = 0.3   # Standard deviation for range measurements

K_values = [1, 2, 3, 4]  # Number of reference points

# True vehicle position
x_true = 0.5
y_true = -0.5

# Generate reference points (evenly spaced on a unit circle)
angles = np.linspace(0, 2 * np.pi, max(K_values), endpoint=False)
x_refs = np.cos(angles)
y_refs = np.sin(angles)

# Generate range measurements with noise
r_true = np.sqrt((x_true - x_refs)**2 + (y_true - y_refs)**2)
r_meas = r_true + np.random.normal(0, sigma_r, size=r_true.shape)

# Define the MAP objective function
def map_objective(position, x_refs, y_refs, r_meas, sigma_r, sigma_x, sigma_y):
    x, y = position
    prior_term = (x**2) / (2 * sigma_x**2) + (y**2) / (2 * sigma_y**2)
    likelihood_term = np.sum(((r_meas - np.sqrt((x - x_refs)**2 + (y - y_refs)**2))**2) / (2 * sigma_r**2))
    return prior_term + likelihood_term

# Perform MAP estimation for different values of K
for K in K_values:
    x_refs_K = x_refs[:K]
    y_refs_K = y_refs[:K]
    r_meas_K = r_meas[:K]
    
    # Initial guess for [x, y]
    initial_guess = [0.0, 0.0]
    
    # Minimize the MAP objective
    result = minimize(map_objective, initial_guess, args=(x_refs_K, y_refs_K, r_meas_K, sigma_r, sigma_x, sigma_y), method='BFGS')
    x_map, y_map = result.x
    
    # Plot the results
    plt.figure(figsize=(6, 6))
    plt.scatter(x_refs_K, y_refs_K, color='blue', marker='o', label='Reference Points')
    plt.scatter(x_true, y_true, color='green', marker='+', s=100, label='True Position')
    plt.scatter(x_map, y_map, color='red', marker='x', s=100, label='MAP Estimate')
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.title(f'MAP Estimate with K={K} Reference Points')
    plt.legend()
    plt.grid(True)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
