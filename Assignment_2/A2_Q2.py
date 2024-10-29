import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from hw2q2 import hw2q2

# Generate dataset using hw2q2
xTrain, yTrain, xValidate, yValidate = hw2q2()

# Separate features and labels for training and validation
x_train = xTrain[0, :]
y_train = yTrain
x_validate = xValidate[0, :]
y_validate = yValidate

# Initial guess for weights
w_init = np.zeros(4)

# Cubic polynomial model
def cubic_model(x, w):
    return w[0] + w[1] * x + w[2] * x**2 + w[3] * x**3

# Negative log-likelihood function for ML estimation
def neg_log_likelihood(w, x, y):
    y_pred = cubic_model(x, w)
    residuals = y - y_pred
    return np.sum(residuals**2)

# Negative log-posterior for MAP estimation (Gaussian prior)
def neg_log_posterior(w, x, y, gamma):
    log_prior = np.sum(w**2) / (2 * gamma**2)  # Gaussian prior with mean 0 and variance gamma^2
    log_likelihood = neg_log_likelihood(w, x, y)
    return log_likelihood + log_prior

# Maximum Likelihood Estimation (MLE)
result_ml = minimize(neg_log_likelihood, w_init, args=(x_train, y_train), method='BFGS')
w_ml = result_ml.x
print(f"ML Estimate of w: {w_ml}")

# Maximum A Posteriori Estimation (MAP)
gamma_values = [0.1, 1, 10, 100]  # Different values for prior variance
map_results = []
for gamma in gamma_values:
    result_map = minimize(neg_log_posterior, w_init, args=(x_train, y_train, gamma), method='BFGS')
    w_map = result_map.x
    map_results.append((gamma, w_map))
    print(f"MAP Estimate of w for gamma={gamma}: {w_map}")

# Plotting results
plt.figure(figsize=(10, 8))
plt.scatter(x_train, y_train, label='Training Data', color='gray', alpha=0.5)

# Plot ML fit
y_ml = cubic_model(x_train, w_ml)
plt.plot(x_train, y_ml, label='ML Fit', color='blue')

# Plot MAP fits for different gamma values
colors = ['red', 'green', 'orange', 'purple']
for (gamma, w_map), color in zip(map_results, colors):
    y_map = cubic_model(x_train, w_map)
    plt.plot(x_train, y_map, label=f'MAP Fit (gamma={gamma})', color=color)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('ML and MAP Fits for Cubic Polynomial Model')
plt.grid(True)
plt.show()
