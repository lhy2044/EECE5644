import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Generate training datasets D20_train, D200_train, D2000_train
np.random.seed(44)

num_samples_list = [20, 200, 2000]
means_L0 = [np.array([-0.9, -1.1]), np.array([0.8, 0.75])]
means_L1 = [np.array([-1.1, 0.9]), np.array([0.9, -0.75])]
C = np.array([[0.75, 0], [0, 1.25]])
weights = [0.5, 0.5]

# Sample generation function
def generate_samples(num_samples, means, cov, weights, label):
    samples = []
    labels = []
    for _ in range(num_samples):
        component = np.random.choice(len(means), p=weights)
        sample = np.random.multivariate_normal(means[component], cov)
        samples.append(sample)
        labels.append(label)
    return np.array(samples), np.array(labels)

# Generate datasets
datasets = []
for num_samples in num_samples_list:
    samples_L0, labels_L0 = generate_samples(num_samples // 2, means_L0, C, weights, label=0)
    samples_L1, labels_L1 = generate_samples(num_samples // 2, means_L1, C, weights, label=1)
    samples = np.vstack((samples_L0, samples_L1))
    labels = np.hstack((labels_L0, labels_L1))
    datasets.append((samples, labels))

# Generate validation dataset D10K_validate
num_samples_validate = 10000
samples_L0_validate, labels_L0_validate = generate_samples(num_samples_validate // 2, means_L0, C, weights, label=0)
samples_L1_validate, labels_L1_validate = generate_samples(num_samples_validate // 2, means_L1, C, weights, label=1)
D10K_validate_samples = np.vstack((samples_L0_validate, samples_L1_validate))
D10K_validate_labels = np.hstack((labels_L0_validate, labels_L1_validate))

# Logistic-quadratic function
def logistic_function_quadratic(w, x):
    z = np.dot(x, w)
    return 1 / (1 + np.exp(-z))

# Negative log-likelihood function for quadratic logistic regression
def neg_log_likelihood_quadratic(w, X, y):
    y_pred = logistic_function_quadratic(w, X)
    return -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Train logistic-quadratic classifier and evaluate performance
for i, (X_train, y_train) in enumerate(datasets):
    # Add quadratic features and bias term to features
    X_train_quad = np.hstack((
        np.ones((X_train.shape[0], 1)),
        X_train,
        X_train[:, 0:1] ** 2,
        X_train[:, 0:1] * X_train[:, 1:2],
        X_train[:, 1:2] ** 2
    ))
    
    # Initialize weights
    w_init = np.zeros(X_train_quad.shape[1])
    
    # Optimize negative log-likelihood
    result = minimize(neg_log_likelihood_quadratic, w_init, args=(X_train_quad, y_train), method='BFGS')
    w_opt = result.x
    
    # Apply classifier to validation set D10K_validate
    X_validate_quad = np.hstack((
        np.ones((D10K_validate_samples.shape[0], 1)),
        D10K_validate_samples,
        D10K_validate_samples[:, 0:1] ** 2,
        D10K_validate_samples[:, 0:1] * D10K_validate_samples[:, 1:2],
        D10K_validate_samples[:, 1:2] ** 2
    ))
    scores = logistic_function_quadratic(w_opt, X_validate_quad)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(D10K_validate_labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'Logistic-Quadratic Classifier (D{num_samples_list[i]}_train, AUC = {roc_auc:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic-Quadratic Classifiers')
plt.legend()
plt.grid()
plt.show()
