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

# Logistic-linear function
def logistic_function(w, x):
    z = np.dot(x, w)
    return 1 / (1 + np.exp(-z))

# Negative log-likelihood function
def neg_log_likelihood(w, X, y):
    y_pred = logistic_function(w, X)
    return -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Train logistic-linear classifier and evaluate performance
for i, (X_train, y_train) in enumerate(datasets):
    # Add bias term to features
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    
    # Initialize weights
    w_init = np.zeros(X_train.shape[1])
    
    # Optimize negative log-likelihood
    result = minimize(neg_log_likelihood, w_init, args=(X_train, y_train), method='BFGS')
    w_opt = result.x
    
    # Apply classifier to validation set D10K_validate
    X_validate = np.hstack((np.ones((D10K_validate_samples.shape[0], 1)), D10K_validate_samples))
    scores = logistic_function(w_opt, X_validate)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(D10K_validate_labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'Logistic-Linear Classifier (D{num_samples_list[i]}_train, AUC = {roc_auc:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic-Linear Classifiers')
plt.legend()
plt.grid()
plt.show()

# Plot decision boundary for each classifier
x_min, x_max = D10K_validate_samples[:, 0].min() - 1, D10K_validate_samples[:, 0].max() + 1
y_min, y_max = D10K_validate_samples[:, 1].min() - 1, D10K_validate_samples[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

for i, (X_train, y_train) in enumerate(datasets):
    # Apply classifier to the mesh grid
    X_mesh = np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()]
    Z = logistic_function(w_opt, X_mesh)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
    plt.scatter(X_train[:, 1], X_train[:, 2], c=y_train, cmap='bwr', edgecolor='k', alpha=0.5) if X_train.shape[1] > 2 else plt.scatter(X_train[:, 1], [0] * X_train.shape[0], c=y_train, cmap='bwr', edgecolor='k', alpha=0.5)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Decision Boundary for Logistic-Linear Classifier (D{num_samples_list[i]}_train)')
    plt.grid()
    plt.show()
