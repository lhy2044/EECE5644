import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Define parameters
m01 = np.array([-0.9, -1.1])
m02 = np.array([0.8, 0.75])
m11 = np.array([-1.1, 0.9])
m12 = np.array([0.9, -0.75])

C = np.array([[0.75, 0], [0, 1.25]])

P_L0 = 0.6
P_L1 = 0.4
w01 = w02 = w11 = w12 = 0.5

# Gaussian PDF function
def gaussian_pdf(x, mean, cov):
    return multivariate_normal.pdf(x, mean=mean, cov=cov)

# Theoretically optimal classifier
def optimal_classifier(x):
    p_x_L0 = w01 * gaussian_pdf(x, m01, C) + w02 * gaussian_pdf(x, m02, C)
    p_x_L1 = w11 * gaussian_pdf(x, m11, C) + w12 * gaussian_pdf(x, m12, C)
    
    # Posterior probabilities (unnormalized)
    posterior_L0 = P_L0 * p_x_L0
    posterior_L1 = P_L1 * p_x_L1
    
    # Discriminant score
    return posterior_L1 / (posterior_L0 + posterior_L1)

# Generate validation dataset D10K_validate
np.random.seed(44)
num_samples = 10000

# Sample generation function
def generate_samples(num_samples, means, cov, weights, label):
    samples = []
    labels = []
    for _ in range(num_samples):
        # Choose component based on weights
        component = np.random.choice(len(means), p=weights)
        sample = np.random.multivariate_normal(means[component], cov)
        samples.append(sample)
        labels.append(label)
    return np.array(samples), np.array(labels)

means_L0 = [m01, m02]
means_L1 = [m11, m12]
weights = [0.5, 0.5]

samples_L0, labels_L0 = generate_samples(num_samples // 2, means_L0, C, weights, label=0)
samples_L1, labels_L1 = generate_samples(num_samples // 2, means_L1, C, weights, label=1)

D10K_validate_samples = np.vstack((samples_L0, samples_L1))
D10K_validate_labels = np.hstack((labels_L0, labels_L1))

# Apply classifier to validation set
scores = [optimal_classifier(x) for x in D10K_validate_samples]

# Calculate ROC curve
fpr, tpr, _ = roc_curve(D10K_validate_labels, scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label=f'Theoretically Optimal Classifier (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Theoretically Optimal Classifier')
plt.legend()
plt.grid()
plt.show()

# Plot decision boundary
x_min, x_max = D10K_validate_samples[:, 0].min() - 1, D10K_validate_samples[:, 0].max() + 1
y_min, y_max = D10K_validate_samples[:, 1].min() - 1, D10K_validate_samples[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

# Compute decision for each point in the mesh
mesh_samples = np.c_[xx.ravel(), yy.ravel()]
mesh_scores = [optimal_classifier(x) for x in mesh_samples]
mesh_scores = np.array(mesh_scores).reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, mesh_scores, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
plt.scatter(D10K_validate_samples[:, 0], D10K_validate_samples[:, 1], c=D10K_validate_labels, cmap='bwr', edgecolor='k', alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary for Theoretically Optimal Classifier')
plt.grid()
plt.show()
