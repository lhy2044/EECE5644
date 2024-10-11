import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(44)

# Define the Gaussian parameters for each class
means = [
    np.array([0, 0, 0]),   # Mean for class 1
    np.array([3, 3, 3]),   # Mean for class 2
    np.array([-3, -3, -3]), # Mean for one component of class 3
    np.array([3, -3, 0])   # Mean for another component of class 3
]

covariances = [
    np.eye(3),  # Covariance for class 1
    np.eye(3),  # Covariance for class 2
    np.eye(3),  # Covariance for one component of class 3
    np.eye(3)   # Covariance for another component of class 3
]

priors = [0.3, 0.3, 0.4]

# Generate samples
n_samples = 10000
samples = []
labels = []

for _ in range(n_samples):
    # Choose a class based on the priors
    class_label = np.random.choice([1, 2, 3], p=priors)
    if class_label == 1:
        sample = np.random.multivariate_normal(means[0], covariances[0])
    elif class_label == 2:
        sample = np.random.multivariate_normal(means[1], covariances[1])
    else:
        # For class 3, choose one of the two components with equal probability
        component = np.random.choice([2, 3])
        sample = np.random.multivariate_normal(means[component], covariances[component])
    
    samples.append(sample)
    labels.append(class_label)

samples = np.array(samples)
labels = np.array(labels)

print("Data generation complete.")

# Define the Gaussian distributions
rv1 = multivariate_normal(mean=means[0], cov=covariances[0])
rv2 = multivariate_normal(mean=means[1], cov=covariances[1])
rv3_component1 = multivariate_normal(mean=means[2], cov=covariances[2])
rv3_component2 = multivariate_normal(mean=means[3], cov=covariances[3])

# Loss matrices for ERM classification
loss_10 = np.array([[0, 10, 10],
                    [1, 0, 10],
                    [1, 1, 0]])

loss_100 = np.array([[0, 100, 100],
                     [1, 0, 100],
                     [1, 1, 0]])

def erm_classification(samples, rv1, rv2, rv3_component1, rv3_component2, priors, loss_matrix):
    predicted_labels = []
    for sample in samples:
        # Calculate the posterior expected risk for each class
        p_x_given_1 = rv1.pdf(sample) * priors[0]
        p_x_given_2 = rv2.pdf(sample) * priors[1]
        p_x_given_3 = 0.5 * (rv3_component1.pdf(sample) + rv3_component2.pdf(sample)) * priors[2]
        posteriors = [p_x_given_1, p_x_given_2, p_x_given_3]

        # Calculate the risk for each decision
        risks = [
            sum([loss_matrix[i, j] * posteriors[j] for j in range(3)])
            for i in range(3)
        ]

        # Assign the class with the minimum risk
        predicted_class = np.argmin(risks) + 1
        predicted_labels.append(predicted_class)
    
    return np.array(predicted_labels)

# Classify using ERM with loss matrix 10
predicted_labels_10 = erm_classification(samples, rv1, rv2, rv3_component1, rv3_component2, priors, loss_10)

# Classify using ERM with loss matrix 100
predicted_labels_100 = erm_classification(samples, rv1, rv2, rv3_component1, rv3_component2, priors, loss_100)

# Confusion matrix estimation for loss matrix 10
confusion_matrix_10 = np.zeros((3, 3))
for true_label, predicted_label in zip(labels, predicted_labels_10):
    confusion_matrix_10[true_label - 1, predicted_label - 1] += 1

print("Confusion Matrix for Loss Matrix 10:")
print(confusion_matrix_10)

# Confusion matrix estimation for loss matrix 100
confusion_matrix_100 = np.zeros((3, 3))
for true_label, predicted_label in zip(labels, predicted_labels_100):
    confusion_matrix_100[true_label - 1, predicted_label - 1] += 1

print("Confusion Matrix for Loss Matrix 100:")
print(confusion_matrix_100)

# Visualization of the results for loss matrix 10
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = {1: 'g', 2: 'b', 3: 'r'}
markers = {1: 'o', 2: '^', 3: 's'}

for i in range(len(samples)):
    color = 'green' if labels[i] == predicted_labels_10[i] else 'red'
    marker = markers[labels[i]]
    ax.scatter(samples[i, 0], samples[i, 1], samples[i, 2], c=color, marker=marker, alpha=0.5)

ax.set_title("3D Scatter Plot of Classification Results (Loss Matrix 10)")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.show()

# Visualization of the results for loss matrix 100
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(samples)):
    color = 'green' if labels[i] == predicted_labels_100[i] else 'red'
    marker = markers[labels[i]]
    ax.scatter(samples[i, 0], samples[i, 1], samples[i, 2], c=color, marker=marker, alpha=0.5)

ax.set_title("3D Scatter Plot of Classification Results (Loss Matrix 100)")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.show()