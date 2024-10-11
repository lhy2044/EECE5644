import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from tqdm import tqdm
import time

# Parameters for the Gaussian distributions
m0 = np.array([-1, -1, -1, -1])
C0 = np.array([[2, -0.5, 0.3, 0],
               [-0.5, 1, -0.5, 0],
               [0.3, -0.5, 1, 0],
               [0, 0, 0, 2]])

m1 = np.array([1, 1, 1, 1])
C1 = np.array([[1, 0.3, -0.2, 0],
               [0.3, 2, 0.3, 0],
               [-0.2, 0.3, 1, 0],
               [0, 0, 0, 3]])

P_L0 = 0.35
P_L1 = 0.65

# Generate 10,000 samples
n_samples = 10000
labels = np.random.choice([0, 1], size=n_samples, p=[P_L0, P_L1])
print("Labels generated.")

samples = []
for label in tqdm(labels, desc="Generating samples"):
    if label == 0:
        sample = np.random.multivariate_normal(m0, C0)
        samples.append(sample)
    else:
        sample = np.random.multivariate_normal(m1, C1)
        samples.append(sample)

samples = np.array(samples)
print("All samples generated.")

# Likelihood ratio test and ROC curve
rv0 = multivariate_normal(mean=m0, cov=C0)
rv1 = multivariate_normal(mean=m1, cov=C1)

gammas = np.linspace(0, 20, 1000)
tpr = []
fpr = []

start_time = time.time()
for gamma in tqdm(gammas, desc="Evaluating gammas"):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    for i, x in enumerate(samples):
        likelihood_ratio = rv1.pdf(x) / rv0.pdf(x)
        if likelihood_ratio > gamma:
            if labels[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if labels[i] == 0:
                tn += 1
            else:
                fn += 1
    
    tpr.append(tp / (tp + fn))
    fpr.append(fp / (fp + tn))

elapsed_time = time.time() - start_time
print(f"Evaluation completed in {elapsed_time:.2f} seconds.")

# Plotting the ROC curve
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Minimum Expected Risk Classifier')
plt.grid()
plt.legend()
plt.show()

# Part A, Question 3: Finding the Threshold for Minimum Probability of Error
min_error = float('inf')
optimal_gamma = None
min_error_tpr = None
min_error_fpr = None

for i, gamma in enumerate(gammas):
    p_error = fpr[i] * P_L0 + (1 - tpr[i]) * P_L1
    if p_error < min_error:
        min_error = p_error
        optimal_gamma = gamma
        min_error_tpr = tpr[i]
        min_error_fpr = fpr[i]

print(f"Minimum probability of error: {min_error:.4f} at gamma = {optimal_gamma:.4f}")

# Superimposing the point of minimum probability of error on the ROC curve
plt.plot(fpr, tpr, label='ROC Curve')
plt.scatter(min_error_fpr, min_error_tpr, color='red', marker='o', label='Min P(error)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Minimum Probability of Error')
plt.grid()
plt.legend()
plt.show()