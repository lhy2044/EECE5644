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

# Naive Bayesian Classifier with Diagonal Covariance Matrices
C0_naive = np.diag(np.diag(C0))
C1_naive = np.diag(np.diag(C1))

rv0_naive = multivariate_normal(mean=m0, cov=C0_naive)
rv1_naive = multivariate_normal(mean=m1, cov=C1_naive)

gammas = np.linspace(0, 20, 1000)
tpr_naive = []
fpr_naive = []

start_time = time.time()
for gamma in tqdm(gammas, desc="Evaluating gammas (Naive Bayes)"):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    for i, x in enumerate(samples):
        likelihood_ratio = rv1_naive.pdf(x) / rv0_naive.pdf(x)
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
    
    tpr_naive.append(tp / (tp + fn))
    fpr_naive.append(fp / (fp + tn))

elapsed_time = time.time() - start_time
print(f"Naive Bayes evaluation completed in {elapsed_time:.2f} seconds.")

# Plotting the ROC curve for Naive Bayesian Classifier
plt.plot(fpr_naive, tpr_naive, label='Naive Bayes ROC Curve', color='orange')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Naive Bayesian Classifier')
plt.grid()
plt.legend()
plt.show()

# Finding the Threshold for Minimum Probability of Error (Naive Bayes)
min_error_naive = float('inf')
optimal_gamma_naive = None
min_error_tpr_naive = None
min_error_fpr_naive = None

for i, gamma in enumerate(gammas):
    p_error = fpr_naive[i] * P_L0 + (1 - tpr_naive[i]) * P_L1
    if p_error < min_error_naive:
        min_error_naive = p_error
        optimal_gamma_naive = gamma
        min_error_tpr_naive = tpr_naive[i]
        min_error_fpr_naive = fpr_naive[i]

print(f"Naive Bayes minimum probability of error: {min_error_naive:.4f} at gamma = {optimal_gamma_naive:.4f}")

# Superimposing the point of minimum probability of error on the Naive Bayes ROC curve
plt.plot(fpr_naive, tpr_naive, label='Naive Bayes ROC Curve', color='orange')
plt.scatter(min_error_fpr_naive, min_error_tpr_naive, color='red', marker='o', label='Naive Bayes Min P(error)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayesian ROC Curve with Minimum Probability of Error')
plt.grid()
plt.legend()
plt.show()