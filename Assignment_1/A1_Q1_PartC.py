import numpy as np
import matplotlib.pyplot as plt
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

# Part C: Fisher Linear Discriminant Analysis (LDA) Classifier
# Estimating the class means and covariance matrices
mean_0_est = np.mean(samples[labels == 0], axis=0)
mean_1_est = np.mean(samples[labels == 1], axis=0)

cov_0_est = np.cov(samples[labels == 0], rowvar=False)
cov_1_est = np.cov(samples[labels == 1], rowvar=False)

# Within-class scatter matrix (SW) and between-class scatter matrix (SB)
SW = cov_0_est + cov_1_est
SB = np.outer(mean_1_est - mean_0_est, mean_1_est - mean_0_est)

# Finding the LDA projection vector
SW_inv = np.linalg.inv(SW)
w_LDA = SW_inv @ (mean_1_est - mean_0_est)

# Projecting the data onto the LDA vector
projections = samples @ w_LDA

tpr_lda = []
fpr_lda = []
thresh_lda = np.linspace(np.min(projections), np.max(projections), 1000)

start_time = time.time()
for tau in tqdm(thresh_lda, desc="Evaluating thresholds (LDA)"):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    for i, proj in enumerate(projections):
        if proj > tau:
            if labels[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if labels[i] == 0:
                tn += 1
            else:
                fn += 1
    
    tpr_lda.append(tp / (tp + fn))
    fpr_lda.append(fp / (fp + tn))

elapsed_time = time.time() - start_time
print(f"LDA evaluation completed in {elapsed_time:.2f} seconds.")

# Plotting the ROC curve for LDA Classifier
plt.plot(fpr_lda, tpr_lda, label='LDA ROC Curve', color='green')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for LDA Classifier')
plt.grid()
plt.legend()
plt.show()

# Finding the Threshold for Minimum Probability of Error (LDA)
min_error_lda = float('inf')
optimal_tau_lda = None
min_error_tpr_lda = None
min_error_fpr_lda = None

for i, tau in enumerate(thresh_lda):
    p_error = fpr_lda[i] * P_L0 + (1 - tpr_lda[i]) * P_L1
    if p_error < min_error_lda:
        min_error_lda = p_error
        optimal_tau_lda = tau
        min_error_tpr_lda = tpr_lda[i]
        min_error_fpr_lda = fpr_lda[i]

print(f"LDA minimum probability of error: {min_error_lda:.4f} at threshold = {optimal_tau_lda:.4f}")

# Superimposing the point of minimum probability of error on the LDA ROC curve
plt.plot(fpr_lda, tpr_lda, label='LDA ROC Curve', color='green')
plt.scatter(min_error_fpr_lda, min_error_tpr_lda, color='red', marker='o', label='LDA Min P(error)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LDA ROC Curve with Minimum Probability of Error')
plt.grid()
plt.legend()
plt.show()