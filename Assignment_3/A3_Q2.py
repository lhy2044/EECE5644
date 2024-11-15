import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from collections import Counter
import matplotlib.pyplot as plt

# Specify the true GMM parameters
true_means = np.array([[0, 0],
                       [2, 2],
                       [5, 0],
                       [0, 5]])

true_covariances = np.array([[[1, 0], [0, 1]],
                             [[1, 0], [0, 1]],
                             [[1, 0], [0, 1]],
                             [[1, 0], [0, 1]]])

true_weights = np.array([0.3, 0.3, 0.2, 0.2])

def generate_samples_from_gmm(n_samples, weights, means, covariances):

    n_components = len(weights)
    component_labels = np.random.choice(n_components, size=n_samples, p=weights)
    samples = np.zeros((n_samples, means.shape[1]))
    for i in range(n_components):
        n_i = np.sum(component_labels == i)
        if n_i > 0:
            samples[component_labels == i, :] = np.random.multivariate_normal(
                means[i], covariances[i], size=n_i)
    return samples

# Generate datasets of different sizes
dataset_sizes = [10, 100, 1000]
n_repetitions = 100  # Number of repetitions for the experiment
max_K = 10  # Maximum number of components to evaluate

selection_results = {size: [] for size in dataset_sizes}

for size in dataset_sizes:
    print(f"Processing dataset size: {size}")
    for repetition in range(n_repetitions):
        data = generate_samples_from_gmm(size, true_weights, true_means, true_covariances)
        n_splits = min(10, size)  # Adjust the number of splits for small datasets
        kf = KFold(n_splits=n_splits, shuffle=True)

        avg_log_likelihoods = {}
        for K in range(1, max_K + 1):
            log_likelihoods = []
            for train_index, val_index in kf.split(data):
                X_train, X_val = data[train_index], data[val_index]
                try:
                    gmm = GaussianMixture(n_components=K, covariance_type='full',
                                          init_params='kmeans', max_iter=100, n_init=1)
                    gmm.fit(X_train)
                    log_likelihood = gmm.score(X_val)  # Average log-likelihood per sample
                    log_likelihoods.append(log_likelihood)
                except Exception as e:
                    log_likelihoods.append(-np.inf)
            avg_log_likelihoods[K] = np.mean(log_likelihoods)
        best_K = max(avg_log_likelihoods, key=avg_log_likelihoods.get)
        selection_results[size].append(best_K)

# Summarize and report the results
for size in dataset_sizes:
    counts = Counter(selection_results[size])
    total = sum(counts.values())
    print(f"\nDataset size: {size}")
    print("Model Order Selection Frequencies:")
    for K in range(1, max_K + 1):
        frequency = counts.get(K, 0)
        percentage = 100 * frequency / total
        print(f"  Model order {K}: selected {frequency} times ({percentage:.1f}%)")

    frequencies = [counts.get(K, 0) for K in range(1, max_K + 1)]
    plt.figure()
    plt.bar(range(1, max_K + 1), frequencies, color='skyblue', edgecolor='black')
    plt.title(f"Model Order Selection Frequencies (Dataset size {size})")
    plt.xlabel("Model Order (Number of Components K)")
    plt.ylabel("Frequency")
    plt.xticks(range(1, max_K + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()