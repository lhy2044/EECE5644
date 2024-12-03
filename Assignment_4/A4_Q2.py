import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from skimage import io, transform
import urllib.request

image_url = "./163085.jpg"
image = io.imread(image_url)

max_size = 200
if max(image.shape[0], image.shape[1]) > max_size:
    scaling_factor = max_size / max(image.shape[0], image.shape[1])
    image = transform.rescale(image, scaling_factor, anti_aliasing=True, channel_axis=-1)
    image = (image * 255).astype(np.uint8)

rows, cols, channels = image.shape
row_indices, col_indices = np.indices((rows, cols))
features = np.stack((row_indices, col_indices, image[:, :, 0], image[:, :, 1], image[:, :, 2]), axis=-1)
features = features.reshape(-1, 5)

scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)

def cross_validate_gmm(X, n_components, n_splits=10):
    kf = KFold(n_splits=n_splits)
    log_likelihoods = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
        gmm.fit(X_train)
        log_likelihood = gmm.score(X_test)
        log_likelihoods.append(log_likelihood)

    return np.mean(log_likelihoods)

component_range = range(2, 11)
best_n_components = None
best_score = -np.inf
scores = []

for n_components in component_range:
    score = cross_validate_gmm(normalized_features, n_components)
    scores.append(score)
    if score > best_score:
        best_score = score
        best_n_components = n_components
    print(f'Components: {n_components}, Average Log-Likelihood: {score}')

print(f'\nBest number of components: {best_n_components}')

best_gmm = GaussianMixture(n_components=best_n_components, covariance_type='full', random_state=0)
best_gmm.fit(normalized_features)

labels = best_gmm.predict(normalized_features)
labels_image = labels.reshape(rows, cols)

unique_labels = np.unique(labels)
grayscale_values = np.linspace(0, 255, num=unique_labels.size, dtype=np.uint8)
label_to_grayscale = dict(zip(unique_labels, grayscale_values))
grayscale_image = np.vectorize(label_to_grayscale.get)(labels_image)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(grayscale_image, cmap='gray')
ax[1].set_title('GMM-based Segmentation')
ax[1].axis('off')

plt.show()
