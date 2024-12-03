import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle
from sklearn.exceptions import ConvergenceWarning
import warnings
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore", category=ConvergenceWarning)

r_minus1 = 2
r_plus1 = 4
sigma = 1
n_train = 1000
n_test = 10000

def generate_data(n_samples):
    labels = np.random.choice([-1, 1], size=n_samples)
    theta = np.random.uniform(-np.pi, np.pi, n_samples)
    data = np.zeros((n_samples, 2))
    for idx, l in enumerate(labels):
        r = r_minus1 if l == -1 else r_plus1
        x = r * np.array([np.cos(theta[idx]), np.sin(theta[idx])]) + np.random.normal(0, sigma, 2)
        data[idx] = x
    return data, labels

X_train, y_train = generate_data(n_train)
X_train, y_train = shuffle(X_train, y_train, random_state=44)

X_test, y_test = generate_data(n_test)

plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train==-1][:, 0], X_train[y_train==-1][:,1], label='Class -1', alpha=0.5)
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:,1], label='Class +1', alpha=0.5)
plt.title('Training Data Scatter Plot')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

# 10-fold cross-validation for SVM
C_values = [0.1, 1, 10, 100, 1000]
gamma_values = [0.01, 0.1, 1, 10, 100]

param_grid_svm = {'C': C_values, 'gamma': gamma_values}
svm = SVC(kernel='rbf', verbose=False)
grid_svm = GridSearchCV(svm, param_grid_svm, cv=10, scoring='accuracy', verbose=2, n_jobs=-1)
grid_svm.fit(X_train, y_train)

print("Best parameters for SVM:", grid_svm.best_params_)
print("Best cross-validation accuracy for SVM:", grid_svm.best_score_)

scores_svm = grid_svm.cv_results_['mean_test_score'].reshape(len(C_values), len(gamma_values))

plt.figure(figsize=(8, 6))
for idx, val in enumerate(C_values):
    plt.plot(gamma_values, scores_svm[idx], label=f'C = {val}')
plt.title('SVM Cross-Validation Results')
plt.xlabel('Gamma')
plt.ylabel('Mean Accuracy')
plt.legend()
plt.xscale('log')
plt.show()

# 10-fold cross-validation for MLP
hidden_neurons = [5, 10, 15, 20, 25]

class QuadraticActivationMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(10,), max_iter=1000):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.mlp = None
        self.loss_curve_ = None

    def fit(self, X, y):
        X_quad = np.hstack((X, X**2))
        self.mlp = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                 activation='identity',
                                 max_iter=self.max_iter,
                                 solver='adam',
                                 verbose=False,
                                 random_state=44)
        self.mlp.fit(X_quad, y)
        self.loss_curve_ = self.mlp.loss_curve_
        self.classes_ = self.mlp.classes_
        return self

    def predict(self, X):
        X_quad = np.hstack((X, X**2))
        return self.mlp.predict(X_quad)

    def predict_proba(self, X):
        X_quad = np.hstack((X, X**2))
        return self.mlp.predict_proba(X_quad)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

param_grid_mlp = {'hidden_layer_sizes': [(n,) for n in hidden_neurons]}
mlp = QuadraticActivationMLP(max_iter=1000)
grid_mlp = GridSearchCV(mlp, param_grid_mlp, cv=10, scoring='accuracy', verbose=2, n_jobs=-1)
grid_mlp.fit(X_train, y_train)

print("Best parameters for MLP:", grid_mlp.best_params_)
print("Best cross-validation accuracy for MLP:", grid_mlp.best_score_)

scores_mlp = grid_mlp.cv_results_['mean_test_score']

plt.figure(figsize=(8, 6))
plt.plot(hidden_neurons, scores_mlp, marker='o')
plt.title('MLP Cross-Validation Results')
plt.xlabel('Number of Hidden Neurons')
plt.ylabel('Mean Accuracy')
plt.show()

# Train final models with the best hyperparameters
best_svm = grid_svm.best_estimator_
best_svm.fit(X_train, y_train)

best_mlp = grid_mlp.best_estimator_
best_mlp.fit(X_train, y_train)

plt.figure(figsize=(8, 6))
plt.plot(best_mlp.loss_curve_)
plt.title('MLP Training Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

# Evaluate on test data
y_pred_svm = best_svm.predict(X_test)
test_accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

y_pred_mlp = best_mlp.predict(X_test)
test_accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)

print(f"SVM Test Accuracy: {test_accuracy_svm * 100:.2f}%")
print("SVM Confusion Matrix:")
print(conf_matrix_svm)

print(f"MLP Test Accuracy: {test_accuracy_mlp * 100:.2f}%")
print("MLP Confusion Matrix:")
print(conf_matrix_mlp)

# Visualization of decision boundaries
def plot_decision_boundary(clf, X, y, title):
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:,1].max() + 1
    h = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    if isinstance(clf, QuadraticActivationMLP):
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        X_grid_quad = np.hstack((X_grid, X_grid**2))
        Z = clf.mlp.predict(X_grid_quad)
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(X[y==-1][:, 0], X[y==-1][:,1], c='red', label='Class -1', edgecolor='k', s=20)
    plt.scatter(X[y==1][:, 0], X[y==1][:,1], c='blue', label='Class +1', edgecolor='k', s=20)
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()

sample_indices = np.random.choice(len(X_test), 1000, replace=False)
X_test_sample = X_test[sample_indices]
y_test_sample = y_test[sample_indices]

plot_decision_boundary(best_svm, X_test_sample, y_test_sample, 'SVM Decision Boundary on Test Data')
plot_decision_boundary(best_mlp, X_test_sample, y_test_sample, 'MLP Decision Boundary on Test Data')