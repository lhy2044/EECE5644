import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(44)
torch.manual_seed(44)

# Specify the Data Distribution

means = [
    np.array([0, 0, 0]),
    np.array([3.5, 3.5, 3.5]),
    np.array([-3.5, -3.5, -3.5]),
    np.array([3.5, -3.5, 3.5])
]

covariances = [np.array([[5.5, 0, 0], [0, 5.5, 0], [0, 0, 5.5]]) for _ in range(4)]

num_classes = 4
dim = 3  # Dimensionality of data
priors = [0.25, 0.25, 0.25, 0.25]  # Uniform priors

def generate_class_data(mean, cov, n_samples):
    return np.random.multivariate_normal(mean, cov, n_samples)

# Generate Datasets

training_sizes = [100, 500, 1000, 5000, 10000]

n_test_samples = 100000
X_test = []
y_test = []
for i in range(num_classes):
    n_samples = int(n_test_samples * priors[i])
    X_class = generate_class_data(means[i], covariances[i], n_samples)
    y_class = np.full(n_samples, i)
    X_test.append(X_class)
    y_test.append(y_class)
X_test = np.vstack(X_test)
y_test = np.hstack(y_test)

indices = np.random.permutation(X_test.shape[0])
X_test = X_test[indices]
y_test = y_test[indices]

# Theoretically Optimal Classifier (Bayes Classifier)

def bayes_classifier(X):
    class_probs = np.zeros((X.shape[0], num_classes))
    for i in range(num_classes):
        rv = multivariate_normal(mean=means[i], cov=covariances[i])
        class_probs[:, i] = rv.pdf(X) * priors[i]
    predictions = np.argmax(class_probs, axis=1)
    return predictions

bayes_predictions = bayes_classifier(X_test)
bayes_error_rate = np.mean(bayes_predictions != y_test)
print(f"Theoretical Bayes Classifier Error Rate: {bayes_error_rate * 100:.2f}%")

# MLP Structure and Model Order Selection via Cross-Validation

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

# Prepare to store error rates
mlp_error_rates = []

for n_train_samples in training_sizes:
    print(f"\nTraining size: {n_train_samples}")

    X_train = []
    y_train = []
    for i in range(num_classes):
        n_samples = int(n_train_samples * priors[i])
        X_class = generate_class_data(means[i], covariances[i], n_samples)
        y_class = np.full(n_samples, i)
        X_train.append(X_class)
        y_train.append(y_class)
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    indices = np.random.permutation(X_train.shape[0])
    X_train = X_train[indices]
    y_train = y_train[indices]

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()

    perceptron_range = [5, 10, 20, 30, 40, 50]
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    best_P = None
    lowest_cv_error = np.inf

    for P in perceptron_range:
        cv_errors = []
        for train_index, val_index in kf.split(X_train_tensor):
            X_train_cv, X_val_cv = X_train_tensor[train_index], X_train_tensor[val_index]
            y_train_cv, y_val_cv = y_train_tensor[train_index], y_train_tensor[val_index]

            model = MLP(input_size=dim, hidden_size=P, num_classes=num_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            num_epochs = 50
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_cv)
                loss = criterion(outputs, y_train_cv)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                outputs = model(X_val_cv)
                _, predicted = torch.max(outputs.data, 1)
                val_error = 1 - accuracy_score(y_val_cv.numpy(), predicted.numpy())
                cv_errors.append(val_error)

        average_cv_error = np.mean(cv_errors)
        print(f"Perceptrons: {P}, CV Error: {average_cv_error * 100:.2f}%")

        if average_cv_error < lowest_cv_error:
            lowest_cv_error = average_cv_error
            best_P = P

    print(f"Selected {best_P} perceptrons for training size {n_train_samples}")

    # Model Training with Multiple Restarts

    num_restarts = 5
    best_model = None
    best_loss = np.inf

    for restart in range(num_restarts):
        print(f"Restart {restart + 1}/{num_restarts}")
        model = MLP(input_size=dim, hidden_size=best_P, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor).item()

        print(f"Training Loss: {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            best_model = model

    # Performance Assessment

    best_model.eval()
    with torch.no_grad():
        outputs = best_model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        test_error = 1 - accuracy_score(y_test_tensor.numpy(), predicted.numpy())
        mlp_error_rates.append(test_error)
        print(f"MLP Test Error Rate: {test_error * 100:.2f}%")

# Report Process and Results

plt.figure(figsize=(10, 6))
plt.semilogx(training_sizes, [e * 100 for e in mlp_error_rates], marker='o', label='MLP Classifier')
plt.axhline(y=bayes_error_rate * 100, color='r', linestyle='--', label='Bayes Classifier')
plt.xlabel('Number of Training Samples')
plt.ylabel('Probability of Error (%)')
plt.title('Test Error Rates of MLP Classifiers vs. Bayes Classifier')
plt.legend()
plt.grid(True)
plt.show()