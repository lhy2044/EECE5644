import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

# Load the Human Activity Recognition dataset
# Dataset URL: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

# Load the data from txt files
X_train = pd.read_csv('human+activity+recognition+using+smartphones/UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
Y_train = pd.read_csv('human+activity+recognition+using+smartphones/UCI HAR Dataset/train/y_train.txt', delim_whitespace=True, header=None)
X_test = pd.read_csv('human+activity+recognition+using+smartphones/UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None)
Y_test = pd.read_csv('human+activity+recognition+using+smartphones/UCI HAR Dataset/test/y_test.txt', delim_whitespace=True, header=None)

# Load feature names and activity labels
features = pd.read_csv('human+activity+recognition+using+smartphones/UCI HAR Dataset/features.txt', delim_whitespace=True, header=None, usecols=[1])[1].tolist()
activity_labels = pd.read_csv('human+activity+recognition+using+smartphones/UCI HAR Dataset/activity_labels.txt', delim_whitespace=True, header=None, index_col=0)[1].to_dict()

# Assign feature names to columns
X_train.columns = features
X_test.columns = features

# Map activity labels to Y_train and Y_test
Y_train = Y_train[0].map(activity_labels)
Y_test = Y_test[0].map(activity_labels)

# Combine X and Y for train and test datasets
har_train = X_train.copy()
har_train['activity'] = Y_train
har_test = X_test.copy()
har_test['activity'] = Y_test

# Combine training and testing datasets
har_data = pd.concat([har_train, har_test], ignore_index=True)

# Map activity labels to integers for classification
activity_mapping = {label: idx for idx, label in enumerate(activity_labels.values())}
har_data['activity'] = har_data['activity'].map(activity_mapping)

# Part A: Dataset Summary Statistics

# 1. Dataset Overview
print("\nDataset Overview:\n")
print(har_data.describe())

# Visualizations for EDA
plt.figure(figsize=(10, 6))
har_data[features[:10]].hist(bins=20, figsize=(15, 10), edgecolor='black')
plt.suptitle('Feature Distributions')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_data = har_data.select_dtypes(include=[np.number])
sns.heatmap(pd.get_dummies(har_data['activity']).corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 2. Class Distribution
print("\nClass Distribution:\n")
print(har_data['activity'].value_counts())
har_data['activity'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Class Distribution')
plt.xlabel('Activity Label')
plt.ylabel('Frequency')
plt.show()

# Part B: Model Training and Evaluation

# Splitting the dataset into training and testing sets
X = har_data.drop(['activity'], axis=1)
y = har_data['activity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Selection and Training
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Neural Network': MLPClassifier(max_iter=1000)
}

results = {}

for name, model in tqdm(models.items(), desc="Training models"):
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else 'N/A'
    print(f"Accuracy: {accuracy:.2f}")
    if roc_auc != 'N/A':
        print(f"ROC AUC: {roc_auc:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Store results for comparison
    results[name] = {
        'Accuracy': accuracy,
        'ROC AUC': roc_auc
    }

# Part C: Comparison and Insights

print("\nModel Comparison:\n")
comparison_df = pd.DataFrame(results).T
print(comparison_df)

# Bar plot for comparison
comparison_df['Accuracy'].plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()

# Part D: Key Takeaways
# The printed results and plots provide insights into model performance and trade-offs. Use these to select the best model based on accuracy, interpretability, and other specific requirements.