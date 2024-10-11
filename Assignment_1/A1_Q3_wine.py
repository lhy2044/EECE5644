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

# Load the wine quality datasets
red_wine = pd.read_csv('wine+quality/winequality-red.csv', sep=';')
white_wine = pd.read_csv('wine+quality/winequality-white.csv', sep=';')

# Add a 'type' column to differentiate red and white wines
red_wine['type'] = 'red'
white_wine['type'] = 'white'

# Combine the datasets
wine_data = pd.concat([red_wine, white_wine], ignore_index=True)

# Define target variable: quality (good vs bad)
wine_data['target'] = wine_data['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Dataset Overview
print("\nDataset Overview:\n")
print(wine_data.describe())

# Visualizations for EDA
plt.figure(figsize=(10, 6))
wine_data.hist(bins=20, figsize=(15, 10), edgecolor='black')
plt.suptitle('Feature Distributions')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
numeric_data = wine_data.select_dtypes(include=[np.number])
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Class Distribution
print("\nClass Distribution:\n")
print(wine_data['target'].value_counts())
wine_data['target'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Class Distribution')
plt.xlabel('Class Label (0 = Bad, 1 = Good)')
plt.ylabel('Frequency')
plt.show()

# Splitting the dataset into training and testing sets
X = wine_data.drop(['quality', 'type', 'target'], axis=1)
y = wine_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Selection and Training
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Neural Network': MLPClassifier()
}

results = {}

for name, model in tqdm(models.items(), desc="Training models"):
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 'N/A'
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

print("\nModel Comparison:\n")
comparison_df = pd.DataFrame(results).T
print(comparison_df)

# Bar plot for comparison
comparison_df['Accuracy'].plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()
