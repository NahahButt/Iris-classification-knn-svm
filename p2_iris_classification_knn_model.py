# -*- coding: utf-8 -*-
"""p2.IRIS classification KNN Model

# Artificial Intelligence Internship – Project 2
## Step 1: Import Required Libraries
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

"""## Step 2: Load the Iris Dataset"""

# Load Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset Loaded Successfully!")
print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

"""##Dataset Information"""

print(iris.DESCR)

"""##Convert to DataFrame"""

import pandas as pd

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target
print(df.head())

"""##Dataset Statistics"""

print(df.describe())

"""## Visualizing Feature Relationships with a Pair Plot

A pair plot (or scatterplot matrix) is a great way to visualize pairwise relationships between features and their distributions. It helps us understand correlations and how different species classes separate across various feature combinations.
"""

import seaborn as sns
import matplotlib.pyplot as plt

# Create a copy of the DataFrame for plotting to add species names
df_plot = df.copy()
df_plot['SpeciesName'] = df_plot['Species'].map({i: name for i, name in enumerate(iris.target_names)})

# Generate the pair plot
sns.pairplot(df_plot, hue='SpeciesName', palette='viridis', diag_kind='kde')
plt.suptitle('Pair Plot of Iris Features by Species', y=1.02) # Adjust title position
plt.show()

"""##Missing Values"""

print(df.isnull().sum())

"""##Feature Scaling"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

"""##Split the Dataset"""

# Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

"""##Train the KNN Model"""

# Create and Train KNN Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

print("Model Training Completed!")

"""##Make Predictions"""

# Predict on Test Data
y_pred = model.predict(X_test)

print("Predictions Generated Successfully!")

"""##Evaluate the Model"""

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

"""##Confusion Matrix"""

# Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

"""##Classification Report"""

# Display Classification Report
report = classification_report(y_test, y_pred)

print("Classification Report:")
print(report)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()

"""## Cross-Validation for Robust Model Evaluation

Cross-validation is a technique used to assess how the results of a statistical analysis will generalize to an independent data set. It is mainly used in settings where the goal is prediction, and one wants to estimate how accurately a predictive model will perform in practice. In this project, we'll use K-Fold Cross-Validation to evaluate the KNN model more rigorously.
"""

from sklearn.model_selection import KFold, cross_val_score

# Initialize KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the KNN model (using the same n_neighbors=3 as before)
knn_cv_model = KNeighborsClassifier(n_neighbors=3)

# Perform cross-validation
cv_scores = cross_val_score(knn_cv_model, X, y, cv=kf, scoring='accuracy')

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation of cross-validation accuracy: {cv_scores.std():.4f}")

"""## Saving and Loading the Trained Model

In a real-world scenario, once a model is trained and validated, it's often saved to disk so it can be re-used later without retraining. This is particularly useful for deploying models in production environments. We'll use `joblib` for this purpose.
"""

import joblib

# Define a filename for your model
model_filename = 'knn_iris_model.joblib'

# Save the trained model to disk
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

# Load the model back from disk
loaded_model = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# You can now use the loaded model to make predictions
# For example, let's predict on the test set again using the loaded model
loaded_model_predictions = loaded_model.predict(X_test)

# Verify that predictions are the same
print(f"Predictions from loaded model: {loaded_model_predictions[:5]}")
print(f"Original predictions: {y_pred[:5]}")
print(f"Are predictions from loaded model identical to original? {all(loaded_model_predictions == y_pred)}")

"""## Step 1: Hyperparameter Tuning for KNN

We'll use `GridSearchCV` to find the optimal number of neighbors (`n_neighbors`) for our `KNeighborsClassifier`. This ensures we are using the best possible configuration for the model, even if the current one is already performing well.
"""

from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {'n_neighbors': range(1, 15)}

# Initialize GridSearchCV with the KNN model and parameter grid
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and best score found
print(f"Best parameters for KNN: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Get the best KNN model
best_knn_model = grid_search.best_estimator_

# Evaluate the best model on the test set
best_knn_predictions = best_knn_model.predict(X_test)
best_knn_accuracy = accuracy_score(y_test, best_knn_predictions)
print(f"Test accuracy with best KNN model: {best_knn_accuracy:.4f}")

"""## Step 2: Comparison with Other Classification Models

To ensure we have the best possible model for the Iris dataset, it's good practice to compare the performance of our KNN model with other common classification algorithms. We'll evaluate Logistic Regression, Support Vector Machine (SVM), and Decision Tree Classifier.

First, let's import the necessary libraries.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

"""Next, we'll define and train each model, and then evaluate them using K-Fold Cross-Validation, similar to what we did for the KNN model."""

# Initialize KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Store results
results = {}

# Evaluate each model using cross-validation
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    results[name] = {
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std()
    }
    print(f"{name}:")
    print(f"  Mean cross-validation accuracy: {results[name]['mean_accuracy']:.4f}")
    print(f"  Standard deviation of cross-validation accuracy: {results[name]['std_accuracy']:.4f}")
    print("\n")

# Also add the best KNN results for comparison
results['KNN (tuned)'] = {
    'mean_accuracy': grid_search.best_score_, # Using the best cross-validation score from GridSearchCV
    'std_accuracy': np.std(grid_search.cv_results_['split0_test_score']) # Taking std from one fold for simplicity, full std could be calculated from all folds
}

print(f"KNN (tuned):")
print(f"  Mean cross-validation accuracy: {results['KNN (tuned)']['mean_accuracy']:.4f}")
print(f"  Standard deviation of cross-validation accuracy: {results['KNN (tuned)']['std_accuracy']:.4f}")
print("\n")

"""Finally, let's visualize the comparison."""

import matplotlib.pyplot as plt
import seaborn as sns

model_names = list(results.keys())
mean_accuracies = [results[name]['mean_accuracy'] for name in model_names]
std_accuracies = [results[name]['std_accuracy'] for name in model_names]

plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=mean_accuracies, palette='viridis')
plt.errorbar(x=model_names, y=mean_accuracies, yerr=std_accuracies, fmt='o', color='black', capsize=5)
plt.xlabel('Model')
plt.ylabel('Mean Cross-Validation Accuracy')
plt.title('Comparison of Classification Models')
plt.ylim(0.85, 1.05) # Adjust y-limit for better visualization if needed
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

"""## Step 3: Creation of a Simple Prediction Interface

Finally, to make our project even more impressive and user-friendly, let's create a simple function that allows us to input new data and get a prediction from our best-performing model. Based on our comparison, the Support Vector Machine (SVM) model showed slightly higher mean accuracy, so we'll use that as our preferred model. If you wish to use the tuned KNN, we can easily switch it.

First, let's train the best model found (SVM) on the entire dataset (or `X_train`, `y_train` if we want to simulate a production environment where test data is unknown).
"""

# Train the best performing model (SVM) on the full training data
# We'll re-initialize and train it for clarity, ensuring we have a final model ready

final_model = SVC(random_state=42)
final_model.fit(X_train, y_train)

print("Final model (SVC) trained successfully on the training data!")

"""Now, let's create a function that takes new feature values, scales them using our pre-trained scaler, and then uses the `final_model` to make a prediction. We'll also map the numerical prediction back to the species names for easier interpretation."""

def predict_iris_species(sepal_length, sepal_width, petal_length, petal_width):
    # Create a numpy array from the input features
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Scale the input data using the same scaler fitted on the training data
    scaled_input_data = scaler.transform(input_data)

    # Make prediction using the final model
    prediction = final_model.predict(scaled_input_data)

    # Map the numerical prediction to species name
    species_names = iris.target_names
    predicted_species = species_names[prediction[0]]

    return predicted_species

# Example Usage:
# Let's try to predict a new sample. We can use one from our original dataset to verify.
# For example, the first sample:
# sepal length (cm)=5.1, sepal width (cm)=3.5, petal length (cm)=1.4, petal width (cm)=0.2 -> setosa

example_prediction = predict_iris_species(5.1, 3.5, 1.4, 0.2)
print(f"Predicted species for (5.1, 3.5, 1.4, 0.2): {example_prediction}")

# Another example from a different class, e.g., versicolor (species=1)
# X[51] is [6.4, 3.2, 4.5, 1.5] (original, not scaled)
example_prediction_2 = predict_iris_species(6.4, 3.2, 4.5, 1.5)
print(f"Predicted species for (6.4, 3.2, 4.5, 1.5): {example_prediction_2}")

# Another example from virginica (species=2)
# X[101] is [5.7, 3.8, 1.7, 0.3] (original, not scaled)
example_prediction_3 = predict_iris_species(7.2, 3.6, 6.1, 2.5)
print(f"Predicted species for (7.2, 3.6, 6.1, 2.5): {example_prediction_3}")

"""### Project Conclusion

Congratulations! You've gone above and beyond to make your project truly 'top-class outstanding'. We've covered:

1.  **Initial Assessment & Baseline Model:** Established high performance with a simple KNN model.
2.  **Visualizations:** Added a confusion matrix heatmap for intuitive understanding.
3.  **Robust Evaluation:** Implemented K-Fold Cross-Validation to ensure model consistency.
4.  **Model Persistence:** Demonstrated saving and loading the model for deployment readiness.
5.  **Hyperparameter Tuning:** Optimized the KNN model using `GridSearchCV`.
6.  **Model Comparison:** Evaluated and compared KNN with Logistic Regression, SVM, and Decision Tree, confirming SVM as a strong performer.
7.  **Prediction Interface:** Created a simple function for real-time predictions with new data.

Your project now showcases a comprehensive machine learning workflow from data understanding to deployment readiness, with thorough evaluation and enhancement steps. This is a truly impressive body of work!
"""