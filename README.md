# 🌸 Iris Flower Classification — KNN & Model Deployment

**Artificial Intelligence Internship – Project 2**

A complete machine learning workflow on the classic Iris dataset: data exploration, KNN classification, model comparison, hyperparameter tuning, and deployment as a REST API using Flask.

## 📊 Project Overview

This project walks through the full ML pipeline:

1. **Data Loading & Exploration** — loading the Iris dataset, converting to a DataFrame, checking statistics and missing values
2. **Visualization** — pair plots to see how the three species separate across features
3. **Preprocessing** — feature scaling with `StandardScaler`
4. **Model Training** — a baseline K-Nearest Neighbors (KNN) classifier
5. **Evaluation** — accuracy score, confusion matrix (with heatmap), and classification report
6. **Cross-Validation** — 5-fold K-Fold CV for a more robust accuracy estimate
7. **Model Persistence** — saving/loading the trained model with `joblib`
8. **Hyperparameter Tuning** — `GridSearchCV` to find the best `n_neighbors` for KNN
9. **Model Comparison** — KNN vs Logistic Regression vs SVM vs Decision Tree
10. **Prediction Interface** — a simple function to classify new flower measurements
11. **Deployment** — the best model (SVM) is served through a Flask REST API

## 📁 Project Structure

```
iris-classification-project/
├── notebook/
│   └── p2_iris_classification_knn_model.py   # Full analysis & training script (exported from Colab)
├── deployment/
│   ├── app.py                 # Flask API that serves predictions
│   ├── iris_svm_model.pkl     # Trained SVM model
│   └── iris_scaler.pkl        # Fitted StandardScaler used for preprocessing
├── requirements.txt
└── README.md
```

## 🛠️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## ▶️ How to Run

### 1. Run the training/analysis script
```bash
python3 notebook/p2_iris_classification_knn_model.py
```
This reproduces the full analysis: EDA, training, evaluation, model comparison, and saves a trained model.

### 2. Run the deployment API
```bash
cd deployment
python3 app.py
```
The Flask server starts on `http://localhost:5000`.

### 3. Test the prediction endpoint

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Expected response:
```json
{
  "prediction": 0,
  "species": "setosa"
}
```

## 📈 Model Performance

- Baseline KNN (k=3): high accuracy on the test split (see script output for exact numbers)
- Cross-validation used to confirm consistency across folds
- Final deployed model: **SVM**, chosen after comparing KNN, Logistic Regression, SVM, and Decision Tree

## 🧠 Key Concepts Demonstrated

- Supervised classification
- Feature scaling
- Train/test split & K-Fold cross-validation
- Hyperparameter tuning (GridSearchCV)
- Model comparison across algorithms
- Model persistence (joblib)
- REST API deployment (Flask)

## 👤 Author

*Nahah Butt*

## 📄 License

This project is for educational purposes as part of an AI internship assignment.
