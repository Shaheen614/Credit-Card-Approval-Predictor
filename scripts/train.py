import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Synthetic dataset (replace with Kaggle dataset if available)
np.random.seed(42)
n = 500
df = pd.DataFrame({
    "age": np.random.randint(18, 70, size=n),
    "income": np.random.randint(20000, 150000, size=n),
    "credit_score": np.random.randint(300, 850, size=n),
    "loan_amount": np.random.randint(1000, 50000, size=n),
})

# Synthetic approval probability
prob = (
    0.4*(df["credit_score"]-300)/(850-300)
    + 0.4*(df["income"]-20000)/(150000-20000)
    - 0.3*(df["loan_amount"]-1000)/(50000-1000)
)
prob = (prob - prob.min())/(prob.max()-prob.min())
df["approved"] = (np.random.rand(n) < prob).astype(int)

# Save sample data
df.to_csv("data/sample_applicants.csv", index=False)

# Features and labels
X = df.drop("approved", axis=1)
y = df["approved"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train, evaluate, and save each model
for name, model in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    # Save pipeline (includes scaler + model)
    joblib.dump(pipeline, f"models/{name}_model.pkl")

