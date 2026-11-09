import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# This script uses wine_quality.csv in the current folder.
# Replace the included 50-row preview with the complete official UCI red-wine CSV
# before recording final lab evaluation results.
df = pd.read_csv("wine_quality.csv")
df = df.drop_duplicates().copy()
y = (df["quality"] >= 6).astype(int)
X = df.drop(columns=["quality"])  # Keep target out of the input features.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"))
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, target_names=["Needs improvement", "Good quality"], zero_division=0))
joblib.dump(model, "wine_model.pkl")
print("Saved wine_model.pkl")
