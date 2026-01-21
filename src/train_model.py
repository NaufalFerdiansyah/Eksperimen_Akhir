import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

from xgboost import XGBClassifier

# =========================
# Load Dataset
# =========================
df = pd.read_csv("data/external/telco_customer_churn.csv")

# Target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]

# =========================
# Preprocessing
# =========================
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

# =========================
# Model
# =========================
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model),
    ]
)

# =========================
# Training
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# =========================
# Save Model
# =========================
joblib.dump(pipeline, "models/best_model.pkl")
joblib.dump(preprocessor, "models/preprocessing.pkl")

print("Model & preprocessing berhasil disimpan!")
