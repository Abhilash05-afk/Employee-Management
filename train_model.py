import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

MODEL_PATH = "models/attrition_model.pkl"


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n==========================================")
print("EMPLOYEE ATTRITION PREDICTION SYSTEM")
print("==========================================")

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")


# ============================================================
# 2. BASIC DATA INFORMATION
# ============================================================

print("\n==========================================")
print("DATASET INFORMATION")
print("==========================================")

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\n==========================================")
print("MISSING VALUES")
print("==========================================")

missing_values = df.isnull().sum()

print(missing_values)

total_missing = missing_values.sum()

print(f"\nTotal missing values: {total_missing}")


# ============================================================
# 4. CHECK TARGET DISTRIBUTION
# ============================================================

print("\n==========================================")
print("ATTRITION DISTRIBUTION")
print("==========================================")

print(df["Attrition"].value_counts())

print("\nAttrition percentage:")

attrition_percentage = (
    df["Attrition"]
    .value_counts(normalize=True)
    * 100
)

print(attrition_percentage.round(2))


# ============================================================
# 5. REMOVE IRRELEVANT COLUMNS
# ============================================================

columns_to_drop = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df = df.drop(
    columns=columns_to_drop
)

print("\nRemoved irrelevant columns:")
print(columns_to_drop)

print(f"\nRemaining columns: {df.shape[1]}")


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["Attrition"]
)

y = df["Attrition"]


# ============================================================
# 7. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["str"]
).columns.tolist()


print("\n==========================================")
print("FEATURE INFORMATION")
print("==========================================")

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n==========================================")
print("TRAIN / TEST SPLIT")
print("==========================================")

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# 9. NUMERICAL PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 10. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# 11. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 12. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
}


# ============================================================
# 13. TRAIN AND COMPARE MODELS
# ============================================================

results = {}

best_model_name = None
best_model_pipeline = None
best_f1_score = -1


print("\n==========================================")
print("MODEL TRAINING")
print("==========================================")


for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="Yes",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="Yes",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="Yes",
        zero_division=0
    )

    results[model_name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Select the model with the highest F1 score
    if f1 > best_f1_score:

        best_f1_score = f1

        best_model_name = model_name

        best_model_pipeline = pipeline


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

results_df = pd.DataFrame(
    results
).T

print(
    results_df.round(4)
)


# ============================================================
# 15. BEST MODEL
# ============================================================

print("\n==========================================")
print("BEST MODEL")
print("==========================================")

print(f"Best model: {best_model_name}")

print(
    f"Best F1 Score: {best_f1_score:.4f}"
)


# ============================================================
# 16. FINAL MODEL PREDICTIONS
# ============================================================

best_predictions = best_model_pipeline.predict(
    X_test
)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

confusion = confusion_matrix(
    y_test,
    best_predictions
)

print(confusion)


# ============================================================
# 19. SAVE BEST MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model_pipeline,
    MODEL_PATH
)

print("\n==========================================")
print("MODEL SAVED")
print("==========================================")

print(
    f"Saved model to: {MODEL_PATH}"
)


# ============================================================
# 20. COMPLETE
# ============================================================

print("\nTraining process completed successfully! 🚀")
