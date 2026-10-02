# ============================================================
# CREDIT SCORE / CREDIT RISK PREDICTION
# Kaggle Credit Risk Dataset
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("Credit_risk_dataset.csv")

print("\n==============================")
print("CREDIT RISK DATASET")
print("==============================")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# ============================================================
# 5. HANDLE OUTLIERS / INVALID AGE
# ============================================================

# The dataset contains a few unusual age values.
# We keep realistic age values between 18 and 100.

df = df[
    (df["person_age"] >= 18) &
    (df["person_age"] <= 100)
]


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

# Debt-to-Income related feature
# The dataset already contains loan_percent_income.
# We create an additional loan-to-income ratio.

df["loan_to_income"] = (
    df["loan_amnt"] /
    (df["person_income"] + 1)
)


# Income per employment year
df["income_per_emp_year"] = (
    df["person_income"] /
    (df["person_emp_length"] + 1)
)


# Loan amount compared with credit history
df["loan_per_credit_year"] = (
    df["loan_amnt"] /
    (df["cb_person_cred_hist_length"] + 1)
)


print("\nNew engineered features:")

print(
    df[
        [
            "loan_to_income",
            "income_per_emp_year",
            "loan_per_credit_year"
        ]
    ].head()
)


# ============================================================
# 7. TARGET VARIABLE
# ============================================================

# loan_status:
# 0 = Non-default
# 1 = Default

print("\nTarget distribution:")

print(df["loan_status"].value_counts())


# ============================================================
# 8. SELECT FEATURES
# ============================================================

features = [

    "person_age",

    "person_income",

    "person_home_ownership",

    "person_emp_length",

    "loan_intent",

    "loan_grade",

    "loan_amnt",

    "loan_int_rate",

    "loan_percent_income",

    "cb_person_default_on_file",

    "cb_person_cred_hist_length",

    # Feature engineering
    "loan_to_income",

    "income_per_emp_year",

    "loan_per_credit_year"
]


X = df[features]

y = df["loan_status"]


print("\nFeatures used:")
print(features)


# ============================================================
# 9. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = [

    "person_age",

    "person_income",

    "person_emp_length",

    "loan_amnt",

    "loan_int_rate",

    "loan_percent_income",

    "cb_person_cred_hist_length",

    "loan_to_income",

    "income_per_emp_year",

    "loan_per_credit_year"
]


categorical_features = [

    "person_home_ownership",

    "loan_intent",

    "loan_grade",

    "cb_person_default_on_file"
]


# ============================================================
# 10. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n==============================")
print("TRAIN TEST SPLIT")
print("==============================")

print("Training data:", X_train.shape)

print("Testing data:", X_test.shape)




# ============================================================
# 11. PREPROCESSING
# ============================================================

# Numerical preprocessing

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


# Categorical preprocessing

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


# Combine numerical and categorical preprocessing

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
# 12. CREATE LOGISTIC REGRESSION MODEL
# ============================================================

logistic_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )

    ]
)


# ============================================================
# 13. CREATE DECISION TREE MODEL
# ============================================================

decision_tree_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            DecisionTreeClassifier(
                max_depth=6,
                random_state=42
            )
        )

    ]
)


# ============================================================
# 14. CREATE RANDOM FOREST MODEL
# ============================================================

random_forest_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        )

    ]
)


# ============================================================
# 15. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

logistic_model.fit(
    X_train,
    y_train
)

logistic_prediction = logistic_model.predict(
    X_test
)

logistic_probability = logistic_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 16. TRAIN DECISION TREE
# ============================================================

print("Training Decision Tree...")

decision_tree_model.fit(
    X_train,
    y_train
)

decision_tree_prediction = decision_tree_model.predict(
    X_test
)

decision_tree_probability = (
    decision_tree_model.predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# 17. TRAIN RANDOM FOREST
# ============================================================

print("Training Random Forest...")

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_prediction = (
    random_forest_model.predict(
        X_test
    )
)

random_forest_probability = (
    random_forest_model.predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# 18. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_test,
    prediction,
    probability
):

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probability
    )


    print("\n")
    print("=" * 60)

    print(model_name)

    print("=" * 60)

    print(
        "Accuracy :",
        round(accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall   :",
        round(recall, 4)
    )

    print(
        "F1 Score :",
        round(f1, 4)
    )

    print(
        "ROC-AUC  :",
        round(roc_auc, 4)
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            prediction,
            zero_division=0
        )
    )


    return [

        accuracy,
        precision,
        recall,
        f1,
        roc_auc

    ]


# ============================================================
# 19. EVALUATE LOGISTIC REGRESSION
# ============================================================

logistic_results = evaluate_model(

    "LOGISTIC REGRESSION",

    y_test,

    logistic_prediction,

    logistic_probability

)


# ============================================================
# 20. EVALUATE DECISION TREE
# ============================================================

decision_tree_results = evaluate_model(

    "DECISION TREE",

    y_test,

    decision_tree_prediction,

    decision_tree_probability

)


# ============================================================
# 21. EVALUATE RANDOM FOREST
# ============================================================

random_forest_results = evaluate_model(

    "RANDOM FOREST",

    y_test,

    random_forest_prediction,

    random_forest_probability

)


# ============================================================
# 22. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model": [

        "Logistic Regression",

        "Decision Tree",

        "Random Forest"

    ],

    "Accuracy": [

        logistic_results[0],

        decision_tree_results[0],

        random_forest_results[0]

    ],

    "Precision": [

        logistic_results[1],

        decision_tree_results[1],

        random_forest_results[1]

    ],

    "Recall": [

        logistic_results[2],

        decision_tree_results[2],

        random_forest_results[2]

    ],

    "F1 Score": [

        logistic_results[3],

        decision_tree_results[3],

        random_forest_results[3]

    ],

    "ROC-AUC": [

        logistic_results[4],

        decision_tree_results[4],

        random_forest_results[4]

    ]

})


print("\n")

print("=" * 70)

print("MODEL COMPARISON")

print("=" * 70)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 23. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    random_forest_prediction

)


plt.figure(
    figsize=(7, 5)
)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

)


plt.title(
    "Random Forest Confusion Matrix"
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.show()


# ============================================================
# 24. ROC CURVE
# ============================================================

logistic_fpr, logistic_tpr, _ = roc_curve(

    y_test,

    logistic_probability

)


tree_fpr, tree_tpr, _ = roc_curve(

    y_test,

    decision_tree_probability

)


forest_fpr, forest_tpr, _ = roc_curve(

    y_test,

    random_forest_probability

)


plt.figure(
    figsize=(8, 6)
)


plt.plot(

    logistic_fpr,

    logistic_tpr,

    label="Logistic Regression"

)


plt.plot(

    tree_fpr,

    tree_tpr,

    label="Decision Tree"

)


plt.plot(

    forest_fpr,

    forest_tpr,

    label="Random Forest"

)


plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--"

)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ROC Curve Comparison"
)


plt.legend()


plt.show()


# ============================================================
# 25. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

# Get the trained preprocessing part

trained_preprocessor = (
    random_forest_model
    .named_steps["preprocessor"]
)


# Get feature names after encoding

feature_names = (
    trained_preprocessor
    .get_feature_names_out()
)


# Get Random Forest model

trained_random_forest = (
    random_forest_model
    .named_steps["model"]
)


# Get feature importance

importance_values = (
    trained_random_forest
    .feature_importances_
)


feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance_values

})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\n")

print("=" * 60)

print("FEATURE IMPORTANCE")

print("=" * 60)

print(
    feature_importance
    .head(15)
    .to_string(index=False)
)


# Plot top 15 features

top_features = (
    feature_importance
    .head(15)
    .sort_values(
        by="Importance"
    )
)


plt.figure(
    figsize=(10, 7)
)


sns.barplot(

    data=top_features,

    x="Importance",

    y="Feature"

)


plt.title(
    "Top 15 Random Forest Feature Importance"
)


plt.xlabel(
    "Importance"
)


plt.ylabel(
    "Feature"
)


plt.show()


# ============================================================
# 26. SAVE RANDOM FOREST MODEL
# ============================================================

joblib.dump(

    random_forest_model,

    "credit_score_model.pkl"
    "credit_score_scaler.pkl"

)


# ============================================================
# 27. SAVE MODEL RESULTS
# ============================================================

results.to_csv(

    "model_results.csv",

    index=False

)


# ============================================================
# 28. FINAL MESSAGE
# ============================================================

print("\n")

print("=" * 70)

print(
    "CREDIT SCORE MODEL COMPLETED SUCCESSFULLY!"
)

print("=" * 70)

print("\nFiles created:")

print("1. credit_score_model.pkl")

print("2. model_results.csv")

print("\nModels trained:")

print("1. Logistic Regression")

print("2. Decision Tree")

print("3. Random Forest")

print("\nEvaluation metrics:")

print("Accuracy")

print("Precision")

print("Recall")

print("F1-Score")

print("ROC-AUC")

print("\nProject completed successfully!")