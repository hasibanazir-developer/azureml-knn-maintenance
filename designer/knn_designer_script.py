# KNN inside Azure ML Designer (Execute Python Script)
# Dataset1 = training rows, Dataset2 = test rows
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

TARGET = "machine_failure"
NUM_COLS = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
CAT_COLS = ["type"]
K = 1               # best K from the notebook
WEIGHTS = "uniform" # best weights from the notebook
P = 1               # best distance (1 = Manhattan) from the notebook

def azureml_main(dataframe1=None, dataframe2=None):
    train, test = dataframe1, dataframe2
    X_train, y_train = train[NUM_COLS + CAT_COLS], train[TARGET].astype(int)
    X_test, y_test = test[NUM_COLS + CAT_COLS], test[TARGET].astype(int)
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    model = Pipeline([("prep", prep),
                      ("knn", KNeighborsClassifier(n_neighbors=K, weights=WEIGHTS, p=P))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    scored = test.copy()
    scored["predicted_failure"] = pred
    scored["failure_probability"] = prob.round(3)
    metrics = pd.DataFrame({
        "metric": ["k", "accuracy", "precision", "recall", "f1", "auc"],
        "value": [K, accuracy_score(y_test, pred),
                  precision_score(y_test, pred, zero_division=0),
                  recall_score(y_test, pred), f1_score(y_test, pred),
                  roc_auc_score(y_test, prob)],
    })
    print(metrics)
    return scored, metrics
