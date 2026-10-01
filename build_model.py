import os
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import portable

URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"
MODEL_DIR = Path("model")

history = pd.read_csv(URL + "history_week6.csv")
new = pd.read_csv(URL + "new_week6.csv")

leaks = {
    "student_id", "cohort", "left",
    "ects_passed_sem1", "deregistration_form_opened", "last_login_week"
}
features = [c for c in new.columns if c not in leaks]
numeric = [c for c in features if c not in ["programme", "gender"]]
categorical = ["programme", "gender"]

prepare = ColumnTransformer([
    ("num", make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    ), numeric),
    ("cat", OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=20,
        sparse_output=False
    ), categorical),
])

model = make_pipeline(
    prepare,
    xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    ),
)

train = history[history["cohort"] <= 2024]
val = history[history["cohort"] == 2025].copy()
model.fit(train[features], train["left"])

val["risk"] = model.predict_proba(val[features])[:, 1]
print("Validation AUC:", round(roc_auc_score(val["left"], val["risk"]), 3))

new["risk"] = model.predict_proba(new[features])[:, 1]
top40 = new.nlargest(40, "risk").copy()
top40.to_csv("contact_list_2026_top40.csv", index=False)

MODEL_DIR.mkdir(exist_ok=True)
portable.export(model, MODEL_DIR)
print("Saved model to:", MODEL_DIR.resolve())
print("Saved contact list to:", Path("contact_list_2026_top40.csv").resolve())
