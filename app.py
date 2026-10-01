from pathlib import Path

import pandas as pd
import streamlit as st

import portable

st.set_page_config(page_title="Study Office Risk Support", page_icon="🎓", layout="wide")

URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"
MODEL_DIR = Path("model")

LEAKS = {
    "student_id", "cohort", "left",
    "ects_passed_sem1", "deregistration_form_opened", "last_login_week"
}


@st.cache_data(ttl=3600)
def load_data():
    """Download the two datasets once per hour instead of on every rerun."""
    history = pd.read_csv(URL + "history_week6.csv")
    new = pd.read_csv(URL + "new_week6.csv")
    return history, new


@st.cache_resource
def load_model():
    """Load the portable model once and reuse it across reruns and users."""
    if not (MODEL_DIR / "booster.json").exists():
        return None
    return portable.Model(MODEL_DIR)


@st.cache_resource
def train_fallback(history):
    """Fallback for first deployment if model/ has not yet been uploaded."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import xgboost as xgb

    features = [c for c in history.columns if c not in LEAKS]
    numeric = [c for c in features if c not in ["programme", "gender"]]
    categorical = ["programme", "gender"]

    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=20,
            sparse_output=False
        ), categorical),
    ])

    pipe = make_pipeline(
        prep,
        xgb.XGBClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42, eval_metric="logloss", n_jobs=2, tree_method="hist"
        )
    )
    train = history[history["cohort"] <= 2024]
    pipe.fit(train[features], train["left"])
    return pipe, features


@st.cache_resource
def load_app():
    """Prepare model inference once; widget changes only rerun the cheap UI code."""
    history, new = load_data()
    portable_model = load_model()

    if portable_model is not None:
        features = portable_model.features
        predictor = portable_model
        new = new.copy()
        new["risk"] = predictor.predict_proba(new[features])
        val = history[history["cohort"] == 2025].copy()
        val["risk"] = predictor.predict_proba(val[features])
    else:
        pipe, features = train_fallback(history)
        predictor = None
        new = new.copy()
        new["risk"] = pipe.predict_proba(new[features])[:, 1]
        val = history[history["cohort"] == 2025].copy()
        val["risk"] = pipe.predict_proba(val[features])[:, 1]

    new_sorted = new.sort_values("risk", ascending=False).reset_index(drop=True)
    show_cols = [
        "student_id", "risk", "top_40", "international", "programme",
        "fees_owed", "submitted_share", "missed_last3", "weeks_since_login"
    ]
    new_sorted["top_40"] = False
    new_sorted.loc[:39, "top_40"] = True

    top40_val = val.nlargest(40, "risk")
    val_left_total = int(val["left"].sum())
    top40_left = int(top40_val["left"].sum())
    top40_recall = top40_left / val_left_total if val_left_total else 0.0
    top40_precision = top40_left / 40 if len(top40_val) else 0.0

    global_top40 = val["risk"].rank(ascending=False, method="first") <= 40
    group = val.groupby("international").agg(
        students=("left", "size"),
        leave_rate=("left", "mean"),
        mean_risk=("risk", "mean"),
    )
    group_recall = (
        val.loc[val["left"] == 1]
        .assign(top40=global_top40[val["left"] == 1])
        .groupby("international")["top40"]
        .mean()
        .rename("recall_top40")
    )
    group = group.join(group_recall).reset_index()
    group["group"] = group["international"].map({0: "Domestic", 1: "International"})

    ranked_csv = new_sorted[show_cols].to_csv(index=False).encode("utf-8")

    return (
        history, new, val, predictor, new_sorted, show_cols, ranked_csv,
        top40_left, val_left_total, top40_recall, top40_precision, group
    )


@st.cache_data(max_entries=100)
def get_student_contribution(student_id):
    """Cache the expensive XGBoost pred_contribs calculation per student."""
    _, new, _, predictor, _, _, _, _, _, _, _, _ = load_app()
    if predictor is None:
        return None

    row = new.loc[new["student_id"] == student_id]
    if row.empty:
        return None

    return predictor.contributions(row).iloc[0].sort_values()


(
    history, new, val, predictor, new_sorted, show_cols, ranked_csv,
    top40_left, val_left_total, top40_recall, top40_precision, group
) = load_app()


st.title("🎓 Study Office Risk Support")
st.caption(
    "Decision support at the end of week 6 — the model provides a risk signal; an adviser makes the decision."
)

st.info(
    "The list is not an automatic decision. Review the student's context before contacting them, "
    "and use a supportive invitation rather than describing them as a predicted dropout."
)

st.header("1. This week's list")
st.dataframe(
    new_sorted[show_cols].style.format({
        "risk": "{:.1%}",
        "submitted_share": "{:.0%}"
    }),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download ranked 2026 list",
    data=ranked_csv,
    file_name="study_office_2026_ranked.csv",
    mime="text/csv",
)

st.header("2. What mistakes does the rule make?")
cutoff = st.slider("Risk cut-off on the 2025 validation cohort", 0.0, 1.0, 0.20, 0.01)
contacted = val["risk"] >= cutoff

tn = int(((val["left"] == 0) & (~contacted)).sum())
fp = int(((val["left"] == 0) & contacted).sum())
fn = int(((val["left"] == 1) & (~contacted)).sum())
tp = int(((val["left"] == 1) & contacted).sum())

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reached in time (TP)", tp)
c2.metric("Worried for nothing (FP)", fp)
c3.metric("Missed (FN)", fn)
c4.metric("Stayed, not contacted (TN)", tn)

c5, c6 = st.columns(2)
c5.metric("Precision", f"{precision:.1%}")
c6.metric("Recall", f"{recall:.1%}")

st.write(
    f"At a **{cutoff:.0%}** cut-off, the rule contacts **{int(contacted.sum())}** students. "
    f"Of those contacted, **{precision:.1%}** later left (precision). "
    f"Of the students who later left, the rule reached **{recall:.1%}** (recall)."
)

st.subheader("Capacity rule: top 40")
st.write(
    f"The top-40 rule reaches **{top40_left}** of {val_left_total} students who later left: "
    f"recall **{top40_recall:.1%}**. Among the 40 contacted, **{top40_precision:.1%}** later left: "
    f"precision **{top40_precision:.1%}**."
)

st.header("3. Per group")
st.dataframe(
    group[["group", "students", "leave_rate", "mean_risk", "recall_top40"]]
    .style.format({
        "leave_rate": "{:.1%}",
        "mean_risk": "{:.1%}",
        "recall_top40": "{:.1%}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.header("4. Why is a student on the list?")
student = st.selectbox("Choose a 2026 student", new_sorted["student_id"].tolist())
row = new_sorted.loc[new_sorted["student_id"] == student].iloc[0]
st.write(f"**Predicted risk: {row['risk']:.1%}**")

if predictor is not None:
    contrib = get_student_contribution(student)
    if contrib is not None:
        st.dataframe(
            pd.DataFrame({
                "feature": list(contrib.index),
                "model contribution": list(contrib.values),
            })
            .sort_values(
                "model contribution",
                key=lambda s: s.abs(),
                ascending=False
            )
            .head(8)
            .style.format({"model contribution": "{:+.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.write("Portable model files are not present, so feature contributions are unavailable in fallback mode.")

st.caption(
    "Important limitation: the model learns statistical patterns from historical cohorts. "
    "It does not establish why an individual student is at risk, and platform activity may not represent "
    "the same circumstances for every student."
)
