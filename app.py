import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from utils import compute_metrics, encode_target, CLASS_NAMES

TARGET = "Result"
MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "support_vector_machine",
}

st.set_page_config(page_title="Phishing Website Classifier Lab", page_icon="🎣", layout="wide")
st.title("🎣 Phishing Website Detection: Model Evaluation Lab")
st.caption("Evaluate five pre-trained classifiers on a held-out test CSV. "
           "No retraining happens in this app.")

with st.sidebar:
    st.header("Controls")
    choice = st.selectbox("Select model", list(MODELS))
    file = st.file_uploader("Upload test CSV (30 features + 'Result')", type="csv")
    st.markdown("**Target `Result`:** `-1` = Phishing, `1` = Legitimate")
    st.markdown("Phishing is treated as the positive class for Precision / Recall / F1.")


@st.cache_resource
def load_model(key):
    return joblib.load(f"model/{key}.joblib")


if file is None:
    st.info("Upload `test_data.csv` from the repo to begin.")
    st.stop()

df = pd.read_csv(file)
if TARGET not in df.columns:
    st.error(f"Missing target column '{TARGET}'.")
    st.stop()

model = load_model(MODELS[choice])
expected = list(model.feature_names_in_)
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()

unknown = set(df[TARGET].unique()) - {-1, 1}
if unknown:
    st.error(f"Unexpected values in '{TARGET}': {unknown} (expected -1 or 1).")
    st.stop()

X, y = df[expected], encode_target(df[TARGET])
pred = model.predict(X)

st.success(f"Loaded {len(df):,} rows, {len(expected)} features.")
tab1, tab2, tab3 = st.tabs(["📊 Metrics", "🧩 Confusion matrix & report", "🏆 All models"])

with tab1:
    st.subheader(f"{choice}: evaluation metrics")
    m = compute_metrics(model, X, y)
    for c, (k, v) in zip(st.columns(6), m.items()):
        c.metric(k, f"{v:.4f}")
    st.subheader("Class distribution in uploaded data")
    counts = pd.Series(y).map(dict(enumerate(CLASS_NAMES))).value_counts()
    st.bar_chart(counts)

with tab2:
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        rep = classification_report(y, pred, labels=[0, 1], target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3))

with tab3:
    st.subheader("All five models on the uploaded data")
    table = pd.DataFrame(
        [{"Model": n, **compute_metrics(load_model(k), X, y)} for n, k in MODELS.items()]
    ).round(4).set_index("Model")
    st.dataframe(table.style.highlight_max(axis=0, color="#b7e4c7"))
    st.caption(f"Best MCC: **{table['MCC'].idxmax()}**")
