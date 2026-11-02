"""Train and save the five classifiers on the UCI Phishing Websites dataset.

Usage:
    python train.py                       # auto-detect the dataset file, else download from UCI
    python train.py path/to/dataset.csv   # .csv, .arff or .xlsx

Outputs: model/*.joblib, model/metrics_table.csv, results_table.md,
         test_data.csv (held-out 20%), train_data.csv (80%)
"""
import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from utils import compute_metrics, encode_target

TARGET = "Result"
CANDIDATES = ["Training Dataset.arff", "phishing.csv", "Phishing_Websites.csv",
              "dataset.csv", "Training Dataset.csv"]


def load_raw():
    paths = ([sys.argv[1]] if len(sys.argv) > 1 else []) + CANDIDATES
    for p in paths:
        if os.path.exists(p):
            print("Loading", p)
            if p.lower().endswith(".arff"):
                from scipy.io import arff
                data, _ = arff.loadarff(p)
                df = pd.DataFrame(data)
                return df.apply(lambda c: c.str.decode("utf-8") if c.dtype == object else c)
            if p.lower().endswith(".xlsx"):
                return pd.read_excel(p)
            return pd.read_csv(p)
    print("No local file found; downloading from the UCI repository (id=327)...")
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=327)
    return pd.concat([ds.data.features, ds.data.targets], axis=1)


df = load_raw()
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={c: TARGET for c in df.columns if c.lower() == TARGET.lower()})
df = df.drop(columns=[c for c in df.columns if c.lower() in ("id", "index")])
df = df.apply(pd.to_numeric).astype(int)
print("Shape:", df.shape)
print(df[TARGET].value_counts())

df = df.dropna().drop_duplicates().reset_index(drop=True)
print("Shape after removing duplicates:", df.shape)

X = df.drop(columns=[TARGET])
y = encode_target(df[TARGET])          # 1 = Phishing, 0 = Legitimate

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", C=10, probability=True, random_state=42),
}

os.makedirs("model", exist_ok=True)
rows = []
for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    joblib.dump(pipe, f"model/{name.replace(' ', '_').lower()}.joblib")
    rows.append({"ML Model Name": name, **compute_metrics(pipe, X_te, y_te)})
    print("done:", name)

# Test CSV: raw features + original 'Result' column (-1 phishing, 1 legitimate)
X_te.assign(**{TARGET: df.loc[X_te.index, TARGET]}).to_csv("test_data.csv", index=False)
X_tr.assign(**{TARGET: df.loc[X_tr.index, TARGET]}).to_csv("train_data.csv", index=False)

res = pd.DataFrame(rows).round(4)
res.to_csv("model/metrics_table.csv", index=False)
md = res.to_markdown(index=False)
open("results_table.md", "w").write(md + "\n")
print(md)   # paste this into README.md
