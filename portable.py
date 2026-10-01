"""Portable XGBoost model loader/exporter for the study-office assignment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

def export(pipe, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    prep, clf = pipe[0], pipe[-1]
    num_pipe, num_cols = prep.named_transformers_["num"], list(prep.transformers_[0][2])
    imputer, scaler = num_pipe[0], num_pipe[-1]
    onehot, cat_cols = prep.named_transformers_["cat"], list(prep.transformers_[1][2])
    infrequent = getattr(onehot, "infrequent_categories_", None)
    if infrequent is None:
        infrequent = [None] * len(cat_cols)
    spec = {
        "numeric": [{"name": c, "median": float(imputer.statistics_[i]), "mean": float(scaler.mean_[i]), "scale": float(scaler.scale_[i])} for i,c in enumerate(num_cols)],
        "categorical": [{"name": c, "frequent": [str(v) for v in onehot.categories_[i] if infrequent[i] is None or v not in infrequent[i]], "has_infrequent": infrequent[i] is not None} for i,c in enumerate(cat_cols)],
        "columns": [n.split("__",1)[1] for n in prep.get_feature_names_out()],
    }
    (folder/"preprocess.json").write_text(json.dumps(spec, indent=2))
    clf.get_booster().save_model(str(folder/"booster.json"))

class Model:
    def __init__(self, folder):
        folder=Path(folder)
        self.spec=json.loads((folder/"preprocess.json").read_text())
        self.booster=xgb.Booster()
        self.booster.load_model(str(folder/"booster.json"))
        self.features=[n["name"] for n in self.spec["numeric"]]+[c["name"] for c in self.spec["categorical"]]

    def transform(self, df):
        parts=[]
        for n in self.spec["numeric"]:
            x=pd.to_numeric(df[n["name"]], errors="coerce").astype(float).fillna(n["median"])
            parts.append(((x-n["mean"])/n["scale"]).to_numpy()[:,None])
        for c in self.spec["categorical"]:
            v=df[c["name"]].astype(str).to_numpy()
            frequent=c["frequent"]
            if frequent:
                parts.append(np.stack([v==level for level in frequent],axis=1).astype(float))
            if c["has_infrequent"]:
                parts.append((~np.isin(v,frequent)).astype(float)[:,None])
        return np.hstack(parts)

    def _dmatrix(self, df):
        return xgb.DMatrix(self.transform(df))

    def predict_proba(self, df):
        return self.booster.predict(self._dmatrix(df))

    def contributions(self, df):
        contrib=self.booster.predict(self._dmatrix(df), pred_contribs=True)[:,:-1]
        owner=[]
        for col in self.spec["columns"]:
            clean=col.split("__",1)[1] if "__" in col else col
            owner.append(next((f for f in self.features if clean==f or clean.startswith(f+"_")), clean))
        return pd.DataFrame(contrib,columns=owner,index=df.index).T.groupby(level=0).sum().T[self.features]
