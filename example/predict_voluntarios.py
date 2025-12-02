#!/usr/bin/env python3
"""
Example script to load `voluntarios.json`, preprocess and predict using saved `best_model.joblib`.

Usage:
    python example/predict_voluntarios.py --model best_model.joblib --input voluntarios.json --out predictions.csv

"""
import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import numpy as np


# Local copy of preprocess to ensure same behavior when running example
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df = df.copy()
    for c in zero_invalid:
        if c in df.columns:
            df.loc[df[c] == 0, c] = np.nan

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    # only operate on columns that exist
    exist_cols = [c for c in zero_invalid if c in df.columns]
    if exist_cols:
        df[exist_cols] = imputer.fit_transform(df[exist_cols])

    return df


def load_voluntarios(path: str) -> pd.DataFrame:
    df = pd.read_json(path)
    # keep order of expected features; pipeline expects numeric columns
    return df


def main():
    parser = argparse.ArgumentParser(description="Predict voluntarios with saved model")
    parser.add_argument("--model", default="best_model.joblib", help="Caminho para o modelo salvo (joblib)")
    parser.add_argument("--input", default="voluntarios.json", help="Arquivo JSON com voluntários")
    parser.add_argument("--out", default="predictions.csv", help="Arquivo de saída CSV com probabilidades e rótulos")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Modelo não encontrado: {args.model}")
    if not Path(args.input).exists():
        raise SystemExit(f"Arquivo de entrada não encontrado: {args.input}")

    df = load_voluntarios(args.input)
    ids = None
    if "id" in df.columns:
        ids = df["id"].copy()
        df = df.drop(columns=["id"])

    # Apply same preprocessing as training (impute zeros etc.)
    df_proc = preprocess(df)

    saved = joblib.load(args.model)
    model = saved.get("model") if isinstance(saved, dict) else saved
    feature_cols = saved.get("feature_cols") if isinstance(saved, dict) else None

    if feature_cols is not None:
        X = df_proc[feature_cols]
    else:
        X = df_proc

    proba = None
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        # fallback to predict
        preds = model.predict(X)
    else:
        preds = (proba >= 0.5).astype(int)

    out = pd.DataFrame({
        "id": ids if ids is not None else range(len(preds)),
        "prediction": preds,
        "probability": proba if proba is not None else None,
    })

    out.to_csv(args.out, index=False)
    print(f"Predictions saved to {args.out}")


if __name__ == "__main__":
    main()
