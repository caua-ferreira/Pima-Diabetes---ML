#!/usr/bin/env python3
"""
pima_diabetes_pipeline.py

Fluxo em pandas + scikit-learn que replica o notebook:
- carrega a base a partir de URLs (Tenta UCI e depois GitHub)
- substitui zeros inválidos por NaN e imputa pela mediana
- escala as features
- treina LogisticRegression e RandomForest com GridSearch (AUC)
- avalia no conjunto de teste e salva o melhor pipeline em disco

Uso:
    python pima_diabetes_pipeline.py --model-out best_model.joblib

Requisitos:
    pip install pandas scikit-learn joblib matplotlib seaborn

"""
import argparse
import logging
import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_URLS: List[str] = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
]


def download_df(urls: List[str], cols: List[str]) -> pd.DataFrame:
    """Tenta carregar a tabela a partir das URLs em ordem. Retorna DataFrame."""
    for url in urls:
        try:
            df = pd.read_csv(url, header=None)
            if df.shape[1] == len(cols):
                df.columns = cols
            else:
                # tentar assumir que último é label se houver +1
                if df.shape[1] == len(cols):
                    df.columns = cols
            print(f"Carregado de: {url} - shape={df.shape}")
            return df
        except Exception as e:
            print(f"Falha ao carregar de {url}: {e}")
    raise RuntimeError("Não foi possível baixar a base de dados das URLs fornecidas")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df = df.copy()
    # Substituir zeros clinicamente inválidos por NaN
    for c in zero_invalid:
        if c in df.columns:
            df.loc[df[c] == 0, c] = np.nan

    # Imputar medianas
    imputer = SimpleImputer(strategy="median")
    df[zero_invalid] = imputer.fit_transform(df[zero_invalid])

    return df


def build_and_search(X_train, y_train):
    """Constrói pipelines e executa GridSearch para LR e RF; retorna dict com resultados."""
    results = {}

    # Logistic Regression pipeline
    pipe_lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(solver="liblinear", max_iter=200))])
    param_lr = {
        "lr__C": [0.01, 0.1, 1.0],
        "lr__penalty": ["l2"]
    }
    gs_lr = GridSearchCV(pipe_lr, param_grid=param_lr, scoring="roc_auc", cv=5, n_jobs=-1)
    gs_lr.fit(X_train, y_train)
    results["lr"] = gs_lr

    # Random Forest pipeline
    pipe_rf = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(random_state=42))])
    param_rf = {
        "rf__n_estimators": [50, 100],
        "rf__max_depth": [5, 10]
    }
    gs_rf = GridSearchCV(pipe_rf, param_grid=param_rf, scoring="roc_auc", cv=5, n_jobs=-1)
    gs_rf.fit(X_train, y_train)
    results["rf"] = gs_rf

    return results


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba) if proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, preds)
    }
    return metrics, preds, proba


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pima diabetes pipeline (pandas + scikit-learn)")
    parser.add_argument("--model-out", default="best_model.joblib", help="Caminho para salvar o melhor modelo/pipeline")
    parser.add_argument("--shap", action="store_true", help="Gerar análise SHAP (se shap instalado)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

    df = download_df(DEFAULT_URLS, cols)
    df = preprocess(df)

    feature_cols = [c for c in cols if c != "Outcome"]
    X = df[feature_cols].astype(float)
    y = df["Outcome"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    logging.info(f"Treino: {X_train.shape}, Teste: {X_test.shape}")

    # Grid search
    results = build_and_search(X_train, y_train)

    # Escolher melhor por AUC
    best_name = None
    best_score = -1.0
    best_model = None
    for name, gs in results.items():
        score = gs.best_score_
        logging.info(f"Melhor CV {name}: {score:.4f} - params: {gs.best_params_}")
        if score > best_score:
            best_score = score
            best_name = name
            best_model = gs.best_estimator_

    logging.info(f"Modelo selecionado: {best_name} (CV AUC={best_score:.4f})")

    # Avaliar no teste
    metrics, preds, proba = evaluate_model(best_model, X_test, y_test)
    logging.info(f"Metrics test: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, ROC AUC={metrics['roc_auc']}")
    logging.info(f"Confusion matrix:\n{metrics['confusion_matrix']}")

    # Salvar modelo
    joblib.dump({"model": best_model, "feature_cols": feature_cols}, args.model_out)
    logging.info(f"Melhor pipeline salvo em: {args.model_out}")

    # Opcional: SHAP (treina um RF sklearn na amostra para explicabilidade se solicitado)
    if args.shap:
        try:
            import shap
            import matplotlib.pyplot as plt

            # treinar um RF simples na amostra (usamos treino completo já escalado dentro do pipeline)
            sample_X = X_train.sample(frac=0.2, random_state=42)
            sample_y = y_train.loc[sample_X.index]
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(sample_X, sample_y)

            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(sample_X)
            shap.summary_plot(shap_values, sample_X, show=False)
            plt.tight_layout()
            plt.savefig("shap_summary.png")
            logging.info("SHAP summary salvo em shap_summary.png")
        except Exception as e:
            logging.warning(f"SHAP falhou: {e} - instale 'shap' e 'matplotlib' para gerar plots")


if __name__ == "__main__":
    main()
