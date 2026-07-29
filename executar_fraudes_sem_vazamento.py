from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from xgboost import XGBClassifier


URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
OUTPUT = Path("resultado_sem_vazamento.txt")
RANDOM_STATE = 42
inicio = perf_counter()


def metricas(y_true, y_pred, y_prob):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def formatar_metricas(nome, valores):
    return f"""\
{nome}
Precisão da fraude: {valores['precision']:.6f}
Recall da fraude: {valores['recall']:.6f}
F1-score da fraude: {valores['f1']:.6f}
Average Precision (PR-AUC): {valores['average_precision']:.6f}
ROC-AUC: {valores['roc_auc']:.6f}
Matriz de confusão: TN={valores['tn']}, FP={valores['fp']}, FN={valores['fn']}, TP={valores['tp']}
"""


print("Carregando dados...", flush=True)
df = pd.read_csv(URL)

# Engenharia determinística: não aprende estatísticas do conjunto completo.
df["Amount_log"] = np.log1p(df["Amount"])
X = df.drop(columns="Class")
y = df["Class"]

# O teste é separado antes de qualquer SMOTE, ajuste ou escolha de threshold.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(
    f"Treino: {len(X_train)} registros ({int(y_train.sum())} fraudes); "
    f"teste intocado: {len(X_test)} registros ({int(y_test.sum())} fraudes).",
    flush=True,
)

# Referência sem SMOTE, equivalente ao melhor modelo confiável anterior.
print("Treinando XGBoost de referência...", flush=True)
baseline = XGBClassifier(
    scale_pos_weight=10,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=2,
)
baseline.fit(X_train, y_train)
baseline_probs = baseline.predict_proba(X_test)[:, 1]
baseline_pred = (baseline_probs >= 0.5).astype(int)
resultado_baseline = metricas(y_test, baseline_pred, baseline_probs)

# O Pipeline garante que o SMOTE seja aplicado somente à parcela de treino
# de cada dobra, nunca à validação e nunca ao teste final.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
pipeline_smote = Pipeline(
    steps=[
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        (
            "model",
            XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=2,
            ),
        ),
    ]
)

param_grid = {
    "model__max_depth": [3, 5],
    "model__n_estimators": [50, 100],
}

print("Selecionando hiperparâmetros sem acesso ao teste...", flush=True)
grid = GridSearchCV(
    estimator=pipeline_smote,
    param_grid=param_grid,
    scoring="average_precision",
    cv=cv,
    n_jobs=-1,
    refit=True,
)
grid.fit(X_train, y_train)

print(f"Melhores parâmetros: {grid.best_params_}", flush=True)

# O threshold é escolhido com probabilidades out-of-fold do conjunto de
# treino. Cada observação é prevista por um modelo que não a utilizou no ajuste.
print("Gerando probabilidades out-of-fold para escolher o threshold...", flush=True)
oof_probs = cross_val_predict(
    grid.best_estimator_,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

precision_oof, recall_oof, thresholds_oof = precision_recall_curve(
    y_train, oof_probs
)
precision_threshold = precision_oof[:-1]
recall_threshold = recall_oof[:-1]
denominador = precision_threshold + recall_threshold
f1_oof = np.divide(
    2 * precision_threshold * recall_threshold,
    denominador,
    out=np.zeros_like(denominador),
    where=denominador != 0,
)
best_idx = int(np.argmax(f1_oof))
best_threshold = float(thresholds_oof[best_idx])
best_f1_oof = float(f1_oof[best_idx])

# Somente agora o teste intocado é utilizado.
final_model = grid.best_estimator_
test_probs = final_model.predict_proba(X_test)[:, 1]

pred_padrao = (test_probs >= 0.5).astype(int)
pred_otimizado = (test_probs >= best_threshold).astype(int)

resultado_smote_padrao = metricas(y_test, pred_padrao, test_probs)
resultado_smote_otimizado = metricas(y_test, pred_otimizado, test_probs)

tempo = perf_counter() - inicio

relatorio = f"""\
EXPERIMENTO DE DETECÇÃO DE FRAUDES SEM VAZAMENTO DE DADOS

Registros: {len(df)}
Fraudes totais: {int(y.sum())}
Treino: {len(X_train)} registros, com {int(y_train.sum())} fraudes
Teste intocado: {len(X_test)} registros, com {int(y_test.sum())} fraudes

Prevenções adotadas:
- separação do teste antes de qualquer reamostragem;
- SMOTE restrito à parcela de treino de cada dobra;
- hiperparâmetros escolhidos somente no treino;
- threshold escolhido por previsões out-of-fold do treino;
- teste utilizado uma única vez, na avaliação final;
- random_state=42 em todas as etapas estocásticas.

Melhores parâmetros: {grid.best_params_}
Average Precision média da validação cruzada: {grid.best_score_:.6f}
Threshold escolhido no treino: {best_threshold:.8f}
F1 out-of-fold correspondente: {best_f1_oof:.6f}

{formatar_metricas("XGBOOST DE REFERÊNCIA, SEM SMOTE", resultado_baseline)}
{formatar_metricas("XGBOOST COM SMOTE E THRESHOLD 0,5", resultado_smote_padrao)}
{formatar_metricas("XGBOOST COM SMOTE E THRESHOLD ESCOLHIDO NO TREINO", resultado_smote_otimizado)}
Relatório detalhado do modelo final:
{classification_report(y_test, pred_otimizado, digits=6, zero_division=0)}

Referências das execuções anteriores:
- XGBoost sem SMOTE: precisão 0,94; recall 0,78; F1 0,85.
- SMOTE com vazamento e threshold otimizado: precisão 0,92; recall 0,99; F1 0,95.

Tempo total: {tempo:.2f} segundos
"""

OUTPUT.write_text(relatorio, encoding="utf-8")
print("\n" + relatorio, flush=True)
