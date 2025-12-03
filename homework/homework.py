# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".

import gzip
import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (precision_score, balanced_accuracy_score,
                                recall_score, f1_score, confusion_matrix)

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)
    df["EDUCATION"] = df["EDUCATION"].replace(0, np.nan)
    df["MARRIAGE"] = df["MARRIAGE"].replace(0, np.nan)
    df.dropna(inplace=True)
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
    return df

# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.

def split_data(df):
    X = df.drop(columns=["default"])
    y = df["default"]
    return X, y

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.

def create_pipeline(df):
    categorical = ['SEX', 'EDUCATION', 'MARRIAGE']
    numerical = [col for col in df.columns if col not in categorical]
    preprocessor = ColumnTransformer(transformers=[
        ("num", MinMaxScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("select_k_best", SelectKBest(score_func=f_classif)),
        ("model", LogisticRegression())
    ])
    return pipeline

# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.

def optimize_hyperparameters(pipeline, x_train, y_train):
    param_grid = {
        "model__C": [0.7, 0.8, 0.9],
        "model__solver": ["liblinear", "saga"],
        "model__max_iter": [1500],
        "select_k_best__k": [1, 2, 5, 10]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), scoring='balanced_accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    return grid_search

# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.

def save_model(model, file_path):
    os.makedirs('files/models', exist_ok=True)
    with gzip.open(file_path, 'wb') as file:
        pickle.dump(model, file)

# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}

def calculate_metrics(model, x, y, dataset_name):
    y_pred = model.predict(x)
    metric = {
        'type': 'metrics',
        'dataset': dataset_name,
        'precision': precision_score(y, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y, y_pred),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1_score': f1_score(y, y_pred, zero_division=0)
    }
    return metric

# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}

def calculate_confusion_matrix(model, x, y, dataset_name):
    y_pred = model.predict(x)
    cm = confusion_matrix(y, y_pred)
    cm_dict = {
        'type': 'cm_matrix',
        'dataset': dataset_name,
        'true_0': {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        'true_1': {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
    }
    return cm_dict

def main():
    train_df = load_and_clean_data("files/input/train_data.csv.zip")
    test_df = load_and_clean_data("files/input/test_data.csv.zip")

    x_train, y_train = split_data(train_df)
    x_test, y_test = split_data(test_df)

    pipeline = create_pipeline(x_train)
    model = optimize_hyperparameters(pipeline, x_train, y_train)
    save_model(model, "files/models/model.pkl.gz")

    metrics = []
    metrics.append(calculate_metrics(model, x_train, y_train, 'train'))
    metrics.append(calculate_metrics(model, x_test, y_test, 'test'))
    metrics.append(calculate_confusion_matrix(model, x_train, y_train, 'train'))
    metrics.append(calculate_confusion_matrix(model, x_test, y_test, 'test'))

    os.makedirs('files/output', exist_ok=True)
    with open("files/output/metrics.json", 'w', encoding='utf-8') as file:
        for metric in metrics:
            file.write(json.dumps(metric) + '\n')

if __name__ == "__main__":
    main()