import os
import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient  # <-- Nova importação necessária

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "flowcast_demand_forecasting"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment and experiment.lifecycle_stage == "deleted":
    print(f"Restaurando experimento '{EXPERIMENT_NAME}' da lixeira do MLflow...")
    client.restore_experiment(experiment.experiment_id)

mlflow.set_experiment(EXPERIMENT_NAME)

def calculate_smape(y_true, y_pred):
    """Calcula a métrica SMAPE (Symmetric Mean Absolute Percentage Error)."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

def train():
    db_path = "/app/data/database.duckdb"
    print(f"Conectando ao DuckDB em {db_path}...")
    conn = duckdb.connect(db_path)

    print("Carregando tabela 'features_sales'...")
    df = conn.execute("SELECT * FROM features_sales").df()
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    split_date = '2017-10-01'
    train_df = df[df['date'] < split_date]
    val_df = df[df['date'] >= split_date]

    print(f"Registros de Treino: {len(train_df):,} | Validação: {len(val_df):,}")

    ignore_cols = ['date', 'store', 'item', 'sales']
    features = [c for c in df.columns if c not in ignore_cols]
    target = 'sales'

    print(f"Atributos utilizados ({len(features)}): {features}")

    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'n_jobs': -1
    }

    with mlflow.start_run():
        print("\nTreinando o modelo LightGBM...")

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params=params,
            train_set=dtrain,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )

        preds = model.predict(X_val)

        smape_score = calculate_smape(y_val.values, preds)
        mae_score = mean_absolute_error(y_val, preds)
        rmse_score = root_mean_squared_error(y_val, preds)
        bias_score = np.mean(preds - y_val.values)

        mlflow.log_params(params)

        mlflow.log_metrics({
            "smape": smape_score,
            "mae": mae_score,
            "rmse": rmse_score,
            "bias": bias_score
        })

        mlflow.lightgbm.log_model(model, artifact_path="model")

        print("\n--- Resultado da Validação ---")
        print(f"SMAPE: {smape_score:.4f}%")
        print(f"MAE:   {mae_score:.4f}")
        print(f"RMSE:  {rmse_score:.4f}")
        print(f"Bias:  {bias_score:.4f}")
        print("\nExperimento e métricas registrados com sucesso no MLflow!")


if __name__ == "__main__":
    train()