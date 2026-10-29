import json
import os
import duckdb
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient  # <-- Nova importação necessária

from config import DB_PATH, DECISION_METRIC, EXPERIMENT_NAME, HOLDOUT_DAYS, get_holdout_start
from lineage import fingerprint_dataframe, get_git_commit
from metrics import compute_all, lgb_feval

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment and experiment.lifecycle_stage == "deleted":
    print(f"Restaurando experimento '{EXPERIMENT_NAME}' da lixeira do MLflow...")
    client.restore_experiment(experiment.experiment_id)

mlflow.set_experiment(EXPERIMENT_NAME)

def train():
    print(f"Conectando ao DuckDB em {DB_PATH}...")
    conn = duckdb.connect(DB_PATH)

    print("Carregando tabela 'features_sales'...")
    df = conn.execute("SELECT * FROM features_sales").df()
    holdout_start = pd.Timestamp(get_holdout_start(conn))
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    train_df = df[df['date'] < holdout_start]
    val_df = df[df['date'] >= holdout_start]

    print(f"Holdout: ultimos {HOLDOUT_DAYS} dias, a partir de {holdout_start.date()}")
    print(f"Registros de Treino: {len(train_df):,} | Validação: {len(val_df):,}")

    ignore_cols = ['date', 'store', 'item', 'sales']
    features = [c for c in df.columns if c not in ignore_cols]
    target = 'sales'

    print(f"Atributos utilizados ({len(features)}): {features}")

    git_commit = get_git_commit()
    if git_commit is None:
        print("AVISO: hash do commit indisponivel (.git nao montado em /app/.git); tag git_commit='unknown'.")
    reproducibility_tags = {
        "git_commit": git_commit or "unknown",
        "data_rows": str(len(df)),
        "data_train_rows": str(len(train_df)),
        "data_holdout_rows": str(len(val_df)),
        "data_max_date": str(df['date'].max().date()),
        "data_sha256": fingerprint_dataframe(df, ['date', 'store', 'item']),
    }
    print(f"Rastreabilidade: {reproducibility_tags}")

    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]

    params = {
        'objective': 'regression',
        # Desliga a metrica embutida: o early stopping usa a metrica de decisao via feval.
        'metric': 'None',
        'boosting_type': 'gbdt',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'n_jobs': -1
    }

    with mlflow.start_run(tags=reproducibility_tags) as run:
        print("\nTreinando o modelo LightGBM...")

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params=params,
            train_set=dtrain,
            valid_sets=[dval],
            feval=lgb_feval(DECISION_METRIC),
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )

        preds = model.predict(X_val)

        scores = compute_all(y_val.values, preds)

        mlflow.log_params(params)
        mlflow.log_params({
            "holdout_days": HOLDOUT_DAYS,
            "holdout_start": str(holdout_start.date()),
            "early_stopping_metric": DECISION_METRIC,
        })

        mlflow.log_metrics(scores)

        mlflow.lightgbm.log_model(model, artifact_path="model")

        print("\n--- Resultado da Validação ---")
        print(f"SMAPE: {scores['smape']:.4f}%")
        print(f"MAE:   {scores['mae']:.4f}")
        print(f"RMSE:  {scores['rmse']:.4f}")
        print(f"Bias:  {scores['bias']:.4f}")
        print(f"Early stopping por '{DECISION_METRIC}' na iteracao {model.best_iteration}")
        print("\nExperimento e métricas registrados com sucesso no MLflow!")
        run_id = run.info.run_id

    # Ultima linha do stdout: contrato lido pelo Airflow via XCom. Nada deve ser impresso depois.
    print(json.dumps({"run_id": run_id}), flush=True)


if __name__ == "__main__":
    train()