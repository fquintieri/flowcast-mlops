import os
import sys
import duckdb
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from sklearn.metrics import root_mean_squared_error

from config import DB_PATH, EXPERIMENT_NAME, HOLDOUT_DAYS, MODEL_NAME, get_holdout_start
from registry import CHAMPION_ALIAS, get_champion_version

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

def load_holdout():
    """Carrega o holdout comum (ultimos HOLDOUT_DAYS dias) usado para avaliar os dois modelos."""
    conn = duckdb.connect(DB_PATH, read_only=True)
    try:
        holdout_start = get_holdout_start(conn)
        holdout = conn.execute(
            "SELECT * FROM features_sales WHERE date >= ?", [holdout_start]
        ).df()
    finally:
        conn.close()

    if holdout.empty:
        raise ValueError(f"Holdout vazio: nenhuma linha em 'features_sales' a partir de {holdout_start}.")

    print(f"Holdout comum: ultimos {HOLDOUT_DAYS} dias, a partir de {holdout_start} ({len(holdout):,} linhas)")
    return holdout

def score_on_holdout(model, holdout, label):
    # Usa as colunas com que o proprio modelo foi treinado, na mesma ordem.
    features = model.feature_name()
    missing = [c for c in features if c not in holdout.columns]
    if missing:
        raise ValueError(f"{label}: atributos ausentes no holdout: {missing}")

    preds = model.predict(holdout[features])
    rmse = root_mean_squared_error(holdout["sales"].values, preds)
    print(f"{label} RMSE no holdout: {rmse:.4f}")
    return rmse

def evaluate():
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print("Erro: Experimento não encontrado no MLflow.")
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        print("Erro: Nenhuma execução encontrada para avaliar.")
        sys.exit(1)

    challenger_run_id = runs[0].info.run_id
    holdout = load_holdout()

    print(f"Carregando Challenger (run {challenger_run_id})...")
    challenger_model = mlflow.lightgbm.load_model(f"runs:/{challenger_run_id}/model")
    challenger_rmse = score_on_holdout(challenger_model, holdout, "Challenger")

    champion_rmse = None
    champion_version = get_champion_version(client)
    if champion_version is None:
        print(f"Nenhum modelo @{CHAMPION_ALIAS} encontrado. Este e um Cold Start.")
    else:
        print(f"Carregando Champion (versao {champion_version.version}, run {champion_version.run_id})...")
        champion_model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}/{champion_version.version}")
        champion_rmse = score_on_holdout(champion_model, holdout, "Champion")

    if champion_rmse is None:
        print("Como nao ha Champion anterior, o Challenger vence por W.O.")
        print("promover_modelo_champion")

    elif challenger_rmse < champion_rmse:
        print(f"O Challenger superou o Champion! ({challenger_rmse:.4f} < {champion_rmse:.4f})")
        print("promover_modelo_champion")

    else:
        print(f"O Challenger NAO superou o Champion. ({challenger_rmse:.4f} >= {champion_rmse:.4f})")
        print("rejeitar_modelo_challenger")

if __name__ == "__main__":
    evaluate()
