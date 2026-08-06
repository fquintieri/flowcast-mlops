import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "flowcast_demand_model")
EXPERIMENT_NAME = "flowcast_demand_forecasting"

def promote_challenger_to_champion():
    print("--- Iniciando Promocao do Modelo Challenger para Champion ---")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"Erro: Experimento '{EXPERIMENT_NAME}' nao encontrado.")
        sys.exit(1)

    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],  # <--- Agora usa a variável no lugar do "0"
        order_by=["attribute.start_time DESC"],
        max_results=1
    )

    if not runs:
        print("Erro: Nenhuma execucao encontrada no MLflow para promover.")
        sys.exit(1)

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    print(f"Challenger localizado | Run ID: {run_id}")

    print(f"Registrando versao no Model Registry ('{MODEL_NAME}')...")
    model_version = mlflow.register_model(model_uri, MODEL_NAME)
    version_number = model_version.version

    print(f"Atribuindo o alias 'Champion' a versao {version_number}...")
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="Champion",
        version=version_number
    )

    print(f"Sucesso! Versao {version_number} do modelo agora e o 'Champion' ativo em Producao.\n")

if __name__ == "__main__":
    promote_challenger_to_champion()