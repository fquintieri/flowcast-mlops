import argparse
import json
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

from config import EXPERIMENT_NAME, MODEL_NAME
from registry import CHAMPION_ALIAS

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

def promote_challenger_to_champion(run_id):
    print("--- Iniciando Promocao do Modelo Challenger para Champion ---")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"Erro: Experimento '{EXPERIMENT_NAME}' nao encontrado.")
        sys.exit(1)

    # Promove exatamente o run avaliado; nunca "o mais recente".
    run = client.get_run(run_id)
    if run.info.experiment_id != experiment.experiment_id:
        print(f"Erro: o run {run_id} nao pertence ao experimento '{EXPERIMENT_NAME}'.")
        sys.exit(1)

    decision = run.data.tags.get("decision")
    if decision != "promote":
        print(f"Erro: o run {run_id} nao foi aprovado na avaliacao (tag decision={decision!r}).")
        sys.exit(1)

    model_uri = f"runs:/{run_id}/model"

    print(f"Challenger localizado | Run ID: {run_id}")

    print(f"Registrando versao no Model Registry ('{MODEL_NAME}')...")
    model_version = mlflow.register_model(model_uri, MODEL_NAME)
    version_number = model_version.version

    print(f"Atribuindo o alias '{CHAMPION_ALIAS}' a versao {version_number}...")
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=version_number
    )

    print(f"Sucesso! Versao {version_number} do modelo agora e o '{CHAMPION_ALIAS}' ativo em Producao.\n")
    print(json.dumps({"promoted_run_id": run_id, "model": MODEL_NAME, "version": version_number}), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promove o run avaliado a Champion no Model Registry.")
    parser.add_argument("--run-id", required=True, help="run_id aprovado pelo evaluate_model.py")
    promote_challenger_to_champion(parser.parse_args().run_id)
