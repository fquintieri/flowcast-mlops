import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "flowcast_demand_forecasting"
MODEL_NAME = os.getenv("MODEL_NAME", "flowcast_demand_model")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

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

    challenger_run = runs[0]
    
    challenger_rmse = challenger_run.data.metrics.get("rmse") or challenger_run.data.metrics.get("RMSE")
        
    if challenger_rmse is None:
        print("Erro: O Challenger nao possui a metrica 'rmse' registrada no MLflow.")
        sys.exit(1)

    print(f"Challenger RMSE: {challenger_rmse:.4f}")

    champion_rmse = None
    try:
        model_version_details = client.get_model_version_by_alias(name=MODEL_NAME, alias="Champion")
        champion_run_id = model_version_details.run_id
        champion_run = client.get_run(champion_run_id)
        
        champion_rmse = champion_run.data.metrics.get("rmse") or champion_run.data.metrics.get("RMSE")
        print(f"Champion RMSE atual: {champion_rmse:.4f}")

    except Exception:
        print("Nenhum modelo @Champion encontrado. Este e um Cold Start.")

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