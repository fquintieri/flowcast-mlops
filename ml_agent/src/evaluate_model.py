import os
import sys
import duckdb
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

from config import (
    DB_PATH, DECISION_METRIC, EXPERIMENT_NAME, HOLDOUT_DAYS, MAX_BIAS_DEGRADATION,
    MIN_IMPROVEMENT, MODEL_NAME, get_holdout_start,
)
from metrics import compute_all
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
    scores = compute_all(holdout["sales"].values, preds)
    print(f"{label} no holdout: " + " | ".join(f"{k}={v:.4f}" for k, v in scores.items()))
    return scores

def decide(challenger, champion):
    """Retorna (decisao, motivo). Promove so com melhora minima e sem piora de bias."""
    if champion is None:
        return "promote", "cold_start"

    required = champion[DECISION_METRIC] * (1 - MIN_IMPROVEMENT)
    print(
        f"Criterio: {DECISION_METRIC} do Challenger <= {required:.4f} "
        f"(Champion {champion[DECISION_METRIC]:.4f} com melhora minima de {MIN_IMPROVEMENT:.2%})"
    )
    if challenger[DECISION_METRIC] > required:
        return "reject", "insufficient_improvement"

    bias_degradation = abs(challenger["bias"]) - abs(champion["bias"])
    print(f"Trava de bias: piora de |bias| = {bias_degradation:+.4f} (limite {MAX_BIAS_DEGRADATION})")
    if bias_degradation > MAX_BIAS_DEGRADATION:
        return "reject", "bias_degradation"

    return "promote", "improvement"

def log_decision(run_id, challenger, champion, champion_version, decision, reason):
    """Registra no run do Challenger a metrica, a margem e o resultado da decisao."""
    client.log_param(run_id, "decision_metric", DECISION_METRIC)
    client.log_param(run_id, "decision_min_improvement", MIN_IMPROVEMENT)
    client.log_param(run_id, "decision_max_bias_degradation", MAX_BIAS_DEGRADATION)
    for name, value in challenger.items():
        client.log_metric(run_id, f"eval_{name}_challenger", value)
    for name, value in (champion or {}).items():
        client.log_metric(run_id, f"eval_{name}_champion", value)
    client.set_tag(run_id, "decision", decision)
    client.set_tag(run_id, "decision_reason", reason)
    client.set_tag(run_id, "decision_champion_version", champion_version.version if champion_version else "none")

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
    challenger_scores = score_on_holdout(challenger_model, holdout, "Challenger")

    champion_scores = None
    champion_version = get_champion_version(client)
    if champion_version is None:
        print(f"Nenhum modelo @{CHAMPION_ALIAS} encontrado. Este e um Cold Start.")
    else:
        print(f"Carregando Champion (versao {champion_version.version}, run {champion_version.run_id})...")
        champion_model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}/{champion_version.version}")
        champion_scores = score_on_holdout(champion_model, holdout, "Champion")

    decision, reason = decide(challenger_scores, champion_scores)
    log_decision(challenger_run_id, challenger_scores, champion_scores, champion_version, decision, reason)
    print(f"Decisao: {decision} (motivo: {reason}, metrica: {DECISION_METRIC})")

    if decision == "promote":
        print("promover_modelo_champion")
    else:
        print("rejeitar_modelo_challenger")

if __name__ == "__main__":
    evaluate()
