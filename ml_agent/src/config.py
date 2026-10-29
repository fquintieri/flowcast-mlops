import os
from datetime import timedelta

DB_PATH = "/app/data/database.duckdb"
EXPERIMENT_NAME = "flowcast_demand_forecasting"
MODEL_NAME = os.getenv("MODEL_NAME") or "flowcast_demand_model"


def _int_env(name, default):
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"Variavel de ambiente {name}='{raw}' deve ser um inteiro.")
    if value <= 0:
        raise ValueError(f"Variavel de ambiente {name}={value} deve ser maior que zero.")
    return value


# Janela de holdout (validacao do treino e avaliacao Champion vs Challenger),
# relativa a data maxima dos dados. Definida apenas aqui.
HOLDOUT_DAYS = _int_env("HOLDOUT_DAYS", 90)


def get_holdout_start(conn):
    """Primeiro dia do holdout: os ultimos HOLDOUT_DAYS dias de features_sales."""
    max_date = conn.execute("SELECT MAX(date) FROM features_sales").fetchone()[0]
    if max_date is None:
        raise ValueError("A tabela 'features_sales' esta vazia; impossivel definir o holdout.")
    return max_date - timedelta(days=HOLDOUT_DAYS - 1)
