import os
from datetime import timedelta

from metrics import DECISION_METRICS

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


def _float_env(name, default, min_value, max_value):
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"Variavel de ambiente {name}='{raw}' deve ser numerica.")
    if not min_value <= value <= max_value:
        raise ValueError(f"Variavel de ambiente {name}={value} fora do intervalo [{min_value}, {max_value}].")
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


# Metrica unica de decisao: usada no early stopping do treino e na promocao.
DECISION_METRIC = (os.getenv("DECISION_METRIC") or "smape").lower()
if DECISION_METRIC not in DECISION_METRICS:
    raise ValueError(
        f"DECISION_METRIC='{DECISION_METRIC}' invalida. Opcoes: {sorted(DECISION_METRICS)}."
    )

# Melhora relativa minima do Challenger sobre o Champion para promover (0.01 = 1%).
MIN_IMPROVEMENT = _float_env("MIN_IMPROVEMENT", 0.01, 0.0, 0.99)

# Trava de bias: rejeita se |bias| do Challenger superar o do Champion em mais que
# este valor, em unidades de venda.
MAX_BIAS_DEGRADATION = _float_env("MAX_BIAS_DEGRADATION", 0.5, 0.0, float("inf"))
