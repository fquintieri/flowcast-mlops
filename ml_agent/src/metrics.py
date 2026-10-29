import numpy as np


def smape(y_true, y_pred):
    """SMAPE (Symmetric Mean Absolute Percentage Error), em %. Pares 0/0 contam como erro 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.divide(
        np.abs(y_pred - y_true), denominator,
        out=np.zeros_like(denominator), where=denominator != 0,
    )
    return 100 * np.mean(diff)


def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def bias(y_true, y_pred):
    """Erro medio (previsto - real): positivo = superestima a demanda."""
    return np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))


# Metricas de erro elegiveis como metrica de decisao (todas: menor e melhor).
DECISION_METRICS = {"smape": smape, "mae": mae, "rmse": rmse}


def compute_all(y_true, y_pred):
    scores = {name: float(fn(y_true, y_pred)) for name, fn in DECISION_METRICS.items()}
    scores["bias"] = float(bias(y_true, y_pred))
    return scores


def lgb_feval(metric_name):
    """Metrica customizada do LightGBM para usar a metrica de decisao no early stopping."""
    fn = DECISION_METRICS[metric_name]

    def feval(preds, eval_data):
        return metric_name, float(fn(eval_data.get_label(), preds)), False

    return feval
