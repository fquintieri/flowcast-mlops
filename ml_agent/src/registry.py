from mlflow.exceptions import MlflowException

from config import MODEL_NAME

CHAMPION_ALIAS = "Champion"


def _is_missing_alias_error(exc):
    # MLflow 2.14 responde INVALID_PARAMETER_VALUE "Registered model alias X not found."
    # para alias inexistente; RESOURCE_DOES_NOT_EXIST cobre o modelo nao registrado.
    if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
        return True
    return exc.error_code == "INVALID_PARAMETER_VALUE" and "alias" in exc.message and "not found" in exc.message


def get_champion_version(client):
    """Retorna a ModelVersion @Champion, ou None se ainda nao houver Champion.

    Qualquer outro erro (MLflow fora do ar, permissao etc.) e propagado.
    """
    try:
        return client.get_model_version_by_alias(name=MODEL_NAME, alias=CHAMPION_ALIAS)
    except MlflowException as e:
        if _is_missing_alias_error(e):
            return None
        raise
