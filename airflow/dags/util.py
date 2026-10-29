import json
import os
import re
import docker

# Pool de 1 slot que serializa todo acesso ao arquivo DuckDB (o lock e por processo:
# um escritor bloqueia qualquer outro processo, inclusive leitores read_only).
DUCKDB_POOL = "duckdb_writer"

# Parametros de ML repassados aos containers apenas quando definidos; os valores
# padrao vivem em ml_agent/src/config.py.
ML_CONFIG_VARS = ["HOLDOUT_DAYS", "DECISION_METRIC", "MIN_IMPROVEMENT", "MAX_BIAS_DEGRADATION"]


def get_ml_config_env():
    return {name: os.environ[name] for name in ML_CONFIG_VARS if os.environ.get(name)}

RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def parse_json_output(raw, source):
    """Le o JSON da ultima linha nao vazia publicada no XCom por um DockerOperator.

    O XCom traz o ultimo bloco do stream de logs (stdout + stderr), que pode ter
    mais de uma linha; o contrato dos scripts e terminar o stdout com uma linha JSON.
    """
    if raw is None:
        raise ValueError(f"A tarefa '{source}' nao publicou saida no XCom.")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        raise ValueError(f"A saida de '{source}' nao termina com uma linha JSON valida: {raw!r}")
    if not isinstance(payload, dict):
        raise ValueError(f"A saida de '{source}' deveria ser um objeto JSON: {payload!r}")
    return payload


def extract_run_id(raw, source="xcom"):
    """Extrai e valida o run_id (32 hex). A validacao impede injecao no comando do container."""
    run_id = parse_json_output(raw, source).get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.match(run_id):
        raise ValueError(f"run_id invalido na saida de '{source}': {run_id!r}")
    return run_id


def get_host_project_root():

    try:
        client = docker.DockerClient(base_url='unix://var/run/docker.sock')
        
        for container in client.containers.list():
            for mount in container.attrs.get('Mounts', []):
                if mount.get('Destination') == '/opt/airflow/dags':
                    host_dags_path = mount['Source'].replace('\\', '/')
                    if '/airflow/dags' in host_dags_path:
                        return host_dags_path.rsplit('/airflow/dags', 1)[0]
    except Exception as e:
        print(f"Aviso: Fallback ativado. Erro ao rastrear volumes: {e}")
        
    return os.getcwd()