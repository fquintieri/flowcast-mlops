import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from util import (
    DUCKDB_POOL, extract_run_id, get_host_project_root, get_ml_config_env, parse_json_output,
)

HOST_ROOT = get_host_project_root()
HOST_DATA_PATH = f"{HOST_ROOT}/data"
HOST_SRC_PATH = f"{HOST_ROOT}/ml_agent/src"
HOST_GIT_PATH = f"{HOST_ROOT}/.git"

default_args = {
    "owner": "flowcast_mlops",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

TRAIN_TASK_ID = "treinar_modelo_challenger"
EVALUATE_TASK_ID = "avaliar_challenger_vs_champion"

def route_evaluation(**context):
    ti = context['ti']
    trained_run_id = extract_run_id(ti.xcom_pull(task_ids=TRAIN_TASK_ID), TRAIN_TASK_ID)
    decision = parse_json_output(ti.xcom_pull(task_ids=EVALUATE_TASK_ID), EVALUATE_TASK_ID)
    print(f"Decisao recebida: {decision}")

    if decision.get("run_id") != trained_run_id:
        raise ValueError(
            f"O run avaliado ({decision.get('run_id')}) difere do run treinado ({trained_run_id})."
        )
    if decision.get("decision") == "promote":
        return "promover_modelo_champion"
    if decision.get("decision") == "reject":
        return "rejeitar_modelo_challenger"
    raise ValueError(f"Decisao desconhecida na saida da avaliacao: {decision.get('decision')!r}")

with DAG(
    dag_id="flowcast_weekly_retrain",
    default_args=default_args,
    description="Pipeline Semanal de Retreinamento, Avaliacao Champion vs Challenger e Promocao",
    schedule="0 4 * * 0",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "weekly", "training"],
    user_defined_filters={"run_id": extract_run_id},
) as dag:

    ingest_data = DockerOperator(
        task_id="ingestao_de_dados",
        pool=DUCKDB_POOL,
        image="flowcast-mlagent:latest",
        api_version="auto",
        auto_remove="force",
        command="python -u src/data_prep.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            Mount(source=HOST_DATA_PATH, target="/app/data", type="bind"),
            Mount(source=HOST_SRC_PATH, target="/app/src", type="bind")
        ],
        environment={"KAGGLE_API_TOKEN": os.getenv("KAGGLE_API_TOKEN", "")}
    )

    feature_engineering = DockerOperator(
        task_id="engenharia_de_atributos",
        pool=DUCKDB_POOL,
        image="flowcast-mlagent:latest",
        api_version="auto",
        auto_remove="force",
        command="python -u src/feature_engineering.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            Mount(source=HOST_DATA_PATH, target="/app/data", type="bind"),
            Mount(source=HOST_SRC_PATH, target="/app/src", type="bind")
        ]
    )

    train_challenger = DockerOperator(
        task_id=TRAIN_TASK_ID,
        pool=DUCKDB_POOL,
        image="flowcast-mlagent:latest",
        api_version="auto",
        auto_remove="force",
        command="python -u src/train.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="flowcast-mlops_default",
        mounts=[
            Mount(source=HOST_DATA_PATH, target="/app/data", type="bind"),
            Mount(source=HOST_SRC_PATH, target="/app/src", type="bind"),
            # Somente leitura: permite registrar o hash do commit no MLflow
            Mount(source=HOST_GIT_PATH, target="/app/.git", type="bind", read_only=True),
        ],
        environment={
            "MLFLOW_TRACKING_URI": "http://flowcast_mlflow_ui:5000",
            **get_ml_config_env(),
        },
        do_xcom_push=True, # Publica {"run_id": ...} (ultima linha do stdout) para avaliacao e promocao
    )

    evaluate_model = DockerOperator(
        task_id=EVALUATE_TASK_ID,
        pool=DUCKDB_POOL,
        image="flowcast-mlagent:latest",
        api_version="auto",
        auto_remove="force",
        command=f"python -u src/evaluate_model.py --run-id {{{{ ti.xcom_pull(task_ids='{TRAIN_TASK_ID}') | run_id }}}}",
        docker_url="unix://var/run/docker.sock",
        network_mode="flowcast-mlops_default",
        mounts=[
            Mount(source=HOST_DATA_PATH, target="/app/data", type="bind"),
            Mount(source=HOST_SRC_PATH, target="/app/src", type="bind")
        ],
        environment={
            "MLFLOW_TRACKING_URI": "http://flowcast_mlflow_ui:5000",
            **get_ml_config_env(),
        },
        do_xcom_push=True, # Publica a decisao em JSON (ultima linha) para o BranchPythonOperator
    )

    branch_decision = BranchPythonOperator(
        task_id="decisao_de_promocao",
        python_callable=route_evaluation,
    )

    promote_champion = DockerOperator(
        task_id="promover_modelo_champion",
        image="flowcast-mlagent:latest",
        api_version="auto",
        auto_remove="force",
        command=f"python -u src/promote_model.py --run-id {{{{ ti.xcom_pull(task_ids='{EVALUATE_TASK_ID}') | run_id }}}}",
        docker_url="unix://var/run/docker.sock",
        network_mode="flowcast-mlops_default",
        mounts=[
            Mount(source=HOST_DATA_PATH, target="/app/data", type="bind"),
            Mount(source=HOST_SRC_PATH, target="/app/src", type="bind")
        ],
        environment={
            "MLFLOW_TRACKING_URI": "http://flowcast_mlflow_ui:5000",
        }
    )

    reject_challenger = BashOperator(
        task_id="rejeitar_modelo_challenger",
        bash_command="echo 'Modelo Challenger rejeitado. O Champion atual continua em producao.'",
    )

    refresh_predictions = DockerOperator(
        task_id="atualizar_previsoes_pos_treino",
        pool=DUCKDB_POOL,
        image="flowcast-mlagent:latest",
        api_version="auto",
        auto_remove="force",
        command="python -u src/predict.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="flowcast-mlops_default",
        mounts=[
            Mount(source=HOST_DATA_PATH, target="/app/data", type="bind"),
            Mount(source=HOST_SRC_PATH, target="/app/src", type="bind")
        ],
        environment={
            "MLFLOW_TRACKING_URI": "http://flowcast_mlflow_ui:5000",
            "MODEL_NAME": "flowcast_demand_model"
        },
        trigger_rule="none_failed_min_one_success",
    )

    ingest_data >> feature_engineering >> train_challenger >> evaluate_model >> branch_decision
    
    branch_decision >> promote_champion >> refresh_predictions
    branch_decision >> reject_challenger >> refresh_predictions