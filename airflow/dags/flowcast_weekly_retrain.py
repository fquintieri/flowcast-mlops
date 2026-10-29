import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from util import DUCKDB_POOL, get_host_project_root

HOST_ROOT = get_host_project_root()
HOST_DATA_PATH = f"{HOST_ROOT}/data"
HOST_SRC_PATH = f"{HOST_ROOT}/ml_agent/src"

default_args = {
    "owner": "flowcast_mlops",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

def route_evaluation(**context):
    eval_output = context['ti'].xcom_pull(task_ids='avaliar_challenger_vs_champion')
    
    if eval_output and "promover_modelo_champion" in eval_output:
        return "promover_modelo_champion"
    return "rejeitar_modelo_challenger"

with DAG(
    dag_id="flowcast_weekly_retrain",
    default_args=default_args,
    description="Pipeline Semanal de Retreinamento, Avaliacao Champion vs Challenger e Promocao",
    schedule="0 4 * * 0",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "weekly", "training"],
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
        task_id="treinar_modelo_challenger",
        pool=DUCKDB_POOL,
        image="flowcast-mlagent:latest",
        api_version="auto",
        auto_remove="force",
        command="python -u src/train.py",
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

    evaluate_model = DockerOperator(
        task_id="avaliar_challenger_vs_champion",
        image="flowcast-mlagent:latest",
        api_version="auto",
        auto_remove="force",
        command="python -u src/evaluate_model.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="flowcast-mlops_default",
        mounts=[
            Mount(source=HOST_DATA_PATH, target="/app/data", type="bind"),
            Mount(source=HOST_SRC_PATH, target="/app/src", type="bind")
        ],
        environment={
            "MLFLOW_TRACKING_URI": "http://flowcast_mlflow_ui:5000",
        },
        do_xcom_push=True, # Habilita a captura dos logs para o BranchPythonOperator ler
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
        command="python -u src/promote_model.py",
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