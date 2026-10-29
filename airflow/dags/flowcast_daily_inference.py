import os
from datetime import datetime, timedelta
from airflow import DAG
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
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="flowcast_daily_inference",
    default_args=default_args,
    description="Pipeline Diario de Ingestao e Inferencia de Vendas",
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "daily", "inference"],
) as dag:

    ingest_data = DockerOperator(
        task_id="ingestao_de_dados_diaria",
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

    update_features = DockerOperator(
        task_id="atualizar_atributos_lags",
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

    generate_predictions = DockerOperator(
        task_id="gerar_previsoes_diarias",
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
        }
    )

    ingest_data >> update_features >> generate_predictions