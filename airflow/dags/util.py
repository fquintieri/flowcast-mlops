import os
import docker

# Pool de 1 slot que serializa todo acesso ao arquivo DuckDB (o lock e por processo:
# um escritor bloqueia qualquer outro processo, inclusive leitores read_only).
DUCKDB_POOL = "duckdb_writer"

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