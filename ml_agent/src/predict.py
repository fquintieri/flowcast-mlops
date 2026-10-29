import os
import sys
import tempfile
import duckdb
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from config import MODEL_NAME
from registry import CHAMPION_ALIAS, get_champion_version

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DB_PATH = "/app/data/database.duckdb"
SUBMISSION_PATH = "/app/data/submission.csv"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def write_csv_atomically(df, path):
    """Grava em arquivo temporario no mesmo diretorio e troca com os.replace (atomico),
    para que leitores (ex.: dashboard) nunca vejam um arquivo parcialmente escrito."""
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".submission-", suffix=".csv.tmp")
    try:
        with os.fdopen(fd, "w", newline="") as tmp_file:
            df.to_csv(tmp_file, index=False)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def load_champion():
    """Carrega somente o modelo aprovado (@Champion). Sem Champion, falha explicitamente."""
    print(f"Buscando modelo em producao: models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    champion_version = get_champion_version(MlflowClient())

    if champion_version is None:
        print(
            f"ERRO: nenhum modelo '{MODEL_NAME}' com alias @{CHAMPION_ALIAS} no Model Registry. "
            "A inferencia nao usa modelos nao aprovados. "
            "Execute o DAG 'flowcast_weekly_retrain', que trata o cold start promovendo o primeiro Champion."
        )
        sys.exit(1)

    print(f"Champion: versao {champion_version.version} (run {champion_version.run_id})")
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{champion_version.version}")

def predict():
    model = load_champion()

    print("Modelo carregado com sucesso! Lendo dados de teste do DuckDB...")
    
    conn = duckdb.connect(DB_PATH, read_only=True)
    
    df_test = conn.execute("SELECT * FROM features_test").df()
    conn.close()

    if df_test.empty:
        print("Erro: A tabela 'features_test' esta vazia no DuckDB.")
        sys.exit(1)
        
    print(f"Gerando previsoes para {len(df_test):,} registros...")

    test_ids = df_test['id']
    
    ignore_cols = ['id', 'date', 'store', 'item']
    features = [c for c in df_test.columns if c not in ignore_cols]
    X_test = df_test[features]

    predictions = model.predict(X_test)

    submission_df = pd.DataFrame({
        'id': test_ids,
        'sales': predictions
    })

    print(f"Salvando resultados em {SUBMISSION_PATH}...")
    write_csv_atomically(submission_df, SUBMISSION_PATH)
    print("Sucesso! Arquivo submission.csv gerado. O Dashboard ja pode ser atualizado.")

if __name__ == "__main__":
    predict()