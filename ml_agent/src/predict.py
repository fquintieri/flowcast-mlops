import os
import sys
import duckdb
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "flowcast_demand_model")
DB_PATH = "/app/data/database.duckdb"
SUBMISSION_PATH = "/app/data/submission.csv"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_champion_or_fallback():
    client = MlflowClient()
    
    try:
        print(f"Buscando modelo em producao: models:/{MODEL_NAME}@Champion")
        return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@Champion")
    except Exception:
        print("Nenhum modelo marcado como @Champion foi encontrado.")

    try:
        versions = client.get_latest_versions(MODEL_NAME)
        if versions:
            latest_version = versions[0].version
            print(f"Fallback ativado: Carregando a versao mais recente (v{latest_version})...")
            return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{latest_version}")
    except Exception:
        pass

    return None

def predict():
    model = load_champion_or_fallback()
    
    if model is None:
        print("[COLD START] Nenhum modelo encontrado no MLflow.")
        print("Dica: Execute primeiro o pipeline de treinamento ('train.py' + 'promote_model.py').")
        sys.exit(0)

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
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Sucesso! Arquivo submission.csv gerado. O Dashboard ja pode ser atualizado.")

if __name__ == "__main__":
    predict()