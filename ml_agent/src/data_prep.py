import os
import zipfile
import duckdb
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION_NAME = "demand-forecasting-kernels-only"
RAW_DATA_PATH = "/app/data/raw"
DB_PATH = "/app/data/database.duckdb"

def download_kaggle_dataset():

    train_csv_path = os.path.join(RAW_DATA_PATH, "train.csv")
    
    if os.path.exists(train_csv_path):
        print(f"Arquivo {train_csv_path} ja existe. Pulando download.")
        return

    print("Autenticando na API do Kaggle...")
    api = KaggleApi()
    api.authenticate()

    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    
    print(f"Baixando dados da competicao '{COMPETITION_NAME}'...")
    api.competition_download_files(COMPETITION_NAME, path=RAW_DATA_PATH)

    for file in os.listdir(RAW_DATA_PATH):
        if file.endswith(".zip"):
            zip_file_path = os.path.join(RAW_DATA_PATH, file)
            print(f"Descompactando {file}...")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DATA_PATH)
            os.remove(zip_file_path)
            
    print("Download e extracao concluidos com sucesso!")

def load_to_duckdb():
    """Le os arquivos CSV baixados (treino e teste) e carrega no DuckDB."""
    train_csv_path = os.path.join(RAW_DATA_PATH, "train.csv")
    test_csv_path = os.path.join(RAW_DATA_PATH, "test.csv")
    
    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        raise FileNotFoundError("Arquivos CSV de treino ou teste nao foram encontrados.")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    print(f"Conectando ao DuckDB em {DB_PATH}...")
    conn = duckdb.connect(DB_PATH)

    print("Carregando tabela 'raw_sales' (Treino)...")
    conn.execute("""
        CREATE OR REPLACE TABLE raw_sales AS 
        SELECT 
            CAST(date AS DATE) AS date,
            CAST(store AS INTEGER) AS store,
            CAST(item AS INTEGER) AS item,
            CAST(sales AS DOUBLE) AS sales
        FROM read_csv_auto(?);
    """, [train_csv_path])

    print("Carregando tabela 'raw_test' (Teste)...")
    conn.execute("""
        CREATE OR REPLACE TABLE raw_test AS 
        SELECT 
            CAST(id AS BIGINT) AS id,
            CAST(date AS DATE) AS date,
            CAST(store AS INTEGER) AS store,
            CAST(item AS INTEGER) AS item
        FROM read_csv_auto(?);
    """, [test_csv_path])

    total_train = conn.execute("SELECT COUNT(*) FROM raw_sales").fetchone()[0]
    total_test = conn.execute("SELECT COUNT(*) FROM raw_test").fetchone()[0]
    
    print(f"\nSucesso! Tabelas criadas/atualizadas no DuckDB:")
    print(f" -> raw_sales (Treino): {total_train:,} linhas")
    print(f" -> raw_test  (Teste):  {total_test:,} linhas")
    
    conn.close()

if __name__ == "__main__":
    download_kaggle_dataset()
    load_to_duckdb()