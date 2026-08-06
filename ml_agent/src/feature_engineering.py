import duckdb

DB_PATH = "/app/data/database.duckdb"

def build_features():
    print(f"Conectando ao DuckDB em {DB_PATH}...")
    conn = duckdb.connect(DB_PATH)

    print("Unificando dados de treino e teste para cálculo de atributos...")

    conn.execute("""
        CREATE OR REPLACE TEMP TABLE full_dataset AS
        SELECT NULL AS id, date, store, item, sales, 'train' AS dataset_type
        FROM raw_sales
        UNION ALL
        SELECT id, date, store, item, NULL AS sales, 'test' AS dataset_type
        FROM raw_test;
    """)

    print("Calculando os 13 atributos unificados...")
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE full_features AS
        SELECT 
            id,
            date,
            store,
            item,
            sales,
            dataset_type,
            -- Atributos Temporais (7)
            DAYOFWEEK(date) AS dayofweek,
            DAYOFMONTH(date) AS dayofmonth,
            DAYOFYEAR(date) AS dayofyear,
            MONTH(date) AS month,
            YEAR(date) AS year,
            QUARTER(date) AS quarter,
            CASE WHEN DAYOFWEEK(date) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
            -- Atributos de Lags / Janelas Móveis (6)
            LAG(sales, 91) OVER (PARTITION BY store, item ORDER BY date) AS lag_91,
            LAG(sales, 98) OVER (PARTITION BY store, item ORDER BY date) AS lag_98,
            LAG(sales, 105) OVER (PARTITION BY store, item ORDER BY date) AS lag_105,
            LAG(sales, 112) OVER (PARTITION BY store, item ORDER BY date) AS lag_112,
            LAG(sales, 182) OVER (PARTITION BY store, item ORDER BY date) AS lag_182,
            LAG(sales, 364) OVER (PARTITION BY store, item ORDER BY date) AS lag_364
        FROM full_dataset;
    """)

    print("Salvando tabela 'features_sales'...")
    conn.execute("""
        CREATE OR REPLACE TABLE features_sales AS
        SELECT * EXCLUDE (id, dataset_type)
        FROM full_features
        WHERE dataset_type = 'train' AND sales IS NOT NULL;
    """)

    print("Salvando tabela 'features_test'...")
    conn.execute("""
        CREATE OR REPLACE TABLE features_test AS
        SELECT * EXCLUDE (sales, dataset_type)
        FROM full_features
        WHERE dataset_type = 'test';
    """)

    total_sales = conn.execute("SELECT COUNT(*) FROM features_sales").fetchone()[0]
    total_test = conn.execute("SELECT COUNT(*) FROM features_test").fetchone()[0]

    print(f"\nSucesso! Atributos gerados com alinhamento perfeito:")
    print(f" -> features_sales: {total_sales:,} linhas")
    print(f" -> features_test:  {total_test:,} linhas")

    conn.close()

if __name__ == "__main__":
    build_features()