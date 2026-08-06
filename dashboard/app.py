import os
import streamlit as st
import duckdb
import pandas as pd

DB_PATH = "/app/data/database.duckdb"
SUBMISSION_PATH = "/app/data/submission.csv"

st.set_page_config(
    page_title="Flowcast - Painel de Previsao de Demanda",
    layout="wide"
)

st.title("Flowcast - Painel de Previsao de Demanda")
st.markdown("Visualizacao de vendas historicas e projecoes geradas pelo modelo de Machine Learning.")

@st.cache_data(ttl=43200)
def load_data():

    # Verifica se os arquivos base já existem (Evita o erro de Cold Start)
    if not os.path.exists(DB_PATH) or not os.path.exists(SUBMISSION_PATH):
        return None, None

    conn = duckdb.connect(DB_PATH, read_only=True)
    
    # Carregar vendas historicas
    df_history = conn.execute("""
        SELECT date, store, item, sales 
        FROM raw_sales
    """).df()
    df_history['date'] = pd.to_datetime(df_history['date'])
    
    # Carregar registros de teste
    df_test = conn.execute("""
        SELECT id, date, store, item 
        FROM raw_test
    """).df()
    df_test['date'] = pd.to_datetime(df_test['date'])
    
    conn.close()
    
    # Carregar previsoes do submission.csv
    df_sub = pd.read_csv(SUBMISSION_PATH)
    
    # Unir dados de teste com as previsoes calculadas
    df_predict = pd.merge(df_test, df_sub, on="id")
    
    return df_history, df_predict

try:
    df_history, df_predict = load_data()

    # Se os dados forem None, o pipeline ainda não rodou.
    if df_history is None or df_predict is None:
        st.warning("Ambiente recém-criado! O pipeline de Machine Learning ainda não gerou as previsões.")
        st.info("Acesse o Airflow (http://localhost:8080) e execute a DAG 'flowcast_weekly_retrain' para popular o painel.")
        st.stop() # Interrompe a renderização do resto da página aqui

    # Barra lateral de filtros
    st.sidebar.header("Filtros de Selecao")
    
    stores = sorted(df_history['store'].unique().tolist())
    selected_store = st.sidebar.selectbox("Selecione a Loja:", stores)
    
    items = sorted(df_history['item'].unique().tolist())
    selected_item = st.sidebar.selectbox("Selecione o Produto:", items)

    # Filtrar dados para a loja e produto selecionados
    hist_filtered = df_history[
        (df_history['store'] == selected_store) & 
        (df_history['item'] == selected_item)
    ].sort_values('date')

    pred_filtered = df_predict[
        (df_predict['store'] == selected_store) & 
        (df_predict['item'] == selected_item)
    ].sort_values('date')

    # Cartoes de Metricas Principais
    col1, col2, col3 = st.columns(3)
    
    avg_hist_sales = hist_filtered['sales'].mean()
    total_pred_sales = pred_filtered['sales'].sum()
    avg_pred_sales = pred_filtered['sales'].mean()

    col1.metric("Media Historica Diaria", f"{avg_hist_sales:.2f} un")
    col2.metric("Media Prevista Diaria", f"{avg_pred_sales:.2f} un")
    col3.metric("Total Previsto no Periodo", f"{total_pred_sales:,.0f} un")

    st.markdown("---")

    # Grafico temporal comparativo
    st.subheader(f"Vendas Historicas vs Previsao (Loja {selected_store} - Produto {selected_item})")

    hist_filtered_subset = hist_filtered[['date', 'sales']].copy()
    hist_filtered_subset['Tipo'] = 'Historico'

    pred_filtered_subset = pred_filtered[['date', 'sales']].copy()
    pred_filtered_subset['Tipo'] = 'Previsao'

    # Filtrar apenas os ultimos 6 meses do historico no grafico
    max_hist_date = hist_filtered_subset['date'].max()
    min_hist_date = max_hist_date - pd.DateOffset(months=6)
    hist_filtered_recent = hist_filtered_subset[hist_filtered_subset['date'] >= min_hist_date]

    combined_df = pd.concat([hist_filtered_recent, pred_filtered_subset])
    chart_data = combined_df.pivot(index='date', columns='Tipo', values='sales')
    
    st.line_chart(chart_data)

    st.markdown("---")
    
    # Secao Refatorada de Analise Detalhada
    st.subheader("Analise Detalhada do Periodo Previsto")

    # Destaques de Pico e Baixa Demanda
    peak_row = pred_filtered.loc[pred_filtered['sales'].idxmax()]
    low_row = pred_filtered.loc[pred_filtered['sales'].idxmin()]

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.info(
            f"Dia de Maior Demanda Prevista: {peak_row['date'].strftime('%d/%m/%Y')} "
            f"({peak_row['sales']:.2f} unidades)"
        )
    with d_col2:
        st.warning(
            f"Dia de Menor Demanda Prevista: {low_row['date'].strftime('%d/%m/%Y')} "
            f"({low_row['sales']:.2f} unidades)"
        )

    # Navegacao por Abas
    tab1, tab2 = st.tabs(["Tabela de Dados Formatada", "Comportamento Semanal"])

    with tab1:
        display_df = pred_filtered[['date', 'store', 'item', 'sales']].copy()
        display_df.rename(columns={'sales': 'previsao_vendas'}, inplace=True)

        # Botao para download dos dados filtrados
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Baixar Tabela de Previsoes em CSV",
            data=csv_data,
            file_name=f"previsao_loja_{selected_store}_item_{selected_item}.csv",
            mime="text/csv"
        )

        # Tabela rica e formatada
        st.dataframe(
            display_df,
            column_config={
                "date": st.column_config.DateColumn("Data da Previsao", format="DD/MM/YYYY"),
                "store": st.column_config.NumberColumn("Loja"),
                "item": st.column_config.NumberColumn("Produto"),
                "previsao_vendas": st.column_config.NumberColumn(
                    "Previsao de Vendas (unidades)",
                    format="%.2f un"
                )
            },
            hide_index=True,
            use_container_width=True
        )

    with tab2:
        st.markdown("Media de vendas diarias previstas por dia da semana para este produto:")
        
        pred_copy = pred_filtered.copy()
        pred_copy['dia_semana'] = pred_copy['date'].dt.day_name()
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_avg = pred_copy.groupby('dia_semana')['sales'].mean().reindex(days_order)
        
        st.bar_chart(weekday_avg)

except Exception as e:
    st.error(f"Erro ao carregar os dados no dashboard: {e}")