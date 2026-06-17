import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Dashboard Executivo de E-commerce", layout="wide")

st.title("📊 Dashboard Executivo de E-commerce")
st.markdown("Análise de vendas, canais de marketing e saúde da base de clientes com inteligência preditiva.")

conn = sqlite3.connect('ecommerce.db')

estados_disponiveis = pd.read_sql_query("SELECT DISTINCT estado FROM dim_clientes ORDER BY estado;", conn)['estado'].tolist()
canais_disponiveis = pd.read_sql_query("SELECT DISTINCT canal_aquisicao FROM fato_vendas ORDER BY canal_aquisicao;", conn)['canal_aquisicao'].tolist()

st.sidebar.header("Filtros do Dashboard")
estados_selecionados = st.sidebar.multiselect("Selecione os Estados:", estados_disponiveis, default=estados_disponiveis)
canais_selecionados = st.sidebar.multiselect("Selecione os Canais de Marketing:", canais_disponiveis, default=canais_disponiveis)

estados_sql = "', '".join(estados_selecionados)
canais_sql = "', '".join(canais_selecionados)

query_kpi = f"""
SELECT 
    COUNT(v.id_venda) as qtd, 
    SUM(v.valor) as fat, 
    AVG(v.valor) as tk_medio 
FROM fato_vendas v
INNER JOIN dim_clientes c ON v.id_cliente = c.id_cliente
WHERE c.estado IN ('{estados_sql}') AND v.canal_aquisicao IN ('{canais_sql}');
"""
df_kpi = pd.read_sql_query(query_kpi, conn)

query_canais = f"""
SELECT v.canal_aquisicao, SUM(v.valor) as faturamento 
FROM fato_vendas v
INNER JOIN dim_clientes c ON v.id_cliente = c.id_cliente
WHERE c.estado IN ('{estados_sql}') AND v.canal_aquisicao IN ('{canais_sql}')
GROUP BY v.canal_aquisicao 
ORDER BY faturamento DESC;
"""
df_canais = pd.read_sql_query(query_canais, conn)

query_recorrencia = f"""
WITH hist AS (
    SELECT v.id_cliente, COUNT(v.id_venda) as compras 
    FROM fato_vendas v
    INNER JOIN dim_clientes c ON v.id_cliente = c.id_cliente
    WHERE c.estado IN ('{estados_sql}') AND v.canal_aquisicao IN ('{canais_sql}')
    GROUP BY v.id_cliente
)
SELECT 
    CASE 
        WHEN compras = 1 THEN '1. Uma única vez'
        WHEN compras BETWEEN 2 AND 4 THEN '2. Recorrente (2-4)'
        ELSE '3. VIP (5+)'
    END as status,
    COUNT(id_cliente) as total_clientes
FROM hist GROUP BY status;
"""
df_recorrencia = pd.read_sql_query(query_recorrencia, conn)

query_vendas_tempo = f"""
SELECT v.data_venda, v.valor
FROM fato_vendas v
INNER JOIN dim_clientes c ON v.id_cliente = c.id_cliente
WHERE c.estado IN ('{estados_sql}') AND v.canal_aquisicao IN ('{canais_sql}');
"""
df_tempo = pd.read_sql_query(query_vendas_tempo, conn)
conn.close()

faturamento = df_kpi['fat'].iloc[0] if df_kpi['fat'].iloc[0] is not None else 0.0
total_vendas = df_kpi['qtd'].iloc[0] if df_kpi['qtd'].iloc[0] is not None else 0
ticket_medio = df_kpi['tk_medio'].iloc[0] if df_kpi['tk_medio'].iloc[0] is not None else 0.0

col1, col2, col3 = st.columns(3)
col1.metric("Faturamento Total", f"R$ {faturamento:,.2f}")
col2.metric("Total de Vendas", f"{total_vendas:,}")
col3.metric("Ticket Médio", f"R$ {ticket_medio:,.2f}")

st.markdown("---")

st.subheader("🔮 Previsão de Faturamento Próximos Meses (Machine Learning)")
if not df_tempo.empty:
    df_tempo['data_venda'] = pd.to_datetime(df_tempo['data_venda'])
    df_mensal = df_tempo.set_index('data_venda').resample('ME')['valor'].sum().reset_index()
    df_mensal.columns = ['Mes', 'Faturamento']
    df_mensal = df_mensal.sort_values('Mes').reset_index(drop=True)

    if len(df_mensal) > 2:
        df_mensal['Mes_Num'] = df_mensal.index + 1

        X = df_mensal[['Mes_Num']]
        y = df_mensal['Faturamento']

        modelo = LinearRegression()
        modelo.fit(X, y)

        ult_mes_num = df_mensal['Mes_Num'].max()
        ult_data = df_mensal['Mes'].max()

        futuro_meses = [ult_mes_num + 1, ult_mes_num + 2, ult_mes_num + 3]
        datas_futuras = [ult_data + pd.DateOffset(months=i) for i in range(1, 4)]

        previsoes = modelo.predict(pd.DataFrame(futuro_meses, columns=['Mes_Num']))

        df_futuro = pd.DataFrame({'Mes': datas_futuras, 'Faturamento': previsoes})

        df_mensal['Tipo'] = 'Histórico Real'
        df_futuro['Tipo'] = 'Previsão (ML)'

        df_ultimo_ponto = df_mensal.tail(1).copy()
        df_ultimo_ponto['Tipo'] = 'Previsão (ML)'
        df_final = pd.concat([df_mensal, df_ultimo_ponto, df_futuro], ignore_index=True)

        fig_prev = px.line(df_final, x='Mes', y='Faturamento', color='Tipo',
                           labels={'Mes': 'Mês', 'Faturamento': 'Faturamento (R$)'},
                           color_discrete_map={'Histórico Real': '#1f77b4', 'Previsão (ML)': '#ff7f0e'})
        st.plotly_chart(fig_prev, use_container_width=True)
    else:
        st.info("Dados históricos insuficientes para traçar uma linha de tendência preditiva.")
else:
    st.warning("Nenhum dado encontrado para gerar previsões.")

st.markdown("---")

col_esq, col_dir = st.columns(2)

with col_esq:
    st.subheader("Faturamento por Canal de Marketing")
    if not df_canais.empty:
        fig_canais = px.bar(df_canais, x='canal_aquisicao', y='faturamento', 
                            labels={'canal_aquisicao': 'Canal', 'faturamento': 'Faturamento (R$)'},
                            color='faturamento', color_continuous_scale='Blues')
        st.plotly_chart(fig_canais, use_container_width=True)

with col_dir:
    st.subheader("Fidelidade da Base de Clientes")
    if not df_recorrencia.empty:
        fig_rec = px.pie(df_recorrencia, values='total_clientes', names='status', 
                         hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_rec, use_container_width=True)
