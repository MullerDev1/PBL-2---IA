import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px  # Para o gráfico de pizza
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Configuração da Página
st.set_page_config(page_title="IA Social - Resumo", page_icon="🏙️", layout="centered")

# 2. IA - Reconstrução
def carregar_modelo_treinado():
    model = Sequential([
        Dense(12, activation='relu', input_shape=(3,)), 
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    arquivo_pesos = 'modelo_pesos.weights.h5'
    if os.path.exists(arquivo_pesos):
        model.load_weights(arquivo_pesos)
        return model, True
    return model, False

modelo, modelo_carregado = carregar_modelo_treinado()

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    scaler_carregado = True
except:
    scaler_carregado = False

# 3. Cabeçalho Minimalista
st.title("🏙️ Diagnóstico de Vulnerabilidade")
st.markdown("Análise simplificada para tomada de decisão imediata.")
st.markdown("---")

# 4. Sidebar de Entrada
st.sidebar.header("⚙️ Parâmetros")
renda = st.sidebar.number_input("Renda Média (R$)", min_value=0.0, value=1200.0)
esgoto = st.sidebar.slider("Saneamento", 0.0, 1.0, 0.50)
alfabetismo = st.sidebar.slider("Educação", 0.0, 1.0, 0.80)

# 5. Resumo Executivo
if st.button("🚀 GERAR RESUMO EXECUTIVO"):
    if modelo_carregado and scaler_carregado:
        # Predição
        colunas = ['renda_media', 'esgoto_sanitario', 'escolaridade']
        dados_norm = scaler.transform(pd.DataFrame([[renda, esgoto, alfabetismo]], columns=colunas))
        probabilidade = float(np.clip(modelo.predict(dados_norm)[0][0], 0.0, 1.0))
        perc_vulneravel = probabilidade * 100
        perc_estavel = 100 - perc_vulneravel

        # Métricas de Resumo
        st.markdown("### 📋 Resumo do Setor")
        c1, c2, c3 = st.columns(3)
        c1.metric("Renda", f"R$ {renda:,.2f}")
        c2.metric("Saneamento", f"{esgoto*100:.0f}%")
        c3.metric("Educação", f"{alfabetismo*100:.0f}%")

        st.markdown("---")

        # Gráfico de Pizza (Composição do Risco)
        st.write("**Distribuição da Probabilidade de Risco**")
        
        df_pizza = pd.DataFrame({
            "Status": ["Risco/Vulnerabilidade", "Estabilidade Social"],
            "Porcentagem": [perc_vulneravel, perc_estavel]
        })
        
        fig = px.pie(
            df_pizza, 
            values='Porcentagem', 
            names='Status', 
            color='Status',
            color_discrete_map={'Risco/Vulnerabilidade': '#ef5350', 'Estabilidade Social': '#66bb6a'},
            hole=0.4  # Transforma em gráfico de rosca (mais moderno)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Veredito Final
        if perc_vulneravel > 50:
            st.error(f"**Veredito:** ALTA VULNERABILIDADE DETECTADA ({perc_vulneravel:.2f}%)")
            st.warning("⚠️ Prioridade máxima para intervenção do poder público.")
        else:
            st.success(f"**Veredito:** SETOR EM CONDIÇÃO ESTÁVEL ({perc_vulneravel:.2f}%)")
            st.info("✅ Manter monitoramento e serviços básicos.")

    else:
        st.error("Erro: Arquivos não encontrados no GitHub.")

st.markdown("---")
st.caption("PBL 2 - Inteligência Artificial | Analista: Gabriel Carvalho")
