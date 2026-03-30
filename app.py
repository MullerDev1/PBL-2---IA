import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Configuração da Página (Layout Wide para caber os gráficos)
st.set_page_config(page_title="IA Social - São Luís", page_icon="🏙️", layout="wide")

# CSS para Estilização
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #007BFF; color: white; font-weight: bold; }
    .metric-container { background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

# 2. Carregamento Seguro do Modelo (Arquitetura Manual + Pesos)
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

# Carregar o Scaler
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    scaler_carregado = True
except:
    scaler_carregado = False

# 3. Cabeçalho e Título
st.title("🏙️ Inteligência Artificial Aplicada à Gestão Social")
st.markdown("Monitoramento de Vulnerabilidade em Setores Censitários de São Luís - MA")
st.markdown("---")

# 4. Sidebar de Entrada (Parâmetros)
st.sidebar.header("⚙️ Entrada de Dados")
st.sidebar.markdown("Ajuste os indicadores do setor:")
renda = st.sidebar.number_input("Renda Média Domiciliar (R$)", min_value=0.0, value=1200.0)
esgoto = st.sidebar.slider("Acesso a Saneamento (0-100%)", 0.0, 1.0, 0.50)
alfabetismo = st.sidebar.slider("Taxa de Alfabetização (0-100%)", 0.0, 1.0, 0.80)

# 5. Processamento e Abas de Visualização
if st.button("🚀 EXECUTAR DIAGNÓSTICO MULTIDIMENSIONAL"):
    if modelo_carregado and scaler_carregado:
        # Preparação dos Dados
        colunas = ['renda_media', 'esgoto_sanitario', 'escolaridade']
        dados_input = pd.DataFrame([[renda, esgoto, alfabetismo]], columns=colunas)
        dados_norm = scaler.transform(dados_input)
        
        # Predição
        pred = modelo.predict(dados_norm)[0][0]
        probabilidade = float(np.clip(pred, 0.0, 1.0))
        perc = probabilidade * 100

        # Métrica Principal de Risco
        st.markdown(f"### Índice de Risco Social: `{perc:.2f}%`")
        st.progress(probabilidade)
        
        st.markdown("---")
        
        # CRIAÇÃO DAS 3 POSSIBILIDADES (ABAS)
        aba1, aba2, aba3 = st.tabs(["📊 Gráfico de Barras", "📈 Gráfico de Área", "📋 Diagnóstico Executivo"])

        # Preparando os dados para os gráficos
        chart_data = pd.DataFrame({
            'Indicador': ['Renda (Normalizada)', 'Saneamento', 'Educação'],
            'Nível (%)': [float(dados_norm[0][0] * 100), float(esgoto * 100), float(alfabetismo * 100)]
        }).set_index('Indicador')

        with aba1:
            st.write("**Comparativo de Indicadores (Qual fator é mais crítico?)**")
            st.bar_chart(chart_data)
            st.caption("O Gráfico de Barras facilita a identificação direta do ponto fraco do setor.")

        with aba2:
            st.write("**Volume de Cobertura Social**")
            st.area_chart(chart_data)
            st.caption("O Gráfico de Área demonstra visualmente o 'preenchimento' dos direitos básicos no setor.")

        with aba3:
            st.write("**Resumo Executivo para Tomada de Decisão**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Renda", f"R$ {renda:,.2f}")
            with c2:
                st.metric("Esgoto", f"{esgoto*100:.0f}%")
            with c3:
                st.metric("Educação", f"{alfabetismo*100:.0f}%")
            
            if perc > 50:
                st.error(f"**Veredito:** ALTA VULNERABILIDADE ({perc:.2f}%)")
                st.info("Ação Sugerida: Priorizar este setor no plano de metas de saneamento e assistência básica.")
            else:
                st.success(f"**Veredito:** SITUAÇÃO ESTÁVEL ({perc:.2f}%)")
                st.info("Ação Sugerida: Manutenção preventiva dos serviços públicos atuais.")

    else:
        st.error("Erro: Arquivos 'modelo_pesos.weights.h5' ou 'scaler.pkl' ausentes no repositório.")

st.markdown("---")
st.caption("PBL 2 - Inteligência Artificial | Protótipo Desenvolvido por Gabriel Carvalho")
