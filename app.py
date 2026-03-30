import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Configuração de Layout e Estilo
st.set_page_config(page_title="IA Social - São Luís", page_icon="🏙️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 3.5em; 
        background-color: #007BFF; color: white; font-weight: bold; 
    }
    </style>
    """, unsafe_allow_html=True)

# 2. IA - Reconstrução da Arquitetura
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

# 3. Interface e Inputs
st.title("🏙️ Sistema de Monitoramento de Vulnerabilidade")
st.markdown("Diagnóstico via Redes Neurais para Apoio à Gestão Pública")
st.markdown("---")

st.sidebar.header("⚙️ Parâmetros de Entrada")
renda = st.sidebar.number_input("Renda Média (R$)", min_value=0.0, value=1200.0)
esgoto = st.sidebar.slider("Acesso a Saneamento", 0.0, 1.0, 0.50)
alfabetismo = st.sidebar.slider("Taxa de Alfabetização", 0.0, 1.0, 0.80)

# 5. Lógica e Dashboards
if st.button("🚀 EXECUTAR DIAGNÓSTICO"):
    if modelo_carregado and scaler_carregado:
        # Processamento
        colunas = ['renda_media', 'esgoto_sanitario', 'escolaridade']
        dados_raw = pd.DataFrame([[renda, esgoto, alfabetismo]], columns=colunas)
        dados_norm = scaler.transform(dados_raw)
        
        pred = modelo.predict(dados_norm)[0][0]
        probabilidade = float(np.clip(pred, 0.0, 1.0))
        perc = probabilidade * 100

        # Métrica de Cabeçalho
        st.markdown(f"### Índice de Vulnerabilidade Social: `{perc:.2f}%`")
        st.progress(probabilidade)
        st.markdown("---")
        
        # 4 ABAS DE VISUALIZAÇÃO
        aba1, aba2, aba3, aba4 = st.tabs(["📊 Barras", "📉 Linhas", "📈 Área", "📋 Resumo"])

        # Dados para os gráficos
        chart_data = pd.DataFrame({
            'Indicador': ['Renda', 'Saneamento', 'Educação'],
            'Nível (%)': [float(dados_norm[0][0] * 100), float(esgoto * 100), float(alfabetismo * 100)]
        }).set_index('Indicador')

        with aba1:
            st.write("**Comparativo de Volume**")
            st.bar_chart(chart_data)

        with aba2:
            st.write("**Perfil de Oscilação dos Indicadores**")
            st.line_chart(chart_data)
            st.caption("O gráfico de linhas ajuda a identificar quedas bruscas em indicadores específicos.")

        with aba3:
            st.write("**Preenchimento de Direitos Sociais**")
            st.area_chart(chart_data)

        with aba4:
            st.write("**Diagnóstico Final**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Renda", f"R$ {renda:,.2f}")
            c2.metric("Saneamento", f"{esgoto*100:.0f}%")
            c3.metric("Educação", f"{alfabetismo*100:.0f}%")
            
            if perc > 50:
                st.error(f"**ALTA VULNERABILIDADE DETECTADA**")
                st.info("A IA recomenda prioridade em investimentos de infraestrutura básica.")
            else:
                st.success(f"**SITUAÇÃO SOB CONTROLE**")
                st.info("O setor apresenta indicadores estáveis de acordo com o modelo.")
            
    else:
        st.error("Erro: Arquivos de pesos ou scaler não encontrados no GitHub.")

st.markdown("---")
st.caption("PBL 2 - Inteligência Artificial | Protótipo de Gestão Social")
