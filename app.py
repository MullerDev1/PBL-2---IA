import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Configuração Visual - Papel do Gabriel (Analista de Inovação)
st.set_page_config(page_title="Gestão Inteligente - SLZ", page_icon="🏙️")
st.title("🏙️ Monitor de Vulnerabilidade Social")
st.markdown("---")

# Carregar a inteligência salva
modelo = load_model('modelo_vulnerabilidade.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Interface Lateral para Entrada de Dados (Simulando o IBGE)
st.sidebar.header("Dados do Setor Censitário")
renda = st.sidebar.number_input("Renda Média (R$)", value=1000)
esgoto = st.sidebar.slider("Acesso a Saneamento (0 a 1)", 0.0, 1.0, 0.5)
alfabetismo = st.sidebar.slider("Taxa de Alfabetização (0 a 1)", 0.0, 1.0, 0.7)

# Botão de Execução da IA
if st.button("Calcular Risco Social"):
    # Organiza os dados no formato que a rede espera (com nomes para evitar o warning)
    colunas = ['renda_media', 'esgoto_sanitario', 'escolaridade']
    input_data = pd.DataFrame([[renda, esgoto, alfabetismo]], columns=colunas)
    
    # Normalização e Previsão
    input_scaled = scaler.transform(input_data)
    probabilidade = modelo.predict(input_scaled)[0][0]
    
    # Resultado Visual para o Gestor
    if probabilidade > 0.5:
        st.error(f"### ALTA VULNERABILIDADE: {probabilidade*100:.2f}%")
        st.warning("Prioridade de investimento público recomendada para esta área.")
    else:
        st.success(f"### Risco Social Baixo/Médio: {probabilidade*100:.2f}%")
