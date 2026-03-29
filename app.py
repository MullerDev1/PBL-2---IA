import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model # Voltamos ao padrão

st.set_page_config(page_title="Gestão Inteligente - SLZ", page_icon="🏙️")
st.title("🏙️ Monitor de Vulnerabilidade Social")

# Carregando o novo formato .keras
# compile=False evita erros se houver conflito nas métricas de treino
modelo = load_model('modelo_vulnerabilidade.keras', compile=False)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.sidebar.header("Dados do Setor Censitário")
renda = st.sidebar.number_input("Renda Média (R$)", value=1000)
esgoto = st.sidebar.slider("Acesso a Saneamento (0 a 1)", 0.0, 1.0, 0.5)
alfabetismo = st.sidebar.slider("Taxa de Alfabetização (0 a 1)", 0.0, 1.0, 0.7)

if st.button("Calcular Risco Social"):
    colunas = ['renda_media', 'esgoto_sanitario', 'escolaridade']
    input_data = pd.DataFrame([[renda, esgoto, alfabetismo]], columns=colunas)
    input_scaled = scaler.transform(input_data)
    
    # Previsão
    probabilidade = modelo.predict(input_scaled)[0][0]
    
    if probabilidade > 0.5:
        st.error(f"### ALTA VULNERABILIDADE: {probabilidade*100:.2f}%")
    else:
        st.success(f"### Risco Social Baixo/Médio: {probabilidade*100:.2f}%")
