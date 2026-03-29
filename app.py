import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

st.set_page_config(page_title="Gestão Inteligente - SLZ", page_icon="🏙️")
st.title("🏙️ Monitor de Vulnerabilidade Social")

# --- RECONSTRUÇÃO DA ARQUITETURA (Papel do Gabriel) ---
# Criamos a mesma "carcaça" que você treinou no Colab
def carregar_modelo_seguro():
    model = Sequential([
        Input(shape=(3,)), # 3 entradas: Renda, Esgoto, Alfabetismo
        Dense(12, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    # Carrega apenas os pesos (os números aprendidos no Backpropagation)
    model.load_weights('modelo_pesos.weights.h5')
    return model

modelo = carregar_modelo_seguro()

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --- INTERFACE ---
st.sidebar.header("Dados do Setor Censitário")
renda = st.sidebar.number_input("Renda Média (R$)", value=1000)
esgoto = st.sidebar.slider("Acesso a Saneamento (0 a 1)", 0.0, 1.0, 0.5)
alfabetismo = st.sidebar.slider("Taxa de Alfabetização (0 a 1)", 0.0, 1.0, 0.7)

if st.button("Calcular Risco Social"):
    colunas = ['renda_media', 'esgoto_sanitario', 'escolaridade']
    input_data = pd.DataFrame([[renda, esgoto, alfabetismo]], columns=colunas)
    input_scaled = scaler.transform(input_data)
    
    probabilidade = modelo.predict(input_scaled)[0][0]
    
    if probabilidade > 0.5:
        st.error(f"### ALTA VULNERABILIDADE: {probabilidade*100:.2f}%")
        st.caption("Recomendação: Intervenção prioritária em infraestrutura.")
    else:
        st.success(f"### Risco Social Baixo/Médio: {probabilidade*100:.2f}%")
