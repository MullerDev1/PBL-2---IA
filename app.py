import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Configuração da Interface
st.set_page_config(page_title="Monitor de Vulnerabilidade Social", page_icon="🏙️", layout="centered")

st.title("🏙️ Monitor de Vulnerabilidade Social")
st.markdown("---")
st.markdown("Protótipo de IA para identificação de áreas críticas em São Luís.")

# 2. Reconstrução Robusta da Rede Neural
def carregar_modelo_treinado():
    # Definimos a arquitetura de forma direta para evitar o ValueError
    model = Sequential([
        # Passamos o input_shape=(3,) direto na primeira camada oculta
        Dense(12, activation='relu', input_shape=(3,)), 
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    arquivo_pesos = 'modelo_pesos.weights.h5'
    
    if os.path.exists(arquivo_pesos):
        # Carrega os pesos salvos no Colab
        model.load_weights(arquivo_pesos)
        return model, True
    else:
        return model, False

# 3. Inicialização
modelo, modelo_carregado = carregar_modelo_treinado()

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    scaler_carregado = True
except:
    scaler_carregado = False

# 4. Barra Lateral (Parâmetros que você perguntou)
st.sidebar.header("📊 Parâmetros do Setor")
renda = st.sidebar.number_input("Renda Média Domiciliar (R$)", min_value=0.0, value=1200.0)
esgoto = st.sidebar.slider("Acesso a Saneamento (0 a 1)", 0.0, 1.0, 0.5)
alfabetismo = st.sidebar.slider("Taxa de Alfabetização (0 a 1)", 0.0, 1.0, 0.8)

# 5. Execução
if st.button("Executar Análise de IA"):
    if modelo_carregado and scaler_carregado:
        colunas = ['renda_media', 'esgoto_sanitario', 'escolaridade']
        dados_usuario = pd.DataFrame([[renda, esgoto, alfabetismo]], columns=colunas)
        
        # Normalização
        dados_norm = scaler.transform(dados_usuario)
        
        # Predição
        probabilidade = modelo.predict(dados_norm)[0][0]
        
        st.subheader("Resultado")
        if probabilidade > 0.5:
            st.error(f"### ALTA VULNERABILIDADE: {probabilidade*100:.2f}%")
        else:
            st.success(f"### Risco Social Baixo/Médio: {probabilidade*100:.2f}%")
    else:
        st.error("Erro técnico: Arquivos de pesos ou scaler não encontrados no GitHub.")

st.markdown("---")
st.caption("Desenvolvido para o PBL 2 - Inteligência Artificial")
