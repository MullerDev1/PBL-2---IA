import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# 1. Configuração da Interface
st.set_page_config(page_title="Monitor de Vulnerabilidade Social", page_icon="🏙️", layout="centered")

st.title("🏙️ Monitor de Vulnerabilidade Social")
st.markdown("---")
st.markdown("Este sistema utiliza **Redes Neurais Artificiais** para identificar áreas com alta vulnerabilidade socioeconômica, auxiliando a gestão pública na alocação de recursos.")

# 2. Função para reconstruir a arquitetura da Rede Neural (MLP)
# Importante: A estrutura deve ser idêntica à que você treinou no Google Colab
def carregar_modelo_treinado():
    # Criamos a "carcaça" do modelo
    model = Sequential([
        Input(shape=(3,)), # Entradas: Renda, Saneamento e Escolaridade
        Dense(12, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Nome do arquivo de pesos que você subiu para o GitHub
    arquivo_pesos = 'modelo_pesos.weights.h5'
    
    if os.path.exists(arquivo_pesos):
        # Carrega apenas os números (pesos) aprendidos durante o Backpropagation
        model.load_weights(arquivo_pesos)
        return model, True
    else:
        return model, False

# 3. Inicialização dos Componentes
modelo, modelo_carregado = carregar_modelo_treinado()

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    scaler_carregado = True
except:
    scaler_carregado = False

# 4. Interface Lateral (Entrada de Dados)
st.sidebar.header("📊 Parâmetros do Setor")
st.sidebar.markdown("Insira os indicadores do setor censitário:")

renda = st.sidebar.number_input("Renda Média Domiciliar (R$)", min_value=0.0, value=1200.0, step=100.0)
esgoto = st.sidebar.slider("Acesso a Saneamento (0 a 1)", 0.0, 1.0, 0.5)
alfabetismo = st.sidebar.slider("Taxa de Alfabetização (0 a 1)", 0.0, 1.0, 0.8)

# 5. Lógica de Predição
if st.button("Executar Análise de Inteligência Artificial"):
    if modelo_carregado and scaler_carregado:
        # Organização dos dados para o formato que a IA espera
        colunas = ['renda_media', 'esgoto_sanitario', 'escolaridade']
        dados_usuario = pd.DataFrame([[renda, esgoto, alfabetismo]], columns=colunas)
        
        # Normalização dos dados (essencial para redes neurais)
        dados_norm = scaler.transform(dados_usuario)
        
        # Cálculo da probabilidade (Forward Propagation)
        probabilidade = modelo.predict(dados_norm)[0][0]
        
        # Exibição dos Resultados
        st.subheader("Resultado da Análise")
        if probabilidade > 0.5:
            st.error(f"### ALTA VULNERABILIDADE DETECTADA: {probabilidade*100:.2f}%")
            st.warning("⚠️ Este setor apresenta indicadores críticos. Recomenda-se intervenção prioritária da prefeitura.")
        else:
            st.success(f"### Baixa/Média Vulnerabilidade: {probabilidade*100:.2f}%")
            st.info("✅ Os indicadores estão dentro dos padrões de normalidade comparados à base de treinamento.")
            
        st.markdown("---")
        st.caption("Modelo baseado em Perceptron Multicamadas (MLP) treinado com dados do IBGE.")
    else:
        st.error("Erro: Verifique se os arquivos 'modelo_pesos.weights.h5' e 'scaler.pkl' estão na mesma pasta do GitHub.")

# 6. Rodapé
st.markdown("---")
st.write("Protótipo funcional para apoio à decisão estratégica municipal.")
