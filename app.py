import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Configuração Visual da Página
st.set_page_config(page_title="IA Gestão Pública - SLZ", page_icon="🏙️", layout="wide")

# CSS para melhorar o visual (agora com o parâmetro correto: unsafe_allow_html)
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { 
        width: 100%; 
        border-radius: 5px; 
        height: 3.5em; 
        background-color: #007BFF; 
        color: white; 
        font-weight: bold; 
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Reconstrução da Rede Neural (MLP)
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
st.title("🏙️ Sistema Inteligente de Monitoramento Social")
st.markdown("---")

# 4. Painel Lateral (Entradas)
st.sidebar.header("⚙️ Parâmetros de Análise")
renda = st.sidebar.number_input("Renda Média Domiciliar (R$)", min_value=0.0, value=1200.0)
esgoto = st.sidebar.slider("Acesso a Saneamento (0 a 100%)", 0.0, 1.0, 0.50)
alfabetismo = st.sidebar.slider("Taxa de Alfabetização (0 a 100%)", 0.0, 1.0, 0.80)

# 5. Dashboard de Resultados
if st.button("🚀 EXECUTAR DIAGNÓSTICO DE IA"):
    if modelo_carregado and scaler_carregado:
        # Preparação dos dados
        colunas = ['renda_media', 'esgoto_sanitario', 'escolaridade']
        dados_usuario = pd.DataFrame([[renda, esgoto, alfabetismo]], columns=colunas)
        dados_norm = scaler.transform(dados_usuario)
        
        # Predição com a Rede Neural
        pred = modelo.predict(dados_norm)[0][0]
        probabilidade = float(np.clip(pred, 0.0, 1.0))
        perc = probabilidade * 100

        st.markdown("### 📊 Resultado do Diagnóstico")
        
        # Métricas em destaque
        c1, c2, c3 = st.columns(3)
        c1.metric("Renda Analisada", f"R$ {renda:,.2f}")
        c2.metric("Saneamento", f"{esgoto*100:.1f}%")
        c3.metric("Alfabetização", f"{alfabetismo*100:.1f}%")

        st.markdown("---")

        # Layout da Barra de Risco
        col_bar, col_status = st.columns([2, 1])
        
        with col_bar:
            st.write(f"**Índice de Vulnerabilidade:** {perc:.2f}%")
            st.progress(probabilidade)

        with col_status:
            if perc > 50:
                st.error("🚨 ALTA VULNERABILIDADE")
            else:
                st.success("✅ SITUAÇÃO ESTÁVEL")

        # Sugestão de IA Explicável
        st.info(f"**Análise do Modelo:** O sistema calculou um score de **{perc:.2f}%**. " + 
                ("Este setor deve ser priorizado para obras de infraestrutura sanitária." if perc > 50 else 
                 "Os indicadores sugerem resiliência social no setor."))
            
    else:
        st.error("Erro: Verifique se os arquivos 'modelo_pesos.weights.h5' e 'scaler.pkl' estão no seu GitHub.")

st.markdown("---")
st.caption("PBL 2 - Inteligência Artificial | Protótipo Func
