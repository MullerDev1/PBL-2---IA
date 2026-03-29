import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Configuração Visual
st.set_page_config(page_title="IA Gestão Pública - SLZ", page_icon="🏙️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_True=True)

# 2. Reconstrução da Rede Neural
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

# 3. Cabeçalho
st.title("🏙️ Sistema Inteligente de Monitoramento Social")
st.markdown("---")

# 4. Painel Lateral
st.sidebar.header("⚙️ Configurações da Análise")
renda = st.sidebar.number_input("Renda Média Domiciliar (R$)", min_value=0.0, value=1200.0)
esgoto = st.sidebar.slider("Acesso a Saneamento (0 a 1)", 0.0, 1.0, 0.50)
alfabetismo = st.sidebar.slider("Taxa de Alfabetização (0 a 1)", 0.0, 1.0, 0.80)

# 5. Execução e Visualização
if st.button("🚀 EXECUTAR DIAGNÓSTICO"):
    if modelo_carregado and scaler_carregado:
        colunas = ['renda_media', 'esgoto_sanitario', 'escolaridade']
        dados_usuario = pd.DataFrame([[renda, esgoto, alfabetismo]], columns=colunas)
        dados_norm = scaler.transform(dados_usuario)
        
        # Fazendo a predição
        pred = modelo.predict(dados_norm)[0][0]
        
        # SEGURANÇA: Garante que o valor esteja entre 0 e 1 e seja float puro
        probabilidade = float(np.clip(pred, 0.0, 1.0))
        perc = probabilidade * 100

        st.markdown("### 📊 Resultado do Diagnóstico")
        
        # Métricas em destaque
        c1, c2, c3 = st.columns(3)
        c1.metric("Renda Analisada", f"R$ {renda:,.2f}")
        c2.metric("Saneamento", f"{esgoto*100:.0f}%")
        c3.metric("Alfabetização", f"{alfabetismo*100:.0f}%")

        st.markdown("---")

        # Layout da Barra e Status
        col_bar, col_txt = st.columns([2, 1])
        
        with col_bar:
            st.write(f"**Índice de Vulnerabilidade:** {perc:.2f}%")
            # A barra agora está protegida pelo 'np.clip'
            st.progress(probabilidade)

        with col_txt:
            if perc > 50:
                st.error("🚨 ALTA VULNERABILIDADE")
            else:
                st.success("✅ SITUAÇÃO ESTÁVEL")

        # Recomendações Técnicas
        st.markdown("#### 💡 Plano de Ação Sugerido")
        if perc > 60:
            st.warning("**Atenção:** O modelo sugere prioridade máxima em obras de saneamento para este setor.")
        else:
            st.info("**Nota:** O setor demonstra indicadores equilibrados. Manter monitoramento preventivo.")
            
    else:
        st.error("Erro: Arquivos não encontrados no GitHub. Verifique 'modelo_pesos.weights.h5' e 'scaler.pkl'.")

st.markdown("---")
st.caption("Protótipo IA - Gestão Pública | Gabriel Carvalho - PBL 2")
