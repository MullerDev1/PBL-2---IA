import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Configuração da Página
st.set_page_config(page_title="IA Gestão Pública - SLZ", page_icon="🏙️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #007BFF; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 2. IA - Reconstrução (Mesma lógica estável)
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
st.sidebar.header("⚙️ Parâmetros de Análise")
renda = st.sidebar.number_input("Renda Média Domiciliar (R$)", min_value=0.0, value=1200.0)
esgoto = st.sidebar.slider("Acesso a Saneamento (0-100%)", 0.0, 1.0, 0.50)
alfabetismo = st.sidebar.slider("Taxa de Alfabetização (0-100%)", 0.0, 1.0, 0.80)

# 5. Dashboard
if st.button("🚀 EXECUTAR DIAGNÓSTICO DE IA"):
    if modelo_carregado and scaler_carregado:
        colunas = ['renda_media', 'esgoto_sanitario', 'escolaridade']
        dados_usuario = pd.DataFrame([[renda, esgoto, alfabetismo]], columns=colunas)
        dados_norm = scaler.transform(dados_usuario)
        
        pred = modelo.predict(dados_norm)[0][0]
        probabilidade = float(np.clip(pred, 0.0, 1.0))
        perc = probabilidade * 100

        # --- NOVA ÁREA VISUAL ---
        st.markdown("### 📊 Relatório de Vulnerabilidade")
        
        # Métricas no topo
        m1, m2, m3 = st.columns(3)
        m1.metric("Renda Analisada", f"R$ {renda:,.2f}")
        m2.metric("Saneamento", f"{esgoto*100:.0f}%")
        m3.metric("Alfabetização", f"{alfabetismo*100:.0f}%")

        st.markdown("---")

        # Divisão em duas colunas: Gráfico à esquerda, Status à direita
        col_graf, col_status = st.columns([1.5, 1])

        with col_graf:
            st.write("**Perfil de Desenvolvimento do Setor (0 a 100%)**")
            # Criando um gráfico de barras simples com Streamlit
            chart_data = pd.DataFrame({
                'Indicador': ['Renda (Normalizada)', 'Saneamento', 'Educação'],
                'Nível (%)': [dados_norm[0][0] * 100, esgoto * 100, alfabetismo * 100]
            }).set_index('Indicador')
            
            st.bar_chart(chart_data)

        with col_status:
            st.write(f"**Índice Final de Risco: {perc:.2f}%**")
            st.progress(probabilidade)
            
            if perc > 50:
                st.error("🚨 ALTA VULNERABILIDADE")
                st.markdown("**Recomendação:** Intervenção em infraestrutura e apoio financeiro.")
            else:
                st.success("✅ SITUAÇÃO ESTÁVEL")
                st.markdown("**Recomendação:** Manutenção e programas educacionais.")

        st.info(f"O modelo analisou que este setor tem um risco de {perc:.2f}% baseado nos pesos treinados via Backpropagation.")
            
    else:
        st.error("Erro: Arquivos não encontrados no GitHub.")

st.markdown("---")
st.caption("PBL 2 - Inteligência Artificial | Protótipo Funcional - São Luís")
