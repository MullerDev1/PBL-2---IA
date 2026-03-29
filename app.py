import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Configuração Visual da Página
st.set_page_config(page_title="IA Gestão Pública - SLZ", page_icon="🏙️", layout="wide")

# Custom CSS para melhorar a estética
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white; }
    .result-card { padding: 20px; border-radius: 10px; border: 1px solid #ddd; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. Reconstrução da Rede Neural (Exatamente como a anterior)
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

# 3. Título e Descrição Principal
st.title("🏙️ Sistema Inteligente de Monitoramento Social")
st.subheader("Análise Preditiva de Vulnerabilidade por Setor Censitário")
st.markdown("---")

# 4. Painel Lateral (Inputs)
st.sidebar.header("⚙️ Configurações da Análise")
renda = st.sidebar.number_input("Renda Média Domiciliar (R$)", min_value=0.0, value=1200.0)
esgoto = st.sidebar.slider("Acesso a Saneamento (0 a 1)", 0.0, 1.0, 0.50)
alfabetismo = st.sidebar.slider("Taxa de Alfabetização (0 a 1)", 0.0, 1.0, 0.80)

# 5. Dashboard de Resultados
if st.button("🚀 EXECUTAR DIAGNÓSTICO DE INTELIGÊNCIA ARTIFICIAL"):
    if modelo_carregado and scaler_carregado:
        # Lógica de Predição
        colunas = ['renda_media', 'esgoto_sanitario', 'escolaridade']
        dados_usuario = pd.DataFrame([[renda, esgoto, alfabetismo]], columns=colunas)
        dados_norm = scaler.transform(dados_usuario)
        probabilidade = modelo.predict(dados_norm)[0][0]
        perc = probabilidade * 100

        # --- PARTE VISUAL MELHORADA ---
        st.markdown("### 📊 Resultado do Diagnóstico")
        
        # Colunas de Métricas
        c1, c2, c3 = st.columns(3)
        c1.metric("Renda Analisada", f"R$ {renda:,.2f}")
        c2.metric("Saneamento", f"{esgoto*100:.0f}%")
        c3.metric("Alfabetização", f"{alfabetismo*100:.0f}%")

        st.markdown("---")

        # Barra de Progresso e Score
        col_bar, col_txt = st.columns([2, 1])
        
        with col_bar:
            st.write(f"**Índice de Vulnerabilidade:** {perc:.2f}%")
            # Muda a cor da barra conforme o risco
            cor_barra = "red" if perc > 50 else "green"
            st.progress(probabilidade)

        with col_txt:
            if perc > 50:
                st.error("🚨 ALTA VULNERABILIDADE")
            else:
                st.success("✅ SITUAÇÃO ESTÁVEL")

        # Cards Informativos Explicativos (IA Explicável)
        st.markdown("#### 💡 Plano de Ação Sugerido")
        if perc > 60:
            st.warning("""
            **Prioridade 1:** O modelo detectou que a combinação de baixa infraestrutura sanitária e renda moderada gera um risco alto. 
            - Sugestão: Mutirão de saneamento básico e programas de transferência de renda.
            """)
        else:
            st.info("""
            **Monitoramento:** O setor apresenta resiliência social. 
            - Sugestão: Manter serviços básicos e focar em educação continuada.
            """)
            
    else:
        st.error("Erro técnico: Certifique-se de que os arquivos de pesos e scaler estão no GitHub.")

# Rodapé profissional
st.markdown("---")
st.caption("Desenvolvido por Gabriel Carvalho | Protótipo IA - Gestão Pública Municipal")
