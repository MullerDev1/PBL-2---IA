import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Configuração da Página
st.set_page_config(page_title="IA Social - São Luís", page_icon="🏙️", layout="wide")

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

# 3. Título Principal
st.title("🏙️ Diagnóstico de Vulnerabilidade Social")
st.markdown("Análise preditiva por setor para suporte à gestão pública municipal.")
st.markdown("---")

# 4. Painel Lateral (Sidebar)
st.sidebar.header("📍 Localização e Dados")

# Campo de escolher o bairro
lista_bairros = [
    "Coroadinho", "Cidade Operária", "Vila Embratel", "Anjo da Guarda", 
    "Centro", "Renascença", "Turu", "Cohatrac", "Vila Luizão", "Bairro de Fátima"
]
bairro_selecionado = st.sidebar.selectbox("Selecione o Bairro/Setor:", lista_bairros)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Indicadores do Setor")
renda = st.sidebar.number_input("Renda Média Domiciliar (R$)", min_value=0.0, value=1200.0)
esgoto = st.sidebar.slider("Nível de Saneamento", 0.0, 1.0, 0.50)
alfabetismo = st.sidebar.slider("Taxa de Alfabetização", 0.0, 1.0, 0.80)

# 5. Dashboard de Diagnóstico
if st.button("🚀 GERAR RELATÓRIO EXECUTIVO"):
    if modelo_carregado and scaler_carregado:
        # Predição
        colunas = ['renda_media', 'esgoto_sanitario', 'escolaridade']
        dados_input = pd.DataFrame([[renda, esgoto, alfabetismo]], columns=colunas)
        dados_norm = scaler.transform(dados_input)
        
        pred = modelo.predict(dados_norm)[0][0]
        probabilidade = float(np.clip(pred, 0.0, 1.0))
        perc = probabilidade * 100

        # Cabeçalho do Relatório com o Nome do Bairro
        st.markdown(f"## 📋 Relatório de Impacto: {bairro_selecionado}")
        
        # Métricas de Resumo
        c1, c2, c3 = st.columns(3)
        c1.metric("Renda Analisada", f"R$ {renda:,.2f}")
        c2.metric("Saneamento", f"{esgoto*100:.1f}%")
        c3.metric("Educação", f"{alfabetismo*100:.1f}%")

        st.markdown("---")

        # Gráfico de Níveis (Barras)
        st.write("### 📊 Gráfico de Níveis dos Indicadores")
        
        # Preparando dados para o gráfico de barras
        # Normalizamos a renda para 0-100 para comparação visual justa
        niveis_data = pd.DataFrame({
            'Indicador': ['Poder de Renda', 'Cobertura Sanitária', 'Nível Educacional'],
            'Nível (%)': [float(dados_norm[0][0] * 100), float(esgoto * 100), float(alfabetismo * 100)]
        }).set_index('Indicador')
        
        st.bar_chart(niveis_data)

        st.markdown("---")

        # Veredito Final da IA
        col_risk, col_msg = st.columns([1, 2])
        
        with col_risk:
            st.write(f"**Risco Calculado pela IA:**")
            st.header(f"{perc:.2f}%")
            st.progress(probabilidade)

        with col_msg:
            if perc > 50:
                st.error(f"🚨 **ALERTA DE VULNERABILIDADE EM {bairro_selecionado.upper()}**")
                st.markdown("O modelo identificou que este setor necessita de intervenção prioritária em políticas públicas.")
            else:
                st.success(f"✅ **SITUAÇÃO ESTÁVEL EM {bairro_selecionado.upper()}**")
                st.markdown("Os indicadores sugerem uma condição social resiliente para este setor.")

        st.info(f"Análise realizada via Rede Neural MLP com base nos pesos treinados para a região de São Luís.")
            
    else:
        st.error("Erro: Arquivos de pesos ou scaler não encontrados no GitHub.")

st.markdown("---")
st.caption("PBL 2 - Inteligência Artificial | Protótipo Funcional | Gabriel Carvalho")
