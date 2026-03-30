import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go # Para o gráfico de colunas finas
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

# 3. Cabeçalho Principal
st.title("🏙️ Diagnóstico de Vulnerabilidade Social")
st.markdown("Análise preditiva por setor para suporte à gestão pública municipal.")
st.markdown("---")

# 4. Painel Lateral (Sidebar)
st.sidebar.header("📍 Localização e Dados")

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
if st.button("🚀 GERAR RELATÓRIO"):
    if modelo_carregado and scaler_carregado:
        # Predição
        colunas_nomes = ['renda_media', 'esgoto_sanitario', 'escolaridade']
        dados_input = pd.DataFrame([[renda, esgoto, alfabetismo]], columns=colunas_nomes)
        dados_norm = scaler.transform(dados_input)
        
        pred = modelo.predict(dados_norm)[0][0]
        probabilidade = float(np.clip(pred, 0.0, 1.0))
        perc = probabilidade * 100

        # Cabeçalho do Relatório
        st.markdown(f"## 📋 Relatório de Impacto: {bairro_selecionado}")
        
        # Métricas
        c1, c2, c3 = st.columns(3)
        c1.metric("Renda Analisada", f"R$ {renda:,.2f}")
        c2.metric("Saneamento", f"{esgoto*100:.1f}%")
        c3.metric("Educação", f"{alfabetismo*100:.1f}%")

        st.markdown("---")

        # Gráfico de Níveis com colunas finas (Plotly)
        st.write("### 📊 Níveis dos Indicadores (%)")
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Poder de Renda', 'Saneamento', 'Educação'],
                y=[float(dados_norm[0][0] * 100), float(esgoto * 100), float(alfabetismo * 100)],
                width=0.3, # Deixa a coluna bem mais fina
                marker_color='#007BFF'
            )
        ])
        
        fig.update_layout(
            height=400,
            yaxis=dict(range=[0, 100]),
            margin=dict(l=20, r=20, t=20, b=20),
            template="simple_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Diagnóstico de Risco
        st.write(f"**Risco Calculado: {perc:.2f}%**")
        st.progress(probabilidade)

        if perc > 70:
            st.error(f"🚨 **RISCO CRÍTICO EM {bairro_selecionado.upper()}**")
        elif 40 <= perc <= 70:
            st.warning(f"⚠️ **ATENÇÃO: RISCO MÉDIO EM {bairro_selecionado.upper()}**")
        else:
            st.success(f"✅ **SITUAÇÃO ESTÁVEL EM {bairro_selecionado.upper()}**")

    else:
        st.error("Erro: Arquivos técnicos não carregados.")

# 6. Rodapé Final (Exatamente como solicitado)
st.markdown("---")
st.caption("PBL 2 - Inteligência Artificial - São Luís")
