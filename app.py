import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
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
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Renda Analisada", f"R$ {renda:,.2f}")
        c2.metric("Saneamento", f"{esgoto*100:.1f}%")
        c3.metric("Educação", f"{alfabetismo*100:.1f}%")

        st.markdown("---")

        # Gráfico de Níveis (Colunas Finas)
        st.write("### 📊 Níveis dos Indicadores (%)")
        
        # Define as cores das colunas com base no nível (OPCIONAL)
        # Se preferir, pode usar uma cor fixa. Aqui, usei cores dinâmicas.
        cores_colunas = ['#28a745' if float(dados_norm[0][0] * 100) > 40 else '#dc3545',
                         '#28a745' if float(esgoto * 100) > 40 else '#dc3545',
                         '#28a745' if float(alfabetismo * 100) > 40 else '#dc3545']
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Poder de Renda', 'Saneamento', 'Educação'],
                y=[float(dados_norm[0][0] * 100), float(esgoto * 100), float(alfabetismo * 100)],
                width=0.25,
                marker_color=cores_colunas # Aplica cores dinâmicas às colunas
            )
        ])
        fig.update_layout(height=350, yaxis=dict(range=[0, 100]), template="simple_white", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # SEÇÃO DO RISCO COM FONTE GIGANTE E COR DINÂMICA
        # Lógica de cor para o número
        if perc > 70:
            cor_risco = "#dc3545" # Vermelho
        elif 40 <= perc <= 70:
            cor_risco = "#ffc107" # Amarelo
        else:
            cor_risco = "#28a745" # Verde

        # Texto com fonte aumentada e cor dinâmica
        st.markdown(f"<h1 style='text-align: center; color: {cor_risco}; font-size: 52px; font-weight: bold;'>Risco Calculado: {perc:.2f}%</h1>", unsafe_allow_html=True)
        
        # Barra de Progresso
        st.progress(probabilidade)

        # Marcadores (Legenda abaixo da barra)
        m1, m2, m3 = st.columns([40, 30, 30])
        with m1:
            st.markdown("<p style='color: #28a745; font-weight: bold; font-size: 16px;'>🟢 Estável (0-40%)</p>", unsafe_allow_html=True)
        with m2:
            st.markdown("<p style='color: #ffc107; font-weight: bold; font-size: 16px; text-align: center;'>🟡 Médio (40-70%)</p>", unsafe_allow_html=True)
        with m3:
            st.markdown("<p style='color: #dc3545; font-weight: bold; font-size: 16px; text-align: right;'>🔴 Vulnerável (70-100%)</p>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Diagnóstico de Risco
        if perc > 70:
            st.error(f"🚨 **RISCO CRÍTICO EM {bairro_selecionado.upper()}**")
            st.warning("⚠️ Prioridade máxima para intervenção em políticas públicas.")
        elif 40 <= perc <= 70:
            st.warning(f"⚠️ **ATENÇÃO: RISCO MÉDIO EM {bairro_selecionado.upper()}**")
            st.info("💡 Ação preventiva recomendada para monitorar indicadores de base.")
        else:
            st.success(f"✅ **SITUAÇÃO ESTÁVEL EM {bairro_selecionado.upper()}**")
            st.info("✅ Manter monitoramento e serviços básicos.")

    else:
        st.error("Erro: Arquivos técnicos não carregados.")

# 6. Rodapé Final
st.markdown("---")
st.caption("PBL 2 - Inteligência Artificial - São Luís")
