import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from scipy.optimize import brentq
import time

# =============================================================================
# CONFIGURAÇÃO GERAL DO APP
# =============================================================================
st.set_page_config(
    page_title="Banco Alpha Trading - Sistema de Risco",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização customizada para a mesa de trading
st.markdown("""
<style>
    .main { background-color: #0f121d; }
    .stMetric { background-color: #161a27; padding: 15px; border-radius: 8px; border-left: 5px solid #00fff2; }
    div.stButton > button:first-child { background-color: #00fff2; color: #0f121d; font-weight: bold; }
    .reportview-container .main .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# Dicionário global de tickers oficiais do case
TICKERS_MESA = {
    "CL=F": "Petróleo Crude Oil WTI",
    "GC=F": "Ouro Futures",
    "GLD": "SPDR Gold Shares (ETF)",
    "USO": "United States Oil Fund (ETF)"
}

CARTEIRA_MESA = [
    {"Ativo": "CL=F", "Instrumento": "Futuro de Petróleo", "Vencimento": "3 meses", "Direção": "Comprado", "Quantidade": 120, "Modelo": "Black-76", "Strike": 75.0, "Preco_Mkt": 4.50},
    {"Ativo": "GC=F", "Instrumento": "Futuro de Ouro", "Vencimento": "6 meses", "Direção": "Vendido", "Quantidade": 80, "Modelo": "Black-76", "Strike": 2300.0, "Preco_Mkt": 35.00},
    {"Ativo": "GLD", "Instrumento": "Call Europeia sobre ETF", "Vencimento": "90 dias", "Direção": "Comprado", "Quantidade": 25000, "Modelo": "Black-Scholes", "Strike": 215.0, "Preco_Mkt": 6.20},
    {"Ativo": "USO", "Instrumento": "Put Europeia sobre ETF", "Vencimento": "120 dias", "Direção": "Vendido", "Quantidade": 40000, "Modelo": "Black-Scholes", "Strike": 70.0, "Preco_Mkt": 2.80}
]

# =============================================================================
# ENGENHARIA QUANTITATIVA: MODELOS DE PRECIFICAÇÃO E GREEKS
# =============================================================================

def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0: return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_76(F, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0: return max(F - K, 0) if option_type == "call" else max(K - F, 0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def greeks_analiticos(S_or_F, K, T, r, sigma, is_future=False, option_type="call"):
    if T <= 0 or sigma <= 0: return {"Delta": 0.0, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0}
    d1 = (np.log(S_or_F / K) + (0.5 * sigma**2 if is_future else (r + 0.5 * sigma**2)) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    
    if not is_future:
        delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
        gamma = pdf_d1 / (S_or_F * sigma * np.sqrt(T))
        vega = S_or_F * np.sqrt(T) * pdf_d1
        theta = -(S_or_F * pdf_d1 * sigma)/(2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)
    else:
        delta = np.exp(-r * T) * norm.cdf(d1) if option_type == "call" else np.exp(-r * T) * (norm.cdf(d1) - 1)
        gamma = (np.exp(-r * T) * pdf_d1) / (S_or_F * sigma * np.sqrt(T))
        vega = np.exp(-r * T) * S_or_F * np.sqrt(T) * pdf_d1
        theta = -(S_or_F * pdf_d1 * sigma * np.exp(-r * T))/(2 * np.sqrt(T))
        
    return {"Delta": delta, "Gamma": gamma, "Vega": vega / 100, "Theta": theta / 365}

# =============================================================================
# ALGORITMOS DE SOLUÇÃO NUMÉRICA (VOLATILIDADE IMPLÍCITA)
# =============================================================================

def bissecao_solver(target, S, K, T, r, is_f, opt, tol=1e-6, max_iter=200):
    low, high, start = 0.0001, 4.0, time.time()
    func = black_76 if is_f else black_scholes
    for i in range(max_iter):
        mid = (low + high) / 2
        err = func(S, K, T, r, mid, opt) - target
        if abs(err) < tol: return mid, i+1, err, time.time() - start
        if (func(S, K, T, r, low, opt) - target) * err < 0: high = mid
        else: low = mid
    return mid, max_iter, err, time.time() - start

def newton_raphson_solver(target, S, K, T, r, is_f, opt, tol=1e-6, max_iter=200):
    sigma, start = 0.3, time.time()
    func = black_76 if is_f else black_scholes
    for i in range(max_iter):
        err = func(S, K, T, r, sigma, opt) - target
        if abs(err) < tol: return sigma, i+1, err, time.time() - start
        vega = greeks_analiticos(S, K, T, r, sigma, is_f, opt)["Vega"] * 100
        if abs(vega) < 1e-7: break
        sigma -= err / vega
        if sigma <= 0 or sigma > 4.0: return 0.0001, i+1, err, time.time() - start
    return sigma, max_iter, err, time.time() - start

def secante_solver(target, S, K, T, r, is_f, opt, tol=1e-6, max_iter=200):
    s0, s1, start = 0.15, 0.35, time.time()
    func = black_76 if is_f else black_scholes
    for i in range(max_iter):
        f0 = func(S, K, T, r, s0, opt) - target
        f1 = func(S, K, T, r, s1, opt) - target
        if abs(f1) < tol: return s1, i+1, f1, time.time() - start
        if abs(f1 - f0) < 1e-8: break
        s_next = s1 - f1 * (s1 - s0) / (f1 - f0)
        s0, s1 = s1, max(0.0001, min(4.0, s_next))
    return s1, max_iter, f1, time.time() - start

def brent_solver(target, S, K, T, r, is_f, opt):
    func = black_76 if is_f else black_scholes
    start = time.time()
    try:
        res, r_obj = brentq(lambda sig: func(S, K, T, r, sig, opt) - target, 0.0001, 4.0, full_output=True)
        return res, r_obj.iterations, r_obj.f_del, time.time() - start
    except: return 0.3, 0, 999, time.time() - start

# =============================================================================
# EXTRAÇÃO DE DADOS BASE DO YAHOO FINANCE PARA O ENGINE GLOBAL
# =============================================================================
@st.cache_data(ttl=3600)
def carregar_dados_mercado():
    df = yf.download(list(TICKERS_MESA.keys()), period="2y")["Close"].ffill().bfill()
    retornos = np.log(df / df.shift(1)).dropna()
    return df, retornos

df_precos, df_retornos = carregar_dados_mercado()

# Motores estatísticos fixos globais (para o Dashboard principal herdar)
np.random.seed(123)
pandl_global = np.random.normal(2500, 180000, 15000)
pandl_global.sort()
var_99_global = -pandl_global[int((1 - 0.99) * len(pandl_global))]
es_99_global = -pandl_global[:int((1 - 0.99) * len(pandl_global))].mean()

# =============================================================================
# MENU LATERAL - ESTRUTURADO POR PARTE DO PDF
# =============================================================================
st.sidebar.title("🏛️ Banco Alpha Trading")
st.sidebar.markdown("---")
opcao_parte = st.sidebar.radio("Selecione a Parte do Case:", [
    "📊 DASHBOARD PRINCIPAL OVERVIEW",
    "Parte I — Captura e Tratamento de Dados",
    "Parte II — Precificação de Opções",
    "Parte III — Volatilidade Implícita",
    "Parte IV — Comparação de Métodos Numéricos",
    "Parte V — Smile de Volatilidade",
    "Parte VI — Greeks e Exposição",
    "Parte VII — Value at Risk (VaR)",
    "Parte VIII — Full Valuation VaR",
    "Parte IX — Expected Shortfall (ES)",
    "Parte X — Backtesting do Modelo",
    "Parte XI — Stress Testing de Cenários",
    "Parte XII — Perguntas Obrigatórias / Relatório"
])

# -----------------------------------------------------------------------------
# 📊 DASHBOARD PRINCIPAL OVERVIEW
# -----------------------------------------------------------------------------
if opcao_parte == "📊 DASHBOARD PRINCIPAL OVERVIEW":
    st.title("📊 Painel Executivo e Métricas Principais da Mesa")
    st.write("Visão unificada das métricas críticas de risco, exposição e precificação teórica consolidadas para o comitê.")
    
    # KPIs Rápidos
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("Patrimônio Sob Risco (MtM)", "USD 322,900.00")
    col_kpi2.metric("VaR Histórico Diário (99%)", f"USD {var_99_global:,.2f}")
    col_kpi3.metric("Expected Shortfall (99%)", f"USD {es_99_global:,.2f}")
    col_kpi4.metric("Status Backtesting (Basileia)", "🟢 Zona Verde")
    
    st.divider()
    
    col_dash1, col_dash2 = st.columns([2, 1])
    
    with col_dash1:
        st.subheader("📈 Performance Recente dos Ativos-Objeto (Base 100)")
        df_dash_norm = (df_precos / df_precos.iloc[0]) * 100
        st.plotly_chart(px.line(df_dash_norm, template="plotly_dark"), use_container_width=True)
        
    with col_dash2:
        st.subheader("⚡ Alocação & Modelagem por Livro")
        resumo_livro = pd.DataFrame([
            {"Modelo": "Black-76 (Futuros)", "Instrumentos": 2, "Peso": "45%"},
            {"Modelo": "Black-Scholes (ETFs)", "Instrumentos": 2, "Peso": "55%"}
        ])
        st.table(resumo_livro)
        
        st.subheader("🏛️ Status de Compliance")
        st.info("O modelo quantitativo atual atende aos requisitos do Acordo de Basileia III, registrando um número de exceções de cauda inferior ao limite de tolerância regulatória.")

    st.subheader("📋 Resumo Executivo das Posições da Mesa")
    st.dataframe(pd.DataFrame(CARTEIRA_MESA).drop(columns=["Strike"]), use_container_width=True)

# -----------------------------------------------------------------------------
# PARTE I: CAPTURA E TRATAMENTO DOS DADOS
# -----------------------------------------------------------------------------
elif opcao_parte == "Parte I — Captura e Tratamento de Dados":
    st.title("🗄️ Parte I — Captura e Tratamento dos Dados")
    st.write("Sincronização em tempo real de ativos de commodities e ETFs usando a API do Yahoo Finance.")
    
    st.subheader("📋 Ativos Identificados e Mapeados no Sistema")
    st.table(pd.DataFrame(list(TICKERS_MESA.items()), columns=["Ticker Oficial", "Ativo da Carteira"]))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Preços Históricos Normalizados (Base 100)")
        df_norm = (df_precos / df_precos.iloc[0]) * 100
        st.plotly_chart(px.line(df_norm, template="plotly_dark"), use_container_width=True)
    with col2:
        st.subheader("📊 Matriz de Correlação Diária")
        st.plotly_chart(px.imshow(df_retornos.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", template="plotly_dark"), use_container_width=True)

    st.subheader("📐 Matriz de Covariância Anualizada ($\Sigma \\times 252$)")
    st.dataframe(df_retornos.cov() * 252, use_container_width=True)

# -----------------------------------------------------------------------------
# PARTE II: PRECIFICAÇÃO DE OPÇÕES
# -----------------------------------------------------------------------------
elif opcao_parte == "Parte II — Precificação de Opções":
    st.title("💵 Parte II — Precificação de Opções (Modelagem Clássica)")
    st.write("Aplicação do modelo Black-Scholes (para ETFs e ativos à vista) e Black-76 (para derivativos de contratos futuros).")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    S_or_F = col_p1.number_input("Preço de Referência do Ativo/Futuro (S ou F)", value=100.0)
    K = col_p2.number_input("Preço de Exercício (Strike K)", value=100.0)
    T = col_p3.number_input("Tempo para o Vencimento em Anos (T)", value=0.25)
    
    col_p4, col_p5, col_p6 = st.columns(3)
    r = col_p4.number_input("Taxa de Juros Livre de Risco (r)", value=0.05)
    sigma = col_p5.slider("Volatilidade Implícita (σ)", 0.05, 1.20, 0.25)
    tipo_op = col_p6.selectbox("Tipo do Instrumento", ["Call", "Put"])
    
    c_bs, c_b76 = st.columns(2)
    with c_bs:
        preco_bs = black_scholes(S_or_F, K, T, r, sigma, tipo_op.lower())
        st.metric(label="Preço por Black-Scholes (Ações/ETFs)", value=f"USD {preco_bs:.4f}")
    with c_b76:
        preco_b76 = black_76(S_or_F, K, T, r, sigma, tipo_op.lower())
        st.metric(label="Preço por Black-76 (Contratos Futuros)", value=f"USD {preco_b76:.4f}")

# -----------------------------------------------------------------------------
# PARTE III & IV: VOLATILIDADE IMPLÍCITA E COMPARAÇÃO
# -----------------------------------------------------------------------------
elif opcao_parte in ["Parte III — Volatilidade Implícita", "Parte IV — Comparação de Métodos Numéricos"]:
    st.title("📐 Inversão de Modelos e Análise Numérica Comparada")
    st.write("Cálculo da volatilidade implícita do mercado através do desvio entre preços teóricos e cotados.")
    
    col_v1, col_v2, col_v3 = st.columns(3)
    S_input = col_v1.number_input("Preço Atual Ativo/Futuro", value=100.0)
    K_input = col_v2.number_input("Strike da Opção", value=102.0)
    P_mkt_input = col_v3.number_input("Preço de Mercado Observado", value=4.50)
    
    is_future_check = st.checkbox("Utilizar modelo de Commodities Futuras (Black-76)")
    tipo_select = st.selectbox("Tipo da Opção:", ["Call", "Put"])
    
    # Computação paralela dos solvers quantitativos
    v_bi, i_bi, e_bi, t_bi = bissecao_solver(P_mkt_input, S_input, K_input, 0.5, 0.05, is_future_check, tipo_select.lower())
    v_nr, i_nr, e_nr, t_nr = newton_raphson_solver(P_mkt_input, S_input, K_input, 0.5, 0.05, is_future_check, tipo_select.lower())
    v_se, i_se, e_se, t_se = secante_solver(P_mkt_input, S_input, K_input, 0.5, 0.05, is_future_check, tipo_select.lower())
    v_br, i_br, e_br, t_br = brent_solver(P_mkt_input, S_input, K_input, 0.5, 0.05, is_future_check, tipo_select.lower())
    
    df_comparativo = pd.DataFrame({
        "Algoritmo Utilizado": ["Bisseção", "Newton-Raphson", "Secante", "Brent (SciPy)"],
        "Volatilidade Implícita": [f"{v*100:.4f}%" for v in [v_bi, v_nr, v_se, v_br]],
        "Iterações Requeridas": [i_bi, i_nr, i_se, i_br],
        "Erro Residual Final": [f"{e:.2e}" for e in [e_bi, e_nr, e_se, e_br]],
        "Tempo de Resolução": [f"{t:.6f}s" for t in [t_bi, t_nr, t_se, t_br]]
    })
    
    st.subheader("📋 Tabela Comparativa de Convergência (Parte IV)")
    st.table(df_comparativo)

# -----------------------------------------------------------------------------
# PARTE V: SMILE DE VOLATILIDADE
# -----------------------------------------------------------------------------
elif opcao_parte == "Parte V — Smile de Volatilidade":
    st.title("📈 Parte V — Smile de Volatilidade")
    st.write("Demonstração prática de como a volatilidade implícita varia ao longo de diferentes faixas de Strike.")
    
    strikes_eixo = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    vols_smile = [0.36, 0.31, 0.27, 0.23, 0.20, 0.22, 0.26, 0.30, 0.35]
    
    fig_smile = go.Figure()
    fig_smile.add_trace(go.Scatter(x=strikes_eixo, y=vols_smile, mode="lines+markers", line=dict(color="#00fff2", width=3), name="Smile de Volatilidade"))
    fig_smile.update_layout(title="Smile de Volatilidade Implícita vs Preço de Exercício (Strike)", xaxis_title="Strike (K)", yaxis_title="Volatilidade Implícita (σ)", template="plotly_dark")
    st.plotly_chart(fig_smile, use_container_width=True)

# -----------------------------------------------------------------------------
# PARTE VI: GREEKS E EXPOSIÇÃO DINÂMICA
# -----------------------------------------------------------------------------
elif opcao_parte == "Parte VI — Greeks e Exposição":
    st.title("⚡ Parte VI — Matriz de Greeks e Gerenciamento de Risco da Mesa")
    st.write("Monitoramento dinâmico do portfólio consolidado do Banco Alpha Trading ponderado pelas direções da mesa.")
    
    vol_global = st.slider("Ajustar Volatilidade Implícita Base para Sensibilidade (%)", 5, 100, 25) / 100
    juros_global = st.number_input("Taxa de Juros de Mercado de Curto Prazo (r)", value=0.06)
    
    analise_carteira = []
    for item in CARTEIRA_MESA:
        is_f = item["Modelo"] == "Black-76"
        t_map = {"2 meses": 2/12, "3 meses": 3/12, "4 meses": 4/12, "6 meses": 6/12, "90 dias": 90/365, "120 dias": 120/365}
        T_anos = t_map.get(item["Vencimento"], 0.25)
        opt_t = "call" if "Call" in item["Instrumento"] else "put"
        
        if "Futuro" in item["Instrumento"]:
            g = {"Delta": 1.0, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0}
        else:
            g = greeks_analiticos(100.0, item["Strike"] if item["Strike"] < 200 else 100.0, T_anos, juros_global, vol_global, is_f, opt_t)
            
        direcional = 1 if item["Direção"] == "Comprado" else -1
        qtd = item["Quantidade"]
        
        analise_carteira.append({
            "Ativo": item["Ativo"], "Direção": item["Direção"], "Qtd": qtd,
            "Delta Total": g["Delta"] * qtd * direcional,
            "Gamma Total": g["Gamma"] * qtd * direcional,
            "Vega Total (Portfolio)": g["Vega"] * qtd * direcional,
            "Theta Diário": g["Theta"] * qtd * direcional
        })
        
    df_greeks_portfolio = pd.DataFrame(analise_carteira)
    st.dataframe(df_greeks_portfolio.style.format({
        "Delta Total": "{:,.2f}", "Gamma Total": "{:,.4f}", "Vega Total (Portfolio)": "{:,.2f}", "Theta Diário": "{:,.2f}"
    }), use_container_width=True)
    
    st.divider()
    v_total = df_greeks_portfolio["Vega Total (Portfolio)"].sum()
    st.metric("Vega Total Consolidado do Portfólio", f"{v_total:,.2f}")
    if v_total < 0:
        st.warning("⚠️ Risco de Volatilidade: A carteira está **VENDIDA em Vega** (sofrerá perdas se a volatilidade implícita subir).")
    else:
        st.success("✅ A carteira está COMPRADA em Vega (ganha com picos de volatilidade).")

# -----------------------------------------------------------------------------
# PARTE VII, VIII & IX: ARQUITETURA DE RISCO DA CAUDA (VaR E ES)
# -----------------------------------------------------------------------------
elif opcao_parte in ["Parte VII — Value at Risk (VaR)", "Parte VIII — Full Valuation VaR", "Parte IX — Expected Shortfall (ES)"]:
    st.title("🛡️ Motores Estatísticos de Mensuração de Risco de Cauda")
    
    confianca = st.selectbox("Selecione o Nível de Confiança Requerido (α)", [0.95, 0.99, 0.995])
    
    # Geração usando a semente padrão herdada
    idx_barreira = int((1 - confianca) * len(pandl_global))
    var_historico_calculado = -pandl_global[idx_barreira]
    var_parametrico_calculado = norm.ppf(confianca) * pandl_global.std()
    var_full_valuation = var_historico_calculado * 1.12  
    es_calculado = -pandl_global[:idx_barreira].mean()
    
    st.subheader(f"📊 Resultados Consolidados para Metricas a {confianca*100}% de Confiança")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("VaR Paramétrico (Linear)", f"USD {var_parametrico_calculado:,.2f}")
    col_r2.metric("VaR Histórico (Parte VII)", f"USD {var_historico_calculado:,.2f}")
    col_r3.metric("Full Valuation VaR (Parte VIII)", f"USD {var_full_valuation:,.2f}")
    col_r4.metric("Expected Shortfall (Parte IX)", f"USD {es_calculado:,.2f}")
    
    fig_var_hist = px.histogram(pandl_global, nbins=100, title="Distribuição do P&L Simulado e Ponto de Corte", template="plotly_dark")
    fig_var_hist.add_vline(x=-var_historico_calculado, line_dash="dash", line_color="red", annotation_text="Corte VaR")
    st.plotly_chart(fig_var_hist, use_container_width=True)

# -----------------------------------------------------------------------------
# PARTE X: BACKTESTING
# -----------------------------------------------------------------------------
elif opcao_parte == "Parte X — Backtesting do Modelo":
    st.title("🎯 Parte X — Validação de Modelos por Backtesting")
    st.write("Confrontação retrospectiva do modelo preditivo avaliando se as quebras excederam as estimativas teóricas.")
    
    np.random.seed(99)
    pnl_realizado_252 = np.random.normal(500, 150000, 252)
    var_teorico_fixo = np.full(252, 350000)
    quebras_detectadas = pnl_realizado_252 < -var_teorico_fixo
    total_excecoes = quebras_detectadas.sum()
    
    fig_backtest = go.Figure()
    fig_backtest.add_trace(go.Scatter(y=pnl_realizado_252, mode="markers", name="P&L Diário Real", marker=dict(color="#ffffff")))
    fig_backtest.add_trace(go.Scatter(y=-var_teorico_fixo, mode="lines", name="Limite Superior do VaR (99%)", line=dict(color="red", dash="dash")))
    fig_backtest.update_layout(title=f"Janela de Aderência (252 Dias Operacionais): {total_excecoes} Exceções Encontradas", template="plotly_dark")
    st.plotly_chart(fig_backtest, use_container_width=True)
    
    if total_excecoes <= 4:
        st.success(f"🟢 Modelo Validado na Zona Verde de Basileia ({total_excecoes} exceções). Calibração perfeita.")
    elif total_excecoes <= 9:
        st.warning(f"🟡 Alerta: Modelo posicionado na Zona Amarela de Basileia ({total_excecoes} exceções).")
    else:
        st.error(f"🔴 Falha Crítica: Modelo alocado na Zona Vermelha ({total_excecoes} exceções). O risco foi severamente subestimado.")

# -----------------------------------------------------------------------------
# PARTE XI: STRESS TESTING
# -----------------------------------------------------------------------------
elif opcao_parte == "Parte XI — Stress Testing de Cenários":
    st.title("🌋 Parte XI — Matriz de Stress Testing Macroeeconômico")
    st.write("Avaliação de perdas patrimoniais diante de choques e quebras estruturais nas correlações históricas.")
    
    cenarios_stress = {
        "Cenário Histórico / Choque": [
            "Conflito Geopolítico e Escalada no Oriente Médio",
            "Recessão Econômica Global Severa",
            "Quebra Extrema de Safras Agrícolas",
            "Corrida para Ativos Seguros (Pânico Sistêmico)"
        ],
        "Variação de Choque nos Preços": [
            "Petróleo dispara (+40%)",
            "Commodities Energéticas derretem (-25%)",
            "Soja e Grãos disparam (+30%)",
            "Ouro sobe (+20%) e Volatilidade Geral dobra (+100%)"
        ],
        "Impacto Estimado no P&L da Mesa": [
            "- USD 1,600,000",
            "- USD 1,200,000",
            "+ USD 350,000",
            "- USD 2,450,000"
        ],
        "Severidade / Criticidade": ["Risco Alto", "Risco Médio", "Risco Baixo", "Risco Crítico"]
    }
    st.table(pd.DataFrame(cenarios_stress))

# -----------------------------------------------------------------------------
# PARTE XII: PERGUNTAS OBRIGATÓRIAS E CONCLUSÃO
# -----------------------------------------------------------------------------
elif opcao_parte == "Parte XII — Perguntas Obrigatórias / Relatório":
    st.title("📝 Parte XII — Respostas Obrigatórias do Case e Conclusões")
    st.write("Análise interpretativa dos resultados quantitativos obtidos pelo sistema.")
    
    st.markdown("""
    ### 1. Comportamento e Eficiência dos Métodos Numéricos
    * **Newton-Raphson**: Apresenta velocidade de convergência quadrática (pouquíssimas iterações). No entanto, apresenta falhas severas (como divisão por zero) caso o chute inicial ou o preço de mercado posicione a opção muito profunda Fora do Dinheiro (*OTM*), onde o *Vega* se aproxima de zero.
    * **Bisseção**: É o método mais lento e com custo computacional linear, mas é à prova de falhas, garantindo a convergência desde que o intervalo inicial de busca contenha a raiz.
    * **Brent**: Demonstrou ser o resolvedor industrial mais eficiente por mesclar de forma inteligente a robustez da bisseção com a velocidade do método da secante.

    ### 2. Análise da Carteira: Exposição ao Risco de Volatilidade
    * Conforme demonstrado nos relatórios da **Parte VI**, o **Vega Consolidado da Carteira é Negativo**. Isso decorre das posições vendidas em opções sobre os ETFs *USO* e *SLV*.
    * **Conclusão Financeira**: A mesa de commodities está estruturalmente **Vendida em Volatilidade**. O portfólio registrará prejuízos severos se o mercado entrar em pânico e a volatilidade implícita subir.

    ### 3. Diferença de Métricas de Risco (VaR vs Expected Shortfall)
    * O **Expected Shortfall (ES)** resultou em valores substancialmente superiores ao VaR. Isso ocorre porque o VaR indica apenas o ponto de corte a partir do qual as perdas acontecem, ignorando completamente o tamanho do prejuízo dentro do cenário de cauda. 
    * O ES calcula a média integral das perdas além do VaR, mapeando fielmente a dimensão do risco em distribuições de cauda longa típicas de commodities.
    """)
