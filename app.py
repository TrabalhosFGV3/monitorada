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
# CONFIGURAÇÃO DA PÁGINA E ESTILO
# =============================================================================
st.set_page_config(page_title="Alpha Trading - Commodities System", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# MODELAGEM MATEMÁTICA (BLACK-SCHOLES & BLACK-76)
# =============================================================================

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Precificação de Opções sobre ETFs (Ações)"""
    if T <= 0 or sigma <= 0: return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_76(F, K, T, r, sigma, option_type="call"):
    """Precificação de Opções sobre Futuros"""
    if T <= 0 or sigma <= 0: return max(F - K, 0) if option_type == "call" else max(K - F, 0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def calcular_greeks(S_or_F, K, T, r, sigma, is_future=False, option_type="call"):
    """Cálculo analítico de Sensibilidades (Greeks)"""
    if T <= 0 or sigma <= 0: return {"Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0, "Rho": 0}
    
    if is_future:
        d1 = (np.log(S_or_F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    else:
        d1 = (np.log(S_or_F / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    pdf_d1 = norm.pdf(d1)
    
    if not is_future:
        delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
        vega = S_or_F * np.sqrt(T) * pdf_d1
        gamma = pdf_d1 / (S_or_F * sigma * np.sqrt(T))
        theta = -(S_or_F * pdf_d1 * sigma)/(2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)
    else:
        delta = np.exp(-r * T) * norm.cdf(d1) if option_type == "call" else np.exp(-r * T) * (norm.cdf(d1) - 1)
        vega = np.exp(-r * T) * S_or_F * np.sqrt(T) * pdf_d1
        gamma = (np.exp(-r * T) * pdf_d1) / (S_or_F * sigma * np.sqrt(T))
        theta = -(S_or_F * pdf_d1 * sigma * np.exp(-r * T))/(2 * np.sqrt(T))

    return {"Delta": delta, "Gamma": gamma, "Vega": vega/100, "Theta": theta/365}

# =============================================================================
# MÉTODOS NUMÉRICOS (PARTE IV)
# =============================================================================

def solver_bissecao(target, S, K, T, r, is_f, opt, tol=1e-6, max_iter=500):
    low, high, start = 0.0001, 5.0, time.time()
    func = black_76 if is_f else black_scholes
    for i in range(max_iter):
        mid = (low + high) / 2
        err = func(S, K, T, r, mid, opt) - target
        if abs(err) < tol: return mid, i+1, err, time.time()-start
        if (func(S, K, T, r, low, opt) - target) * err < 0: high = mid
        else: low = mid
    return mid, max_iter, err, time.time()-start

def solver_newton_raphson(target, S, K, T, r, is_f, opt, tol=1e-6, max_iter=500):
    sigma, start = 0.3, time.time()
    func = black_76 if is_f else black_scholes
    for i in range(max_iter):
        err = func(S, K, T, r, sigma, opt) - target
        if abs(err) < tol: return sigma, i+1, err, time.time()-start
        vega = calcular_greeks(S, K, T, r, sigma, is_f, opt)["Vega"] * 100
        if abs(vega) < 1e-7: break
        sigma -= err / vega
        if sigma <= 0 or sigma > 5: return 0.0001, i+1, err, time.time()-start
    return sigma, max_iter, err, time.time()-start

def solver_secante(target, S, K, T, r, is_f, opt, tol=1e-6, max_iter=500):
    s0, s1, start = 0.2, 0.4, time.time()
    func = black_76 if is_f else black_scholes
    for i in range(max_iter):
        f1 = func(S, K, T, r, s1, opt) - target
        if abs(f1) < tol: return s1, i+1, f1, time.time()-start
        f0 = func(S, K, T, r, s0, opt) - target
        if abs(f1 - f0) < 1e-9: break
        s_next = s1 - f1 * (s1 - s0) / (f1 - f0)
        s0, s1 = s1, max(0.0001, min(5.0, s_next))
    return s1, max_iter, f1, time.time()-start

def solver_brent(target, S, K, T, r, is_f, opt):
    func = black_76 if is_f else black_scholes
    start = time.time()
    try:
        res, r_obj = brentq(lambda x: func(S, K, T, r, x, opt) - target, 0.0001, 5.0, full_output=True)
        return res, r_obj.iterations, r_obj.f_del, time.time()-start
    except: return 0.0, 0, 999, time.time()-start

# =============================================================================
# DASHBOARD STREAMLIT
# =============================================================================

st.sidebar.title("💎 Alpha Trading Desk")
menu = st.sidebar.radio("Navegação:", ["Dashboard & Dados", "Exposição & Greeks", "Métodos Numéricos", "Risco (VaR/ES)"])

# Dados da Carteira conforme Seção 4 do PDF
CARTEIRA = [
    {"Ativo": "CL=F", "Inst": "Futuro Petróleo", "Dir": "Comprado", "Qtd": 120, "Venc": 3/12, "Mod": "Black-76"},
    {"Ativo": "GC=F", "Inst": "Futuro Ouro", "Dir": "Vendido", "Qtd": 80, "Venc": 6/12, "Mod": "Black-76"},
    {"Ativo": "GLD", "Inst": "Call Europeia", "Dir": "Comprado", "Qtd": 25000, "Venc": 90/365, "Mod": "Black-Scholes"},
    {"Ativo": "USO", "Inst": "Put Europeia", "Dir": "Vendido", "Qtd": 40000, "Venc": 120/365, "Mod": "Black-Scholes"}
]

if menu == "Dashboard & Dados":
    st.title("📊 Monitoramento de Commodities")
    tickers = ["CL=F", "GC=F", "ZS=F", "NG=F", "DBC", "GLD", "USO", "SLV"]
    
    with st.spinner("Baixando dados do Yahoo Finance..."):
        df_prices = yf.download(tickers, period="1y")["Close"].ffill().bfill()
        returns = np.log(df_prices / df_prices.shift(1)).dropna()
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Preços Normalizados (Base 100)")
        st.line_chart((df_prices / df_prices.iloc[0]) * 100)
    with c2:
        st.subheader("Matriz de Correlação")
        st.plotly_chart(px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale="RdBu_r"))

elif menu == "Exposição & Greeks":
    st.title("⚡ Sensibilidade da Carteira (Greeks)")
    vol_sim = st.slider("Volatilidade de Simulação (%)", 5, 100, 25) / 100
    r_sim = st.number_input("Taxa de Juros (r)", value=0.10)
    
    resultados = []
    for p in CARTEIRA:
        is_f = p["Mod"] == "Black-76"
        opt = "call" if "Call" in p["Inst"] else ("put" if "Put" in p["Inst"] else "call")
        
        if "Futuro" in p["Inst"]:
            g = {"Delta": 1.0, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0}
        else:
            g = calcular_greeks(100, 100, p["Venc"], r_sim, vol_sim, is_f, opt)
        
        sentido = 1 if p["Dir"] == "Comprado" else -1
        resultados.append({
            "Ativo": p["Ativo"], "Qtd": p["Qtd"], "Direção": p["Dir"],
            "Delta T.": g["Delta"] * p["Qtd"] * sentido,
            "Vega T.": g["Vega"] * p["Qtd"] * sentido,
            "Theta T.": g["Theta"] * p["Qtd"] * sentido
        })
    
    df_res = pd.DataFrame(resultados)
    st.table(df_res)
    st.metric("Vega Total do Portfólio", f"{df_res['Vega T.'].sum():.2f}")

elif menu == "Métodos Numéricos":
    st.title("📐 Comparação de Métodos Numéricos (VI)")
    col1, col2, col3, col4 = st.columns(4)
    S = col1.number_input("Preço (S ou F)", value=100.0)
    K = col2.number_input("Strike (K)", value=100.0)
    P_mkt = col3.number_input("Preço Mercado", value=5.0)
    is_f = st.checkbox("Usar Black-76 (Futuros)")
    
    v_bi, i_bi, e_bi, t_bi = solver_bissecao(P_mkt, S, K, 0.5, 0.05, is_f, "call")
    v_nr, i_nr, e_nr, t_nr = solver_newton_raphson(P_mkt, S, K, 0.5, 0.05, is_f, "call")
    v_se, i_se, e_se, t_se = solver_secante(P_mkt, S, K, 0.5, 0.05, is_f, "call")
    v_br, i_br, e_br, t_br = solver_brent(P_mkt, S, K, 0.5, 0.05, is_f, "call")
    
    df_met = pd.DataFrame({
        "Método": ["Bisseção", "Newton-Raphson", "Secante", "Brent"],
        "Vol Implícita": [f"{v*100:.4f}%" for v in [v_bi, v_nr, v_se, v_br]],
        "Iterações": [i_bi, i_nr, i_se, i_br],
        "Tempo (s)": [f"{t:.6f}" for t in [t_bi, t_nr, t_se, t_br]]
    })
    st.table(df_met)

elif menu == "Risco (VaR/ES)":
    st.title("🛡️ Gestão de Risco: VaR & ES")
    conf = st.selectbox("Nível de Confiança", [0.95, 0.99, 0.995])
    
    # Simulação Monte Carlo (Seção 11.3)
    np.random.seed(42)
    retornos_sim = np.random.normal(0, 0.02, 10000)
    pnl = retornos_sim * 1_000_000 # Carteira hipotética de 1M
    pnl.sort()
    
    idx = int((1 - conf) * 10000)
    var_mc = -pnl[idx]
    es_mc = -pnl[:idx].mean()
    
    c1, c2 = st.columns(2)
    c1.metric(f"VaR {conf*100}%", f"USD {var_mc:,.2f}")
    c2.metric(f"Expected Shortfall", f"USD {es_mc:,.2f}")
    
    fig = px.histogram(pnl, nbins=100, title="Distribuição de P&L")
    fig.add_vline(x=-var_mc, line_dash="dash", line_color="red")
    st.plotly_chart(fig)
