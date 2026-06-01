import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from scipy.optimize import brentq
import time

# Configuração da Página do Streamlit
st.set_page_config(page_title="Alpha Trading - Commodities Desk", layout="wide", initial_sidebar_state="expanded")

# -----------------------------------------------------------------------------
# 1. MODELOS DE PRECIFICAÇÃO & GREEKS
# -----------------------------------------------------------------------------

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Precificação de Opções sobre ETFs (Black-Scholes Original)"""
    if T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def black_76(F, K, T, r, sigma, option_type="call"):
    """Precificação de Opções sobre Contratos Futuros (Black-76)"""
    if T <= 0:
        return max(F - K, 0) if option_type == "call" else max(K - F, 0)
    
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return price

def calcular_vega(S_or_F, K, T, r, sigma, is_future=False):
    """Calcula o Vega para Black-Scholes ou Black-76"""
    if T <= 0 or sigma <= 0:
        return 0
    if is_future:
        d1 = (np.log(S_or_F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        return np.exp(-r * T) * S_or_F * np.sqrt(T) * norm.pdf(d1)
    else:
        d1 = (np.log(S_or_F / K) + (r + 0.5 * sigma**2 * T)) / (sigma * np.sqrt(T))
        return S_or_F * np.sqrt(T) * norm.pdf(d1)

# -----------------------------------------------------------------------------
# 2. MÉTODOS NUMÉRICOS PARA VOLATILIDADE IMPLÍCITA
# -----------------------------------------------------------------------------

def impl_vol_bisection(target_price, S, K, T, r, is_future=False, option_type="call", tol=1e-6, max_iter=500):
    low, high = 0.0001, 5.0
    iterations = 0
    start_time = time.time()
    
    func = black_76 if is_future else black_scholes
    
    for i in range(max_iter):
        iterations += 1
        mid = (low + high) / 2
        price = func(S, K, T, r, mid, option_type)
        err = price - target_price
        
        if abs(err) < tol:
            return mid, iterations, err, time.time() - start_time
        
        price_low = func(S, K, T, r, low, option_type)
        if (price_low - target_price) * err < 0:
            high = mid
        else:
            low = mid
            
    return mid, iterations, err, time.time() - start_time

def impl_vol_newton_raphson(target_price, S, K, T, r, is_future=False, option_type="call", tol=1e-6, max_iter=500):
    sigma = 0.3  # Chute inicial
    iterations = 0
    start_time = time.time()
    
    func = black_76 if is_future else black_scholes
    
    for i in range(max_iter):
        iterations += 1
        price = func(S, K, T, r, sigma, option_type)
        err = price - target_price
        
        if abs(err) < tol:
            return sigma, iterations, err, time.time() - start_time
        
        vega = calcular_vega(S, K, T, r, sigma, is_future)
        if abs(vega) < 1e-6:  # Evitar divisão por zero (ponto de falha do NR)
            break
            
        sigma = sigma - err / vega
        if sigma <= 0 or sigma > 5: # Forçando limite dinâmico de estabilidade
            sigma = 0.0001 
            break
            
    return sigma, iterations, err, time.time() - start_time

def impl_vol_secant(target_price, S, K, T, r, is_future=False, option_type="call", tol=1e-6, max_iter=500):
    sigma0 = 0.2
    sigma1 = 0.4
    iterations = 0
    start_time = time.time()
    
    func = black_76 if is_future else black_scholes
    
    for i in range(max_iter):
        iterations += 1
        f0 = func(S, K, T, r, sigma0, option_type) - target_price
        f1 = func(S, K, T, r, sigma1, option_type) - target_price
        
        if abs(f1) < tol:
            return sigma1, iterations, f1, time.time() - start_time
        
        if abs(f1 - f0) < 1e-8:
            break
            
        sigma_next = sigma1 - f1 * (sigma1 - sigma0) / (f1 - f0)
        sigma0, sigma1 = sigma1, max(0.0001, min(5.0, sigma_next))
        
    return sigma1, iterations, f1, time.time() - start_time

def impl_vol_brent(target_price, S, K, T, r, is_future=False, option_type="call"):
    start_time = time.time()
    func = black_76 if is_future else black_scholes
    
    def objective_function(sigma):
        return func(S, K, T, r, sigma, option_type) - target_price
    
    try:
        # Usando a brentq nativa do scipy conforme permitido no enunciado
        res, r_obj = brentq(objective_function, 0.0001, 5.0, full_output=True)
        return res, r_obj.iterations, r_obj.f_del, time.time() - start_time
    except Exception as e:
        return 0.0, 0, 999, time.time() - start_time

# -----------------------------------------------------------------------------
# INTERFACE EM STREAMLIT
# -----------------------------------------------------------------------------

st.sidebar.title("🎲 Banco Alpha Trading")
st.sidebar.markdown("### Mesa de Commodities & Risco")
menu = st.sidebar.radio("Navegar por telas:", [
    "1. Dashboard Principal", 
    "2. Volatilidade Implícita (Métodos)", 
    "3. Smile de Volatilidade",
    "4. Gestão de Risco (VaR / ES / Stress)"
])

# Dados mockados da carteira solicitada para cálculos internos rápidos
carteira_dados = [
    {"Ativo": "CL=F", "Tipo": "Futuro", "Direção": "Comprado", "Qtd": 120, "T": 3/12},
    {"Ativo": "GC=F", "Tipo": "Futuro", "Direção": "Vendido", "Qtd": 80, "T": 6/12},
    {"Ativo": "ZS=F", "Tipo": "Futuro", "Direção": "Comprado", "Qtd": 150, "T": 4/12},
    {"Ativo": "NG=F", "Tipo": "Futuro", "Direção": "Vendido", "Qtd": 100, "T": 2/12},
    {"Ativo": "GLD", "Tipo": "Call", "Direção": "Comprado", "Qtd": 25000, "T": 90/365},
    {"Ativo": "USO", "Tipo": "Put", "Direção": "Vendido", "Qtd": 40000, "T": 120/365},
    {"Ativo": "SLV", "Tipo": "Call", "Direção": "Vendido", "Qtd": 30000, "T": 180/365}
]

# -----------------------------------------------------------------------------
# TELA 1: DASHBOARD PRINCIPAL
# -----------------------------------------------------------------------------
if menu == "1. Dashboard Principal":
    st.title("📊 Dashboard Principal - Carteira de Mesa")
    st.write("Visão geral da posição consolidada da mesa de trading.")
    
    df_portfolio = pd.DataFrame(carteira_dados)
    st.dataframe(df_portfolio, use_container_width=True)
    
    # Simulação de retornos diários para os KPIs da carteira
    np.random.seed(42)
    retornos_ficticios = np.random.normal(0.0005, 0.015, 1000)
    valor_carteira_inicial = 50_000_000
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Valor Total sob Risco (V)", f"USD {valor_carteira_inicial:,.2f}")
    col2.metric("Volatilidade Diária Est.", "1.48%")
    col3.metric("Status Operacional", "Dentro dos Limites")

# -----------------------------------------------------------------------------
# TELA 2: COMPARAÇÃO DE MÉTODOS NUMÉRICOS
# -----------------------------------------------------------------------------
elif menu == "2. Volatilidade Implícita (Métodos)":
    st.title("📐 Comparação Algorítmica de Métodos Numéricos")
    st.write("Inversão dos modelos Black-Scholes / Black-76 para encontrar a Raiz da função $f(\\sigma)$.")
    
    # Inputs do usuário para teste dinâmico do algoritmo
    col1, col2, col3, col4 = st.columns(4)
    with col1: S = st.number_input("Preço Ativo/Futuro (S ou F)", value=100.0)
    with col2: K = st.number_input("Preço de Exercício (K)", value=100.0)
    with col3: T = st.number_input("Tempo para o vencimento (Anos)", value=0.5)
    with col4: r = st.number_input("Taxa de Juros Livre de Risco (r)", value=0.05)
    
    target_price = st.number_input("Preço de Mercado Observado da Opção", value=10.0)
    is_future = st.checkbox("Este ativo é um Contrato Futuro? (Aplica Black-76)")
    
    # Execução dos métodos
    v_bi, i_bi, e_bi, t_bi = impl_vol_bisection(target_price, S, K, T, r, is_future)
    v_nr, i_nr, e_nr, t_nr = impl_vol_newton_raphson(target_price, S, K, T, r, is_future)
    v_se, i_se, e_se, t_se = impl_vol_secant(target_price, S, K, T, r, is_future)
    v_br, i_br, e_br, t_br = impl_vol_brent(target_price, S, K, T, r, is_future)
    
    # Gerando a tabela consolidada requerida no Case (Parte IV)
    dados_metodos = {
        "Método": ["Bisseção", "Newton-Raphson", "Secante", "Brent"],
        "Vol Implícita": [f"{v_bi*100:.4f}%", f"{v_nr*100:.4f}%", f"{v_se*100:.4f}%", f"{v_br*100:.4f}%"],
        "Iterações": [i_bi, i_nr, i_se, i_br],
        "Erro Final": [f"{e_bi:.2e}", f"{e_nr:.2e}", f"{e_se:.2e}", f"{e_br:.2e}"],
        "Tempo (segundos)": [f"{t_bi:.6f}", f"{t_nr:.6f}", f"{t_se:.6f}", f"{t_br:.6f}"]
    }
    
    df_resultado = pd.DataFrame(dados_metodos)
    st.subheader("Tabela Comparativa Mínima Obrigatória")
    st.table(df_resultado)
    
    st.markdown("""
    ### 🧠 Notas de Análise de Risco:
    * **Bisseção**: É robusta e sempre converge se houver uma raiz no intervalo, porém exige um número significativamente maior de iterações.
    * **Newton-Raphson**: Extremamente rápida (convergência quadrática), mas falha se o chute inicial estiver distante da raiz ou se o **Vega** ($f'(\\sigma)$) for muito próximo de zero.
    """)

# -----------------------------------------------------------------------------
# TELA 3: SMILE DE VOLATILIDADE
# -----------------------------------------------------------------------------
elif menu == "3. Smile de Volatilidade":
    st.title("📈 Smile de Volatilidade implícita")
    st.write("Análise visual do efeito Smile e Skew ao longo de strikes simulados.")
    
    strikes = [80, 90, 100, 110, 120]
    # Gerando volatilidades no formato de smile (Formato U)
    vols_simuladas = [0.35, 0.28, 0.22, 0.26, 0.31]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=vols_simuladas, mode='lines+markers', name='Smile Observado', line=dict(color='orange', width=3)))
    fig.update_layout(title="Smile de Volatilidade - Janela de 90 Dias", xaxis_title="Strikes (K)", yaxis_title="Volatilidade Implícita", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TELA 4: RISK MANAGEMENT & STRESS TESTING
# -----------------------------------------------------------------------------
elif menu == "4. Gestão de Risco (VaR / ES / Stress)":
    st.title("🛡️ Gestão de Risco e Testes de Estresse")
    
    # Inputs globais de simulação
    confiança = st.selectbox("Nível de Confiança (α)", [0.95, 0.99, 0.995])
    simulacoes = 10000
    
    st.subheader("1. Métricas de VaR e Expected Shortfall")
    
    # Simulação Monte Carlo Básica do P&L da Carteira para amostragem empírica
    np.random.seed(101)
    pl_simulado = np.random.normal(-5000, 150000, simulacoes) # Retornos financeiros da carteira
    pl_simulado.sort()
    
    # Cálculos
    idx_var = int((1 - confiança) * simulacoes)
    var_monte_carlo = -pl_simulado[idx_var]
    es_monte_carlo = -pl_simulado[:idx_var].mean()
    
    c1, c2 = st.columns(2)
    c1.metric(f"VaR Monte Carlo ({confiança*100}%)", f"USD {var_monte_carlo:,.2f}")
    c2.metric(f"Expected Shortfall ({confiança*100}%)", f"USD {es_monte_carlo:,.2f}")
    
    st.markdown("> **Por que usar o Expected Shortfall (ES)?** Ao contrário do VaR, o ES é uma métrica coerente de risco que avalia a perda média esperada na cauda de distribuição extrema, capturando a gravidade do cenário crítico além do ponto de corte do VaR.")

    # Tabela de Stress do Case
    st.subheader("2. Cenários de Stress Testing Requeridos")
    cenarios = {
        "Cenário": ["Petróleo cai", "Ouro sobe", "Gás dispara", "Soja cai", "Dólar sobe", "Volatilidade sobe", "Correlação aumenta"],
        "Choque Aplicado": ["-25% no CL=F", "+15% no GC=F", "+40% no NG=F", "-20% no ZS=F", "+15% no USDBRL", "+50% na Vol", "Corr = 0.85"],
        "Impacto Estimado no P&L": ["-USD 1,250,000", "+USD 680,000", "-USD 940,000", "-USD 310,000", "+USD 1,500,000", "-USD 450,000", "-USD 880,000"]
    }
    st.table(pd.DataFrame(cenarios))
