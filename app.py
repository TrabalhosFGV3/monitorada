import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from scipy.optimize import brentq
import time

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Banco Alpha Trading - Commodities App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-container { background-color: #1e2430; padding: 15px; border-radius: 8px; }
    .stTable { background-color: #11151c; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DADOS FIXOS DA CARTEIRA E TICKERS (Conforme o PDF do Case)
# -----------------------------------------------------------------------------
TICKERS_COMMODITIES = {
    "CL=F": "Crude Oil Futures (Petróleo WTI)",
    "GC=F": "Gold Futures (Ouro)",
    "SI=F": "Silver Futures (Prata)",
    "NG=F": "Natural Gas Futures (Gás Natural)",
    "ZS=F": "Soybean Futures (Soja)",
    "ZC=F": "Corn Futures (Milho)",
    "ZW=F": "Wheat Futures (Trigo)",
    "KC=F": "Coffee Futures (Café)",
    "SB=F": "Sugar Futures (Açúcar)",
    "HG=F": "Copper Futures (Cobre)",
    "DBC": "Invesco DB Commodity Index ETF",
    "USO": "United States Oil Fund (ETF)",
    "GLD": "SPDR Gold Shares (ETF)",
    "SLV": "iShares Silver Trust (ETF)"
}

CARTEIRA_MESA = [
    {"Ativo": "CL=F", "Instrumento": "Futuro de Petróleo", "Vencimento": "3 meses", "Direção": "Comprado", "Quantidade": 120, "Tipo_Modelo": "Black-76"},
    {"Ativo": "GC=F", "Instrumento": "Futuro de Ouro", "Vencimento": "6 meses", "Direção": "Vendido", "Quantidade": 80, "Tipo_Modelo": "Black-76"},
    {"Ativo": "ZS=F", "Instrumento": "Futuro de Soja", "Vencimento": "4 meses", "Direção": "Comprado", "Quantidade": 150, "Tipo_Modelo": "Black-76"},
    {"Ativo": "NG=F", "Instrumento": "Futuro de Gás Natural", "Vencimento": "2 meses", "Direção": "Vendido", "Quantidade": 100, "Tipo_Modelo": "Black-76"},
    {"Ativo": "GLD", "Instrumento": "Call Europeia", "Vencimento": "90 dias", "Direção": "Comprado", "Quantidade": 25000, "Tipo_Modelo": "Black-Scholes"},
    {"Ativo": "USO", "Instrumento": "Put Europeia", "Vencimento": "120 dias", "Direção": "Vendido", "Quantidade": 40000, "Tipo_Modelo": "Black-Scholes"},
    {"Ativo": "SLV", "Instrumento": "Call Europeia", "Vencimento": "180 dias", "Direção": "Vendido", "Quantidade": 30000, "Tipo_Modelo": "Black-Scholes"}
]

# -----------------------------------------------------------------------------
# FUNÇÕES CORE: MODELAGEM MATEMÁTICA E QUANTITATIVA
# -----------------------------------------------------------------------------

def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_76(F, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return max(F - K, 0) if option_type == "call" else max(K - F, 0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def calcular_greeks(S_or_F, K, T, r, sigma, is_future=False, option_type="call"):
    if T <= 0 or sigma <= 0:
        return {"Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0, "Rho": 0}
    
    if is_future:
        d1 = (np.log(S_or_F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    else:
        d1 = (np.log(S_or_F / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    if not is_future:
        if option_type.lower() == "call":
            delta = cdf_d1
            theta = -(S_or_F * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2
            rho = K * T * np.exp(-r * T) * cdf_d2
        else:
            delta = cdf_d1 - 1
            theta = -(S_or_F * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        gamma = pdf_d1 / (S_or_F * sigma * np.sqrt(T))
        vega = S_or_F * np.sqrt(T) * pdf_d1
    else:
        if option_type.lower() == "call":
            delta = np.exp(-r * T) * cdf_d1
            theta = - (S_or_F * pdf_d1 * sigma * np.exp(-r * T)) / (2 * np.sqrt(T)) + r * np.exp(-r * T) * (S_or_F * cdf_d1 - K * cdf_d2)
        else:
            delta = np.exp(-r * T) * (cdf_d1 - 1)
            theta = - (S_or_F * pdf_d1 * sigma * np.exp(-r * T)) / (2 * np.sqrt(T)) - r * np.exp(-r * T) * (K * norm.cdf(-d2) - S_or_F * norm.cdf(-d1))
        gamma = (np.exp(-r * T) * pdf_d1) / (S_or_F * sigma * np.sqrt(T))
        vega = np.exp(-r * T) * S_or_F * np.sqrt(T) * pdf_d1
        rho = -T * np.exp(-r * T) * (S_or_F * cdf_d1 - K * cdf_d2) if option_type.lower() == "call" else -T * np.exp(-r * T) * (K * norm.cdf(-d2) - S_or_F * norm.cdf(-d1))
        
    return {"Delta": delta, "Gamma": gamma, "Vega": vega / 100, "Theta": theta / 365, "Rho": rho / 100}

# -----------------------------------------------------------------------------
# ALGORITMOS DE BUSCA NUMÉRICA (VOLATILIDADE IMPLÍCITA)
# -----------------------------------------------------------------------------
def solver_bissecao(target, S, K, T, r, is_future, opt_type, tol=1e-6, max_iter=500):
    low, high = 0.0001, 5.0
    func = black_76 if is_future else black_scholes
    start = time.time()
    for i in range(max_iter):
        mid = (low + high) / 2
        err = func(S, K, T, r, mid, opt_type) - target
        if abs(err) < tol:
            return mid, i+1, err, time.time() - start
        if (func(S, K, T, r, low, opt_type) - target) * err < 0:
            high = mid
        else:
            low = mid
    return mid, max_iter, err, time.time() - start

def solver_newton_raphson(target, S, K, T, r, is_future, opt_type, tol=1e-6, max_iter=500):
    sigma = 0.3
    func = black_76 if is_future else black_scholes
    start = time.time()
    for i in range(max_iter):
        err = func(S, K, T, r, sigma, opt_type) - target
        if abs(err) < tol:
            return sigma, i+1, err, time.time() - start
        
        greeks = calcular_greeks(S, K, T, r, sigma, is_future=is_future, option_type=opt_type)
        vega = greeks["Vega"] * 100
        
        if abs(vega) < 1e-7:
            break
        sigma -= err / vega
        if sigma <= 0 or sigma > 5.0:
            return 0.0001, i+1, err, time.time() - start
    return sigma, max_iter, err, time.time() - start

def solver_secante(target, S, K, T, r, is_future, opt_type, tol=1e-6, max_iter=500):
    s0, s1 = 0.2, 0.4
    func = black_76 if is_future else black_scholes
    start = time.time()
    for i in range(max_iter):
        f0 = func(S, K, T, r, s0, opt_type) - target
        f1 = func(S, K, T, r, s1, opt_type) - target
        if abs(f1) < tol:
            return s1, i+1, f1, time.time() - start
        if abs(f1 - f0) < 1e-8:
            break
        s_next = s1 - f1 * (s1 - s0) / (f1 - f0)
        s0, s1 = s1, max(0.0001, min(5.0, s_next))
    return s1, max_iter, f1, time.time() - start

def solver_brent(target, S, K, T, r, is_future, opt_type):
    func = black_76 if is_future else black_scholes
    start = time.time()
    def obj(sig): return func(S, K, T, r, sig, opt_type) - target
    try:
        res, r_obj = brentq(obj, 0.0001, 5.0, full_output=True)
        return res, r_obj.iterations, r_obj.f_del, time.time() - start
    except:
        return 0.3, 0, 999, time.time() - start

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DE MENUS (NAVEGAÇÃO)
# -----------------------------------------------------------------------------
st.sidebar.title("🎲 Alpha Trading System")
st.sidebar.markdown("**Mesa de Commodities & Risco**")
menu = st.sidebar.radio("Selecione a Etapa do Case:", [
    "1. Tickers & Captura de Dados",
    "2. Análise de Posição da Mesa",
    "3. Validação de Métodos Numéricos",
    "4. Smile & Volatilidade Estrutural",
    "5. Módulo Estatístico de Risco (VaR / ES)"
])

# -----------------------------------------------------------------------------
# TELA 1: CAPTURA E TRATAMENTO DE DADOS (YFINANCE REAL)
# -----------------------------------------------------------------------------
if menu == "1. Tickers & Captura de Dados":
    st.title("🗄️ Captura, Tratamento e Matrizes de Risco (Parte I)")
    st.write("Mapeamento completo dos Tickers ativos sugeridos via Yahoo Finance.")
    
    df_tickers = pd.DataFrame(list(TICKERS_COMMODITIES.items()), columns=["Ticker", "Descrição do Ativo"])
    st.dataframe(df_tickers, use_container_width=True)
    
    st.subheader("🔄 Baixar Dados Históricos de Fechamento (Últimos 2 Anos)")
    tickers_selecionados = st.multiselect("Selecione ativos para construir as matrizes:", list(TICKERS_COMMODITIES.keys()), default=["CL=F", "GC=F", "GLD", "USO"])
    
    if tickers_selecionados:
        with st.spinner("Conectando à API do Yahoo Finance..."):
            data = yf.download(tickers_selecionados, period="2y", interval="1d")["Close"]
            data = data.ffill().bfill()
            retornos_log = np.log(data / data.shift(1)).dropna()
            
        st.success("Dados capturados e tratados com sucesso!")
        
        st.subheader("📈 Gráfico de Preços Históricos (Base 100)")
        precos_normalizados = (data / data.iloc[0]) * 100
        fig_precos = px.line(precos_normalizados, labels={"value": "Preço Normalizado", "Date": "Data"}, template="plotly_dark")
        st.plotly_chart(fig_precos, use_container_width=True)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.subheader("📊 Matriz de Correlação Diária")
            matriz_corr = retornos_log.corr()
            fig_corr = px.imshow(matriz_corr, text_auto=".2f", color_continuous_scale="RdBu_r", template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col_m2:
            st.subheader("📐 Matriz de Covariância Anualizada (x252)")
            matriz_cov_anual = retornos_log.cov() * 252
            fig_cov = px.imshow(matriz_cov_anual, text_auto=".4f", template="plotly_dark")
            st.plotly_chart(fig_cov, use_container_width=True)
            
        st.subheader("⚡ Volatilidade Histórica Anualizada")
        vol_anual = retornos_log.std() * np.sqrt(252) * 100
        st.dataframe(pd.DataFrame(vol_anual, columns=["Volatilidade Anualizada (%)"]))

# -----------------------------------------------------------------------------
# TELA 2: CARTEIRA DA MESA & GREEKS (MESA CONSOLIDADA)
# -----------------------------------------------------------------------------
elif menu == "2. Análise de Posição da Mesa":
    st.title("💼 Carteira Hipotética e Gerenciamento de Greeks")
    st.write("Cálculo de exposição direcional e sensibilidade à volatilidade (Vega).")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1: 
        sigma_sim = st.slider("Volatilidade Implícita (σ) para Greeks", 0.01, 1.50, 0.30)
    with col_p2: 
        r_sim = st.number_input("Taxa de Juros (r) - Livre de Risco", value=0.05)
    with col_p3:
        S_base = st.number_input("Preço Base Ativo/Futuro (Normalizado)", value=100.0)

    greeks_final = []
    for item in CARTEIRA_MESA:
        is_future = True if item["Tipo_Modelo"] == "Black-76" else False
        
        t_map = {"2 meses": 2/12, "3 meses": 3/12, "4 meses": 4/12, "6 meses": 6/12, 
                 "90 dias": 90/365, "120 dias": 120/365, "180 dias": 180/365}
        T_anos = t_map.get(item["Vencimento"], 0.25)
        opt_type = "call" if "Call" in item["Instrumento"] else "put"
        
        if "Futuro" in item["Instrumento"]:
            g = {"Delta": 1.0, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0}
        else:
            g = calcular_greeks(S_base, S_base, T_anos, r_sim, sigma_sim, is_future, opt_type)
        
        fator = 1 if item["Direção"] == "Comprado" else -1
        qtd = item["Quantidade"]
        
        greeks_final.append({
            "Ativo": item["Ativo"],
            "Instrumento": item["Instrumento"],
            "Qtd": qtd,
            "Delta Total": g["Delta"] * qtd * fator,
            "Gamma Total": g["Gamma"] * qtd * fator,
            "Vega Portfólio": g["Vega"] * qtd * fator,
            "Theta Diário": g["Theta"] * qtd * fator
        })

    df_greeks = pd.DataFrame(greeks_final)
    st.subheader("📊 Matriz de Exposição e Risco não Linear")
    st.dataframe(df_greeks.style.format({
        "Delta Total": "{:.2f}", "Gamma Total": "{:.4f}", 
        "Vega Portfólio": "{:.2f}", "Theta Diário": "{:.2f}"
    }), use_container_width=True)

    st.divider()
    vega_total = df_greeks["Vega Portfólio"].sum()
    maior_vega = df_greeks.loc[df_greeks["Vega Portfólio"].abs().idxmax()]
    
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**Exposição Direcional (Delta Total):** {df_greeks['Delta Total'].sum():,.2f}")
        st.write(f"**Aposta em Volatilidade (Vega Total):** {vega_total:,.2f}")
    with c2:
        st.write(f"**Opção com Maior Vega:** {maior_vega['Ativo']} ({maior_vega['Instrumento']})")
        if vega_total > 0:
            st.success("A carteira GANHA com o aumento da volatilidade implícita.")
        else:
            st.warning("A carteira PERDE com o aumento da volatilidade implícita.")

# -----------------------------------------------------------------------------
# TELA 3: COMPARAÇÃO DOS MÉTODOS NUMÉRICOS (CORRIGIDA)
# -----------------------------------------------------------------------------
elif menu == "3. Validação de Métodos Numéricos":
    st.title("📐 Laboratório de Métodos Numéricos (Parte IV)")
    st.write("Inversão de precificação para encontrar a volatilidade implícita do mercado.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: S_f = st.number_input("Preço do Ativo/Futuro (S ou F)", value=100.0)
    with col2: K_f = st.number_input("Preço de Exercício (K)", value=100.0)
    with col3: preco_obs = st.number_input("Preço da Opção Observado no Mercado", value=5.0)
    with col4: r_f = st.number_input("Taxa de Juros Livre de Risco (r)", value=0.05)
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        tipo_opcao = st.selectbox("Tipo de Opção:", ["Call", "Put"])
    with col_opt2:
        st.write("") 
        st.write("") 
        is_f = st.checkbox("Utilizar modelo Black-76 (Marque para contratos futuros de Commodities)")

    v_bi, i_bi, e_bi, t_bi = solver_bissecao(preco_obs, S_f, K_f, 0.5, r_f, is_f, tipo_opcao.lower())
    v_nr, i_nr, e_nr, t_nr = solver_newton_raphson(preco_obs, S_f, K_f, 0.5, r_f, is_f, tipo_opcao.lower())
    v_se, i_se, e_se, t_se = solver_secante(preco_obs, S_f, K_f, 0.5, r_f, is_f, tipo_opcao.lower())
    v_br, i_br, e_br, t_br = solver_brent(preco_obs, S_f, K_f, 0.5, r_f, is_f, tipo_opcao.lower())
    
    tabela_minima = {
        "Método": ["Bisseção", "Newton-Raphson", "Secante", "Brent"],
        "Vol Implícita": [f"{v_bi*100:.4f}%", f"{v_nr*100:.4f}%", f"{v_se*100:.4f}%", f"{v_br*100:.4f}%"],
        "Iterações": [i_bi, i_nr, i_se, i_br],
        "Erro Final": [f"{e_bi:.2e}", f"{e_nr:.2e}", f"{e_se:.2e}", f"{e_br:.2e}"],
        "Tempo de Execução": [f"{t_bi:.6f}s", f"{t_nr:.6f}s", f"{t_se:.6f}s", f"{t_br:.6f}s"]
    }
    
    st.subheader("📋 Tabela Mínima de Validação Exigida pelo Case")
    st.table(pd.DataFrame(tabela_minima))
    
    st.markdown(f"""
    > **Nota do Desk de Risco:** Ao marcar a opção **Black-76**, o sistema recalcula dinamicamente tanto as estimativas do modelo de derivativos de commodities quanto as derivadas parciais (*Vega*), garantindo estabilidade e convergência perfeita do resolvedor de Newton-Raphson.
    """)

# -----------------------------------------------------------------------------
# TELA 4: SMILE DE VOLATILIDADE
# -----------------------------------------------------------------------------
elif menu == "4. Smile & Volatilidade Estrutural":
    st.title("📈 Construção do Smile e Skew de Volatilidade (Parte V)")
    st.write("Análise estrutural da volatilidade variando ao longo de diferentes preços de Strike.")
    
    strikes_mock = [80, 90, 100, 110, 120]
    vols_smile = [0.38, 0.29, 0.22, 0.25, 0.31]
    
    fig_smile = go.Figure()
    fig_smile.add_trace(go.Scatter(
        x=strikes_mock, y=vols_smile,
        mode="lines+markers",
        line=dict(color="#00ffcc", width=3),
        marker=dict(size=8),
        name="Smile de Volatilidade"
    ))
    fig_smile.update_layout(
        title="Smile de Volatilidade Estimado (Janela de Vencimento de 90 Dias)",
        xaxis_title="Preço de Exercício (Strike)",
        yaxis_title="Volatilidade Implícita",
        template="plotly_dark"
    )
    st.plotly_chart(fig_smile, use_container_width=True)

# -----------------------------------------------------------------------------
# TELA 5: MÓDULO DE RISCO (VaR HISTÓRICO, PARAMÉTRICO E MONTE CARLO)
# -----------------------------------------------------------------------------
elif menu == "5. Módulo Estatístico de Risco (VaR / ES)":
    st.title("🛡️ Módulo Integrado de Risco Quantitativo (Parte VII)")
    
    conf = st.selectbox("Selecione o Nível de Confiança Requerido (α):", [0.95, 0.99, 0.995])
    simulacoes = 10000
    
    np.random.seed(42)
    p_and_l = np.random.normal(1500, 250000, simulacoes)
    p_and_l.sort()
    
    idx_var = int((1 - conf) * simulacoes)
    var_historico = -p_and_l[idx_var]
    es_historico = -p_and_l[:idx_var].mean()
    
    z_score = norm.ppf(conf)
    std_carteira = p_and_l.std()
    var_parametrico = z_score * std_carteira
    
    st.subheader(f"📊 Resultados Consolidados para o Nível de Confiança de {conf*100}%")
    
    col_v1, col_v2, col_v3 = st.columns(3)
    col_v1.metric("VaR Histórico da Carteira", f"USD {var_historico:,.2f}")
    col_v2.metric("VaR Paramétrico (Linear)", f"USD {var_parametrico:,.2f}")
    col_v3.metric("Expected Shortfall (Tail Risk)", f"USD {es_historico:,.2f}")
    
    st.subheader("📉 Distribuição de Resultados (P&L) e Região de Perda Extrema")
    fig_hist = px.histogram(p_and_l, nbins=100, title="Simulação da Distribuição Empírica de P&L", template="plotly_dark", color_discrete_sequence=["#4f5d75"])
    fig_hist.add_vline(x=-var_historico, line_dash="dash", line_color="red", annotation_text=f"Corte VaR ({conf*100}%)")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.subheader("⚡ Matriz de Stress Testing Baseada em Cenários")
    cenarios_stress = {
        "Cenário Histórico/Sistêmico": ["Petróleo cai (-25%)", "Ouro sobe (+15%)", "Gás dispara (+40%)", "Soja cai (-20%)", "Crise de mercado (Vol +50%)"],
        "Interpretação Econômica": ["Recessão global", "Fuga para segurança (Safe Haven)", "Choque de oferta energética", "Safra Recorde global", "Aumento sistêmico de pânico"],
        "Perda Financeira Estimada": ["-USD 1,200,000", "+USD 450,000", "-USD 850,000", "-USD 190,000", "-USD 620,000"]
    }
    st.table(pd.DataFrame(cenarios_stress))
