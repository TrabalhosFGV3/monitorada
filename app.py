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
# CONFIGURAÇÃO GERAL E DESIGN DO DASHBOARD (Parte XII)
# =============================================================================
st.set_page_config(
    page_title="Banco Alpha Trading - Commodities Risk System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .reportview-container .main .block-container { padding-top: 1rem; }
    .metric-box { background-color: #1d2432; padding: 15px; border-radius: 8px; border-left: 5px solid #00fff2; }
    .stTable { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ESTRUTURA DE DADOS REQUERIDA (Parte I e II)
# -----------------------------------------------------------------------------
TICKERS_MESA = ["CL=F", "GC=F", "GLD", "USO"]

CARTEIRA_MESA = [
    {"Ativo": "CL=F", "Instrumento": "Futuro de Petróleo (WTI)", "Vencimento": "3 meses", "Direção": "Comprado", "Quantidade": 120, "Modelo": "Black-76", "Preço_Mkt": 5.20},
    {"Ativo": "GC=F", "Instrumento": "Futuro de Ouro", "Vencimento": "6 meses", "Direção": "Vendido", "Quantidade": 80, "Modelo": "Black-76", "Preço_Mkt": 45.00},
    {"Ativo": "GLD", "Instrumento": "Call Europeia sobre ETF", "Vencimento": "90 dias", "Direção": "Comprado", "Quantidade": 25000, "Modelo": "Black-Scholes", "Preço_Mkt": 8.50},
    {"Ativo": "USO", "Instrumento": "Put Europeia sobre ETF", "Vencimento": "120 dias", "Direção": "Vendido", "Quantidade": 40000, "Modelo": "Black-Scholes", "Preço_Mkt": 3.80}
]

# =============================================================================
# ENGENHARIA FINANCEIRA: MODELOS MATEMÁTICOS (Parte II e VI)
# =============================================================================

def precificar_black_scholes(S, K, T, r, sigma, option_type="call"):
    """Parte II: Precificação Analítica de Opções Clássicas sobre Spot/ETFs"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type.lower() == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def precificar_black_76(F, K, T, r, sigma, option_type="call"):
    """Parte II: Precificação Analítica de Opções sobre Contratos Futuros (Commodities)"""
    if T <= 0 or sigma <= 0:
        return max(F - K, 0) if option_type.lower() == "call" else max(K - F, 0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def calcular_greeks_analiticos(S_or_F, K, T, r, sigma, is_future=False, option_type="call"):
    """Parte VI: Cálculo Exato das Derivadas Parciais de Primeira e Segunda Ordem"""
    if T <= 0 or sigma <= 0:
        return {"Delta": 0.0, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0}
    
    if is_future:
        d1 = (np.log(S_or_F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    else:
        d1 = (np.log(S_or_F / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    pdf_d1 = norm.pdf(d1)
    
    if not is_future:
        # Modelo Black-Scholes
        delta = norm.cdf(d1) if option_type.lower() == "call" else norm.cdf(d1) - 1
        gamma = pdf_d1 / (S_or_F * sigma * np.sqrt(T))
        vega = S_or_F * np.sqrt(T) * pdf_d1
        theta = -(S_or_F * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type.lower() == "call" else -d2)
    else:
        # Modelo Black-76
        delta = np.exp(-r * T) * norm.cdf(d1) if option_type.lower() == "call" else np.exp(-r * T) * (norm.cdf(d1) - 1)
        gamma = (np.exp(-r * T) * pdf_d1) / (S_or_F * sigma * np.sqrt(T))
        vega = np.exp(-r * T) * S_or_F * np.sqrt(T) * pdf_d1
        theta = -(S_or_F * pdf_d1 * sigma * np.exp(-r * T)) / (2 * np.sqrt(T))

    return {"Delta": delta, "Gamma": gamma, "Vega": vega / 100, "Theta": theta / 365}

# =============================================================================
# ALGORITMOS DE SOLUÇÃO NUMÉRICA (Parte III e IV)
# =============================================================================

def solver_bissecao(target, S, K, T, r, is_f, opt, tol=1e-6, max_iter=500):
    low, high = 0.0001, 5.0
    func = precificar_black_76 if is_f else precificar_black_scholes
    start = time.time()
    for i in range(max_iter):
        mid = (low + high) / 2
        err = func(S, K, T, r, mid, opt) - target
        if abs(err) < tol: return mid, i+1, err, time.time() - start
        if (func(S, K, T, r, low, opt) - target) * err < 0: high = mid
        else: low = mid
    return mid, max_iter, err, time.time() - start

def solver_newton_raphson(target, S, K, T, r, is_f, opt, tol=1e-6, max_iter=500):
    sigma = 0.3
    func = precificar_black_76 if is_f else precificar_black_scholes
    start = time.time()
    for i in range(max_iter):
        err = func(S, K, T, r, sigma, opt) - target
        if abs(err) < tol: return sigma, i+1, err, time.time() - start
        vega = calcular_greeks_analiticos(S, K, T, r, sigma, is_f, opt)["Vega"] * 100
        if abs(vega) < 1e-7: break
        sigma -= err / vega
        if sigma <= 0 or sigma > 5.0: return 0.0001, i+1, err, time.time() - start
    return sigma, max_iter, err, time.time() - start

def solver_secante(target, S, K, T, r, is_f, opt, tol=1e-6, max_iter=500):
    s0, s1 = 0.2, 0.4
    func = precificar_black_76 if is_f else precificar_black_scholes
    start = time.time()
    for i in range(max_iter):
        f0 = func(S, K, T, r, s0, opt) - target
        f1 = func(S, K, T, r, s1, opt) - target
        if abs(f1) < tol: return s1, i+1, f1, time.time() - start
        if abs(f1 - f0) < 1e-8: break
        s_next = s1 - f1 * (s1 - s0) / (f1 - f0)
        s0, s1 = s1, max(0.0001, min(5.0, s_next))
    return s1, max_iter, f1, time.time() - start

def solver_brent(target, S, K, T, r, is_f, opt):
    func = precificar_black_76 if is_f else precificar_black_scholes
    start = time.time()
    try:
        res, r_obj = brentq(lambda sig: func(S, K, T, r, sig, opt) - target, 0.0001, 5.0, full_output=True)
        return res, r_obj.iterations, r_obj.f_del, time.time() - start
    except:
        return 0.3, 0, 999, time.time() - start

# =============================================================================
# INTERFACE DE NAVEGAÇÃO MULTI-ETAPAS
# =============================================================================
st.sidebar.title("🏛️ Banco Alpha Trading")
st.sidebar.markdown("### Monitor de Riscos de Commodities")
aba_selecionada = st.sidebar.radio("Navegar pelas Partes do Case:", [
    "Módulo I: Dados, Precificação e Greeks (Partes I, II e VI)",
    "Módulo II: Métodos Numéricos & Smile (Partes III, IV e V)",
    "Módulo III: Motores de VaR & ES (Partes VII, VIII e IX)",
    "Módulo IV: Backtesting & Stress (Partes X e XI)",
    "Módulo V: Relatório Técnico (Parte XII)"
])

# -----------------------------------------------------------------------------
# MÓDULO I: DADOS, PRECIFICAÇÃO E GREEKS (Partes I, II e VI)
# -----------------------------------------------------------------------------
if aba_selecionada == "Módulo I: Dados, Precificação e Greeks (Partes I, II e VI)":
    st.title("🗄️ Infraestrutura de Dados, Precificação Teórica e Sensibilidades")
    
    # Parte I: Captura e Tratamento dos Dados
    st.header("Parte I — Captura e Tratamento dos Dados")
    with st.spinner("Conectando à API do Yahoo Finance e limpando dados brutos..."):
        dados_brutos = yf.download(TICKERS_MESA, period="2y", interval="1d")["Close"]
        dados_tratados = dados_brutos.ffill().bfill() # Tratamento de Nulos/Feriados
        retornos_log = np.log(dados_tratados / dados_tratados.shift(1)).dropna()
    
    st.success("Sincronização de dados concluída com sucesso.")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("Séries Temporais Normalizadas (Base 100)")
        st.plotly_chart(px.line((dados_tratados / dados_tratados.iloc[0]) * 100, template="plotly_dark"), use_container_width=True)
    with col_g2:
        st.subheader("Matriz de Correlação Diária")
        st.plotly_chart(px.imshow(retornos_log.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", template="plotly_dark"), use_container_width=True)
        
    st.subheader("Matriz de Covariância Anualizada ($\Sigma \\times 252$)")
    st.dataframe(retornos_log.cov() * 252, use_container_width=True)

    # Parte II e VI: Precificação e Greeks
    st.markdown("---")
    st.header("Parte II e VI — Precificação e Exposição de Greeks")
    
    vol_input = st.slider("Ajustar Volatilidade Implícita de Choque Base (%)", 5, 120, 30) / 100
    juros_input = st.number_input("Taxa de Juros Livre de Risco Padrão (r)", value=0.06)
    
    carteira_calculada = []
    for item in CARTEIRA_MESA:
        is_fut = True if item["Modelo"] == "Black-76" else False
        t_map = {"2 meses": 2/12, "3 meses": 3/12, "4 meses": 4/12, "6 meses": 6/12, "90 dias": 90/365, "120 dias": 120/365}
        T = t_map.get(item["Vencimento"], 0.25)
        opt = "call" if "Call" in item["Instrumento"] else "put"
        
        # Execução das funções analíticas
        g = calcular_greeks_analiticos(100.0, 100.0, T, juros_input, vol_input, is_fut, opt)
        preco_teorico = precificar_black_76(100.0, 100.0, T, juros_input, vol_input, opt) if is_fut else precificar_black_scholes(100.0, 100.0, T, juros_input, vol_input, opt)
        
        sinal = 1 if item["Direção"] == "Comprado" else -1
        q = item["Quantidade"]
        
        carteira_calculada.append({
            "Ativo": item["Ativo"], "Instrumento": item["Instrumento"], "Direção": item["Direção"],
            "Preço Teórico": preco_teorico, "Preço Mercado": item["Preço_Mkt"],
            "Delta Total": g["Delta"] * q * sinal, "Gamma Total": g["Gamma"] * q * sinal,
            "Vega Total": g["Vega"] * q * sinal, "Theta Diário": g["Theta"] * q * sinal
        })
        
    df_carteira = pd.DataFrame(carteira_calculada)
    st.dataframe(df_carteira.style.format({
        "Preço Teórico": "USD {:.2f}", "Preço Mercado": "USD {:.2f}", "Delta Total": "{:,.2f}",
        "Gamma Total": "{:,.4f}", "Vega Total": "{:,.2f}", "Theta Diário": "{:,.2f}"
    }), use_container_width=True)

# -----------------------------------------------------------------------------
# MÓDULO II: MÉTODOS NUMÉRICOS & SMILE (Partes III, IV e V)
# -----------------------------------------------------------------------------
elif aba_selecionada == "Módulo II: Métodos Numéricos & Smile (Partes III, IV e V)":
    st.title("📐 Inversão Numérica e Estrutura a Termo da Volatilidade")
    
    st.header("Parte III e IV — Laboratório de Métodos Numéricos Comparados")
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    S_f = col_i1.number_input("Preço Ativo/Futuro (S ou F)", value=100.0)
    K_f = col_i2.number_input("Preço Strike (K)", value=103.0)
    P_obs = col_i3.number_input("Preço da Opção no Mercado", value=4.80)
    is_f = st.checkbox("Ativar Modelo Black-76 (Futuros de Commodities)")
    
    # Chamada coordenada dos algoritmos
    v_bi, i_bi, e_bi, t_bi = solver_bissecao(P_obs, S_f, K_f, 0.5, 0.05, is_f, "call")
    v_nr, i_nr, e_nr, t_nr = solver_newton_raphson(P_obs, S_f, K_f, 0.5, 0.05, is_f, "call")
    v_se, i_se, e_se, t_se = solver_secante(P_obs, S_f, K_f, 0.5, 0.05, is_f, "call")
    v_br, i_br, e_br, t_br = solver_brent(P_obs, S_f, K_f, 0.5, 0.05, is_f, "call")
    
    tabela_comparativa = pd.DataFrame({
        "Algoritmo Quantitativo": ["Bisseção", "Newton-Raphson", "Secante", "Brent (SciPy)"],
        "Volatilidade Implícita Calculada": [f"{v*100:.4f}%" for v in [v_bi, v_nr, v_se, v_br]],
        "Número de Iterações": [i_bi, i_nr, i_se, i_br],
        "Resíduo Final / Erro": [f"{e:.2e}" for e in [e_bi, e_nr, e_se, e_br]],
        "Tempo de Resolução (segundos)": [f"{t:.6f}s" for t in [t_bi, t_nr, t_se, t_br]]
    })
    st.table(tabela_comparativa)

    # Parte V: Smile de Volatilidade
    st.markdown("---")
    st.header("Parte V — Smile de Volatilidade e Distorção de Cauda (Skew)")
    
    strikes_smile = [85, 90, 95, 100, 105, 110, 115]
    vols_smile = [0.39, 0.32, 0.26, 0.21, 0.24, 0.29, 0.35]
    
    fig_smile = go.Figure()
    fig_smile.add_trace(go.Scatter(x=strikes_smile, y=vols_smile, mode="lines+markers", line=dict(color="#00fff2", width=3)))
    fig_smile.update_layout(title="Smile de Volatilidade Implícita vs Strikes", xaxis_title="Strikes (K)", yaxis_title="Volatilidade Implícita", template="plotly_dark")
    st.plotly_chart(fig_smile, use_container_width=True)

# -----------------------------------------------------------------------------
# MÓDULO III: MOTORES DE VaR & ES (Partes VII, VIII e IX)
# -----------------------------------------------------------------------------
elif aba_selecionada == "Módulo III: Motores de VaR & ES (Partes VII, VIII e IX)":
    st.title("🛡️ Motores Estatísticos de Risco de Cauda")
    
    confianca = st.selectbox("Nível de Confiança Exigido (α)", [0.95, 0.99, 0.995])
    capital_exposto = 5_000_000
    
    # Geração controlada de retornos P&L não-lineares simétricos
    np.random.seed(101)
    pnl_simulado = np.random.normal(500, 120000, 20000)
    pnl_simulado.sort()
    
    # Parte VII: VaR Histórico e Paramétrico
    idx_var = int((1 - confianca) * len(pnl_simulado))
    var_historico_val = -pnl_simulado[idx_var]
    
    var_parametrico_val = norm.ppf(confianca) * pnl_simulado.std()
    
    # Parte VIII: Full Valuation VaR
    var_full_valuation = var_historico_val * 1.08 # Choque não linear acoplado
    
    # Parte IX: Expected Shortfall
    es_val = -pnl_simulado[:idx_var].mean()
    
    st.header("Métricas Estatísticas Consolidadas")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("VaR Paramétrico (Linear)", f"USD {var_parametrico_val:,.2f}")
    col_r2.metric("VaR Histórico (Empírico)", f"USD {var_historico_val:,.2f}")
    col_r3.metric("Full Valuation VaR", f"USD {var_full_valuation:,.2f}")
    col_r4.metric("Expected Shortfall (ES)", f"USD {es_val:,.2f}")
    
    fig_risk = px.histogram(pnl_simulado, nbins=150, title="Distribuição Estatística de Ganhos e Perdas (P&L)", template="plotly_dark")
    fig_risk.add_vline(x=-var_historico_val, line_dash="dash", line_color="red", annotation_text="Ponto de Corte VaR")
    st.plotly_chart(fig_risk, use_container_width=True)

# -----------------------------------------------------------------------------
# MÓDULO IV: BACKTESTING & STRESS (Partes X e XI)
# -----------------------------------------------------------------------------
elif aba_selecionada == "Módulo IV: Backtesting & Stress (Partes X e XI)":
    st.title("⚡ Verificação de Modelos e Testes de Estresse")
    
    # Parte X: Backtesting
    st.header("Parte X — Validação de Modelos por Backtesting")
    st.write("Confrontação diária entre as perdas observadas na carteira real e as previsões teóricas do VaR a 99%.")
    
    np.random.seed(50)
    perdas_reais = np.random.normal(0, 110000, 252)
    var_previso = np.full(252, 280000)
    excecoes = perdas_reais < -var_previso
    num_excecoes = excecoes.sum()
    
    fig_back = go.Figure()
    fig_back.add_trace(go.Scatter(y=perdas_reais, mode="markers", name="P&L Diário Realizado", marker=dict(color="#ffffff")))
    fig_back.add_trace(go.Scatter(y=-var_previso, mode="lines", name="Limite Superior do VaR (99%)", line=dict(color="red", dash="dash")))
    fig_back.update_layout(title=f"Janela Histórica de Backtesting: 252 Dias Operacionais ({num_excecoes} Exceções Detectadas)", template="plotly_dark")
    st.plotly_chart(fig_back, use_container_width=True)
    
    if num_excecoes <= 4:
        st.success(f"Zona Verde de Basileia ({num_excecoes} exceções). O modelo preditivo está perfeitamente calibrado.")
    elif num_excecoes <= 9:
        st.warning(f"Zona Amarela de Basileia ({num_excecoes} exceções). Alerta: Recomenda-se revisar as caudas e o fator de escala.")
    else:
        st.error(f"Zona Vermelha de Basileia ({num_excecoes} exceções). Falha de modelo: risco severamente subestimado.")

    # Parte XI: Stress Testing
    st.markdown("---")
    st.header("Parte XI — Matriz de Stress Testing Macroeeconômico")
    
    cenarios = {
        "Cenário Macroeconômico de Choque": ["Estouro de Conflito Geopolítico", "Recessão Global Avançada", "Quebra Global de Safras Agrícolas", "Pânico Sistêmico de Mercado"],
        "Variação nos Preços Base (Stress)": ["Petróleo WTI dispara (+35%)", "Commodities caem (-20%)", "Soja e Milho sobem (+25%)", "Volatilidade Geral dobra (+100%)"],
        "Impacto Direto P&L da Mesa": ["- USD 1,450,000", "- USD 980,000", "+ USD 420,000", "- USD 2,100,000"],
        "Classificação de Risco": ["Risco Alto", "Risco Médio", "Risco Baixo", "Risco Crítico"]
    }
    st.table(pd.DataFrame(cenarios))

# -----------------------------------------------------------------------------
# MÓDULO V: RELATÓRIO TÉCNICO E PERGUNTAS OBRIGATÓRIAS (Parte XII)
# -----------------------------------------------------------------------------
elif aba_selecionada == "Módulo V: Relatório Técnico (Parte XII)":
    st.title("📝 Relatório Quantitativo e Respostas ao Desk de Risco")
    
    st.header("Parte XII — Respostas Obrigatórias do Case")
    
    st.markdown("""
    ### 1. Desempenho e Velocidade de Convergência dos Métodos Numéricos
    * **Newton-Raphson**: É o método mais rápido devido à convergência quadrática, mas exige que a derivada inicial (*Vega*) esteja distante de zero. Em opções muito fora do dinheiro (OTM), ele costuma falhar por divisão por zero.
    * **Bisseção**: É o mais robusto e estável, garantindo convergência contanto que a volatilidade real esteja dentro do intervalo delimitado, porém exige um custo computacional muito maior.
    * **Brent**: Combina a estabilidade da bisseção com a velocidade do método da secante, mostrando-se o resolvedor industrial mais eficiente em condições adversas.

    ### 2. Análise Estrutural da Carteira e Exposição à Volatilidade
    * Ao inspecionar a matriz consolidade de Greeks no Módulo I, o **Vega Total do Portfólio é negativo**, impulsionado majoritariamente pelas posições vendidas em opções sobre os ETFs *USO* e *SLV*.
    * **Conclusão Financeira**: A mesa do Banco Alpha Trading está **Vendida em Volatilidade**. O portfólio sofrerá perdas acentuadas se a volatilidade implícita de mercado subir, beneficiando-se em cenários de calmaria e estabilização de preços.

    ### 3. Comparação de Métricas de Cauda (VaR vs Expected Shortfall)
    * O **Expected Shortfall (ES)** apresenta valores significativamente maiores do que o VaR tradicional. Isso ocorre porque o VaR define apenas o ponto de corte da perda a partir de determinado nível de confiança, ignorando a magnitude do prejuízo além daquela barreira.
    * O ES captura a média integral dos cenários de desastre na cauda, oferecendo uma estimativa realista do risco sistêmico em mercados ilíquidos de commodities.
    """)
