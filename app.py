import os
import sys
from pathlib import Path
from datetime import date
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

STREAMLIT_DIR = Path(__file__).resolve().parent
ROOT_DIR = STREAMLIT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from config import settings  # type: ignore
except ModuleNotFoundError:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        load_dotenv = None  # type: ignore
    if load_dotenv:
        load_dotenv()

    class _FallbackSettings:
        APP_NAME: str = os.getenv("APP_NAME", "Dashboard Conciliação Bancária")
        DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
        TOTALBANK_API_KEY: str = os.getenv("TOTALBANK_API_KEY", "mock_key_totalbank")
        TOTALBANK_API_URL: str = os.getenv("TOTALBANK_API_URL", "https://api.totalbank.com.br")
        SAP_HOST: str = os.getenv("SAP_HOST", "localhost")
        SAP_USER: str = os.getenv("SAP_USER", "demo")
        SAP_PASSWORD: str = os.getenv("SAP_PASSWORD", "demo")
        SAP_CLIENT: str = os.getenv("SAP_CLIENT", "800")
        DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./conciliacao_bancaria.db")
        AMOUNT_TOLERANCE: float = float(os.getenv("AMOUNT_TOLERANCE", "0.01"))
        DATE_TOLERANCE_DAYS: int = int(os.getenv("DATE_TOLERANCE_DAYS", "2"))
        FUZZY_MATCH_THRESHOLD: float = float(os.getenv("FUZZY_MATCH_THRESHOLD", "0.80"))
        USE_MOCK_DATA: bool = os.getenv("USE_MOCK_DATA", "True").lower() == "true"

    settings = _FallbackSettings()

try:
    from reconciliation_engine import ReconciliationEngine  # type: ignore
except ModuleNotFoundError:
    import logging
    from datetime import datetime

    from fuzzywuzzy import fuzz

    logger = logging.getLogger(__name__)

    class Transaction:
        def __init__(
            self,
            id: str,
            source: str,
            amount: float,
            transaction_date,
            description: str,
            reference: str | None = None,
            account_number: str | None = None,
            counterparty: str | None = None,
            status: str = "pending",
        ):
            self.id = id
            self.source = source
            self.amount = amount
            self.transaction_date = (
                datetime.fromisoformat(transaction_date)
                if isinstance(transaction_date, str)
                else transaction_date
            )
            self.description = description
            self.reference = reference
            self.account_number = account_number
            self.counterparty = counterparty
            self.status = status
            self.matched_with = None
            self.match_score = 0.0

    class ReconciliationEngine:
        """Fallback reconciliation engine used when the shared module isn't available."""

        def __init__(self):
            self.amount_tolerance = settings.AMOUNT_TOLERANCE
            self.date_tolerance = settings.DATE_TOLERANCE_DAYS
            self.fuzzy_threshold = settings.FUZZY_MATCH_THRESHOLD

        def reconcile(self, totalbank_df: pd.DataFrame, sap_df: pd.DataFrame) -> Dict[str, Any]:
            tb_transactions = [Transaction(**row) for _, row in totalbank_df.iterrows()]
            sap_transactions = [Transaction(**row) for _, row in sap_df.iterrows()]

            matched_pairs: List[Dict[str, Any]] = []
            unmatched_totalbank = tb_transactions.copy()
            unmatched_sap = sap_transactions.copy()

            for tb_tx in tb_transactions:
                for sap_tx in unmatched_sap[:]:
                    if self._is_exact_match(tb_tx, sap_tx):
                        matched_pairs.append(
                            {
                                "totalbank_id": tb_tx.id,
                                "sap_id": sap_tx.id,
                                "totalbank_amount": tb_tx.amount,
                                "sap_amount": sap_tx.amount,
                                "totalbank_date": tb_tx.transaction_date,
                                "sap_date": sap_tx.transaction_date,
                                "totalbank_description": tb_tx.description,
                                "sap_description": sap_tx.description,
                                "match_type": "exact",
                                "match_score": 100.0,
                                "status": "matched",
                            }
                        )
                        unmatched_totalbank.remove(tb_tx)
                        unmatched_sap.remove(sap_tx)
                        break

            for tb_tx in unmatched_totalbank[:]:
                best_match = None
                best_score = 0.0
                best_sap_tx = None
                for sap_tx in unmatched_sap:
                    score = self._fuzzy_match_score(tb_tx, sap_tx)
                    if score > best_score:
                        best_score = score
                        best_match = sap_tx
                        best_sap_tx = sap_tx

                if best_score >= self.fuzzy_threshold and best_match:
                    matched_pairs.append(
                        {
                            "totalbank_id": tb_tx.id,
                            "sap_id": best_sap_tx.id,
                            "totalbank_amount": tb_tx.amount,
                            "sap_amount": best_sap_tx.amount,
                            "totalbank_date": tb_tx.transaction_date,
                            "sap_date": best_sap_tx.transaction_date,
                            "totalbank_description": tb_tx.description,
                            "sap_description": best_sap_tx.description,
                            "match_type": "fuzzy",
                            "match_score": round(best_score, 2),
                            "status": "matched_fuzzy",
                        }
                    )
                    unmatched_totalbank.remove(tb_tx)
                    unmatched_sap.remove(best_sap_tx)

            total_tb_amount = sum(tx.amount for tx in tb_transactions)
            total_sap_amount = sum(tx.amount for tx in sap_transactions)
            total_matched_amount = sum(p["totalbank_amount"] for p in matched_pairs)

            summary = {
                "total_totalbank_transactions": len(tb_transactions),
                "total_sap_transactions": len(sap_transactions),
                "total_matched": len(matched_pairs),
                "total_unmatched_totalbank": len(unmatched_totalbank),
                "total_unmatched_sap": len(unmatched_sap),
                "total_totalbank_amount": round(total_tb_amount, 2),
                "total_sap_amount": round(total_sap_amount, 2),
                "total_matched_amount": round(total_matched_amount, 2),
                "unmatched_totalbank_amount": round(sum(tx.amount for tx in unmatched_totalbank), 2),
                "unmatched_sap_amount": round(sum(tx.amount for tx in unmatched_sap), 2),
                "reconciliation_rate": round(
                    (len(matched_pairs) / max(len(tb_transactions), len(sap_transactions) or 1)) * 100, 2
                ),
            }

            return {
                "matched": matched_pairs,
                "unmatched_totalbank": unmatched_totalbank,
                "unmatched_sap": unmatched_sap,
                "summary": summary,
            }

        def _is_exact_match(self, tb_tx: Transaction, sap_tx: Transaction) -> bool:
            amount_diff = abs(tb_tx.amount - sap_tx.amount)
            if amount_diff > self.amount_tolerance:
                return False

            date_diff = abs((tb_tx.transaction_date - sap_tx.transaction_date).days)
            if date_diff > self.date_tolerance:
                return False
            return True

        def _fuzzy_match_score(self, tb_tx: Transaction, sap_tx: Transaction) -> float:
            scores = []
            weights = []

            amount_diff = abs(tb_tx.amount - sap_tx.amount)
            max_amount = max(tb_tx.amount, sap_tx.amount)
            amount_score = max(0, 1 - (amount_diff / max_amount if max_amount else 1)) * 100
            scores.append(amount_score)
            weights.append(0.40)

            date_diff = abs((tb_tx.transaction_date - sap_tx.transaction_date).days)
            date_score = max(0, 1 - (date_diff / 30)) * 100
            scores.append(date_score)
            weights.append(0.30)

            if tb_tx.description and sap_tx.description:
                desc_score = fuzz.ratio(tb_tx.description.lower(), sap_tx.description.lower())
            else:
                desc_score = 50
            scores.append(desc_score)
            weights.append(0.30)

            final_score = sum(s * w for s, w in zip(scores, weights))
            return final_score

DATA_DIR = STREAMLIT_DIR
SLA_COLOR = "#5CC698"
BAR_COLOR_SAFE = "#353E55"
BAR_COLOR_ALERT = "#E45E6E"
BACKGROUND_COLOR = "#f5f6fb"

st.set_page_config(
    page_title=f"{settings.APP_NAME} | Streamlit",
    layout="wide",
    page_icon=":bar_chart:"
)

CUSTOM_CSS = f"""
<style>
.stApp {{
    background-color: {BACKGROUND_COLOR};
}}

.main > div {{
    padding: 0px 40px 40px;
}}

body, p, div, span, label {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #1e2330;
}}

h1, h2, h3 {{
    font-weight: 600;
    letter-spacing: -0.02em;
}}

.section-card {{
    background: #fff;
    border-radius: 22px;
    padding: 24px 28px;
    box-shadow: 0 18px 35px rgba(15, 23, 42, 0.08);
    margin-bottom: 24px;
}}

.metric-card {{
    background: #fff;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
    border: 1px solid rgba(17, 24, 39, 0.05);
}}

.metric-title {{
    text-transform: uppercase;
    font-size: 12px;
    letter-spacing: 0.2em;
    color: #9198ad;
    margin-bottom: 8px;
}}

.metric-value {{
    font-size: 34px;
    font-weight: 600;
}}

.metric-subtitle {{
    color: #7a8199;
    font-size: 14px;
}}

.footnote {{
    font-size: 12px;
    color: #9ca3af;
    margin-top: 12px;
}}

.filter-label {{
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.18em;
    color: #c5c9d6;
    margin-bottom: 6px;
    display: block;
}}

div[data-baseweb="radio"] > div {{
    flex-wrap: wrap;
    gap: 8px;
}}

div[data-baseweb="radio"] label {{
    border: 1px solid rgba(15,23,42,0.12);
    padding: 6px 18px;
    border-radius: 999px;
    background: #fff;
    font-size: 12px;
    color: #9ca3af;
    margin-top: 12px;
}}

div[data-baseweb="radio"] input {{
    display: none;
}}

div[data-baseweb="radio"] label[data-checked="true"] {{
    background: #101828;
    color: #fff;
    border-color: #101828;
}}

.dataframe th {{
    text-transform: uppercase;
    font-size: 12px !important;
    letter-spacing: 0.08em;
    color: #6b7287 !important;
}}
    /* Force labels for other controls to light color (date, radio) */
    section[data-testid*="stDateInput"] label,
    section[data-testid*="stRadio"] label,
    div[data-testid*="stDateInput"] label,
    div[data-testid*="stRadio"] label {{
        color: #f8fafc !important;
    }}

/* Make selectbox use a dark theme with light text for readability */
div[data-baseweb="select"] > div[role="button"],
div[data-baseweb="select"] > div,
div[class*="st-"][class*="select"] > div {{
    background: #0b1220 !important; /* dark */
    color: #f8fafc !important; /* light text */
    -webkit-text-fill-color: #f8fafc !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
}}

/* Ensure the hidden search input/visible value inherits the same color */
div[data-baseweb="select"] input {{
    color: #f8fafc !important;
    caret-color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    background: transparent !important;
}}

div[data-baseweb="select"] [role="listbox"],
div[data-baseweb="select"] [role="option"],
div[data-baseweb="select"] ul,
div[data-baseweb="select"] li {{
    background: #0b1220 !important;
    color: #f8fafc !important;
    border-radius: 8px !important;
}}

div[data-baseweb="select"] [role="option"]:hover,
div[data-baseweb="select"] li:hover {{
    background: rgba(255,255,255,0.08) !important;
    color: #ffffff !important;
}}

div[data-baseweb="select"] [role="option"][aria-selected="true"],
div[data-baseweb="select"] li[aria-selected="true"] {{
    background: rgba(228,94,110,0.12) !important;
    color: #ffffff !important;
}}

/* Focus / open state with red accent to match mock */
div[data-baseweb="select"] > div[role="button"][aria-expanded="true"],
div[data-baseweb="select"] > div[role="button"]:focus-within,
div[data-baseweb="select"] > div:focus-within {{
    border-color: {BAR_COLOR_ALERT} !important;
    box-shadow: 0 6px 18px rgba(228,94,110,0.12) !important;
}}

/* Force ALL text inside selectbox to be light colored - ULTRA aggressive selectors */
div[data-baseweb="select"] *,
div[data-baseweb="select"] > div[role="button"] *,
div[data-baseweb="select"] > div > div,
div[data-baseweb="select"] > div > div *,
div[data-baseweb="select"] span,
div[data-baseweb="select"] p,
div[data-baseweb="select"] div,
div[data-baseweb="select"] span *,
div[data-baseweb="select"] div *,
div[data-baseweb="select"] label,
div[data-baseweb="select"] [role="option"] *,
div[data-baseweb="select"] li *,
[data-baseweb="select"] *,
section[data-testid*="stSelectbox"] *,
section[data-testid*="stSelectbox"] div,
section[data-testid*="stSelectbox"] span,
/* NOVO: Específico para os itens do dropdown */
li[role="option"] *,
div[role="option"] *,
[role="option"] *,
/* NOVO: Para texto dentro de spans e divs dentro do dropdown */
div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
/* NOVO: Forçar herança de cor */
* [data-baseweb="select"] * {{
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 600 !important;
    opacity: 1 !important;
    text-shadow: none !important;
}}

/* Force the select label itself to light color */
section[data-testid*="stSelectbox"] label,
div[data-testid*="stSelectbox"] label,
div[data-testid*="stSelectbox"] > label,
label[data-testid*="stLabel"] {{
    color: #f8fafc !important;
}}

/* Ensure caret/arrow is visible */
div[data-baseweb="select"] > div[role="button"] svg,
div[data-baseweb="select"] svg,
section[data-testid*="stSelectbox"] svg {{
    color: #ffffff !important;
    fill: #ffffff !important;
}}

/* Make main headings darker so they're readable on a light theme */
h1, h2, h3 {{
    color: #0b1220 !important;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# JavaScript para forçar aplicação do CSS nos elementos do dropdown
js_code = """
<script>
function forceDropdownStyles() {
    console.log('Aplicando estilos nos dropdowns...');
    
    // Esperar um pouco para os elementos carregarem
    setTimeout(function() {
        console.log('Verificando elementos do dropdown...');
        
        // Encontrar todos os elementos de dropdown
        const dropdowns = document.querySelectorAll('[data-baseweb="select"], [role="listbox"], [role="option"]');
        console.log('Encontrados', dropdowns.length, 'dropdowns');
        
        dropdowns.forEach((dropdown, index) => {
            console.log('Processando dropdown', index);
            
            // Forçar cor branca em todos os textos internos
            const allTextElements = dropdown.querySelectorAll('span, div, li, p, label, input');
            console.log('Encontrados', allTextElements.length, 'elementos de texto');
            
            allTextElements.forEach(el => {
                el.style.color = '#ffffff';
                el.style.webkitTextFillColor = '#ffffff';
                if (el.tagName === 'INPUT') {
                    el.style.caretColor = '#ffffff';
                    el.style.backgroundColor = 'transparent';
                }
                el.style.fontWeight = '600';
                el.style.opacity = '1';
                el.style.textShadow = 'none';
            });
            
            // Também aplicar diretamente no elemento
            dropdown.style.color = '#ffffff';
        });
        
        // Especificamente para itens do menu suspenso
        const menuItems = document.querySelectorAll('[role="option"]');
        console.log('Encontrados', menuItems.length, 'itens de menu');
        
        menuItems.forEach(item => {
            item.style.color = '#ffffff';
            item.style.backgroundColor = '#0b1220';
            item.style.fontWeight = '600';
        });
        
        console.log('Estilos aplicados com sucesso!');
    }, 2000); // Aumentado para 2 segundos
}

// Executar quando a página carregar
window.addEventListener('load', forceDropdownStyles);
// Também executar quando houver mudanças no DOM
const observer = new MutationObserver(forceDropdownStyles);
observer.observe(document.body, { childList: true, subtree: true });

// Forçar execução após 3 segundos também
setTimeout(forceDropdownStyles, 3000);
</script>
"""

st.markdown(js_code, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    totalbank_df = pd.read_csv(DATA_DIR / "totalbank_mock.csv")
    sap_df = pd.read_csv(DATA_DIR / "sap_mock.csv")
    for df in (totalbank_df, sap_df):
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return totalbank_df, sap_df

def format_currency(value: float) -> str:
    formatted = f"R$ {value:,.2f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")

def prepare_stage_stats(matched_df: pd.DataFrame):
    if matched_df.empty:
        return pd.DataFrame(), 0.0, 0.0
    stage_df = matched_df.copy()
    stage_df["totalbank_date"] = pd.to_datetime(stage_df["totalbank_date"])
    stage_df["sap_date"] = pd.to_datetime(stage_df["sap_date"])
    stage_df["processing_days"] = (stage_df["sap_date"] - stage_df["totalbank_date"]).dt.days.abs()
    grouped = (
        stage_df.groupby("totalbank_description", as_index=False)
        .agg(
            avg_days=("processing_days", "mean"),
            max_days=("processing_days", "max"),
            volume=("totalbank_amount", "sum"),
            registros=("totalbank_id", "count"),
        )
        .sort_values("avg_days", ascending=False)
    )
    mean_days = float(stage_df["processing_days"].mean()) if not stage_df.empty else 0.0
    max_days = float(stage_df["processing_days"].max()) if not stage_df.empty else 0.0
    return grouped, mean_days, max_days

def build_stage_chart(stage_stats: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if stage_stats.empty:
        fig.add_annotation(
            text="Sem dados conciliados no recorte selecionado",
            showarrow=False,
            font=dict(color="#7a8199", size=16)
        )
        fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=60, b=40)
        )
        return fig
    sla_reference = float(stage_stats["avg_days"].median()) if not stage_stats.empty else 0.0
    colors = [BAR_COLOR_ALERT if value > sla_reference else BAR_COLOR_SAFE for value in stage_stats["avg_days"]]
    fig.add_trace(
        go.Bar(
            x=stage_stats["totalbank_description"],
            y=stage_stats["avg_days"],
            marker_color=colors,
            text=[f"{v:.2f}" for v in stage_stats["avg_days"]],
            textposition="outside",
            textfont=dict(color="#0b1220", size=12),
            name="Tempo medio (dias)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stage_stats["totalbank_description"],
            y=[sla_reference] * len(stage_stats),
            mode="lines",
            name=f"SLA (mediana) {sla_reference:.2f}d",
            line=dict(color=SLA_COLOR, width=3),
            opacity=0.6,
        )
    )
    fig.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=60, b=80),
        plot_bgcolor="#fff",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.25,
        xaxis=dict(tickangle=-40, showgrid=False, tickfont=dict(color="#6b7287")),
        yaxis=dict(title="Tempo (dias)", gridcolor="rgba(15,23,42,0.06)", tickfont=dict(color="#6b7287")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#0b1220"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(15,23,42,0.04)",
        ),
        legend_title_text=None
    )
    # Ensure bar text uses dark color.
    fig.update_traces(textfont=dict(color="#0b1220", size=12), selector=dict(type="bar"))
    return fig

def render_metric_card(container, title: str, value: str, subtitle: str | None = None):
    subtitle_html = f"<p class='metric-subtitle'>{subtitle}</p>" if subtitle else ""
    container.markdown(
        f"""
        <div class='metric-card'>
            <p class='metric-title'>{title}</p>
            <p class='metric-value'>{value}</p>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def prepare_table(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return df
    formatted = df.copy()
    for col in formatted.columns:
        if "date" in col:
            formatted[col] = pd.to_datetime(formatted[col]).dt.strftime("%d/%m/%Y")
    return formatted.rename(columns=rename_map)

def main():
    totalbank_df, sap_df = load_data()
    combined = pd.concat([totalbank_df["transaction_date"], sap_df["transaction_date"]])
    min_date = combined.min().date()
    max_date = combined.max().date()

    st.markdown(
        f"""
        <div class='section-card' style='padding-bottom:16px;'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start; gap:24px; flex-wrap:wrap;'>
                <div>
                    <p class='metric-title' style='margin-bottom:6px;'>Painel financeiro</p>
                    <h1 style='margin:0'>{settings.APP_NAME} - Streamlit</h1>
                    <p style='color:#7a8199; margin-top:6px;'>Integra TotalBank + SAP com visual inspirado no dashboard fornecido.</p>
                </div>
                <div style='text-align:right;'>
                    <p class='metric-title' style='margin-bottom:6px;'>Periodo disponivel</p>
                    <p style='font-size:18px; margin:0;'>{min_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    filters_card = st.container()
    with filters_card:
        month_periods = combined.dt.to_period("M").sort_values().unique()
        month_labels = ["Todos"] + [p.strftime("%b/%Y").upper() for p in month_periods]
        month_lookup = {label: period for label, period in zip(month_labels[1:], month_periods)}
        default_month = month_labels[-1] if len(month_labels) > 1 else month_labels[0]
        initial_month = st.session_state.get("month_filter", default_month)
        if initial_month not in month_labels:
            initial_month = default_month

        cols = st.columns([2, 2, 2])
        with cols[0]:
            st.markdown("<span class='filter-label'>Situacao</span>", unsafe_allow_html=True)
            situation = st.selectbox(
                "Situacao",
                [
                    "Visao geral",
                    "Conciliacoes exatas",
                    "Conciliacoes fuzzy",
                    "Nao conciliadas - TotalBank",
                    "Nao conciliadas - SAP",
                ],
                label_visibility="collapsed",
            )
        with cols[1]:
            st.markdown("<span class='filter-label'>Mes de referencia</span>", unsafe_allow_html=True)
            st.selectbox(
                "Mes de referencia",
                options=month_labels,
                index=month_labels.index(initial_month),
                key="month_filter",
                label_visibility="collapsed",
            )
        with cols[2]:
            st.markdown("<span class='filter-label'>Periodo</span>", unsafe_allow_html=True)
            date_range = st.date_input(
                "Periodo",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                label_visibility="collapsed",
            )

        selected_month = st.session_state.get("month_filter", initial_month)

    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    tb_filtered = totalbank_df[
        (totalbank_df["transaction_date"].dt.date >= start_date)
        & (totalbank_df["transaction_date"].dt.date <= end_date)
    ]
    sap_filtered = sap_df[
        (sap_df["transaction_date"].dt.date >= start_date)
        & (sap_df["transaction_date"].dt.date <= end_date)
    ]

    if selected_month != "Todos":
        target_period = month_lookup[selected_month]
        tb_filtered = tb_filtered[tb_filtered["transaction_date"].dt.to_period("M") == target_period]
        sap_filtered = sap_filtered[sap_filtered["transaction_date"].dt.to_period("M") == target_period]

    engine = ReconciliationEngine()
    reconciliation = engine.reconcile(tb_filtered, sap_filtered)
    matched_df = pd.DataFrame(reconciliation["matched"])
    unmatched_tb_df = pd.DataFrame([
        {
            "id": tx.id,
            "amount": tx.amount,
            "transaction_date": tx.transaction_date,
            "description": tx.description,
            "source": tx.source,
        }
        for tx in reconciliation["unmatched_totalbank"]
    ])
    unmatched_sap_df = pd.DataFrame([
        {
            "id": tx.id,
            "amount": tx.amount,
            "transaction_date": tx.transaction_date,
            "description": tx.description,
            "source": tx.source,
        }
        for tx in reconciliation["unmatched_sap"]
    ])

    if not matched_df.empty:
        matched_df["totalbank_date"] = pd.to_datetime(matched_df["totalbank_date"])
        matched_df["sap_date"] = pd.to_datetime(matched_df["sap_date"])

    view_matched = matched_df.copy()
    if situation == "Conciliacoes exatas":
        view_matched = view_matched[view_matched["match_type"] == "exact"]
    elif situation == "Conciliacoes fuzzy":
        view_matched = view_matched[view_matched["match_type"] == "fuzzy"]

    summary = reconciliation["summary"].copy()
    if situation in {"Conciliacoes exatas", "Conciliacoes fuzzy"}:
        summary["total_matched"] = len(view_matched)
        base_total = max(summary["total_totalbank_transactions"], summary["total_sap_transactions"]) or 1
        summary["reconciliation_rate"] = round((len(view_matched) / base_total) * 100, 2)

    stage_stats, mean_days, max_days = prepare_stage_stats(view_matched)

    metric_cols = st.columns(3)
    render_metric_card(
        metric_cols[0],
        "Transacoes conciliadas",
        f"{summary['total_matched']}",
        f"Taxa de sucesso: {summary['reconciliation_rate']}%",
    )
    render_metric_card(
        metric_cols[1],
        "Tempo medio (dias)",
        f"{mean_days:.2f}",
        "Diferenca media entre TotalBank e SAP",
    )
    render_metric_card(
        metric_cols[2],
        "Tempo maximo (dias)",
        f"{max_days:.2f}",
        "Caso mais critico no periodo",
    )

    with st.container():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        tab_titles = [
            "Conciliacoes",
            "Pendencias TotalBank",
            "Pendencias SAP",
        ]
        tabs = st.tabs(tab_titles)
        with tabs[0]:
            if view_matched.empty:
                st.info("Nenhuma conciliacao para os filtros informados.")
            else:
                matched_view = prepare_table(
                    view_matched[
                        [
                            "totalbank_id",
                            "sap_id",
                            "totalbank_amount",
                            "sap_amount",
                            "totalbank_date",
                            "sap_date",
                            "match_type",
                            "match_score",
                        ]
                    ],
                    {
                        "totalbank_id": "ID TotalBank",
                        "sap_id": "ID SAP",
                        "totalbank_amount": "Valor TotalBank",
                        "sap_amount": "Valor SAP",
                        "totalbank_date": "Data TotalBank",
                        "sap_date": "Data SAP",
                        "match_type": "Tipo",
                        "match_score": "Score",
                    },
                )
                if not matched_view.empty:
                    matched_view["Valor TotalBank"] = matched_view["Valor TotalBank"].apply(format_currency)
                    matched_view["Valor SAP"] = matched_view["Valor SAP"].apply(format_currency)
                st.dataframe(matched_view, use_container_width=True, hide_index=True)
        with tabs[1]:
            if unmatched_tb_df.empty:
                st.success("Nenhuma pendencia no TotalBank para este recorte.")
            else:
                tb_table = prepare_table(
                    unmatched_tb_df,
                    {
                        "id": "ID",
                        "amount": "Valor",
                        "transaction_date": "Data",
                        "description": "Descricao",
                        "source": "Origem",
                    },
                )
                tb_table["Valor"] = tb_table["Valor"].apply(format_currency)
                st.dataframe(tb_table, use_container_width=True, hide_index=True)
        with tabs[2]:
            if unmatched_sap_df.empty:
                st.success("Nenhuma pendencia no SAP para este recorte.")
            else:
                sap_table = prepare_table(
                    unmatched_sap_df,
                    {
                        "id": "ID",
                        "amount": "Valor",
                        "transaction_date": "Data",
                        "description": "Descricao",
                        "source": "Origem",
                    },
                )
                sap_table["Valor"] = sap_table["Valor"].apply(format_currency)
                st.dataframe(sap_table, use_container_width=True, hide_index=True)
        st.markdown(
            "<p class='footnote'>Legenda das colunas: ID/ID SAP/ID TotalBank = identificadores; Valor = montante formatado; Data = data da transacao; Descricao = narrativa da operacao; Origem = sistema de origem; Tipo = tipo de conciliacao; Score = confianca da correspondencia.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Performance por categoria")
        st.write("Tempo medio de conciliacao por descricao de transacao do TotalBank.")
        chart = build_stage_chart(stage_stats)
        st.plotly_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Removed the instructional caption — not needed in the top-level dashboard

if __name__ == "__main__":
    main()
