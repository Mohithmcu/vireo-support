import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from vireo import pipeline as P

st.set_page_config(
    page_title="Vireo Audio | Support Analytics",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Enterprise Theme CSS (Light, High-Contrast, Clean)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
    }
    
    /* Main Background */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04) !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #64748b !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
        margin-bottom: 4px !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #0f172a !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Headers */
    h1 {
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 2px !important;
    }
    h2, h3 {
        color: #1e293b !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #1e293b !important;
    }
    .sidebar-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        color: #475569 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #eff6ff !important;
        color: #2563eb !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading and analyzing support data...")
def build(d="data"):
    raw, a, o, p = P.load(d)
    t = P.clean(raw, a, o, p)
    R, mu, tau, s = P.agent_table(t)
    rel = P.split_half_reliability(s, n_iter=300)
    lot, base = P.lot_alerts(t, o, p)
    fa = P.false_alarm_test(t)
    dq = P.quality_report(raw, t)
    return t, R, mu, tau, rel, lot, base, fa, dq

with st.sidebar:
    st.markdown("### 🎧 Vireo Support Analytics")
    st.caption("Operational & Quality Dashboard")
    st.divider()
    data_dir = st.text_input("Data folder", value="data", help="Directory containing the 5 export CSVs")
    st.divider()
    st.markdown("""
    **Key Executive Findings:**
    - 🔴 **Pulse 2 batches 2510–2512:** 41.7%–47.0% failure rate vs 2%–9% normal baseline
    - 💰 **₹14.5 lakh** total excess replacement cost *(range: ₹13.0L–₹14.5L)*
    - 🛡️ **₹10.9 lakh** avoidable via prompt lot holds
    - 👤 **Only 3 agents** sit below mean *(suggestive candidates for 1-on-1 coaching, not punitive action)*
    """)

try:
    t, R, mu, tau, rel, lot, base, fa, dq = build(data_dir)
except Exception as e:
    st.error(f"Could not load data from '{data_dir}': {e}")
    st.stop()

st.title("Vireo Audio — Support Operations & Defect Analytics")
st.caption("Reporting Period: Jan 2025 – Jun 2026 · 11,750 Helpdesk Tickets · 44 Agents · Bengaluru & Indore")

# Top KPI metrics row
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Total Tickets", f"{len(t):,}")
with col2: st.metric("Mean CSAT (Tier 1)", f"{mu:.2f} / 5.0")
with col3: st.metric("Overall Replacement Rate", f"{t.repl.mean():.1%}")
with col4: st.metric("SLA Breaches", f"{int(t.breach.sum()):,}")
with col5:
    flagged = lot[lot.flagged]
    st.metric("Excess Repl Cost", f"₹{flagged.excess_rs.sum()/100000:.1f} Lakh")

st.write("")

tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Per-Agent (Raw View)",
    "📊 Per-Agent (Adjusted View)",
    "⚠️ Defect Lot Alerts",
    "🔍 Data Quality & SLAs",
])

# ─────────────────────────────────────────────
# TAB 1: RAW VIEW
# ─────────────────────────────────────────────
with tab1:
    st.warning(
        "⚠️ **Raw CSAT primarily reflects ticket routing, not agent capability.** "
        "The bottom agents in this raw table handle 60–65% hardware-fault tickets "
        "from defective Pulse 2 earbud lots—dealing with the most frustrated customers by design. "
        "Review Tab 2 (Adjusted View) before making any coaching or bonus allocation decisions."
    )

    col_a, col_b = st.columns([3, 1])
    with col_b:
        show_team = st.multiselect("Filter by team", sorted(R.agent_team.unique()), default=[])

    display_R = R.copy()
    if show_team:
        display_R = display_R[display_R.agent_team.isin(show_team)]

    raw_sorted = display_R.sort_values("raw").reset_index()
    raw_sorted["rank"] = range(1, len(raw_sorted) + 1)
    raw_sorted["flag_bottom10"] = raw_sorted["rank"] <= 10

    fig1 = px.bar(
        raw_sorted, x="agent_name", y="raw",
        color="flag_bottom10",
        color_discrete_map={True: "#ef4444", False: "#2563eb"},
        hover_data={"agent_team": True, "agent_shift": True, "tickets": True,
                    "hw_share": ":.0%", "scored": True, "flag_bottom10": False},
        labels={"raw": "Raw CSAT", "agent_name": "Agent", "flag_bottom10": "Bottom 10"},
        title="Raw CSAT by Agent (Tier 1 Support)",
    )
    fig1.add_hline(y=mu, line_dash="dash", line_color="#d97706",
                   annotation_text=f"Overall Mean: {mu:.2f}", annotation_position="top right")
    fig1.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_tickangle=-45,
        showlegend=False,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        yaxis=dict(gridcolor="#f1f5f9", range=[2.5, 4.0]),
    )
    st.plotly_chart(fig1, width="stretch")

    cols_show = ["agent_name","agent_team","agent_shift","agent_site","tickets","scored","raw","handle_med","breach_rate","hw_share","pl2_share"]
    disp = raw_sorted[["agent_id"] + cols_show].set_index("agent_id")
    disp.columns = ["Name","Team","Shift","Site","Tickets","Scored","Raw CSAT","Handle Med (min)","Breach %","HW Share","PL2 Share"]
    disp = disp.sort_values("Raw CSAT")
    
    st.dataframe(
        disp.style.background_gradient(subset=["Raw CSAT"], cmap="RdYlGn", vmin=2.8, vmax=3.9)
               .format({"Raw CSAT": "{:.2f}", "Breach %": "{:.0%}", "HW Share": "{:.0%}", "PL2 Share": "{:.0%}", "Handle Med (min)": "{:.0f}"}),
        width="stretch", height=380,
    )
    st.caption("💡 **Note on handle time:** Logistics and Returns Desk agents show median handle times of ~1,465 minutes because they wait on courier confirmations and refund processing. Frontline chat, email, and voice agents average 21–24 minutes.")

# ─────────────────────────────────────────────
# TAB 2: ADJUSTED VIEW
# ─────────────────────────────────────────────
with tab2:
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Between-Agent SD", f"{tau:.3f} pts", help="True standard deviation after Empirical Bayes shrinkage")
    with col2: st.metric("Avg Measurement Error", f"{R.se.mean():.3f} pts", help="Typical standard error of individual agent estimates")
    with col3: st.metric("Adjusted Reliability", f"{rel:.2f}", help="Split-half reliability over 300 iterations (0.25 adjusted vs 0.69 raw)")
    with col4: st.metric("Agents Below Mean (CI)", int((R.hi < mu).sum()), help="Only these 3 are statistically below average")

    st.info(
        "**Statistical Interpretation:** The true between-agent spread (0.076 CSAT pts) is smaller than the measurement error (0.094). "
        "Permutation testing demonstrates that random chance produces up to 3 flags under the null hypothesis (95th percentile). "
        "The 3 flagged agents (all in Chat Frontline) have small adjusted gaps (0.20 to 0.29 pts) and are suggestive candidates for supportive coaching, not confirmed underperformers."
    )

    adj_sorted = R.sort_values("adj").reset_index()
    adj_sorted["below_mean"] = adj_sorted["hi"] < mu

    fig2 = go.Figure()
    colors = ["#ef4444" if b else "#2563eb" for b in adj_sorted.below_mean]
    fig2.add_trace(go.Scatter(
        x=adj_sorted.agent_name, y=adj_sorted.adj,
        mode="markers", marker=dict(size=9, color=colors),
        error_y=dict(type="data",
                     array=(adj_sorted.hi - adj_sorted.adj).values,
                     arrayminus=(adj_sorted.adj - adj_sorted.lo).values,
                     color="rgba(100, 116, 139, 0.4)", width=2),
        name="Adjusted CSAT",
        hovertemplate="<b>%{x}</b><br>Adjusted CSAT: %{y:.2f}<br><extra></extra>",
    ))
    fig2.add_hline(y=mu, line_dash="dash", line_color="#d97706",
                   annotation_text=f"Mean {mu:.2f}", annotation_position="top right")
    fig2.update_layout(
        title="Bias-Adjusted CSAT with 95% Confidence Intervals",
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        yaxis=dict(gridcolor="#f1f5f9", range=[2.7, 4.0]),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig2, width="stretch")

    st.markdown("#### Adjusted Bottom 10 (Controlled for Ticket Complexity & Routing Queue)")
    adj_disp = R.sort_values("shrunk")[["agent_name","agent_team","agent_shift","tickets","scored","raw","adj","lo","hi","shrunk","hw_share","breach_rate"]].copy()
    adj_disp["below_mean"] = adj_disp["hi"] < mu
    adj_disp.columns = ["Name","Team","Shift","Tickets","Scored","Raw","Adj","CI Low","CI High","Shrunk","HW Share","Breach %","Below Mean"]
    st.dataframe(
        adj_disp.head(10).style
            .background_gradient(subset=["Adj"], cmap="RdYlGn", vmin=3.0, vmax=3.8)
            .format({"Raw": "{:.2f}", "Adj": "{:.2f}", "CI Low": "{:.2f}", "CI High": "{:.2f}",
                     "Shrunk": "{:.2f}", "HW Share": "{:.0%}", "Breach %": "{:.0%}"}),
        width="stretch",
    )

    st.markdown("#### Top 5 for Diwali Bonus (Adjusted)")
    top5_adj = R.nlargest(5, "adj")[["agent_name","agent_team","agent_shift","tickets","scored","raw","adj","lo","hi"]].copy()
    top5_adj.columns = ["Name","Team","Shift","Tickets","Scored","Raw CSAT","Adjusted CSAT","CI Low","CI High"]
    st.dataframe(top5_adj.style.format({"Raw CSAT":"{:.2f}","Adjusted CSAT":"{:.2f}","CI Low":"{:.2f}","CI High":"{:.2f}"}), width="stretch")
    st.caption("Top 5 adjusted agents: Sukhwinder Rao, Steven Ghosh, Vihaan Menon, Aarav Pereira, and Thomas Reddy.")

# ─────────────────────────────────────────────
# TAB 3: DEFECT LOT ALERTS
# ─────────────────────────────────────────────
with tab3:
    st.markdown("### Defect Lot Early-Warning Engine")
    flagged = lot[lot.flagged]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Excess Cost", f"₹{flagged.excess_rs.sum():,.0f}",
                  help="Above baseline replacement costs (policy formula: unit cost + ₹340 logistics)")
    with col2:
        st.metric("Avoidable Cost via Alert", f"₹{flagged.avoidable_rs.sum():,.0f}",
                  help="Units sold after automated lot alert triggers, avoidable through warehouse holds")
    with col3:
        st.metric("Lot False-Alarm Rate", "0 / 105 lots (0.0%)",
                  help="Out of 108 lots with >= 30 tickets, only the 3 defective Pulse 2 batches fired")

    st.info(
        f"**3-Sigma Cumulative Control Rule:** Detects replacement surges when cumulative rate exceeds baseline ({base:.1%}) + 3σ after ≥ 30 tickets. "
        "Alerts fired **27 to 34 days** after first sale of each defective lot (19 Nov, 29 Dec, 25 Jan). "
        "The defect is strictly confined to 3 production months (Oct–Dec 2025). Lots 2601 onward return to a normal 2–9% rate."
    )

    lot_disp = flagged[["tickets","repl","units","repl_per_100_units","csat","first_sale","alert_at","days_to_alert","excess_rs","avoidable_rs"]].copy()
    lot_disp.index = [f"{sku} (Lot {ly})" for sku, ly in lot_disp.index]
    lot_disp.columns = ["Tickets","Replacements","Units Sold","Repl / 100 Units","CSAT","First Sale","Alert Date","Days to Alert","Excess ₹","Avoidable ₹"]
    st.dataframe(
        lot_disp.style.background_gradient(subset=["Repl / 100 Units"], cmap="Reds")
                      .format({"Repl / 100 Units":"{:.1f}","CSAT":"{:.2f}","Excess ₹":"{:,.0f}","Avoidable ₹":"{:,.0f}"}),
        width="stretch",
    )

    st.markdown("#### Weekly Replacement Rate vs. 3-Sigma Control Limit")
    sku_sel = st.selectbox("Select product SKU to inspect", sorted(t.product_sku.unique()),
                           index=list(sorted(t.product_sku.unique())).index("VA-EB-PL2") if "VA-EB-PL2" in t.product_sku.unique() else 0)
    w = t[t.product_sku == sku_sel].groupby("week").agg(n=("repl","size"), r=("repl","sum"))
    w = w[w.n >= 5].copy()
    w["rate"] = w.r / w.n
    w["ucl"] = base + 3 * np.sqrt(base * (1 - base) / w.n)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=w.index, y=w.rate, mode="lines+markers", name="Replacement Rate",
                              line=dict(color="#2563eb", width=2), marker=dict(size=5)))
    fig3.add_trace(go.Scatter(x=w.index, y=w.ucl, mode="lines", name="Control Limit (3σ)",
                              line=dict(color="#dc2626", width=1.5, dash="dash")))
    fig3.add_hline(y=base, line_dash="dot", line_color="#d97706",
                   annotation_text=f"Baseline: {base:.1%}", annotation_position="bottom right")
    alert_weeks = w[w.rate > w.ucl]
    if len(alert_weeks):
        fig3.add_trace(go.Scatter(x=alert_weeks.index, y=alert_weeks.rate,
                                  mode="markers", name="Breach Triggered",
                                  marker=dict(color="#dc2626", size=9, symbol="x")))
    fig3.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis_title="Week",
        yaxis_title="Replacement Rate",
        yaxis=dict(gridcolor="#f1f5f9", tickformat=".0%"),
    )
    st.plotly_chart(fig3, width="stretch")

    st.markdown("#### Monthly Replacement Cost Trend by Top SKUs")
    top_skus = t.groupby("product_sku").repl_cost.sum().nlargest(5).index.tolist()
    monthly = t[t.product_sku.isin(top_skus)].groupby(["month","product_sku"]).repl_cost.sum().reset_index()
    fig4 = px.line(monthly, x="month", y="repl_cost", color="product_sku",
                   labels={"repl_cost":"Cost (₹)","month":"Month","product_sku":"Product SKU"})
    fig4.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        yaxis=dict(gridcolor="#f1f5f9"),
    )
    st.plotly_chart(fig4, width="stretch")

# ─────────────────────────────────────────────
# TAB 4: DATA QUALITY & SLAS
# ─────────────────────────────────────────────
with tab4:
    st.markdown("### Data Quality Report & Pipeline Hygiene")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Audits & Fixes Applied")
        for k, v in dq.items():
            icon = "✅" if "0%" not in str(v) or "0" == str(v) else "⚠️"
            st.markdown(f"**{k}:** `{v}`")

    with col2:
        st.markdown("#### Known Structural Realities")
        st.markdown(r"""
        1. **Legacy timestamp fix:** `legacy_fd` records were stored UTC; all other timestamps are IST. Shifting `resolved_at` by +5.5h completely eliminated negative handle times (from 68% down to 0%).
        2. **Order-lot fallback join:** 35% of tickets lacked `order_id`. A `merge_asof` fallback join on customer and product achieved **94.9% accuracy** on tickets with ground truth.
        3. **Agent name collision:** Two agents share the name "Kavya Pandey" (A3006 in Chat, A3029 in Logistics). All pipeline joins strictly use unique `agent_id`.
        4. **False-alarm specificity:** Out of 108 SKU-and-lot groups with $\ge 30$ tickets, only the 3 defective Pulse 2 batches fired (0 false alarms on 105 healthy lots).
        5. **Volume scaling:** Rupee figures reflect observed sample volumes (~150 tickets/week vs ~650 stated), serving as a conservative sample floor.
        """)

    st.divider()
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### CSAT Survey Response Rate by Channel")
        resp = t.groupby("channel").csat_score.apply(lambda s: s.notna().mean()).reset_index()
        resp.columns = ["Channel","Response Rate"]
        fig5 = px.bar(resp, x="Channel", y="Response Rate", color="Response Rate",
                      color_continuous_scale="Blues")
        fig5.update_layout(
            template="plotly_white",
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            yaxis=dict(gridcolor="#f1f5f9", tickformat=".0%"),
            showlegend=False,
        )
        st.plotly_chart(fig5, width="stretch")

    with col_d:
        st.markdown("#### Monthly Ticket Volume & CSAT Trend")
        monthly_t = t.groupby("month").agg(tickets=("ticket_id","size"), csat=("csat_score","mean")).reset_index()
        fig6 = make_subplots(specs=[[{"secondary_y":True}]])
        fig6.add_trace(go.Bar(x=monthly_t.month, y=monthly_t.tickets, name="Ticket Volume",
                              marker_color="#93c5fd", opacity=0.8), secondary_y=False)
        fig6.add_trace(go.Scatter(x=monthly_t.month, y=monthly_t.csat, mode="lines+markers", name="Mean CSAT",
                                  line=dict(color="#2563eb", width=2)), secondary_y=True)
        fig6.update_layout(
            template="plotly_white",
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            yaxis=dict(gridcolor="#f1f5f9"),
            yaxis2=dict(range=[2.5, 4.2]),
        )
        st.plotly_chart(fig6, width="stretch")
