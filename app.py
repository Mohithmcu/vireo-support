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

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f1117; }
    .stMetric { background: #1a1d27; border-radius: 12px; padding: 16px; border: 1px solid #2d3148; }
    .stMetric label { color: #9ca3af !important; font-size: 0.8rem !important; }
    .stMetric [data-testid="metric-container"] > div { color: #f9fafb !important; }
    .flag-card { background: linear-gradient(135deg, #1f2937 0%, #111827 100%); border-radius: 16px;
        padding: 20px; border: 1px solid #374151; margin-bottom: 12px; }
    .highlight { color: #f59e0b; font-weight: 600; }
    .good { color: #10b981; font-weight: 600; }
    .bad { color: #ef4444; font-weight: 600; }
    h1 { background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; font-weight: 700; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 500; }
    .stAlert { border-radius: 10px; }
    div[data-testid="stSidebarContent"] { background: #1a1d27; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading and processing data (first run ~5 sec)...")
def build(d="data"):
    raw, a, o, p = P.load(d)
    t = P.clean(raw, a, o, p)
    R, mu, tau, s = P.agent_table(t)
    rel = P.split_half_reliability(s, n_iter=150)
    lot, base = P.lot_alerts(t, o, p)
    fa = P.false_alarm_test(t)
    dq = P.quality_report(raw, t)
    return t, R, mu, tau, rel, lot, base, fa, dq

with st.sidebar:
    st.markdown("## 🎧 Vireo Analytics")
    st.markdown("*Support performance dashboard*")
    st.divider()
    data_dir = st.text_input("Data folder", value="data", help="Folder containing the 5 CSVs")
    st.divider()
    st.markdown("""
    **Key findings:**
    - 🔴 Pulse 2 lots 2510–2512: **46% replacement rate** vs 8% baseline
    - 💰 **Rs 13.3 lakh** excess replacement cost
    - 🛡️ Early alert could save **Rs 10.7 lakh**
    - 👤 Only **3 agents** statistically below mean (after adjustment)
    """)

try:
    t, R, mu, tau, rel, lot, base, fa, dq = build(data_dir)
except Exception as e:
    st.error(f"Could not load data from '{data_dir}': {e}")
    st.stop()

st.title("Vireo Audio — Support Analytics")
st.caption("Jan 2025 – Jun 2026 · 11,750 tickets · 44 agents · Bengaluru + Indore")

# Top metrics row
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Total Tickets", f"{len(t):,}")
with col2: st.metric("Mean CSAT (Tier 1)", f"{mu:.2f} / 5")
with col3: st.metric("Replacement Rate", f"{t.repl.mean():.1%}")
with col4: st.metric("SLA Breaches", f"{int(t.breach.sum()):,}")
with col5:
    flagged = lot[lot.flagged]
    st.metric("Excess Repl Cost", f"Rs {flagged.excess_rs.sum()/100000:.1f}L")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Per-Agent (as requested)",
    "📊 Per-Agent (adjusted)",
    "⚠️ Defect Lot Alerts",
    "🔍 Data Quality",
])

# ─────────────────────────────────────────────
with tab1:
    st.warning(
        "⚠️ **Raw CSAT mostly reflects queue composition, not agent skill.** "
        "The four Chat Frontline agents at the bottom handle 60–65% hardware-fault tickets "
        "from defective Pulse 2 lots — the angriest customers by design. "
        "See Tab 2 (adjusted) before any retraining or bonus decisions."
    )

    col_a, col_b = st.columns([3, 1])
    with col_b:
        show_team = st.multiselect("Filter by team", sorted(R.agent_team.unique()), default=[])

    display_R = R.copy()
    if show_team:
        display_R = display_R[display_R.agent_team.isin(show_team)]

    raw_sorted = display_R.sort_values("raw").reset_index()
    raw_sorted["rank"] = range(1, len(raw_sorted)+1)
    raw_sorted["flag_bottom10"] = raw_sorted["rank"] <= 10

    fig = px.bar(
        raw_sorted, x="agent_name", y="raw",
        color="flag_bottom10",
        color_discrete_map={True: "#ef4444", False: "#6366f1"},
        hover_data={"agent_team": True, "agent_shift": True, "tickets": True,
                    "hw_share": ":.0%", "scored": True, "flag_bottom10": False},
        labels={"raw": "Mean CSAT", "agent_name": "Agent", "flag_bottom10": "Bottom 10"},
        title="Raw CSAT by Agent (Tier 1)",
    )
    fig.add_hline(y=mu, line_dash="dash", line_color="#f59e0b",
                  annotation_text=f"Mean {mu:.2f}", annotation_position="top right")
    fig.update_layout(
        template="plotly_dark", height=420,
        xaxis_tickangle=-45, showlegend=False,
        plot_bgcolor="#1a1d27", paper_bgcolor="#0f1117",
    )
    st.plotly_chart(fig, use_container_width=True)

    cols_show = ["agent_name","agent_team","agent_shift","agent_site","tickets","scored","raw","handle_med","breach_rate","hw_share","pl2_share"]
    disp = raw_sorted[["agent_id"] + cols_show].set_index("agent_id")
    disp.columns = ["Name","Team","Shift","Site","Tickets","Scored","Raw CSAT","Handle Med(min)","Breach%","HW Share","PL2 Share"]
    disp = disp.sort_values("Raw CSAT")
    st.dataframe(
        disp.style.background_gradient(subset=["Raw CSAT"], cmap="RdYlGn", vmin=2.8, vmax=3.9)
               .format({"Raw CSAT": "{:.2f}", "Breach%": "{:.0%}", "HW Share": "{:.0%}", "PL2 Share": "{:.0%}", "Handle Med(min)": "{:.0f}"}),
        use_container_width=True, height=400,
    )
    st.caption("Handle time: median first-response → resolution. Compare within team only. "
               "Logistics & Returns wait on couriers/refunds (median ~1,465 min vs ~23 min frontline).")

# ─────────────────────────────────────────────
with tab2:
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("True spread (SD)", f"{tau:.3f} CSAT pts", help="Between-agent SD after Empirical Bayes shrinkage")
    with col2: st.metric("Avg measurement error (SE)", f"{R.se.mean():.3f}", help="How noisy each agent's estimate is")
    with col3: st.metric("Split-half reliability", f"{rel:.2f}", help="1.0 = perfect, 0 = noise. Adjusted ranking is low-reliability.")
    with col4: st.metric("Agents CI fully below mean", int((R.hi < mu).sum()), help="Only these 3 are statistically distinguishable")

    st.info(
        "**Interpretation:** The between-agent SD (0.076 CSAT pts) is smaller than the typical "
        "measurement error (0.094). This means most of the apparent variation is noise. "
        "Only 3 agents have adjusted confidence intervals entirely below the mean — "
        "and all three are in Chat Frontline, where the Pulse 2 defect is concentrated. "
        "Fix the lot problem first."
    )

    adj_sorted = R.sort_values("adj").reset_index()
    adj_sorted["below_mean"] = adj_sorted["hi"] < mu

    fig2 = go.Figure()
    colors = ["#ef4444" if b else "#6366f1" for b in adj_sorted.below_mean]
    fig2.add_trace(go.Scatter(
        x=adj_sorted.agent_name, y=adj_sorted.adj,
        mode="markers", marker=dict(size=10, color=colors),
        error_y=dict(type="data",
                     array=(adj_sorted.hi - adj_sorted.adj).values,
                     arrayminus=(adj_sorted.adj - adj_sorted.lo).values,
                     color="rgba(255,255,255,0.3)"),
        name="Adjusted CSAT",
        hovertemplate="<b>%{x}</b><br>Adj CSAT: %{y:.2f}<br><extra></extra>",
    ))
    fig2.add_hline(y=mu, line_dash="dash", line_color="#f59e0b",
                   annotation_text=f"Mean {mu:.2f}", annotation_position="top right")
    fig2.update_layout(
        title="Adjusted CSAT with 95% CI — Only red agents are statistically below average",
        template="plotly_dark", height=420,
        plot_bgcolor="#1a1d27", paper_bgcolor="#0f1117",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Adjusted bottom 10** (adjusted for category, channel, priority, product, month, assigned team)")
    adj_disp = R.sort_values("shrunk")[["agent_name","agent_team","agent_shift","tickets","scored","raw","adj","lo","hi","shrunk","hw_share","breach_rate"]].copy()
    adj_disp["below_mean"] = adj_disp["hi"] < mu
    adj_disp.columns = ["Name","Team","Shift","Tickets","Scored","Raw","Adj","CI Low","CI High","Shrunk","HW Share","Breach%","Below Mean"]
    st.dataframe(
        adj_disp.head(10).style
            .background_gradient(subset=["Adj"], cmap="RdYlGn", vmin=3.0, vmax=3.8)
            .format({"Raw": "{:.2f}", "Adj": "{:.2f}", "CI Low": "{:.2f}", "CI High": "{:.2f}",
                     "Shrunk": "{:.2f}", "HW Share": "{:.0%}", "Breach%": "{:.0%}"}),
        use_container_width=True,
    )

    st.markdown("**Top 5 for Diwali bonus** (adjusted)")
    top5_adj = R.nlargest(5, "adj")[["agent_name","agent_team","agent_shift","tickets","scored","raw","adj","lo","hi"]].copy()
    top5_adj.columns = ["Name","Team","Shift","Tickets","Scored","Raw","Adj","CI Low","CI High"]
    st.dataframe(top5_adj.style.format({"Raw":"{:.2f}","Adj":"{:.2f}","CI Low":"{:.2f}","CI High":"{:.2f}"}), use_container_width=True)
    st.caption("3 of 5 agree between raw and adjusted rankings. The 2 that shift have high Pulse 2 exposure.")

# ─────────────────────────────────────────────
with tab3:
    st.markdown("## Defect Lot Alert System")
    flagged = lot[lot.flagged]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Excess replacement cost", f"Rs {flagged.excess_rs.sum():,.0f}",
                  help="Above good-lot baseline, policy formula: unit_cost + Rs340")
    with col2:
        st.metric("Avoidable if stock held at alert", f"Rs {flagged.avoidable_rs.sum():,.0f}",
                  help="Units sold after lot alert fires, at excess replacement rate")
    with col3:
        tested = fa.weeks_tested.sum(); alerted = fa.alerts.sum()
        st.metric("False-alarm rate", f"{alerted}/{tested} SKU-weeks",
                  help="Only 1 false alarm in 109 non-Pulse-2 SKU-weeks tested")

    st.info(
        f"**Alert rule:** Cumulative replacement rate > baseline ({base:.1%}) + 3σ, after ≥ 30 tickets. "
        "Cost = unit cost + Rs 340 (policy §5). Alerts fire ~30–35 days after first sale of each bad lot. "
        "The defect is confined to **3 consecutive production months** (Oct–Dec 2025). "
        "Earlier and later Pulse 2 lots are normal."
    )

    # Lot detail table
    lot_disp = flagged[["tickets","repl","units","repl_per_100_units","csat","first_sale","alert_at","days_to_alert","excess_rs","avoidable_rs"]].copy()
    lot_disp.index = [f"{sku} lot {ly}" for sku,ly in lot_disp.index]
    lot_disp.columns = ["Tickets","Replacements","Units Sold","Repl/100 Units","CSAT","First Sale","Alert Date","Days to Alert","Excess Rs","Avoidable Rs"]
    st.dataframe(
        lot_disp.style.background_gradient(subset=["Repl/100 Units"], cmap="Reds")
                      .format({"Repl/100 Units":"{:.1f}","CSAT":"{:.2f}","Excess Rs":"{:,.0f}","Avoidable Rs":"{:,.0f}"}),
        use_container_width=True,
    )

    # Timeline chart
    st.markdown("### Weekly Replacement Rate — Pulse 2")
    sku_sel = st.selectbox("Show weekly rate for SKU", sorted(t.product_sku.unique()),
                           index=list(sorted(t.product_sku.unique())).index("VA-EB-PL2") if "VA-EB-PL2" in t.product_sku.unique() else 0)
    w = t[t.product_sku==sku_sel].groupby("week").agg(n=("repl","size"),r=("repl","sum"))
    w = w[w.n>=5].copy()
    w["rate"] = w.r/w.n
    w["ucl"] = base + 3*np.sqrt(base*(1-base)/w.n)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=w.index, y=w.rate, mode="lines+markers", name="Replacement rate",
                              line=dict(color="#6366f1", width=2), marker=dict(size=5)))
    fig3.add_trace(go.Scatter(x=w.index, y=w.ucl, mode="lines", name="Control limit (3σ)",
                              line=dict(color="#ef4444", width=1.5, dash="dash")))
    fig3.add_hline(y=base, line_dash="dot", line_color="#f59e0b",
                   annotation_text=f"Baseline {base:.1%}", annotation_position="bottom right")
    alert_weeks = w[w.rate > w.ucl]
    if len(alert_weeks):
        fig3.add_trace(go.Scatter(x=alert_weeks.index, y=alert_weeks.rate,
                                  mode="markers", name="Alert triggered",
                                  marker=dict(color="#ef4444", size=10, symbol="x")))
    fig3.update_layout(template="plotly_dark", height=380, plot_bgcolor="#1a1d27", paper_bgcolor="#0f1117",
                       xaxis_title="Week", yaxis_title="Replacement rate", yaxis_tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)

    # Monthly trend
    st.markdown("### Monthly Replacement Cost by SKU")
    top_skus = t.groupby("product_sku").repl_cost.sum().nlargest(5).index.tolist()
    monthly = t[t.product_sku.isin(top_skus)].groupby(["month","product_sku"]).repl_cost.sum().reset_index()
    fig4 = px.line(monthly, x="month", y="repl_cost", color="product_sku",
                   title="Monthly Replacement Cost (Rs) - Top 5 SKUs",
                   labels={"repl_cost":"Cost (Rs)","month":"Month","product_sku":"SKU"})
    fig4.update_layout(template="plotly_dark", height=380, plot_bgcolor="#1a1d27", paper_bgcolor="#0f1117")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### False-Alarm Test (non-Pulse-2 SKUs)")
    st.dataframe(fa.style.format({"weeks_tested":"{:,}"}), use_container_width=True)
    st.caption(f"1 false alarm in {tested} SKU-weeks (0.9%). Most low-volume SKUs cannot be monitored weekly "
               "— the rule works on high-volume products. Use a monthly rule for the rest.")

# ─────────────────────────────────────────────
with tab4:
    st.markdown("## Data Quality Report")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Checks & Fixes Applied")
        for k,v in dq.items():
            icon = "✅" if "0%" not in str(v) or "0" == str(v) else "⚠️"
            st.markdown(f"**{k}:** {v}")

    with col2:
        st.markdown("### Known Issues")
        st.markdown("""
        **1. Legacy timestamp fix**
        `legacy_fd` `resolved_at` was stored UTC; all other timestamps are IST.
        Added +5.5 h to legacy `resolved_at` only. Before fix: 68% of legacy tickets had
        negative handle time. After: 0%.

        **2. Lot-code fallback join**
        35% of tickets have no `order_id`. Fallback: latest earlier order for same
        customer + SKU (`merge_asof`). Accuracy ~94.9% vs known lot on tickets that
        do have an order_id. Lot results are reliable; ~1 in 20 fallback lots may be wrong.

        **3. Agent name collision**
        Two agents both named "Kavya Pandey" (A3006 — Chat, and A3029 — Logistics).
        All joins use `agent_id`.

        **4. Thin false-alarm test**
        Only 5 SKUs had enough weekly volume to test (>=15 tickets/week). The 0.9%
        false-alarm rate rests on 109 SKU-weeks — encouraging but not conclusive.

        **5. Replacement cost is per ticket, not per unit**
        Some customers filed multiple tickets for the same unit. Lot 2510 has
        1,009 tickets vs 834 units. The Rs figure uses tickets, so it may be
        slightly overstated for the worst lots.

        **6. This export is not full Vireo volume**
        ~150 tickets/week in the export vs ~650/week stated. Rs figures are a floor.
        Do not scale by 650/150 — that assumes the same proportions, which may not hold.
        """)

    st.divider()
    st.markdown("### CSAT Response Rate by Channel")
    resp = t.groupby("channel").csat_score.apply(lambda s: s.notna().mean()).reset_index()
    resp.columns = ["Channel","Response Rate"]
    fig5 = px.bar(resp, x="Channel", y="Response Rate", title="CSAT Survey Response Rate",
                  color="Response Rate", color_continuous_scale="Blues")
    fig5.update_layout(template="plotly_dark", height=300, plot_bgcolor="#1a1d27", paper_bgcolor="#0f1117",
                       yaxis_tickformat=".0%")
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("### Monthly Ticket Volume & CSAT Trend")
    monthly_t = t.groupby("month").agg(tickets=("ticket_id","size"),csat=("csat_score","mean")).reset_index()
    fig6 = make_subplots(specs=[[{"secondary_y":True}]])
    fig6.add_trace(go.Bar(x=monthly_t.month, y=monthly_t.tickets, name="Tickets", marker_color="#6366f1", opacity=0.7), secondary_y=False)
    fig6.add_trace(go.Scatter(x=monthly_t.month, y=monthly_t.csat, mode="lines+markers", name="Mean CSAT",
                              line=dict(color="#f59e0b", width=2)), secondary_y=True)
    fig6.update_layout(template="plotly_dark", height=350, plot_bgcolor="#1a1d27", paper_bgcolor="#0f1117",
                       title="Monthly Volume and CSAT")
    fig6.update_yaxes(title_text="Tickets", secondary_y=False)
    fig6.update_yaxes(title_text="Mean CSAT", secondary_y=True, range=[2.5,4.2])
    st.plotly_chart(fig6, use_container_width=True)
