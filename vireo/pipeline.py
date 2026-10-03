"""
vireo/pipeline.py - Core data pipeline for Vireo Audio support analytics.
"""
import glob, os, numpy as np, pandas as pd
import statsmodels.formula.api as smf

TARGET_MIN = {"chat": 15, "voice": 120, "social": 240, "email": 480}
BREACH_CREDIT_RS = 350
LOGISTICS_RS = 340
LEGACY_UTC_SHIFT_H = 5.5
HW_CATEGORIES = ["Charging & Battery","Audio Quality","Connectivity","Warranty & Repair","App & Firmware"]
BAD_LOTS = ["2510", "2511", "2512"]

def _find(directory, stem):
    hits = sorted(glob.glob(f"{directory}/**/*{stem}.csv", recursive=True))
    if not hits:
        raise FileNotFoundError(f"No *{stem}.csv found under '{directory}'")
    return hits[0]

def load(directory="data"):
    t = pd.read_csv(_find(directory,"tickets"), parse_dates=["created_at","first_response_at","resolved_at"])
    a = pd.read_csv(_find(directory,"agents"))
    o = pd.read_csv(_find(directory,"orders"), parse_dates=["order_date"])
    p = pd.read_csv(_find(directory,"products"))
    return t, a, o, p

def sig_charge(series):
    m = series.fillna("").str.lower()
    return m.str.contains(r"charg") & m.str.contains(r"left|right")

def clean(raw_t, a, o, p):
    t = raw_t.copy()
    leg = t.source_system.eq("legacy_fd")
    t.loc[leg,"resolved_at"] = t.loc[leg,"resolved_at"] + pd.Timedelta(hours=LEGACY_UTC_SHIFT_H)
    t["frt_min"] = (t.first_response_at - t.created_at).dt.total_seconds()/60
    t["handle_min"] = (t.resolved_at - t.first_response_at).dt.total_seconds()/60
    t["breach"] = t.apply(lambda r: r.frt_min > TARGET_MIN.get(r.channel, 9999), axis=1)
    t["week"] = t.created_at.dt.to_period("W").dt.start_time
    t["month"] = t.created_at.dt.to_period("M").astype(str)
    t["quarter"] = t.created_at.dt.to_period("Q").astype(str)
    t["repl"] = t.replacement_issued.eq("Y").astype(int)
    t["sig_charge"] = sig_charge(t.customer_message)
    prod = p.rename(columns={"sku":"product_sku"})[["product_sku","family","unit_cost_inr","product_name","warranty_months"]]
    t = t.merge(prod, on="product_sku", how="left")
    t["repl_cost"] = np.where(t.repl==1, t.unit_cost_inr+LOGISTICS_RS, 0)
    t["is_pl2"] = t.product_sku.eq("VA-EB-PL2")
    ag = a.rename(columns={"name":"agent_name","site":"agent_site","team":"agent_team","shift":"agent_shift"})
    t = t.merge(ag[["agent_id","agent_name","agent_site","agent_team","agent_shift","tier"]], on="agent_id", how="left")
    ob = o.rename(columns={"sku":"product_sku"})[["order_id","customer_id","product_sku","order_date","lot_code"]]
    t = t.merge(ob[["order_id","lot_code"]], on="order_id", how="left")
    fb_src = ob.sort_values("order_date")[["customer_id","product_sku","order_date","lot_code"]].rename(columns={"lot_code":"fb_lot"})
    t_s = t.sort_values("created_at").reset_index(drop=True)
    fb = pd.merge_asof(t_s[["customer_id","product_sku","created_at"]], fb_src, left_on="created_at", right_on="order_date", by=["customer_id","product_sku"])
    t_s["lot"] = t_s.lot_code.fillna(pd.Series(fb.fb_lot.values))
    t_s["lot_ym"] = t_s.lot.str.split("-").str[1]
    t_s["fb_lot"] = fb.fb_lot.values
    return t_s.sort_values("created_at").reset_index(drop=True)

def quality_report(raw_t, t):
    leg_raw = raw_t.source_system.eq("legacy_fd")
    h_raw = (raw_t.resolved_at - raw_t.first_response_at).dt.total_seconds()/60
    leg_clean = t.source_system.eq("legacy_fd")
    has_order = t.order_id.notna()
    lot_match = (t.loc[has_order,"fb_lot"]==t.loc[has_order,"lot_code"]).mean()
    return {
        "Total tickets": len(raw_t),
        "Date range": f"{raw_t.created_at.min().date()} to {raw_t.created_at.max().date()}",
        "legacy_fd tickets": f"{leg_raw.sum()} ({leg_raw.mean():.0%})",
        "Legacy negative handle BEFORE fix": f"{(h_raw[leg_raw]<0).mean():.0%}",
        "Legacy negative handle AFTER fix": f"{(t.handle_min[leg_clean]<0).mean():.0%}",
        "Tickets blank order_id": f"{raw_t.order_id.isna().mean():.0%}",
        "Lot coverage": f"{t.lot.notna().mean():.1%}",
        "Fallback lot accuracy": f"{lot_match:.1%}",
        "CSAT response rate": f"{t.csat_score.notna().mean():.0%}",
        "Repl+refund both set": int(((t.repl==1)&(t.refund_amount_inr.fillna(0)>0)).sum()),
        "Agents sharing display name": "Kavya Pandey (A3006 vs A3029) - join on agent_id only",
    }

def agent_table(t):
    t1 = t[t.tier==1].copy()
    s = t1.dropna(subset=["csat_score"]).copy()
    for col in ["category","channel","priority","product_sku","assigned_team","month"]:
        s[col] = s[col].astype(str)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress SingularMatrixWarning (rank-deficient month*team dummies)
        model = smf.ols("csat_score ~ C(category)+C(channel)+C(priority)+C(product_sku)+C(month)+C(assigned_team)", data=s).fit()
    s["resid"] = model.resid
    mu = float(s.csat_score.mean())
    g = s.groupby("agent_id").agg(scored=("csat_score","size"),raw=("csat_score","mean"),res=("resid","mean"),sd=("resid","std"))
    g["adj"] = mu + g.res
    g["se"] = g.sd/np.sqrt(g.scored)
    g["lo"] = g.adj - 1.96*g.se
    g["hi"] = g.adj + 1.96*g.se
    tau2 = max(float(g.res.var())-float((g.se**2).mean()), 1e-6)
    tau = float(np.sqrt(tau2))
    g["shrunk"] = mu + g.res*tau2/(tau2+g.se**2)
    info = t1.groupby("agent_id").agg(
        agent_name=("agent_name","first"), agent_team=("agent_team","first"),
        agent_shift=("agent_shift","first"), agent_site=("agent_site","first"),
        tickets=("ticket_id","size"), handle_med=("handle_min","median"),
        breach_rate=("breach","mean"), xfer_rate=("transfers",lambda x:(x>0).mean()),
        hw_share=("category",lambda x: pd.Series(x).isin(HW_CATEGORIES).mean()),
        pl2_share=("is_pl2","mean"),
    )
    R = info.join(g).round(3)
    return R, mu, tau, s

def split_half_reliability(s, n_iter=300, seed=0):
    rng = np.random.default_rng(seed)
    corrs = []
    for _ in range(n_iter):
        h = rng.integers(0, 2, len(s))
        x = s.assign(h=h).groupby(["agent_id","h"]).resid.mean().unstack()
        if x.shape[1]==2:
            corrs.append(float(x.corr().iloc[0,1]))
    return float(np.nanmean(corrs)) if corrs else float("nan")

def lot_alerts(t, o, p, min_n=30):
    baseline = float(t.groupby("product_sku").repl.mean().median())
    ob = o.rename(columns={"sku":"product_sku"}).assign(lot_ym=lambda d: d.lot_code.str.split("-").str[1])
    x = t[t.lot_ym.notna()].sort_values("created_at").copy()
    k = ["product_sku","lot_ym"]
    x["cum_n"] = x.groupby(k).cumcount()+1
    x["cum_r"] = x.groupby(k).repl.cumsum()
    x["ucl"] = baseline + 3*np.sqrt(baseline*(1-baseline)/x.cum_n)
    x["hit"] = (x.cum_n>=min_n)&(x.cum_r/x.cum_n>x.ucl)
    alert_at = x[x.hit].groupby(k).created_at.min().rename("alert_at")
    lot = x.groupby(k).agg(tickets=("ticket_id","size"),repl=("repl","sum"),csat=("csat_score","mean"))
    units_df = ob.groupby(k).agg(units=("qty","sum"),first_sale=("order_date","min"),last_sale=("order_date","max"))
    lot = lot.join(units_df).join(alert_at)
    lot["repl_per_100_units"] = (100*lot.repl/lot.units.replace(0,np.nan)).round(1)
    lot["flagged"] = lot.alert_at.notna()
    lot["days_to_alert"] = (lot.alert_at-lot.first_sale).dt.days
    good = lot[~lot.flagged].groupby("product_sku").agg(r=("repl","sum"),u=("units","sum"))
    good_rate = (good.r/good.u).to_dict()
    cost_map = (p.set_index("sku").unit_cost_inr+LOGISTICS_RS).to_dict()
    lot["excess_rs"] = np.nan
    lot["avoidable_rs"] = np.nan
    for (sku,ly), row in lot[lot.flagged].iterrows():
        gr = good_rate.get(sku, baseline)
        c = cost_map.get(sku, 1820)
        excess_repl = max(row.repl - gr*row.units, 0)
        lot.loc[(sku,ly),"excess_rs"] = excess_repl*c
        if pd.notna(row.alert_at):
            ua = ob[(ob.product_sku==sku)&(ob.lot_ym==ly)&(ob.order_date>row.alert_at.normalize())].qty.sum()
            lot.loc[(sku,ly),"avoidable_rs"] = ua*max(row.repl/max(row.units,1)-gr,0)*c
    return lot.round(2), baseline

def false_alarm_test(t, cutoff="2025-03-01", min_weekly=15):
    rows = []
    subset = t[(~t.is_pl2)&(t.created_at>=cutoff)]
    for sku, d in subset.groupby("product_sku"):
        b = float(d.repl.mean())
        ww = d.groupby("week").agg(n=("repl","size"),r=("repl","sum"))
        ww = ww[ww.n>=min_weekly]
        if len(ww)==0:
            rows.append({"sku":sku,"weeks_tested":0,"alerts":0}); continue
        ww["alerted"] = ww.r/ww.n > b+3*np.sqrt(b*(1-b)/ww.n)
        rows.append({"sku":sku,"weeks_tested":len(ww),"alerts":int(ww.alerted.sum())})
    return pd.DataFrame(rows)

if __name__=="__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv)>1 else "data"
    print(f"Loading from '{data_dir}'...")
    raw_t, a, o, p = load(data_dir)
    t = clean(raw_t, a, o, p)
    print("\n=== Data Quality ===")
    for k,v in quality_report(raw_t,t).items(): print(f"  {k}: {v}")
    lot, base = lot_alerts(t, o, p)
    flagged = lot[lot.flagged]
    print(f"\n=== Lot Alerts (baseline={base:.3f}) ===")
    print(flagged[["tickets","repl","units","repl_per_100_units","first_sale","alert_at","days_to_alert","excess_rs","avoidable_rs"]].to_string())
    print(f"  Total excess Rs: {flagged.excess_rs.sum():,.0f}")
    print(f"  Avoidable Rs:    {flagged.avoidable_rs.sum():,.0f}")
    fa = false_alarm_test(t)
    tested, alerted = fa.weeks_tested.sum(), fa.alerts.sum()
    print(f"\n=== False-Alarm Test ({tested} SKU-weeks) ===")
    print(fa.to_string(index=False))
    print(f"  Rate: {alerted}/{tested} = {alerted/max(tested,1):.3f}")
    R, mu, tau, s = agent_table(t)
    print(f"\n=== Agent Table (mean CSAT={mu:.2f}) ===")
    cols = ["agent_name","agent_team","agent_shift","tickets","scored","raw","adj","hw_share"]
    print("Raw bottom 10:"); print(R.nsmallest(10,"raw")[cols].to_string())
    print("Adj bottom 10:"); print(R.nsmallest(10,"adj")[cols].to_string())
    print(f"  Between-agent SD: {tau:.3f}")
    print(f"  Agents CI below mean: {int((R.hi<mu).sum())}")
    sla = int(t.breach.sum())
    print(f"\n=== SLA: {sla} breaches | Rs {sla*BREACH_CREDIT_RS:,} credits ===")
    print(f"  By channel: {t.groupby('channel').breach.mean().round(3).to_dict()}")
