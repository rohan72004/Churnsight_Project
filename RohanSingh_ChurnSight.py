"""
ChurnSight - Customer Retention & Revenue Intelligence Dashboard
=================================================================
A single-file Python project (backend + ML + frontend) that turns raw
subscription-customer data into business decisions:

    Data -> Information -> Insight -> Decision -> Action

Pages
  1. Executive Overview        - KPIs and trends
  2. Revenue & Plan Analysis   - where the money comes from
  3. Customer Risk & Drivers   - who is likely to churn, and why (ML model)
  4. Risk | Opportunity | Action - recommended decisions + retention what-if

Run:
    pip install -r requirements.txt
    streamlit run RohanSingh_ChurnSight.py

Data:
    Reads `telecom_customers.csv` next to this file. If it does not exist, a
    reproducible synthetic dataset (seed = 42) is generated and saved. You can
    also upload your own CSV with the same columns from the sidebar.
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telecom_customers.csv")

REGIONS = ["North", "South", "East", "West", "Central"]
PLANS = ["Basic", "Standard", "Premium"]
SERVICES = ["Fiber", "5G", "Broadband"]
CONTRACTS = ["Month-to-Month", "1 Year", "2 Year"]
PAYMENTS = ["UPI", "Credit Card", "Net Banking", "Auto-debit"]

REQUIRED_COLUMNS = [
    "customer_id", "signup_date", "churn_date", "region", "age", "plan", "service",
    "contract", "payment_method", "monthly_charges", "support_tickets",
    "avg_satisfaction", "data_usage_gb", "add_ons", "late_payments", "churned",
]
NUM_FEATURES = ["age", "monthly_charges", "support_tickets", "avg_satisfaction",
                "data_usage_gb", "add_ons", "late_payments"]
CAT_FEATURES = ["region", "plan", "service", "contract", "payment_method"]

PALETTE = ["#1f6feb", "#f0883e", "#2ea043", "#a371f7", "#db61a2", "#8b949e"]
LABELS = {"churned": "Churn rate", "mrr": "MRR (₹)", "monthly_charges": "Monthly charges (₹)",
          "ticket_band": "Support tickets raised", "satisfaction_band": "Satisfaction score",
          "contract": "Contract type", "region": "Region", "month": "Month", "importance": "Importance",
          "feature": "", "customers": "Customers", "service": "Service"}
PLAN_COLORS = {"Basic": "#1f6feb", "Standard": "#f0883e", "Premium": "#2ea043"}
RISK_COLORS = {"High": "#d1242f", "Medium": "#f0883e", "Low": "#2ea043"}


# --------------------------------------------------------------------------
# 2. DATA GENERATION (used only when no CSV is available)
# --------------------------------------------------------------------------
def generate_data(n=5000, seed=42):
    """Create a realistic synthetic telecom/broadband subscription dataset."""
    rng = np.random.default_rng(seed)
    obs_end = pd.Timestamp("2025-12-31")

    region = rng.choice(REGIONS, n, p=[0.22, 0.24, 0.20, 0.22, 0.12])
    plan = rng.choice(PLANS, n, p=[0.40, 0.38, 0.22])
    service = rng.choice(SERVICES, n, p=[0.40, 0.35, 0.25])
    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20])
    payment = rng.choice(PAYMENTS, n, p=[0.40, 0.20, 0.15, 0.25])

    age = np.clip(rng.normal(38, 12, n), 18, 75).round().astype(int)
    add_ons = rng.integers(0, 5, n)
    base = pd.Series(plan).map({"Basic": 399, "Standard": 799, "Premium": 1399}).to_numpy()
    svc_uplift = pd.Series(service).map({"Fiber": 1.15, "5G": 1.25, "Broadband": 1.0}).to_numpy()
    monthly = (base * svc_uplift + add_ons * 99 + rng.normal(0, 40, n)).round().astype(int)

    tickets = rng.poisson(1.4 + 0.8 * (service == "Broadband") + 0.6 * (region == "West"))
    satisfaction = np.clip(
        rng.normal(3.8 - 0.25 * tickets + 0.2 * (plan == "Premium"), 0.7), 1, 5
    ).round(1)
    late = rng.poisson(0.5 + 0.6 * (payment == "Net Banking"))
    usage = (rng.lognormal(4.3, 0.5, n) * pd.Series(plan).map(
        {"Basic": 0.6, "Standard": 1.0, "Premium": 1.6}).to_numpy()).round(1)

    # Churn probability: logistic model of the drivers (tenure deliberately excluded)
    z = (
        1.1 * (contract == "Month-to-Month") - 0.6 * (contract == "1 Year")
        - 1.0 * (contract == "2 Year")
        + 0.30 * tickets - 0.6 * (satisfaction - 3.5) + 0.25 * late
        + 0.8 * (region == "West") + 0.3 * (monthly / 1000 - 0.8)
        - 0.12 * add_ons - 0.3 * (service == "Fiber") + 0.25 * (service == "Broadband")
    )
    lo, hi = -10.0, 10.0
    for _ in range(40):  # calibrate the intercept so overall churn is ~24%
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > 0.24:
            hi = mid
        else:
            lo = mid
    p = 1 / (1 + np.exp(-(z + (lo + hi) / 2)))
    churned = (rng.random(n) < p).astype(int)

    # Sign-up dates skew toward recent months (a growing business)
    start = pd.Timestamp("2022-01-01")
    span = (pd.Timestamp("2025-12-20") - start).days
    signup = start + pd.to_timedelta((span * rng.beta(1.4, 1.0, n)).astype(int), unit="D")
    avail = (obs_end - signup).days.to_numpy()
    churn_offset = np.clip(rng.uniform(0.15, 1.0, n) * avail, 1, avail).astype(int)
    churn_date = pd.Series(signup + pd.to_timedelta(churn_offset, unit="D"))
    churn_date = churn_date.where(churned == 1, pd.NaT)

    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "signup_date": signup, "churn_date": churn_date, "region": region, "age": age,
        "plan": plan, "service": service, "contract": contract, "payment_method": payment,
        "monthly_charges": monthly, "support_tickets": tickets, "avg_satisfaction": satisfaction,
        "data_usage_gb": usage, "add_ons": add_ons, "late_payments": late, "churned": churned,
    })

    # Make the raw file realistically "dirty": missing values + duplicate rows
    df.loc[rng.choice(n, int(n * 0.015), replace=False), "avg_satisfaction"] = np.nan
    df.loc[rng.choice(n, int(n * 0.010), replace=False), "data_usage_gb"] = np.nan
    df = pd.concat([df, df.sample(25, random_state=seed)], ignore_index=True)
    return df


def load_raw(uploaded):
    """Load the CSV: uploaded file > local file > freshly generated data."""
    if uploaded is not None:
        return pd.read_csv(uploaded)
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    raw = generate_data()
    raw.to_csv(DATA_FILE, index=False)
    return raw


# --------------------------------------------------------------------------
# 3. DATA CLEANING & FEATURE ENGINEERING
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def clean_data(raw):
    """Remove duplicates, fix types, impute missing values, engineer features."""
    df = raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    df = df.drop_duplicates(subset="customer_id").reset_index(drop=True)
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["churn_date"] = pd.to_datetime(df["churn_date"], errors="coerce")
    df = df.dropna(subset=["signup_date"])

    for col in NUM_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    for col in CAT_FEATURES:
        df[col] = df[col].fillna(df[col].mode()[0]).astype(str).str.strip()
    df["avg_satisfaction"] = df["avg_satisfaction"].clip(1, 5)
    df["churned"] = df["churned"].fillna(0).astype(int)

    obs_end = max(df["signup_date"].max(), df["churn_date"].max()).to_period("M").end_time.normalize()
    end = df["churn_date"].fillna(obs_end)
    df["tenure_months"] = ((end - df["signup_date"]).dt.days / 30.44).round(1).clip(lower=0)
    df["lifetime_revenue"] = df["monthly_charges"] * df["tenure_months"]
    df["ticket_band"] = pd.cut(df["support_tickets"], [-1, 0, 1, 2, 3, 100],
                               labels=["0", "1", "2", "3", "4+"]).astype(str)
    df["satisfaction_band"] = pd.cut(df["avg_satisfaction"], [0, 2, 3, 4, 5],
                                     labels=["1-2", "2-3", "3-4", "4-5"]).astype(str)
    return df, obs_end


def monthly_trend(df, obs_end):
    """Active customers, MRR, new sign-ups and churned customers for each month."""
    rows = []
    for m in pd.period_range(df["signup_date"].min(), obs_end, freq="M"):
        end = m.to_timestamp(how="end")
        active = df[(df["signup_date"] <= end) & (df["churn_date"].isna() | (df["churn_date"] > end))]
        rows.append({
            "month": m.to_timestamp(),
            "active": len(active),
            "mrr": active["monthly_charges"].sum(),
            "new": int((df["signup_date"].dt.to_period("M") == m).sum()),
            "churned": int((df["churn_date"].dt.to_period("M") == m).sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. MACHINE LEARNING - CHURN PREDICTION
# --------------------------------------------------------------------------
def encode(df, columns=None):
    x = pd.get_dummies(df[NUM_FEATURES + CAT_FEATURES], columns=CAT_FEATURES).astype(float)
    return x if columns is None else x.reindex(columns=columns, fill_value=0.0)


@st.cache_data(show_spinner="Training churn model...")
def train_model(df):
    """Random Forest churn classifier. Returns model, metrics and driver importances."""
    x = encode(df)
    y = df["churned"]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=15,
                                   class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    model.fit(x_tr, y_tr)
    pred, proba = model.predict(x_te), model.predict_proba(x_te)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_te, pred),
        "ROC-AUC": roc_auc_score(y_te, proba),
        "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
    }
    imp = pd.Series(model.feature_importances_, index=x.columns)
    grouped = {}
    for col, v in imp.items():
        base = next((c for c in CAT_FEATURES if col.startswith(c + "_")), col)
        grouped[base] = grouped.get(base, 0) + v
    importance = pd.Series(grouped).sort_values(ascending=False)
    return model, list(x.columns), metrics, importance


def risk_band(p):
    return "High" if p >= 0.6 else ("Medium" if p >= 0.35 else "Low")


# --------------------------------------------------------------------------
# 5. BUSINESS LOGIC - KPIs AND INSIGHTS
# --------------------------------------------------------------------------
def inr(x):
    """Format rupees in Indian style (Lakh / Crore)."""
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.2f} L"
    return f"₹{x:,.0f}"


def high_value_at_risk(df, threshold=0.6):
    active = df[df["churned"] == 0]
    return active[(active["risk_score"] >= threshold)
                  & (active["monthly_charges"] >= active["monthly_charges"].median())]


def build_insights(df):
    """Generate Risk / Opportunity / Action statements straight from the data."""
    overall = df["churned"].mean()
    by_region = df.groupby("region")["churned"].mean().sort_values(ascending=False)
    worst, worst_rate = by_region.index[0], by_region.iloc[0]
    by_contract = df.groupby("contract")["churned"].mean()
    active = df[df["churned"] == 0]

    seg = df.groupby(["service", "plan"]).agg(n=("churned", "size"), churn=("churned", "mean"),
                                              arpu=("monthly_charges", "mean"))
    seg = seg[seg["n"] >= 80].assign(score=lambda s: s["arpu"] * (1 - s["churn"]))
    best = seg["score"].idxmax()

    m2m_mrr = active.loc[active["contract"] == "Month-to-Month", "monthly_charges"].sum()
    gap = by_contract.get("Month-to-Month", 0) - by_contract.get("1 Year", 0)
    upside = m2m_mrr * 12 * max(gap, 0) * 0.20

    hv = high_value_at_risk(df)
    hv_worst = hv[hv["region"] == worst]
    at_risk_rev = (active["monthly_charges"] * 12 * active["risk_score"]).sum()

    return {
        "risk": [
            f"**{worst}** region churn is **{worst_rate:.1%}** vs {overall:.1%} overall "
            f"({worst_rate / overall:.1f}x the average).",
            f"Month-to-month customers churn at **{by_contract.get('Month-to-Month', 0):.1%}**, "
            f"versus {by_contract.get('2 Year', 0):.1%} on 2-year contracts.",
            f"Expected annual revenue at risk from active customers: **{inr(at_risk_rev)}**.",
        ],
        "opportunity": [
            f"**{best[1]} + {best[0]}** is the strongest segment: ARPU {inr(seg.loc[best, 'arpu'])}, "
            f"churn only {seg.loc[best, 'churn']:.1%}. Grow it with targeted upsell.",
            f"Moving just 20% of month-to-month customers to annual plans could protect "
            f"about **{inr(upside)}** a year.",
        ],
        "action": [
            f"Launch a retention offer for **{len(hv):,}** high-value, high-risk customers "
            f"({len(hv_worst):,} of them in {worst}).",
            f"Investigate service quality in **{worst}**: "
            f"{df.loc[df['region'] == worst, 'support_tickets'].mean():.1f} support tickets per customer "
            f"vs {df['support_tickets'].mean():.1f} overall.",
            "Offer 1-year contract discounts to low-risk month-to-month customers.",
        ],
        "worst_region": worst,
    }


# --------------------------------------------------------------------------
# 6. DASHBOARD PAGES
# --------------------------------------------------------------------------
def style_fig(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.2), title_font_size=15,
                      plot_bgcolor="rgba(0,0,0,0)")
    fig.for_each_xaxis(lambda a: a.update(title_text=LABELS.get(a.title.text, a.title.text)))
    fig.for_each_yaxis(lambda a: a.update(title_text=LABELS.get(a.title.text, a.title.text)))
    return fig


def page_overview(df, trend):
    st.header("Executive Overview")
    st.caption("What is happening in the business right now?")
    active = df[df["churned"] == 0]
    mrr, prev_mrr = trend["mrr"].iloc[-1], trend["mrr"].iloc[-2]
    at_risk = (active["monthly_charges"] * 12 * active["risk_score"]).sum()

    c = st.columns(6)
    c[0].metric("Active Customers", f"{len(active):,}",
                f"{int(trend['active'].iloc[-1] - trend['active'].iloc[-2]):+,} vs last month")
    c[1].metric("MRR", inr(mrr), f"{(mrr / prev_mrr - 1):+.1%} MoM")
    c[2].metric("Churn Rate", f"{df['churned'].mean():.1%}")
    c[3].metric("ARPU", inr(active["monthly_charges"].mean()))
    c[4].metric("Avg Satisfaction", f"{active['avg_satisfaction'].mean():.2f} / 5")
    c[5].metric("Revenue at Risk (12m)", inr(at_risk))

    left, right = st.columns(2)
    fig = px.area(trend, x="month", y="mrr", title="Monthly Recurring Revenue trend",
                  color_discrete_sequence=[PALETTE[0]])
    left.plotly_chart(style_fig(fig), width="stretch")

    last = trend.tail(24).melt(id_vars="month", value_vars=["new", "churned"],
                               var_name="type", value_name="customers")
    fig = px.bar(last, x="month", y="customers", color="type", barmode="group",
                 title="New sign-ups vs churned customers (last 24 months)",
                 color_discrete_map={"new": PALETTE[2], "churned": RISK_COLORS["High"]})
    right.plotly_chart(style_fig(fig), width="stretch")

    ins = build_insights(df)
    st.subheader("Headlines for leadership")
    a, b, c3 = st.columns(3)
    a.error("**Top risk**\n\n" + ins["risk"][0])
    b.success("**Top opportunity**\n\n" + ins["opportunity"][0])
    c3.info("**Recommended action**\n\n" + ins["action"][0])


def page_revenue(df):
    st.header("Revenue & Plan Analysis")
    st.caption("Which plans, services and regions drive revenue?")
    active = df[df["churned"] == 0]

    left, right = st.columns(2)
    by_plan = active.groupby("plan")["monthly_charges"].sum().reindex(PLANS).reset_index()
    fig = px.pie(by_plan, names="plan", values="monthly_charges", hole=0.5,
                 title="MRR share by plan", color="plan", color_discrete_map=PLAN_COLORS)
    left.plotly_chart(style_fig(fig), width="stretch")

    by_region = active.groupby("region")["monthly_charges"].sum().sort_values().reset_index()
    fig = px.bar(by_region, x="monthly_charges", y="region", orientation="h",
                 title="MRR by region", color_discrete_sequence=[PALETTE[0]])
    right.plotly_chart(style_fig(fig), width="stretch")

    left, right = st.columns(2)
    heat = active.pivot_table(index="plan", columns="service", values="monthly_charges",
                              aggfunc="sum").reindex(PLANS)
    fig = px.imshow(heat, text_auto=".2s", color_continuous_scale="Blues",
                    title="MRR heatmap: plan x service")
    left.plotly_chart(style_fig(fig), width="stretch")

    arpu = active.groupby(["service", "plan"])["monthly_charges"].mean().reset_index()
    fig = px.bar(arpu, x="service", y="monthly_charges", color="plan", barmode="group",
                 title="ARPU by service and plan", category_orders={"plan": PLANS},
                 color_discrete_map=PLAN_COLORS)
    right.plotly_chart(style_fig(fig), width="stretch")

    st.subheader("Plan scorecard")
    card = df.groupby("plan").agg(
        customers=("customer_id", "count"), churn_rate=("churned", "mean"),
        avg_monthly_charge=("monthly_charges", "mean"), avg_tenure_months=("tenure_months", "mean"),
        lifetime_revenue=("lifetime_revenue", "sum")).reindex(PLANS)
    st.dataframe(card.style.format({"churn_rate": "{:.1%}", "avg_monthly_charge": "₹{:,.0f}",
                                    "avg_tenure_months": "{:.1f}", "lifetime_revenue": "₹{:,.0f}"}),
                 width="stretch")


def page_risk(df, importance, metrics):
    st.header("Customer Risk & Drivers")
    st.caption("Why do customers leave, and who is most likely to leave next?")
    overall = df["churned"].mean()

    left, right = st.columns(2)
    reg = df.groupby("region")["churned"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(reg, x="region", y="churned", title="Churn rate by region",
                 color_discrete_sequence=[PALETTE[1]])
    fig.add_hline(y=overall, line_dash="dash", annotation_text=f"overall {overall:.1%}")
    fig.update_yaxes(tickformat=".0%")
    left.plotly_chart(style_fig(fig), width="stretch")

    con = df.groupby("contract")["churned"].mean().reindex(CONTRACTS).reset_index()
    fig = px.bar(con, x="contract", y="churned", title="Churn rate by contract type",
                 color_discrete_sequence=[PALETTE[3]])
    fig.update_yaxes(tickformat=".0%")
    right.plotly_chart(style_fig(fig), width="stretch")

    left, right = st.columns(2)
    tk = df.groupby("ticket_band")["churned"].mean().reindex(["0", "1", "2", "3", "4+"]).reset_index()
    fig = px.line(tk, x="ticket_band", y="churned", markers=True,
                  title="Churn rate vs support tickets raised",
                  color_discrete_sequence=[RISK_COLORS["High"]])
    fig.update_xaxes(type="category")
    fig.update_yaxes(tickformat=".0%")
    left.plotly_chart(style_fig(fig), width="stretch")

    sat = df.groupby("satisfaction_band")["churned"].mean().reindex(["1-2", "2-3", "3-4", "4-5"]).reset_index()
    fig = px.bar(sat, x="satisfaction_band", y="churned", title="Churn rate by satisfaction score",
                 color_discrete_sequence=[PALETTE[4]])
    fig.update_yaxes(tickformat=".0%")
    right.plotly_chart(style_fig(fig), width="stretch")

    st.subheader("Churn drivers learned by the ML model")
    m = st.columns(4)
    for col, (k, v) in zip(m, metrics.items()):
        col.metric(k, f"{v:.2f}")
    imp = importance.head(8).sort_values().reset_index()
    imp.columns = ["feature", "importance"]
    fig = px.bar(imp, x="importance", y="feature", orientation="h",
                 title="Feature importance (Random Forest)", color_discrete_sequence=[PALETTE[0]])
    st.plotly_chart(style_fig(fig, 320), width="stretch")

    st.subheader("Priority list: high-value customers at high churn risk")
    hv = high_value_at_risk(df).sort_values("risk_score", ascending=False)
    cols = ["customer_id", "region", "plan", "service", "contract", "monthly_charges",
            "support_tickets", "avg_satisfaction", "risk_score"]
    st.dataframe(hv[cols].head(15).style.format({"risk_score": "{:.0%}", "monthly_charges": "₹{:,.0f}", "avg_satisfaction": "{:.1f}"}),
                 width="stretch", hide_index=True)
    st.download_button("Download full at-risk list (CSV)", hv[cols].to_csv(index=False),
                       "high_value_at_risk_customers.csv", "text/csv")


def page_actions(df):
    st.header("Risk  |  Opportunity  |  Action")
    st.caption("From fact to insight to decision.")
    ins = build_insights(df)

    a, b, c = st.columns(3)
    with a:
        st.subheader("Risks")
        for t in ins["risk"]:
            st.error(t)
    with b:
        st.subheader("Opportunities")
        for t in ins["opportunity"]:
            st.success(t)
    with c:
        st.subheader("Actions")
        for t in ins["action"]:
            st.info(t)

    st.divider()
    st.subheader("Retention campaign simulator")
    st.caption("Estimate the payoff of a targeted retention offer before spending any money.")
    s1, s2, s3 = st.columns(3)
    thr = s1.slider("Target customers with risk score above", 0.30, 0.90, 0.60, 0.05)
    success = s2.slider("Offer success rate (customers retained)", 0.10, 0.60, 0.30, 0.05)
    disc = s3.slider("Discount given (% of monthly bill, for 6 months)", 0.05, 0.40, 0.15, 0.05)

    tgt = high_value_at_risk(df, thr)
    saved = (tgt["monthly_charges"] * 12 * tgt["risk_score"] * success).sum()
    cost = (tgt["monthly_charges"] * disc * 6).sum()
    net = saved - cost
    k = st.columns(4)
    k[0].metric("Customers targeted", f"{len(tgt):,}")
    k[1].metric("Revenue protected (12m)", inr(saved))
    k[2].metric("Campaign cost", inr(cost))
    k[3].metric("Net benefit", inr(net), f"ROI {net / cost:.1f}x" if cost else None)

    fig = go.Figure(go.Bar(x=["Revenue protected", "Campaign cost", "Net benefit"], y=[saved, cost, net],
                           marker_color=[PALETTE[2], RISK_COLORS["High"], PALETTE[0]]))
    fig.update_layout(title="Campaign economics")
    st.plotly_chart(style_fig(fig, 300), width="stretch")


# --------------------------------------------------------------------------
# 7. APP ENTRY POINT
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="ChurnSight", page_icon="📡", layout="wide")
    st.title("📡 ChurnSight: Customer Retention & Revenue Intelligence")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Executive Overview", "Revenue & Plan Analysis",
                                      "Customer Risk & Drivers", "Risk | Opportunity | Action"])
    uploaded = st.sidebar.file_uploader("Optional: upload your own CSV", type="csv")

    try:
        df, obs_end = clean_data(load_raw(uploaded))
    except ValueError as err:
        st.error(str(err))
        st.stop()

    model, columns, metrics, importance = train_model(df)
    df = df.assign(risk_score=model.predict_proba(encode(df, columns))[:, 1])
    df["risk_band"] = df["risk_score"].apply(risk_band)

    st.sidebar.header("Filters")
    regions = st.sidebar.multiselect("Region", REGIONS, default=REGIONS)
    plans = st.sidebar.multiselect("Plan", PLANS, default=PLANS)
    view = df[df["region"].isin(regions) & df["plan"].isin(plans)]
    if view.empty or (view["churned"] == 0).sum() == 0:
        st.warning("No customers match the selected filters.")
        st.stop()
    st.sidebar.caption(f"{len(view):,} customers | data through {obs_end:%b %Y}")

    trend = monthly_trend(view, obs_end)
    if page == "Executive Overview":
        page_overview(view, trend)
    elif page == "Revenue & Plan Analysis":
        page_revenue(view)
    elif page == "Customer Risk & Drivers":
        page_risk(view, importance, metrics)
    else:
        page_actions(view)


main()
