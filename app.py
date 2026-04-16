"""HCZ Streamlit dashboard for Enrollment and Recruitment performance.

Single-file app with:
- Google Sheets loading placeholders
- Mock data fallback
- Objective-aware KPIs and diagnostics
- Defensive data prep and metric calculations
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(page_title="HCZ Performance Dashboard", layout="wide")

SHEET_CAMPAIGN = "Campaign Master Feed"
SHEET_GA4 = "GA4 Master Feed"
SHEET_LP = "LP Master Feed (Weekly)"

OBJECTIVE_OPTIONS = ["All", "Enrollment", "Recruitment"]
AGG_OPTIONS = ["Weekly", "Monthly"]

NUMERIC_CAMPAIGN = [
    "cost",
    "impressions",
    "clicks",
    "career_clicks",
    "applications",
    "enrollment_forms",
    "enrollment_apply_clicks",
]
NUMERIC_GA4 = [
    "paid_traffic",
    "non_paid_traffic",
    "applications_submitted",
    "career_clicks",
    "enrollment_form_submits",
]
NUMERIC_LP = [
    "sessions",
    "total_users",
    "engaged_sessions",
    "views",
    "career_clicks",
    "enrollment_form_submits",
]


# -----------------------------
# Utility helpers
# -----------------------------
def to_snake_case(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("/", " ")
        .replace("-", " ")
        .replace("(", "")
        .replace(")", "")
        .replace("%", " pct")
        .replace("  ", " ")
        .replace(" ", "_")
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake_case(c) for c in df.columns]
    return df


def safe_div(numer: pd.Series | float, denom: pd.Series | float) -> pd.Series | float:
    if isinstance(numer, pd.Series) or isinstance(denom, pd.Series):
        numer_s = numer if isinstance(numer, pd.Series) else pd.Series(numer, index=denom.index)
        denom_s = denom if isinstance(denom, pd.Series) else pd.Series(denom, index=numer.index)
        out = numer_s / denom_s.replace({0: np.nan})
        return out.replace([np.inf, -np.inf], np.nan)
    return np.nan if (denom in [0, None] or pd.isna(denom)) else numer / denom


def ensure_columns(df: pd.DataFrame, required: Iterable[str], fill_value=0) -> pd.DataFrame:
    df = df.copy()
    for c in required:
        if c not in df.columns:
            df[c] = fill_value
    return df


def add_time_columns(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    if date_col not in df.columns:
        return df
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["week_start"] = df[date_col].dt.to_period("W-SUN").dt.start_time
    df["month_start"] = df[date_col].dt.to_period("M").dt.start_time
    return df


def add_lp_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    elif "date" in df.columns:
        df["week_start"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["week_start"] = pd.NaT
    df["month_start"] = df["week_start"].dt.to_period("M").dt.start_time
    return df


def format_delta(curr: float, prev: float) -> str:
    if pd.isna(prev) or prev == 0:
        return "n/a"
    return f"{((curr - prev)/prev):+.1%}"


def apply_common_filters(
    campaign_df: pd.DataFrame,
    ga4_df: pd.DataFrame,
    lp_df: pd.DataFrame,
    filt: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    c = campaign_df.copy()
    g = ga4_df.copy()
    l = lp_df.copy()

    # Date filter (campaign/ga4 use date; LP uses week_start)
    start_d, end_d = filt["date_range"]
    start_ts, end_ts = pd.Timestamp(start_d), pd.Timestamp(end_d)

    if "date" in c.columns:
        c = c[c["date"].between(start_ts, end_ts)]
    if "date" in g.columns:
        g = g[g["date"].between(start_ts, end_ts)]
    if "week_start" in l.columns:
        l = l[l["week_start"].between(start_ts, end_ts)]

    if filt["objective"] != "All" and "objective" in c.columns:
        c = c[c["objective"] == filt["objective"]]

    if filt["platforms"] and "platform" in c.columns:
        c = c[c["platform"].isin(filt["platforms"])]

    if filt["campaigns"] and "campaign_name" in c.columns:
        c = c[c["campaign_name"].isin(filt["campaigns"])]

    if filt["ad_topics"] and "ad_topic" in c.columns:
        c = c[c["ad_topic"].isin(filt["ad_topics"])]

    if filt["meta_ads"] and "ad_name" in c.columns:
        c = c[c["ad_name"].isin(filt["meta_ads"])]

    if filt["landing_pages"] and "landing_page" in l.columns:
        l = l[l["landing_page"].isin(filt["landing_pages"])]

    if filt["devices"] and "device" in l.columns:
        l = l[l["device"].isin(filt["devices"])]

    return c, g, l


def aggregate_timeseries(
    df: pd.DataFrame, agg_level: str, metrics: List[str], dims: Optional[List[str]] = None
) -> pd.DataFrame:
    if df.empty:
        return df
    time_col = "week_start" if agg_level == "Weekly" else "month_start"
    if time_col not in df.columns:
        return pd.DataFrame()
    dims = dims or []
    use_cols = [time_col] + [d for d in dims if d in df.columns]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return pd.DataFrame()
    return (
        df.groupby(use_cols, dropna=False, as_index=False)[metrics]
        .sum(min_count=1)
        .sort_values(time_col)
    )


def add_campaign_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["ctr"] = safe_div(d.get("clicks", 0), d.get("impressions", 0))
    d["cpc"] = safe_div(d.get("cost", 0), d.get("clicks", 0))
    d["cost_per_enrollment_form"] = safe_div(d.get("cost", 0), d.get("enrollment_forms", 0))
    d["enrollment_form_rate"] = safe_div(d.get("enrollment_forms", 0), d.get("clicks", 0))
    d["enrollment_apply_click_rate"] = safe_div(d.get("enrollment_apply_clicks", 0), d.get("clicks", 0))
    d["cost_per_application"] = safe_div(d.get("cost", 0), d.get("applications", 0))
    d["application_rate"] = safe_div(d.get("applications", 0), d.get("clicks", 0))
    d["career_click_rate"] = safe_div(d.get("career_clicks", 0), d.get("clicks", 0))
    return d


def add_lp_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["engagement_rate"] = safe_div(d.get("engaged_sessions", 0), d.get("sessions", 0))
    d["views_per_session"] = safe_div(d.get("views", 0), d.get("sessions", 0))
    d["enrollment_form_submit_rate"] = safe_div(d.get("enrollment_form_submits", 0), d.get("sessions", 0))
    d["career_click_rate_lp"] = safe_div(d.get("career_clicks", 0), d.get("sessions", 0))
    d["enrollment_lp_next_step_rate"] = d["enrollment_form_submit_rate"]
    d["recruitment_lp_next_step_rate"] = d["career_click_rate_lp"]
    return d


def metric_sum(df: pd.DataFrame, col: str) -> float:
    return float(df[col].sum()) if col in df.columns and not df.empty else 0.0


def get_prior_period(df: pd.DataFrame, date_col: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df.iloc[0:0]
    days = (end_date - start_date).days + 1
    prior_end = start_date - pd.Timedelta(days=1)
    prior_start = prior_end - pd.Timedelta(days=days - 1)
    return df[df[date_col].between(prior_start, prior_end)]


# -----------------------------
# Data loading (placeholders + mock)
# -----------------------------
def load_from_google_sheets_placeholder() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace this with real Google Sheets loading logic.

    Example implementation options:
    - gspread + service account in st.secrets
    - googleapiclient Sheets API
    """
    raise NotImplementedError("Google Sheets loader not configured. Using mock data fallback.")


def make_mock_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=210, freq="D")

    platforms = ["Meta", "Google"]
    objectives = ["Enrollment", "Recruitment"]
    campaigns = {
        "Enrollment": ["Enroll | Brand", "Enroll | Program Interest"],
        "Recruitment": ["Recruit | Educators", "Recruit | Ops Roles"],
    }
    ad_topics = ["Community", "Impact", "Benefits", "Urgency"]
    ad_names = ["Meta Ad A", "Meta Ad B", "Meta Ad C", "Meta Ad D"]

    rows = []
    for d in dates:
        for obj in objectives:
            for p in platforms:
                for c in campaigns[obj]:
                    cost = max(10, rng.normal(120 if p == "Meta" else 90, 25))
                    impressions = max(100, int(rng.normal(15000 if p == "Meta" else 9000, 2500)))
                    ctr = 0.012 if p == "Meta" else 0.045
                    clicks = max(5, int(impressions * max(0.003, rng.normal(ctr, 0.004))))
                    career_clicks = int(clicks * max(0.02, rng.normal(0.10, 0.03))) if obj == "Recruitment" else int(clicks * 0.03)
                    applications = int(clicks * max(0.01, rng.normal(0.07, 0.02))) if obj == "Recruitment" else int(clicks * 0.01)
                    enrollment_forms = int(clicks * max(0.01, rng.normal(0.06, 0.02))) if obj == "Enrollment" else int(clicks * 0.01)
                    enrollment_apply_clicks = int(clicks * max(0.02, rng.normal(0.15, 0.04))) if obj == "Enrollment" else int(clicks * 0.02)
                    rows.append(
                        {
                            "date": d,
                            "platform": p,
                            "campaign_name": c,
                            "ad_name": rng.choice(ad_names) if p == "Meta" else "",
                            "cost": float(cost),
                            "impressions": impressions,
                            "clicks": clicks,
                            "career_clicks": career_clicks,
                            "applications": applications,
                            "enrollment_forms": enrollment_forms,
                            "enrollment_apply_clicks": enrollment_apply_clicks,
                            "ad_topic": rng.choice(ad_topics),
                            "objective": obj,
                            "month": d.month,
                            "year": d.year,
                        }
                    )
    campaign_df = pd.DataFrame(rows)

    ga4_rows = []
    for d in dates:
        paid = int(max(30, rng.normal(600, 110)))
        non_paid = int(max(20, rng.normal(850, 160)))
        apps_sub = int(max(0, rng.normal(28, 8)))
        ga4_rows.append(
            {
                "date": d,
                "paid_traffic": paid,
                "non_paid_traffic": non_paid,
                "applications_submitted": apps_sub,
                "career_clicks": int(max(0, rng.normal(42, 12))),
                "enrollment_form_submits": int(max(0, rng.normal(35, 10))),
                "month": d.month,
                "year": d.year,
            }
        )
    ga4_df = pd.DataFrame(ga4_rows)

    lp_weeks = pd.date_range(end=pd.Timestamp.today().normalize(), periods=40, freq="W-MON")
    landing_pages = ["/enroll", "/recruit", "/programs", "/careers"]
    sources = ["google", "meta", "direct"]
    mediums = ["cpc", "paid_social", "none"]
    devices = ["desktop", "mobile", "tablet"]

    lp_rows = []
    for w in lp_weeks:
        for lp in landing_pages:
            for dev in devices:
                sess = int(max(10, rng.normal(300, 80)))
                engaged = int(sess * max(0.2, rng.normal(0.58, 0.1)))
                views = int(sess * max(1.0, rng.normal(1.9, 0.3)))
                lp_rows.append(
                    {
                        "week_start": w,
                        "month": w.month,
                        "landing_page": lp,
                        "source": rng.choice(sources),
                        "medium": rng.choice(mediums),
                        "campaign": rng.choice(["Enroll | Brand", "Recruit | Educators", "Generic"]),
                        "content": rng.choice(["headline_a", "headline_b", "cta_a"]),
                        "term": rng.choice(["hcz", "jobs", "programs"]),
                        "device": dev,
                        "sessions": sess,
                        "total_users": int(sess * max(0.7, rng.normal(0.85, 0.05))),
                        "engaged_sessions": engaged,
                        "views": views,
                        "career_clicks": int(sess * max(0.01, rng.normal(0.09, 0.03))),
                        "enrollment_form_submits": int(sess * max(0.01, rng.normal(0.07, 0.02))),
                    }
                )
    lp_df = pd.DataFrame(lp_rows)

    dq_df = pd.DataFrame(
        [
            {"source": SHEET_CAMPAIGN, "last_refresh": pd.Timestamp.today().normalize(), "row_count": len(campaign_df), "note": "mock"},
            {"source": SHEET_GA4, "last_refresh": pd.Timestamp.today().normalize(), "row_count": len(ga4_df), "note": "mock"},
            {"source": SHEET_LP, "last_refresh": pd.Timestamp.today().normalize(), "row_count": len(lp_df), "note": "mock"},
        ]
    )
    return campaign_df, ga4_df, lp_df, dq_df


@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        campaign_df, ga4_df, lp_df, data_quality_df = load_from_google_sheets_placeholder()
    except Exception:
        campaign_df, ga4_df, lp_df, data_quality_df = make_mock_data()

    campaign_df = normalize_columns(campaign_df)
    ga4_df = normalize_columns(ga4_df)
    lp_df = normalize_columns(lp_df)
    data_quality_df = normalize_columns(data_quality_df)

    campaign_df = ensure_columns(campaign_df, NUMERIC_CAMPAIGN, 0)
    ga4_df = ensure_columns(ga4_df, NUMERIC_GA4, 0)
    lp_df = ensure_columns(lp_df, NUMERIC_LP, 0)

    for c in NUMERIC_CAMPAIGN:
        campaign_df[c] = pd.to_numeric(campaign_df[c], errors="coerce").fillna(0)
    for c in NUMERIC_GA4:
        ga4_df[c] = pd.to_numeric(ga4_df[c], errors="coerce").fillna(0)
    for c in NUMERIC_LP:
        lp_df[c] = pd.to_numeric(lp_df[c], errors="coerce").fillna(0)

    campaign_df = add_time_columns(campaign_df, "date")
    ga4_df = add_time_columns(ga4_df, "date")
    lp_df = add_lp_time_columns(lp_df)

    for cat_col in ["platform", "campaign_name", "ad_name", "ad_topic", "objective"]:
        if cat_col in campaign_df.columns:
            campaign_df[cat_col] = campaign_df[cat_col].fillna("Unknown")
    for cat_col in ["landing_page", "source", "medium", "campaign", "device"]:
        if cat_col in lp_df.columns:
            lp_df[cat_col] = lp_df[cat_col].fillna("Unknown")

    return campaign_df, ga4_df, lp_df, data_quality_df


campaign_df, ga4_df, lp_df, data_quality_df = load_data()


# -----------------------------
# Sidebar filters
# -----------------------------
min_date = min(
    campaign_df["date"].min() if "date" in campaign_df.columns and not campaign_df.empty else pd.Timestamp.today(),
    ga4_df["date"].min() if "date" in ga4_df.columns and not ga4_df.empty else pd.Timestamp.today(),
    lp_df["week_start"].min() if "week_start" in lp_df.columns and not lp_df.empty else pd.Timestamp.today(),
).date()

max_date = max(
    campaign_df["date"].max() if "date" in campaign_df.columns and not campaign_df.empty else pd.Timestamp.today(),
    ga4_df["date"].max() if "date" in ga4_df.columns and not ga4_df.empty else pd.Timestamp.today(),
    lp_df["week_start"].max() if "week_start" in lp_df.columns and not lp_df.empty else pd.Timestamp.today(),
).date()

default_start = max_date - pd.Timedelta(weeks=12)
default_start = max(default_start, min_date)

st.sidebar.title("HCZ Dashboard Filters")
date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    selected_start, selected_end = date_range
else:
    selected_start, selected_end = default_start, max_date

agg_level = st.sidebar.radio("Aggregation", AGG_OPTIONS, index=0)
objective = st.sidebar.selectbox("Objective", OBJECTIVE_OPTIONS, index=0)

platform_values = sorted(campaign_df["platform"].dropna().unique().tolist()) if "platform" in campaign_df.columns else []
selected_platforms = st.sidebar.multiselect("Platform", platform_values, default=platform_values)

camp_values = sorted(campaign_df["campaign_name"].dropna().unique().tolist()) if "campaign_name" in campaign_df.columns else []
selected_campaigns = st.sidebar.multiselect("Campaign name", camp_values)

topic_values = sorted(campaign_df["ad_topic"].dropna().unique().tolist()) if "ad_topic" in campaign_df.columns else []
selected_topics = st.sidebar.multiselect("Ad Topic", topic_values)

lp_values = sorted(lp_df["landing_page"].dropna().unique().tolist()) if "landing_page" in lp_df.columns else []
selected_lp = st.sidebar.multiselect("Landing page", lp_values)

device_values = sorted(lp_df["device"].dropna().unique().tolist()) if "device" in lp_df.columns else []
selected_devices = st.sidebar.multiselect("Device", device_values)

meta_df_all = campaign_df[campaign_df.get("platform", "") == "Meta"] if "platform" in campaign_df.columns else pd.DataFrame()
meta_ad_values = sorted(meta_df_all["ad_name"].dropna().unique().tolist()) if "ad_name" in meta_df_all.columns else []
selected_meta_ads = st.sidebar.multiselect("Meta ad name (optional)", meta_ad_values)

filt = {
    "date_range": (selected_start, selected_end),
    "objective": objective,
    "platforms": selected_platforms,
    "campaigns": selected_campaigns,
    "ad_topics": selected_topics,
    "landing_pages": selected_lp,
    "devices": selected_devices,
    "meta_ads": selected_meta_ads,
}

f_campaign, f_ga4, f_lp = apply_common_filters(campaign_df, ga4_df, lp_df, filt)
f_campaign = add_campaign_metrics(f_campaign)
f_lp = add_lp_metrics(f_lp)


# -----------------------------
# Layout navigation
# -----------------------------
st.title("HCZ Marketing Performance Dashboard")
st.caption("Weekly-first, objective-aware performance reporting for Enrollment and Recruitment.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Executive Summary",
        "Objective Performance",
        "Campaign Performance",
        "Meta Creative",
        "Landing Page Analysis",
        "Data Quality / Definitions",
    ]
)


# -----------------------------
# Tab 1 - Executive Summary
# -----------------------------
with tab1:
    st.subheader("Executive Summary")

    if f_campaign.empty and f_ga4.empty and f_lp.empty:
        st.info("No data available for current filter combination.")
    else:
        start_ts, end_ts = pd.Timestamp(selected_start), pd.Timestamp(selected_end)
        prior_campaign = get_prior_period(campaign_df, "date", start_ts, end_ts)
        prior_ga4 = get_prior_period(ga4_df, "date", start_ts, end_ts)
        prior_lp = get_prior_period(lp_df.rename(columns={"week_start": "date"}), "date", start_ts, end_ts)

        curr = {
            "cost": metric_sum(f_campaign, "cost"),
            "impressions": metric_sum(f_campaign, "impressions"),
            "clicks": metric_sum(f_campaign, "clicks"),
            "enrollment_forms": metric_sum(f_campaign, "enrollment_forms"),
            "applications": metric_sum(f_campaign, "applications"),
            "enrollment_apply_clicks": metric_sum(f_campaign, "enrollment_apply_clicks"),
            "career_clicks": metric_sum(f_campaign, "career_clicks"),
            "paid_traffic": metric_sum(f_ga4, "paid_traffic"),
            "engagement_rate": safe_div(metric_sum(f_lp, "engaged_sessions"), metric_sum(f_lp, "sessions")),
            "enrollment_lp_next_step_rate": safe_div(metric_sum(f_lp, "enrollment_form_submits"), metric_sum(f_lp, "sessions")),
            "recruitment_lp_next_step_rate": safe_div(metric_sum(f_lp, "career_clicks"), metric_sum(f_lp, "sessions")),
        }
        curr["ctr"] = safe_div(curr["clicks"], curr["impressions"])
        curr["cpc"] = safe_div(curr["cost"], curr["clicks"])
        curr["cost_per_enrollment_form"] = safe_div(curr["cost"], curr["enrollment_forms"])
        curr["enrollment_form_rate"] = safe_div(curr["enrollment_forms"], curr["clicks"])
        curr["enrollment_apply_click_rate"] = safe_div(curr["enrollment_apply_clicks"], curr["clicks"])
        curr["cost_per_application"] = safe_div(curr["cost"], curr["applications"])
        curr["application_rate"] = safe_div(curr["applications"], curr["clicks"])
        curr["career_click_rate"] = safe_div(curr["career_clicks"], curr["clicks"])

        prev = {
            "cost": metric_sum(prior_campaign, "cost"),
            "impressions": metric_sum(prior_campaign, "impressions"),
            "clicks": metric_sum(prior_campaign, "clicks"),
            "enrollment_forms": metric_sum(prior_campaign, "enrollment_forms"),
            "applications": metric_sum(prior_campaign, "applications"),
            "enrollment_apply_clicks": metric_sum(prior_campaign, "enrollment_apply_clicks"),
            "career_clicks": metric_sum(prior_campaign, "career_clicks"),
            "paid_traffic": metric_sum(prior_ga4, "paid_traffic"),
            "engagement_rate": safe_div(metric_sum(prior_lp, "engaged_sessions"), metric_sum(prior_lp, "sessions")),
            "enrollment_lp_next_step_rate": safe_div(metric_sum(prior_lp, "enrollment_form_submits"), metric_sum(prior_lp, "sessions")),
            "recruitment_lp_next_step_rate": safe_div(metric_sum(prior_lp, "career_clicks"), metric_sum(prior_lp, "sessions")),
        }
        prev["ctr"] = safe_div(prev["clicks"], prev["impressions"])
        prev["cpc"] = safe_div(prev["cost"], prev["clicks"])
        prev["cost_per_enrollment_form"] = safe_div(prev["cost"], prev["enrollment_forms"])
        prev["enrollment_form_rate"] = safe_div(prev["enrollment_forms"], prev["clicks"])
        prev["enrollment_apply_click_rate"] = safe_div(prev["enrollment_apply_clicks"], prev["clicks"])
        prev["cost_per_application"] = safe_div(prev["cost"], prev["applications"])
        prev["application_rate"] = safe_div(prev["applications"], prev["clicks"])
        prev["career_click_rate"] = safe_div(prev["career_clicks"], prev["clicks"])

        def kpi_card(label: str, key: str, fmt: str = "num"):
            val = curr.get(key, np.nan)
            prev_val = prev.get(key, np.nan)
            delta = format_delta(val, prev_val)
            if fmt == "currency":
                disp = f"${val:,.0f}" if pd.notna(val) else "—"
            elif fmt == "currency2":
                disp = f"${val:,.2f}" if pd.notna(val) else "—"
            elif fmt == "pct":
                disp = f"{val:.1%}" if pd.notna(val) else "—"
            else:
                disp = f"{val:,.0f}" if pd.notna(val) else "—"
            st.metric(label, disp, delta)

        if objective == "All":
            cfg = [
                ("Cost", "cost", "currency"),
                ("Impressions", "impressions", "num"),
                ("Clicks", "clicks", "num"),
                ("CTR", "ctr", "pct"),
                ("CPC", "cpc", "currency2"),
                ("Enrollment Forms", "enrollment_forms", "num"),
                ("Applications (3rd-party)", "applications", "num"),
                ("Paid Traffic (GA4)", "paid_traffic", "num"),
                ("Enrollment Apply Clicks", "enrollment_apply_clicks", "num"),
                ("Career Clicks", "career_clicks", "num"),
            ]
        elif objective == "Enrollment":
            cfg = [
                ("Cost", "cost", "currency"),
                ("Impressions", "impressions", "num"),
                ("Clicks", "clicks", "num"),
                ("Enrollment Forms", "enrollment_forms", "num"),
                ("Enrollment Apply Clicks", "enrollment_apply_clicks", "num"),
                ("Cost / Enrollment Form", "cost_per_enrollment_form", "currency2"),
                ("Enrollment Form Rate", "enrollment_form_rate", "pct"),
                ("Enrollment Apply Click Rate", "enrollment_apply_click_rate", "pct"),
                ("Paid Traffic (GA4)", "paid_traffic", "num"),
                ("Engagement Rate", "engagement_rate", "pct"),
                ("Enrollment LP Next Step Rate", "enrollment_lp_next_step_rate", "pct"),
            ]
        else:
            cfg = [
                ("Cost", "cost", "currency"),
                ("Impressions", "impressions", "num"),
                ("Clicks", "clicks", "num"),
                ("Applications (3rd-party)", "applications", "num"),
                ("Career Clicks", "career_clicks", "num"),
                ("Cost / Application", "cost_per_application", "currency2"),
                ("Application Rate", "application_rate", "pct"),
                ("Career Click Rate", "career_click_rate", "pct"),
                ("Paid Traffic (GA4)", "paid_traffic", "num"),
                ("Engagement Rate", "engagement_rate", "pct"),
                ("Recruitment LP Next Step Rate", "recruitment_lp_next_step_rate", "pct"),
            ]

        cols = st.columns(4)
        for i, (lbl, key, fmt) in enumerate(cfg):
            with cols[i % 4]:
                kpi_card(lbl, key, fmt)

        st.markdown("---")
        ts = aggregate_timeseries(
            f_campaign,
            agg_level,
            ["cost", "impressions", "clicks", "enrollment_forms", "applications", "career_clicks", "enrollment_apply_clicks"],
            [],
        )
        if not ts.empty:
            ts = add_campaign_metrics(ts)
            metric_options = {
                "All": ["cost", "clicks", "ctr", "cpc", "enrollment_forms", "applications", "career_clicks"],
                "Enrollment": ["cost", "clicks", "enrollment_forms", "enrollment_apply_clicks", "cost_per_enrollment_form", "enrollment_form_rate"],
                "Recruitment": ["cost", "clicks", "applications", "career_clicks", "cost_per_application", "application_rate"],
            }
            pick = st.selectbox("Trend metric", metric_options[objective])
            x_col = "week_start" if agg_level == "Weekly" else "month_start"
            fig_ts = px.line(ts, x=x_col, y=pick, markers=True, title="Trend")
            st.plotly_chart(fig_ts, use_container_width=True)

        plat_ts = aggregate_timeseries(f_campaign, agg_level, ["cost", "clicks"], ["platform"])
        if not plat_ts.empty and "platform" in plat_ts.columns:
            x_col = "week_start" if agg_level == "Weekly" else "month_start"
            fig_stack = px.bar(plat_ts, x=x_col, y="cost", color="platform", title="Platform Contribution by Period")
            st.plotly_chart(fig_stack, use_container_width=True)

            plat_sum = add_campaign_metrics(
                f_campaign.groupby("platform", as_index=False)[
                    ["cost", "impressions", "clicks", "enrollment_forms", "applications", "career_clicks", "enrollment_apply_clicks"]
                ].sum()
            )
            st.dataframe(plat_sum, use_container_width=True)


# -----------------------------
# Tab 2 - Objective Performance
# -----------------------------
with tab2:
    st.subheader("Objective Performance")

    obj_df = campaign_df[campaign_df["objective"].isin(["Enrollment", "Recruitment"])].copy()
    obj_df, _, obj_lp = apply_common_filters(obj_df, ga4_df, lp_df, {**filt, "objective": "All"})

    if obj_df.empty:
        st.info("No objective data available for selected filters.")
    else:
        ts_obj = aggregate_timeseries(obj_df, agg_level, ["cost", "enrollment_forms", "applications", "clicks"], ["objective"])
        ts_obj = add_campaign_metrics(ts_obj)
        x_col = "week_start" if agg_level == "Weekly" else "month_start"

        st.plotly_chart(px.area(ts_obj, x=x_col, y="cost", color="objective", title="Spend by Objective"), use_container_width=True)

        ts_obj_long = ts_obj.copy()
        ts_obj_long["primary_outcome"] = np.where(
            ts_obj_long["objective"] == "Enrollment", ts_obj_long["enrollment_forms"], ts_obj_long["applications"]
        )
        st.plotly_chart(
            px.line(ts_obj_long, x=x_col, y="primary_outcome", color="objective", markers=True, title="Primary Outcomes by Objective"),
            use_container_width=True,
        )

        ts_obj_long["cost_per_primary_outcome"] = safe_div(ts_obj_long["cost"], ts_obj_long["primary_outcome"])
        st.plotly_chart(
            px.line(ts_obj_long, x=x_col, y="cost_per_primary_outcome", color="objective", markers=True, title="Cost per Outcome by Objective"),
            use_container_width=True,
        )

        # Funnel-style summary table
        lp_roll = obj_lp[["sessions", "engaged_sessions", "enrollment_form_submits", "career_clicks"]].sum() if not obj_lp.empty else pd.Series(dtype=float)
        ga_roll = f_ga4[["paid_traffic"]].sum() if not f_ga4.empty else pd.Series(dtype=float)
        enroll = obj_df[obj_df["objective"] == "Enrollment"]
        recruit = obj_df[obj_df["objective"] == "Recruitment"]
        funnel = pd.DataFrame(
            [
                {
                    "objective": "Enrollment",
                    "cost": enroll["cost"].sum(),
                    "clicks": enroll["clicks"].sum(),
                    "paid_traffic_ga4": ga_roll.get("paid_traffic", np.nan),
                    "sessions": lp_roll.get("sessions", np.nan),
                    "engaged_sessions": lp_roll.get("engaged_sessions", np.nan),
                    "next_steps": lp_roll.get("enrollment_form_submits", np.nan),
                    "final_outcomes": enroll["enrollment_forms"].sum(),
                },
                {
                    "objective": "Recruitment",
                    "cost": recruit["cost"].sum(),
                    "clicks": recruit["clicks"].sum(),
                    "paid_traffic_ga4": ga_roll.get("paid_traffic", np.nan),
                    "sessions": lp_roll.get("sessions", np.nan),
                    "engaged_sessions": lp_roll.get("engaged_sessions", np.nan),
                    "next_steps": lp_roll.get("career_clicks", np.nan),
                    "final_outcomes": recruit["applications"].sum(),
                },
            ]
        )
        st.dataframe(funnel, use_container_width=True)


# -----------------------------
# Tab 3 - Campaign Performance
# -----------------------------
with tab3:
    st.subheader("Campaign Performance")

    camp_group_cols = ["platform", "campaign_name", "ad_topic"]
    valid_group_cols = [c for c in camp_group_cols if c in f_campaign.columns]

    if f_campaign.empty:
        st.info("No campaign rows for selected filters.")
    else:
        camp_tbl = f_campaign.groupby(valid_group_cols, as_index=False)[
            ["cost", "impressions", "clicks", "enrollment_forms", "enrollment_apply_clicks", "applications", "career_clicks"]
        ].sum()
        camp_tbl = add_campaign_metrics(camp_tbl)

        if objective == "Enrollment":
            cols_show = [
                "platform",
                "campaign_name",
                "ad_topic",
                "cost",
                "impressions",
                "clicks",
                "ctr",
                "cpc",
                "enrollment_forms",
                "enrollment_apply_clicks",
                "cost_per_enrollment_form",
                "enrollment_form_rate",
                "enrollment_apply_click_rate",
            ]
            x_metric = "enrollment_form_rate"
            y_metric = "cost_per_enrollment_form"
        elif objective == "Recruitment":
            cols_show = [
                "platform",
                "campaign_name",
                "ad_topic",
                "cost",
                "impressions",
                "clicks",
                "ctr",
                "cpc",
                "applications",
                "career_clicks",
                "cost_per_application",
                "application_rate",
                "career_click_rate",
            ]
            x_metric = "application_rate"
            y_metric = "cost_per_application"
        else:
            cols_show = [
                "platform",
                "campaign_name",
                "ad_topic",
                "cost",
                "impressions",
                "clicks",
                "ctr",
                "cpc",
                "enrollment_forms",
                "applications",
                "career_clicks",
                "cost_per_enrollment_form",
                "cost_per_application",
            ]
            x_metric = "ctr"
            y_metric = "cpc"

        cols_show = [c for c in cols_show if c in camp_tbl.columns]
        st.dataframe(camp_tbl[cols_show].sort_values("cost", ascending=False), use_container_width=True)

        x_col = "week_start" if agg_level == "Weekly" else "month_start"
        camp_trend = aggregate_timeseries(
            f_campaign,
            agg_level,
            ["cost", "clicks", "enrollment_forms", "applications"],
            ["campaign_name"],
        )
        metric_pick = st.selectbox("Campaign trend metric", ["cost", "clicks", "enrollment_forms", "applications"]) if not camp_trend.empty else None
        if metric_pick:
            st.plotly_chart(px.line(camp_trend, x=x_col, y=metric_pick, color="campaign_name", title="Weekly/Monthly Campaign Trend"), use_container_width=True)

        scatter = camp_tbl.dropna(subset=[x_metric, y_metric])
        if not scatter.empty:
            fig_scatter = px.scatter(
                scatter,
                x=x_metric,
                y=y_metric,
                size="clicks" if "clicks" in scatter.columns else "cost",
                color="platform" if "platform" in scatter.columns else None,
                hover_name="campaign_name" if "campaign_name" in scatter.columns else None,
                title="Campaign Diagnostic Scatterplot",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)


# -----------------------------
# Tab 4 - Meta Creative
# -----------------------------
with tab4:
    st.subheader("Meta Creative")

    meta = f_campaign[f_campaign["platform"] == "Meta"].copy() if "platform" in f_campaign.columns else pd.DataFrame()
    if meta.empty:
        st.info("No Meta data for selected filters.")
    else:
        topic_tbl = meta.groupby("ad_topic", as_index=False)[
            ["cost", "clicks", "enrollment_forms", "enrollment_apply_clicks", "applications", "career_clicks"]
        ].sum()
        topic_tbl = add_campaign_metrics(topic_tbl)

        ad_tbl = meta.groupby("ad_name", as_index=False)[
            ["cost", "clicks", "enrollment_forms", "enrollment_apply_clicks", "applications", "career_clicks"]
        ].sum()
        ad_tbl = add_campaign_metrics(ad_tbl)

        if objective == "Enrollment":
            show = ["cost", "clicks", "enrollment_forms", "enrollment_apply_clicks", "cost_per_enrollment_form", "enrollment_apply_click_rate"]
            trend_metric = "enrollment_forms"
        elif objective == "Recruitment":
            show = ["cost", "clicks", "applications", "career_clicks", "cost_per_application", "career_click_rate"]
            trend_metric = "applications"
        else:
            show = ["cost", "clicks", "enrollment_forms", "applications", "career_clicks"]
            trend_metric = "clicks"

        st.markdown("**Ad Topic Summary**")
        st.dataframe(topic_tbl[["ad_topic"] + [c for c in show if c in topic_tbl.columns]], use_container_width=True)

        st.markdown("**Ad Name Summary**")
        st.dataframe(ad_tbl[["ad_name"] + [c for c in show if c in ad_tbl.columns]], use_container_width=True)

        # Optional parsing helper placeholder
        with st.expander("Ad name parsing helper (optional)"):
            st.write(
                "Add naming-convention parser here, e.g., split ad_name into audience / hook / format / CTA when naming is standardized."
            )

        x_col = "week_start" if agg_level == "Weekly" else "month_start"
        meta_trend = aggregate_timeseries(meta, agg_level, ["cost", "clicks", "enrollment_forms", "applications", "career_clicks"], ["ad_name"])
        if not meta_trend.empty:
            st.plotly_chart(px.line(meta_trend, x=x_col, y=trend_metric, color="ad_name", title="Meta Creative Trend"), use_container_width=True)


# -----------------------------
# Tab 5 - Landing Page Analysis
# -----------------------------
with tab5:
    st.subheader("Landing Page Analysis")

    if f_lp.empty:
        st.info("No landing page data for selected filters.")
    else:
        lp_tbl = f_lp.groupby(["landing_page", "source", "medium"], as_index=False)[
            ["sessions", "total_users", "engaged_sessions", "views", "career_clicks", "enrollment_form_submits"]
        ].sum()
        lp_tbl = add_lp_metrics(lp_tbl)

        if objective == "Enrollment":
            cols_lp = [
                "landing_page",
                "sessions",
                "total_users",
                "engaged_sessions",
                "engagement_rate",
                "views",
                "views_per_session",
                "enrollment_form_submits",
                "enrollment_form_submit_rate",
            ]
            y_scatter = "enrollment_form_submit_rate"
            trend_metric = "enrollment_form_submits"
        elif objective == "Recruitment":
            cols_lp = [
                "landing_page",
                "sessions",
                "total_users",
                "engaged_sessions",
                "engagement_rate",
                "views",
                "views_per_session",
                "career_clicks",
                "career_click_rate_lp",
            ]
            y_scatter = "career_click_rate_lp"
            trend_metric = "career_clicks"
        else:
            cols_lp = [
                "landing_page",
                "sessions",
                "total_users",
                "engaged_sessions",
                "engagement_rate",
                "views",
                "views_per_session",
                "enrollment_form_submits",
                "enrollment_form_submit_rate",
                "career_clicks",
                "career_click_rate_lp",
            ]
            y_scatter = "enrollment_form_submit_rate"
            trend_metric = "sessions"

        st.dataframe(lp_tbl[[c for c in cols_lp if c in lp_tbl.columns]].sort_values("sessions", ascending=False), use_container_width=True)

        color_dim = st.selectbox("Scatter color", ["source", "medium", "campaign"])
        fig_lp_scatter = px.scatter(
            lp_tbl,
            x="engagement_rate",
            y=y_scatter,
            size="sessions",
            color=color_dim if color_dim in lp_tbl.columns else None,
            hover_name="landing_page",
            title="LP Quality Scatter",
        )
        st.plotly_chart(fig_lp_scatter, use_container_width=True)

        x_col = "week_start" if agg_level == "Weekly" else "month_start"
        lp_trend = aggregate_timeseries(f_lp, agg_level, ["sessions", "enrollment_form_submits", "career_clicks"], ["landing_page"])
        if not lp_trend.empty:
            st.plotly_chart(px.line(lp_trend, x=x_col, y=trend_metric, color="landing_page", title="Landing Page Trend"), use_container_width=True)

        st.markdown("**Optional Device Breakdown**")
        dev_tbl = f_lp.groupby("device", as_index=False)[["sessions", "enrollment_form_submits", "career_clicks"]].sum()
        st.dataframe(dev_tbl, use_container_width=True)


# -----------------------------
# Tab 6 - Definitions + Data Quality
# -----------------------------
with tab6:
    st.subheader("Data Quality / Definitions")

    st.markdown(
        """
### KPI Definitions
- **CTR** = clicks / impressions
- **CPC** = cost / clicks
- **Cost per Enrollment Form** = cost / enrollment_forms
- **Enrollment Form Rate** = enrollment_forms / clicks (campaign/media) or enrollment_form_submits / sessions (LP)
- **Enrollment Apply Click Rate** = enrollment_apply_clicks / clicks
- **Cost per Application** = cost / applications
- **Application Rate** = applications / clicks
- **Career Click Rate** = career_clicks / clicks (campaign/media)
- **Engagement Rate** = engaged_sessions / sessions
- **Enrollment LP Next Step Rate** = enrollment_form_submits / sessions
- **Recruitment LP Next Step Rate** = career_clicks / sessions

### Metric Source Notes
- **Applications (3rd-party)** comes from Campaign Master Feed.
- **Applications Submitted (GA4)** comes from GA4 Master Feed.
- These metrics are intentionally shown as separate measures.
        """
    )

    if data_quality_df is not None and not data_quality_df.empty:
        st.markdown("### Source freshness / QA")
        st.dataframe(data_quality_df, use_container_width=True)
    else:
        st.info("No data quality table provided.")
