"""Streamlit dashboard for HCZ marketing performance reporting.

This app reads "HCZ - Master Data File.xlsx" and renders six dashboard sections:
1) Executive Overview
2) Paid Media Performance
3) Website & GA4 Trends
4) Landing Page Performance
5) Campaign / Creative Drilldown
6) Data Explorer
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gspread
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google.oauth2.service_account import Credentials


EXCEL_FILE = "HCZ - Master Data File.xlsx"
DATA_SOURCE_DEFAULT = "google_sheets"  # Options: "google_sheets", "local_file"
ALLOW_LOCAL_FALLBACK = True

SHEET_CAMPAIGN = "Campaign Master Feed"
SHEET_GA4 = "GA4 Master Feed"
SHEET_LP = "LP Master Feed (Weekly)"

# Keep worksheet mapping centralized so minor tab name changes are easy to adjust.
GSHEET_WORKSHEET_MAP = {
    SHEET_CAMPAIGN: "Campaign Master Feed",
    SHEET_GA4: "GA4 Master Feed",
    SHEET_LP: "LP Master Feed (Weekly)",
}

PAGE_OPTIONS = [
    "Executive Overview",
    "Paid Media Performance",
    "Website & GA4 Trends",
    "Landing Page Performance",
    "Campaign / Creative Drilldown",
    "Data Explorer",
]

# Restrained, professional color palette.
COLOR_BLUE = "#2D6FA3"
COLOR_GREEN = "#2F8A63"
COLOR_SLATE = "#4C5A6A"
COLOR_LIGHT = "#EAF0F6"


# -------------------------------
# Utility and formatting helpers
# -------------------------------
def to_snake_case(name: str) -> str:
    """Convert a column name to snake_case while preserving readability."""
    return (
        str(name)
        .strip()
        .replace("/", " ")
        .replace("-", " ")
        .replace("(", "")
        .replace(")", "")
        .replace("__", " ")
        .lower()
        .replace(" ", "_")
    )

def is_valid_number(value):
    try:
        if value is None:
            return False
        if pd.isna(value):
            return False
        value = float(value)
        return math.isfinite(value)
    except Exception:
        return False


def safe_divide(numerator, denominator):
    try:
        if not is_valid_number(numerator) or not is_valid_number(denominator):
            return None
        numerator = float(numerator)
        denominator = float(denominator)
        if denominator == 0:
            return None
        result = numerator / denominator
        return result if math.isfinite(result) else None
    except Exception:
        return None


def fmt_int(value):
    if not is_valid_number(value):
        return "—"
    return f"{int(round(float(value), 0)):,}"


def fmt_currency(value, decimals=0):
    if not is_valid_number(value):
        return "—"
    return f"${float(value):,.{decimals}f}"


def fmt_currency_2(value):
    if not is_valid_number(value):
        return "—"
    return f"${float(value):,.2f}"


def fmt_pct(value):
    if not is_valid_number(value):
        return "—"
    return f"{float(value) * 100:,.1f}%"


def fmt_number(value, decimals=0):
    if not is_valid_number(value):
        return "—"
    return f"{float(value):,.{decimals}f}"


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert column to datetime safely with coercion."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def normalize_categorical(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Fill blank categorical values with 'Unknown' while preserving '(not set)'."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
            df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            df[col] = df[col].fillna("Unknown")
    return df


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Ensure numeric columns are converted safely."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def parse_month_for_sorting(df: pd.DataFrame, month_col: str) -> pd.DataFrame:
    """Add sortable month number where month labels may be text or numeric."""
    if month_col not in df.columns:
        return df

    month_map = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }

    raw = df[month_col].astype(str).str.strip().str.lower()
    month_num = pd.to_numeric(raw, errors="coerce")
    mapped = raw.map(month_map)
    df["month_num"] = month_num.fillna(mapped).fillna(0).astype(int)
    return df


def previous_period_delta(
    df: pd.DataFrame, date_col: str, metric_col: str, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> Optional[float]:
    """Calculate delta versus equivalent prior period for a metric."""
    if df.empty or date_col not in df.columns or metric_col not in df.columns:
        return None

    period_days = (end_date - start_date).days + 1
    if period_days <= 0:
        return None

    prev_end = start_date - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=period_days - 1)

    current_val = df.loc[df[date_col].between(start_date, end_date), metric_col].sum()
    prev_val = df.loc[df[date_col].between(prev_start, prev_end), metric_col].sum()

    if prev_val == 0:
        return None
    return (current_val - prev_val) / prev_val


def filter_summary_text(filters: Dict[str, Any]) -> str:
    """Create compact text describing active filters."""
    active = []
    for label, value in filters.items():
        if value is None:
            continue
        if isinstance(value, list):
            if value and "All" not in value:
                active.append(f"{label}: {', '.join(map(str, value[:3]))}{'…' if len(value) > 3 else ''}")
        else:
            if value != "All":
                active.append(f"{label}: {value}")

    if not active:
        return "Active filters: All data"
    return "Active filters: " + " | ".join(active)


def empty_state(message: str = "No data available for the selected filters.") -> None:
    """Render a standard empty-state callout."""
    st.info(message)


# -------------------------------
# Data loading and transformations
# -------------------------------
@st.cache_data(show_spinner=False)
def load_local_data(file_path: str) -> Dict[str, pd.DataFrame]:
    """Load workbook sheets from a local file and return raw dataframes."""
    workbook = pd.read_excel(
        file_path,
        sheet_name=[SHEET_CAMPAIGN, SHEET_GA4, SHEET_LP],
        engine="openpyxl",
    )
    return workbook


@st.cache_resource(show_spinner=False)
def get_gsheet_client() -> gspread.Client:
    """Build an authenticated gspread client using Streamlit secrets."""
    if "google_service_account" not in st.secrets:
        raise ValueError(
            "Missing [google_service_account] in Streamlit secrets. "
            "Add service account fields in .streamlit/secrets.toml or deployment secrets."
        )

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(
        dict(st.secrets["google_service_account"]),
        scopes=scopes,
    )
    return gspread.authorize(creds)


@st.cache_data(show_spinner=False)
def load_google_sheet_tab(sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    """Load a single worksheet tab from Google Sheets as a DataFrame."""
    client = get_gsheet_client()
    spreadsheet = client.open_by_key(sheet_id)
    worksheet = spreadsheet.worksheet(worksheet_name)
    records = worksheet.get_all_records()
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def load_all_datasets_from_gsheet(
    sheet_id: str, worksheet_map: Dict[str, str]
) -> Dict[str, pd.DataFrame]:
    """Load all dashboard datasets from Google Sheets using configured tab names."""
    datasets: Dict[str, pd.DataFrame] = {}
    for dataset_key, worksheet_name in worksheet_map.items():
        datasets[dataset_key] = load_google_sheet_tab(sheet_id, worksheet_name)
    return datasets


@st.cache_data(show_spinner=False)
def clean_campaign_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean campaign-level paid media data and add derived metrics."""
    df = df.copy()
    df.columns = [to_snake_case(c) for c in df.columns]

    df = ensure_datetime(df, "date")
    df = ensure_numeric(
        df,
        [
            "cost",
            "impressions",
            "clicks",
            "career_clicks",
            "applications",
            "enrollment_forms",
            "enrollment_apply_clicks",
            "year",
        ],
    )

    df = normalize_categorical(
        df,
        ["platform", "campaign_name", "ad_name", "ad_topic", "objective", "month"],
    )

    df = parse_month_for_sorting(df, "month")
    df["ctr"] = safe_divide(df["clicks"], df["impressions"])
    df["cpc"] = safe_divide(df["cost"], df["clicks"])
    df["cpm"] = safe_divide(df["cost"], df["impressions"]) * 1000
    df["cost_per_application"] = safe_divide(df["cost"], df["applications"])
    df["cost_per_career_click"] = safe_divide(df["cost"], df["career_clicks"])

    return df


@st.cache_data(show_spinner=False)
def clean_ga4_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean GA4 trend data and compute paid-traffic-based rates."""
    df = df.copy()
    df.columns = [to_snake_case(c) for c in df.columns]

    df = ensure_datetime(df, "date")
    df = ensure_numeric(
        df,
        [
            "paid_traffic",
            "non_paid_traffic",
            "applications_submitted",
            "career_clicks",
            "enrollment_form_submits",
            "year",
        ],
    )
    df = normalize_categorical(df, ["month"])

    df = parse_month_for_sorting(df, "month")
    df["application_rate_from_paid_traffic"] = safe_divide(
        df["applications_submitted"], df["paid_traffic"]
    )
    df["career_click_rate_from_paid_traffic"] = safe_divide(
        df["career_clicks"], df["paid_traffic"]
    )
    return df


@st.cache_data(show_spinner=False)
def clean_lp_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean weekly landing page dataset and add engagement efficiency fields."""
    df = df.copy()
    df.columns = [to_snake_case(c) for c in df.columns]

    df = ensure_datetime(df, "week_start")
    df = ensure_numeric(
        df,
        [
            "sessions",
            "total_users",
            "engaged_sessions",
            "views",
            "career_clicks",
            "enrollment_form_submits",
        ],
    )
    df = normalize_categorical(
        df,
        ["month", "landing_page", "source", "medium", "campaign", "content", "term", "device"],
    )

    df = parse_month_for_sorting(df, "month")
    df["engagement_rate"] = safe_divide(df["engaged_sessions"], df["sessions"])
    df["views_per_session"] = safe_divide(df["views"], df["sessions"])
    return df


@st.cache_data(show_spinner=False)
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert dataframe to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def apply_filters(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    year: str = "All",
    month: str = "All",
    objective: str = "All",
    platform: str = "All",
    ad_topic: str = "All",
    landing_page: str = "All",
    source: str = "All",
    medium: str = "All",
    device: str = "All",
) -> pd.DataFrame:
    """Apply unified filtering across datasets where columns exist."""
    filtered = df.copy()

    if date_col and date_col in filtered.columns and start_date is not None and end_date is not None:
        filtered = filtered[filtered[date_col].between(start_date, end_date)]

    filter_map = {
        "year": year,
        "month": month,
        "objective": objective,
        "platform": platform,
        "ad_topic": ad_topic,
        "landing_page": landing_page,
        "source": source,
        "medium": medium,
        "device": device,
    }

    for col, selected in filter_map.items():
        if col in filtered.columns and selected != "All":
            filtered = filtered[filtered[col].astype(str) == str(selected)]

    return filtered


def build_filter_options(df: pd.DataFrame, col: str) -> List[str]:
    """Build ordered filter options with an 'All' default."""
    if col not in df.columns:
        return ["All"]
    values = sorted(df[col].astype(str).dropna().unique().tolist())
    return ["All", *values]


# -------------------------------
# UI rendering helpers
# -------------------------------
def render_header(max_data_date: Optional[pd.Timestamp], active_filter_text: str) -> None:
    """Render polished dashboard header area."""
    st.markdown(
        """
        <style>
        .hcz-header {
            background: #ffffff;
            border: 1px solid #DFE8F0;
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }
        .hcz-title {
            font-size: 1.7rem;
            font-weight: 700;
            color: #1F3B58;
            margin-bottom: 0.2rem;
        }
        .hcz-subtitle {
            font-size: 0.98rem;
            color: #47617B;
            margin-bottom: 0.5rem;
        }
        .hcz-meta {
            font-size: 0.85rem;
            color: #5E7388;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #E0E8F0;
            border-radius: 12px;
            padding: 0.5rem 0.7rem;
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    refresh_text = "N/A"
    if isinstance(max_data_date, pd.Timestamp) and pd.notna(max_data_date):
        refresh_text = max_data_date.strftime("%B %d, %Y")

    st.markdown(
        f"""
        <div class="hcz-header">
            <div class="hcz-title">Harlem Children’s Zone Marketing Performance Dashboard</div>
            <div class="hcz-subtitle">Executive reporting across paid media, website behavior, and landing page outcomes.</div>
            <div class="hcz-meta"><b>Last refresh based on available data:</b> {refresh_text}</div>
            <div class="hcz-meta" style="margin-top: 0.25rem;">{active_filter_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(kpis: List[Tuple[str, str, Optional[str]]], cols_per_row: int = 5) -> None:
    """Render KPI cards in a responsive row format."""
    for i in range(0, len(kpis), cols_per_row):
        row_items = kpis[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if j < len(row_items):
                label, value, delta = row_items[j]
                col.metric(label=label, value=value, delta=delta)


def get_weighted_ctr_campaign(df: pd.DataFrame, min_impressions: int = 1000) -> Optional[Tuple[str, float]]:
    """Get best campaign by CTR with minimum impression threshold."""
    if df.empty:
        return None

    grouped = (
        df.groupby("campaign_name", as_index=False)[["clicks", "impressions"]]
        .sum()
        .assign(ctr=lambda d: safe_divide(d["clicks"], d["impressions"]))
    )
    grouped = grouped[grouped["impressions"] >= min_impressions]
    if grouped.empty:
        return None

    top_row = grouped.sort_values("ctr", ascending=False).iloc[0]
    return top_row["campaign_name"], float(top_row["ctr"])


def get_best_cpa_campaign(df: pd.DataFrame, min_apps: int = 3) -> Optional[Tuple[str, float]]:
    """Get best campaign by cost per application with application threshold."""
    if df.empty:
        return None

    grouped = (
        df.groupby("campaign_name", as_index=False)[["cost", "applications"]]
        .sum()
        .assign(cost_per_application=lambda d: safe_divide(d["cost"], d["applications"]))
    )
    grouped = grouped[grouped["applications"] >= min_apps]
    if grouped.empty:
        return None

    top_row = grouped.sort_values("cost_per_application", ascending=True).iloc[0]
    return top_row["campaign_name"], float(top_row["cost_per_application"])


# -------------------------------
# Page renderers
# -------------------------------
def render_executive_overview(campaign_df: pd.DataFrame, ga4_df: pd.DataFrame, lp_df: pd.DataFrame) -> None:
    """Render executive overview page."""
    st.subheader("Executive Overview")
    st.caption("High-level outcomes across paid media, GA4, and landing page engagement.")

    if campaign_df.empty and ga4_df.empty and lp_df.empty:
        empty_state()
        return

    # KPI row
    kpis = [
        ("Total Paid Media Cost", fmt_currency(campaign_df["cost"].sum() if not campaign_df.empty else 0), None),
        ("Total Impressions", fmt_int(campaign_df["impressions"].sum() if not campaign_df.empty else 0), None),
        ("Total Clicks", fmt_int(campaign_df["clicks"].sum() if not campaign_df.empty else 0), None),
        ("Total Applications", fmt_int(campaign_df["applications"].sum() if not campaign_df.empty else 0), None),
        ("Total Career Clicks", fmt_int(campaign_df["career_clicks"].sum() if not campaign_df.empty else 0), None),
        ("Total Paid Traffic", fmt_int(ga4_df["paid_traffic"].sum() if not ga4_df.empty else 0), None),
        ("Total Non Paid Traffic", fmt_int(ga4_df["non_paid_traffic"].sum() if not ga4_df.empty else 0), None),
        (
            "Total GA4 Applications Submitted",
            fmt_int(ga4_df["applications_submitted"].sum() if not ga4_df.empty else 0),
            None,
        ),
        ("Total Landing Page Sessions", fmt_int(lp_df["sessions"].sum() if not lp_df.empty else 0), None),
        (
            "Total Landing Page Career Clicks",
            fmt_int(lp_df["career_clicks"].sum() if not lp_df.empty else 0),
            None,
        ),
    ]
    render_kpi_row(kpis, cols_per_row=5)

    # Trends
    st.markdown("### Trends")
    col1, col2 = st.columns(2)

    with col1:
        if not campaign_df.empty:
            daily_spend = campaign_df.groupby("date", as_index=False)["cost"].sum()
            fig = px.line(daily_spend, x="date", y="cost", title="Paid Media Cost Over Time")
            fig.update_traces(line_color=COLOR_BLUE)
            fig.update_layout(yaxis_title="Cost ($)", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("No campaign data for paid media cost trend.")

    with col2:
        if not campaign_df.empty:
            daily_click_apps = (
                campaign_df.groupby("date", as_index=False)[["clicks", "applications"]].sum().melt(
                    id_vars="date", var_name="metric", value_name="value"
                )
            )
            fig = px.line(
                daily_click_apps,
                x="date",
                y="value",
                color="metric",
                title="Paid Media Clicks and Applications Over Time",
                color_discrete_map={"clicks": COLOR_BLUE, "applications": COLOR_GREEN},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("No campaign data for clicks/applications trend.")

    col3, col4 = st.columns(2)
    with col3:
        if not ga4_df.empty:
            traffic_df = ga4_df.groupby("date", as_index=False)[["paid_traffic", "non_paid_traffic"]].sum()
            traffic_melt = traffic_df.melt(id_vars="date", var_name="type", value_name="traffic")
            fig = px.line(
                traffic_melt,
                x="date",
                y="traffic",
                color="type",
                title="Paid vs Non-Paid Traffic Over Time",
                color_discrete_map={"paid_traffic": COLOR_BLUE, "non_paid_traffic": COLOR_GREEN},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("No GA4 data for traffic trend.")

    with col4:
        if not lp_df.empty:
            weekly_sessions = lp_df.groupby("week_start", as_index=False)["sessions"].sum()
            fig = px.line(weekly_sessions, x="week_start", y="sessions", title="Landing Page Sessions Over Time")
            fig.update_traces(line_color=COLOR_SLATE)
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("No LP data for sessions trend.")

    # Breakdowns
    st.markdown("### Breakdown")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if not campaign_df.empty:
            spend_platform = campaign_df.groupby("platform", as_index=False)["cost"].sum().sort_values("cost", ascending=False)
            fig = px.bar(spend_platform, x="platform", y="cost", title="Spend by Platform", color_discrete_sequence=[COLOR_BLUE])
            st.plotly_chart(fig, use_container_width=True)
    with b2:
        if not campaign_df.empty:
            app_objective = campaign_df.groupby("objective", as_index=False)["applications"].sum().sort_values("applications", ascending=False)
            fig = px.bar(app_objective, x="objective", y="applications", title="Applications by Objective", color_discrete_sequence=[COLOR_GREEN])
            st.plotly_chart(fig, use_container_width=True)
    with b3:
        if not ga4_df.empty:
            traffic_split = pd.DataFrame(
                {
                    "Traffic Type": ["Paid", "Non Paid"],
                    "Value": [ga4_df["paid_traffic"].sum(), ga4_df["non_paid_traffic"].sum()],
                }
            )
            fig = px.pie(traffic_split, names="Traffic Type", values="Value", title="Traffic Split", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)
    with b4:
        if not lp_df.empty:
            top_lp = lp_df.groupby("landing_page", as_index=False)["sessions"].sum().sort_values("sessions", ascending=False).head(10)
            fig = px.bar(top_lp, x="sessions", y="landing_page", orientation="h", title="Top Landing Pages by Sessions", color_discrete_sequence=[COLOR_SLATE])
            st.plotly_chart(fig, use_container_width=True)

    # Highlights
    st.markdown("### Highlights")
    highlights: List[str] = []

    if not campaign_df.empty:
        spend_by_platform = campaign_df.groupby("platform", as_index=False)["cost"].sum()
        if not spend_by_platform.empty:
            r = spend_by_platform.sort_values("cost", ascending=False).iloc[0]
            highlights.append(f"Highest-spend platform: **{r['platform']}** ({fmt_currency_2(r['cost'])}).")

        topic_apps = campaign_df.groupby("ad_topic", as_index=False)["applications"].sum()
        if not topic_apps.empty:
            r = topic_apps.sort_values("applications", ascending=False).iloc[0]
            highlights.append(f"Top ad topic by applications: **{r['ad_topic']}** ({fmt_int(r['applications'])}).")

        best_ctr = get_weighted_ctr_campaign(campaign_df, min_impressions=1000)
        if best_ctr:
            highlights.append(f"Best CTR campaign (min 1,000 impressions): **{best_ctr[0]}** ({fmt_pct(best_ctr[1])}).")
        else:
            best_cpa = get_best_cpa_campaign(campaign_df, min_apps=3)
            if best_cpa:
                highlights.append(f"Best cost/application campaign (min 3 applications): **{best_cpa[0]}** ({fmt_currency_2(best_cpa[1])}).")

    if not lp_df.empty:
        lp_sessions = lp_df.groupby("landing_page", as_index=False)["sessions"].sum()
        if not lp_sessions.empty:
            r = lp_sessions.sort_values("sessions", ascending=False).iloc[0]
            highlights.append(f"Top landing page by sessions: **{r['landing_page']}** ({fmt_int(r['sessions'])}).")

    if highlights:
        for item in highlights:
            st.markdown(f"- {item}")
    else:
        empty_state("No highlight insights available for the selected filters.")


def render_paid_media_page(campaign_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
    """Render paid media performance page."""
    st.subheader("Paid Media Performance")
    st.caption("Daily paid media delivery, efficiency, and ranking insights from Campaign Master Feed.")

    if campaign_df.empty:
        empty_state()
        return

    # KPI cards with period-over-period deltas where possible.
    deltas = {
        "spend": previous_period_delta(campaign_df, "date", "cost", start_date, end_date),
        "clicks": previous_period_delta(campaign_df, "date", "clicks", start_date, end_date),
        "applications": previous_period_delta(campaign_df, "date", "applications", start_date, end_date),
    }

    total_spend = campaign_df["cost"].sum()
    total_impressions = campaign_df["impressions"].sum()
    total_clicks = campaign_df["clicks"].sum()
    total_career_clicks = campaign_df["career_clicks"].sum()
    total_apps = campaign_df["applications"].sum()
    total_forms = campaign_df["enrollment_forms"].sum()

    overall_ctr = safe_divide(total_clicks, total_impressions)
    overall_cpc = safe_divide(total_spend, total_clicks)
    overall_cpa = safe_divide(total_spend, total_apps)
    overall_cpcc = safe_divide(total_spend, total_career_clicks)

    kpis = [
        ("Spend", fmt_currency(total_spend), fmt_pct(deltas["spend"]) if deltas["spend"] is not None else None),
        ("Impressions", fmt_int(total_impressions), None),
        ("Clicks", fmt_int(total_clicks), fmt_pct(deltas["clicks"]) if deltas["clicks"] is not None else None),
        ("CTR", fmt_pct(overall_ctr), None),
        ("CPC", fmt_currency_2(overall_cpc), None),
        ("Career Clicks", fmt_int(total_career_clicks), None),
        ("Applications", fmt_int(total_apps), fmt_pct(deltas["applications"]) if deltas["applications"] is not None else None),
        ("Enrollment Forms", fmt_int(total_forms), None),
        ("Cost per Application", fmt_currency_2(overall_cpa), None),
        ("Cost per Career Click", fmt_currency_2(overall_cpcc), None),
    ]
    render_kpi_row(kpis, cols_per_row=5)

    c1, c2 = st.columns(2)
    with c1:
        daily = campaign_df.groupby("date", as_index=False)[["cost", "clicks", "applications"]].sum()
        melted = daily.melt(id_vars="date", var_name="metric", value_name="value")
        fig = px.line(
            melted,
            x="date",
            y="value",
            color="metric",
            title="Daily Trend: Spend, Clicks, Applications",
            color_discrete_map={"cost": COLOR_BLUE, "clicks": COLOR_GREEN, "applications": COLOR_SLATE},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        scatter = campaign_df.groupby(["campaign_name", "platform"], as_index=False)[
            ["cost", "applications", "clicks"]
        ].sum()
        fig = px.scatter(
            scatter,
            x="cost",
            y="applications",
            size="clicks",
            color="platform",
            hover_name="campaign_name",
            title="Campaign Efficiency: Spend vs Applications",
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4, c5 = st.columns(3)
    with c3:
        spend_platform = campaign_df.groupby("platform", as_index=False)["cost"].sum().sort_values("cost", ascending=False)
        fig = px.bar(spend_platform, x="platform", y="cost", title="Spend by Platform", color_discrete_sequence=[COLOR_BLUE])
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        app_platform = campaign_df.groupby("platform", as_index=False)["applications"].sum().sort_values("applications", ascending=False)
        fig = px.bar(app_platform, x="platform", y="applications", title="Applications by Platform", color_discrete_sequence=[COLOR_GREEN])
        st.plotly_chart(fig, use_container_width=True)
    with c5:
        objective_mix = campaign_df.groupby("objective", as_index=False)["cost"].sum().sort_values("cost", ascending=False)
        fig = px.pie(objective_mix, names="objective", values="cost", title="Objective Mix", hole=0.45)
        st.plotly_chart(fig, use_container_width=True)

    c6, c7 = st.columns(2)
    with c6:
        spend_topic = campaign_df.groupby("ad_topic", as_index=False)["cost"].sum().sort_values("cost", ascending=False).head(15)
        fig = px.bar(spend_topic, x="ad_topic", y="cost", title="Spend by Ad Topic", color_discrete_sequence=[COLOR_SLATE])
        st.plotly_chart(fig, use_container_width=True)
    with c7:
        apps_topic = campaign_df.groupby("ad_topic", as_index=False)["applications"].sum().sort_values("applications", ascending=False).head(15)
        fig = px.bar(apps_topic, x="ad_topic", y="applications", title="Applications by Ad Topic", color_discrete_sequence=[COLOR_GREEN])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Ranked Tables")
    t1, t2 = st.columns(2)
    with t1:
        top_campaign_spend = campaign_df.groupby("campaign_name", as_index=False)["cost"].sum().sort_values("cost", ascending=False).head(15)
        st.markdown("**Top Campaigns by Spend**")
        st.dataframe(top_campaign_spend, use_container_width=True)

        top_ad_apps = campaign_df.groupby("ad_name", as_index=False)["applications"].sum().sort_values("applications", ascending=False).head(15)
        st.markdown("**Top Ads by Applications**")
        st.dataframe(top_ad_apps, use_container_width=True)

    with t2:
        top_campaign_apps = campaign_df.groupby("campaign_name", as_index=False)["applications"].sum().sort_values("applications", ascending=False).head(15)
        st.markdown("**Top Campaigns by Applications**")
        st.dataframe(top_campaign_apps, use_container_width=True)

        low_cpa = (
            campaign_df.groupby("campaign_name", as_index=False)[["cost", "applications"]]
            .sum()
            .query("applications >= 3")
            .assign(cost_per_application=lambda d: safe_divide(d["cost"], d["applications"]))
            .sort_values("cost_per_application", ascending=True)
            .head(15)
        )
        st.markdown("**Lowest Cost per Application (min 3 applications)**")
        st.dataframe(low_cpa, use_container_width=True)

    st.markdown("### Platform Comparison (Meta vs Google)")
    platform_compare = (
        campaign_df[campaign_df["platform"].str.lower().isin(["meta", "google"])]
        .groupby("platform", as_index=False)[["cost", "impressions", "clicks", "applications"]]
        .sum()
    )
    if platform_compare.empty:
        empty_state("No Meta/Google rows in current filter context.")
    else:
        platform_compare["ctr"] = safe_divide(platform_compare["clicks"], platform_compare["impressions"])
        platform_compare["cpc"] = safe_divide(platform_compare["cost"], platform_compare["clicks"])
        platform_compare["cost_per_application"] = safe_divide(
            platform_compare["cost"], platform_compare["applications"]
        )
        st.dataframe(platform_compare, use_container_width=True)


def render_ga4_page(ga4_df: pd.DataFrame) -> None:
    """Render website and GA4 trends page."""
    st.subheader("Website & GA4 Trends")
    st.caption("Site-level traffic and conversion behavior using GA4 Master Feed.")

    if ga4_df.empty:
        empty_state()
        return

    paid = ga4_df["paid_traffic"].sum()
    non_paid = ga4_df["non_paid_traffic"].sum()
    apps = ga4_df["applications_submitted"].sum()
    career = ga4_df["career_clicks"].sum()
    enroll = ga4_df["enrollment_form_submits"].sum()

    app_rate = safe_divide(apps, paid)
    career_rate = safe_divide(career, paid)

    kpis = [
        ("Paid Traffic", fmt_int(paid), None),
        ("Non Paid Traffic", fmt_int(non_paid), None),
        ("Applications Submitted", fmt_int(apps), None),
        ("Career Clicks", fmt_int(career), None),
        ("Enrollment Form Submits", fmt_int(enroll), None),
        ("Application Rate from Paid Traffic", fmt_pct(app_rate), None),
        ("Career Click Rate from Paid Traffic", fmt_pct(career_rate), None),
    ]
    render_kpi_row(kpis, cols_per_row=4)

    g1, g2 = st.columns(2)
    with g1:
        traffic = ga4_df.groupby("date", as_index=False)[["paid_traffic", "non_paid_traffic"]].sum()
        melt = traffic.melt(id_vars="date", var_name="traffic_type", value_name="value")
        fig = px.line(
            melt,
            x="date",
            y="value",
            color="traffic_type",
            title="Daily Paid vs Non-Paid Traffic",
            color_discrete_map={"paid_traffic": COLOR_BLUE, "non_paid_traffic": COLOR_GREEN},
        )
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        fig = px.line(
            ga4_df.groupby("date", as_index=False)["applications_submitted"].sum(),
            x="date",
            y="applications_submitted",
            title="Daily Applications Submitted",
            color_discrete_sequence=[COLOR_GREEN],
        )
        st.plotly_chart(fig, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        fig = px.line(
            ga4_df.groupby("date", as_index=False)["career_clicks"].sum(),
            x="date",
            y="career_clicks",
            title="Daily Career Clicks",
            color_discrete_sequence=[COLOR_SLATE],
        )
        st.plotly_chart(fig, use_container_width=True)

    with g4:
        monthly = (
            ga4_df.groupby(["year", "month", "month_num"], as_index=False)[
                ["paid_traffic", "non_paid_traffic", "applications_submitted"]
            ]
            .sum()
            .sort_values(["year", "month_num"])
        )
        monthly["period"] = monthly["year"].astype(int).astype(str) + "-" + monthly["month"]
        monthly_melt = monthly.melt(
            id_vars=["period"],
            value_vars=["paid_traffic", "non_paid_traffic", "applications_submitted"],
            var_name="metric",
            value_name="value",
        )
        fig = px.bar(
            monthly_melt,
            x="period",
            y="value",
            color="metric",
            barmode="group",
            title="Monthly Summary (Paid, Non-Paid, Applications)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Derived Insight Views")
    d1, d2 = st.columns(2)
    with d1:
        comp = ga4_df.groupby("date", as_index=False)[["paid_traffic", "applications_submitted"]].sum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=comp["date"], y=comp["paid_traffic"], name="Paid Traffic", line=dict(color=COLOR_BLUE)))
        fig.add_trace(go.Scatter(x=comp["date"], y=comp["applications_submitted"], name="Applications Submitted", line=dict(color=COLOR_GREEN), yaxis="y2"))
        fig.update_layout(
            title="Paid Traffic vs Applications Submitted",
            yaxis=dict(title="Paid Traffic"),
            yaxis2=dict(title="Applications", overlaying="y", side="right"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        comp2 = ga4_df.groupby("date", as_index=False)[["paid_traffic", "career_clicks"]].sum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=comp2["date"], y=comp2["paid_traffic"], name="Paid Traffic", line=dict(color=COLOR_BLUE)))
        fig.add_trace(go.Scatter(x=comp2["date"], y=comp2["career_clicks"], name="Career Clicks", line=dict(color=COLOR_SLATE), yaxis="y2"))
        fig.update_layout(
            title="Paid Traffic vs Career Clicks",
            yaxis=dict(title="Paid Traffic"),
            yaxis2=dict(title="Career Clicks", overlaying="y", side="right"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Monthly Summary Table")
    summary = (
        ga4_df.groupby(["year", "month", "month_num"], as_index=False)[
            ["paid_traffic", "non_paid_traffic", "applications_submitted", "career_clicks", "enrollment_form_submits"]
        ]
        .sum()
        .sort_values(["year", "month_num"])
    )
    st.dataframe(summary, use_container_width=True)


def render_lp_page(lp_df: pd.DataFrame) -> None:
    """Render landing page performance page."""
    st.subheader("Landing Page Performance")
    st.caption("Weekly landing page engagement and conversion outcomes.")

    if lp_df.empty:
        empty_state()
        return

    sessions = lp_df["sessions"].sum()
    users = lp_df["total_users"].sum()
    engaged = lp_df["engaged_sessions"].sum()
    views = lp_df["views"].sum()
    career = lp_df["career_clicks"].sum()
    enroll = lp_df["enrollment_form_submits"].sum()

    kpis = [
        ("Sessions", fmt_int(sessions), None),
        ("Total Users", fmt_int(users), None),
        ("Engaged Sessions", fmt_int(engaged), None),
        ("Engagement Rate", fmt_pct(safe_divide(engaged, sessions)), None),
        ("Views", fmt_int(views), None),
        ("Views per Session", f"{safe_divide(views, sessions):.2f}", None),
        ("Career Clicks", fmt_int(career), None),
        ("Enrollment Form Submits", fmt_int(enroll), None),
    ]
    render_kpi_row(kpis, cols_per_row=4)

    l1, l2 = st.columns(2)
    with l1:
        trend = lp_df.groupby("week_start", as_index=False)["sessions"].sum()
        fig = px.line(trend, x="week_start", y="sessions", title="Weekly Sessions Trend", color_discrete_sequence=[COLOR_BLUE])
        st.plotly_chart(fig, use_container_width=True)
    with l2:
        trend2 = lp_df.groupby("week_start", as_index=False)["engaged_sessions"].sum()
        fig = px.line(trend2, x="week_start", y="engaged_sessions", title="Weekly Engaged Sessions Trend", color_discrete_sequence=[COLOR_GREEN])
        st.plotly_chart(fig, use_container_width=True)

    l3, l4 = st.columns(2)
    with l3:
        by_lp = lp_df.groupby("landing_page", as_index=False)["sessions"].sum().sort_values("sessions", ascending=False).head(15)
        fig = px.bar(by_lp, x="sessions", y="landing_page", orientation="h", title="Sessions by Landing Page", color_discrete_sequence=[COLOR_SLATE])
        st.plotly_chart(fig, use_container_width=True)
    with l4:
        by_lp_cc = lp_df.groupby("landing_page", as_index=False)["career_clicks"].sum().sort_values("career_clicks", ascending=False).head(15)
        fig = px.bar(by_lp_cc, x="career_clicks", y="landing_page", orientation="h", title="Career Clicks by Landing Page", color_discrete_sequence=[COLOR_GREEN])
        st.plotly_chart(fig, use_container_width=True)

    l5, l6, l7 = st.columns(3)
    with l5:
        by_device = lp_df.groupby("device", as_index=False)["sessions"].sum().sort_values("sessions", ascending=False)
        fig = px.bar(by_device, x="device", y="sessions", title="Sessions by Device", color_discrete_sequence=[COLOR_BLUE])
        st.plotly_chart(fig, use_container_width=True)
    with l6:
        engage_by_device = (
            lp_df.groupby("device", as_index=False)[["engaged_sessions", "sessions"]]
            .sum()
            .assign(engagement_rate=lambda d: safe_divide(d["engaged_sessions"], d["sessions"]))
            .sort_values("engagement_rate", ascending=False)
        )
        fig = px.bar(engage_by_device, x="device", y="engagement_rate", title="Engagement Rate by Device", color_discrete_sequence=[COLOR_GREEN])
        fig.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)
    with l7:
        sm = lp_df.assign(source_medium=lp_df["source"] + " / " + lp_df["medium"])\
              .groupby("source_medium", as_index=False)["sessions"].sum()\
              .sort_values("sessions", ascending=False).head(15)
        fig = px.bar(sm, x="sessions", y="source_medium", orientation="h", title="Sessions by Source / Medium", color_discrete_sequence=[COLOR_SLATE])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tables")
    t1, t2 = st.columns(2)
    with t1:
        top_lp_sessions = lp_df.groupby("landing_page", as_index=False)["sessions"].sum().sort_values("sessions", ascending=False).head(20)
        st.markdown("**Top Landing Pages by Sessions**")
        st.dataframe(top_lp_sessions, use_container_width=True)

        top_lp_career = lp_df.groupby("landing_page", as_index=False)["career_clicks"].sum().sort_values("career_clicks", ascending=False).head(20)
        st.markdown("**Top Landing Pages by Career Clicks**")
        st.dataframe(top_lp_career, use_container_width=True)

    with t2:
        top_sm = (
            lp_df.assign(source_medium=lp_df["source"] + " / " + lp_df["medium"])\
            .groupby("source_medium", as_index=False)[["sessions", "career_clicks", "enrollment_form_submits"]]
            .sum()
            .sort_values("sessions", ascending=False)
            .head(20)
        )
        st.markdown("**Top Source / Medium Combinations**")
        st.dataframe(top_sm, use_container_width=True)

        device_summary = (
            lp_df.groupby("device", as_index=False)[["sessions", "engaged_sessions", "career_clicks", "enrollment_form_submits"]]
            .sum()
            .assign(engagement_rate=lambda d: safe_divide(d["engaged_sessions"], d["sessions"]))
            .sort_values("sessions", ascending=False)
        )
        st.markdown("**Device Performance Summary**")
        st.dataframe(device_summary, use_container_width=True)

    st.markdown("### Landing Page Funnel Metrics")
    funnel = (
        lp_df.groupby("landing_page", as_index=False)[
            ["sessions", "engaged_sessions", "career_clicks", "enrollment_form_submits"]
        ]
        .sum()
        .sort_values("sessions", ascending=False)
        .head(20)
    )
    st.dataframe(funnel, use_container_width=True)


def render_drilldown_page(campaign_df: pd.DataFrame) -> None:
    """Render campaign and creative drilldown page."""
    st.subheader("Campaign / Creative Drilldown")
    st.caption("Inspect campaign-level and ad-level performance with focused drilldown controls.")

    if campaign_df.empty:
        empty_state()
        return

    campaigns = sorted(campaign_df["campaign_name"].dropna().unique().tolist())
    selected_campaign = st.selectbox("Select campaign name", campaigns)

    subset = campaign_df[campaign_df["campaign_name"] == selected_campaign]
    ad_options = sorted(subset["ad_name"].dropna().unique().tolist())
    selected_ads = st.multiselect("Optional ad name filter", ad_options, default=[])

    if selected_ads:
        subset = subset[subset["ad_name"].isin(selected_ads)]

    if subset.empty:
        empty_state("No rows for selected campaign/ad filters.")
        return

    spend = subset["cost"].sum()
    impressions = subset["impressions"].sum()
    clicks = subset["clicks"].sum()
    apps = subset["applications"].sum()

    kpis = [
        ("Spend", fmt_currency(spend), None),
        ("Impressions", fmt_int(impressions), None),
        ("Clicks", fmt_int(clicks), None),
        ("CTR", fmt_pct(safe_divide(clicks, impressions)), None),
        ("Applications", fmt_int(apps), None),
        ("Cost / Application", fmt_currency_2(safe_divide(spend, apps)), None),
    ]
    render_kpi_row(kpis, cols_per_row=3)

    series = subset.groupby("date", as_index=False)[["cost", "clicks", "applications"]].sum()
    melt = series.melt(id_vars="date", var_name="metric", value_name="value")
    fig = px.line(
        melt,
        x="date",
        y="value",
        color="metric",
        title="Selected Campaign/Ad Time Series",
        color_discrete_map={"cost": COLOR_BLUE, "clicks": COLOR_GREEN, "applications": COLOR_SLATE},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Ad-Level Comparison")
    ad_tbl = (
        subset.groupby("ad_name", as_index=False)[["cost", "impressions", "clicks", "applications"]]
        .sum()
        .assign(
            ctr=lambda d: safe_divide(d["clicks"], d["impressions"]),
            cost_per_application=lambda d: safe_divide(d["cost"], d["applications"]),
        )
        .sort_values("applications", ascending=False)
    )
    st.dataframe(ad_tbl, use_container_width=True)

    st.markdown("### Creative / Theme Summary (Ad Topic)")
    topic_tbl = (
        subset.groupby("ad_topic", as_index=False)[["cost", "clicks", "applications"]]
        .sum()
        .assign(cost_per_application=lambda d: safe_divide(d["cost"], d["applications"]))
        .sort_values("applications", ascending=False)
    )
    st.dataframe(topic_tbl, use_container_width=True)

    csv_bytes = convert_df_to_csv(subset)
    st.download_button(
        "Download filtered campaign/ad data as CSV",
        data=csv_bytes,
        file_name="campaign_creative_drilldown.csv",
        mime="text/csv",
    )


def render_data_explorer(campaign_df: pd.DataFrame, ga4_df: pd.DataFrame, lp_df: pd.DataFrame) -> None:
    """Render data explorer page for ad-hoc exploration and export."""
    st.subheader("Data Explorer")
    st.caption("Select a dataset, inspect rows, search text, and download filtered exports.")

    dataset_name = st.selectbox("Dataset selector", [SHEET_CAMPAIGN, SHEET_GA4, SHEET_LP])

    if dataset_name == SHEET_CAMPAIGN:
        df = campaign_df.copy()
    elif dataset_name == SHEET_GA4:
        df = ga4_df.copy()
    else:
        df = lp_df.copy()

    if df.empty:
        empty_state("The selected dataset has no rows under current filters.")
        return

    all_columns = df.columns.tolist()
    selected_cols = st.multiselect("Columns", all_columns, default=all_columns)
    view = df[selected_cols].copy() if selected_cols else df.copy()

    search_text = st.text_input("Search (case-insensitive, across selected columns)", "").strip()
    if search_text:
        mask = pd.Series(False, index=view.index)
        for col in view.columns:
            mask = mask | view[col].astype(str).str.contains(search_text, case=False, na=False)
        view = view[mask]

    st.write(f"Row count: **{len(view):,}**")
    st.dataframe(view, use_container_width=True, height=450)

    st.download_button(
        "Download filtered dataset as CSV",
        data=convert_df_to_csv(view),
        file_name=f"{dataset_name.lower().replace(' ', '_')}_filtered.csv",
        mime="text/csv",
    )


# -------------------------------
# App orchestration
# -------------------------------
def main() -> None:
    """Entry point for Streamlit app."""
    st.set_page_config(
        page_title="HCZ Marketing Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar: navigation + relevant filters.
    with st.sidebar:
        st.title("HCZ Dashboard")
        page = st.radio("Section", PAGE_OPTIONS)
        data_source = st.selectbox(
            "Data source",
            options=["google_sheets", "local_file"],
            index=0 if DATA_SOURCE_DEFAULT == "google_sheets" else 1,
            help="Default is Google Sheets. Local file is useful for development fallback.",
        )

    # Data loading with Google Sheets as primary source.
    try:
        if data_source == "google_sheets":
            if "google_sheet_id" not in st.secrets:
                raise ValueError(
                    "Missing `google_sheet_id` in Streamlit secrets. "
                    "Add it to .streamlit/secrets.toml or deployment secrets."
                )
            raw = load_all_datasets_from_gsheet(
                str(st.secrets["google_sheet_id"]),
                GSHEET_WORKSHEET_MAP,
            )
        else:
            file_path = Path(EXCEL_FILE)
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Could not find local file: {EXCEL_FILE}. Place it next to app.py."
                )
            raw = load_local_data(str(file_path))
    except Exception as exc:
        if data_source == "google_sheets" and ALLOW_LOCAL_FALLBACK:
            file_path = Path(EXCEL_FILE)
            if file_path.exists():
                st.warning(
                    "Google Sheets load failed; using local file fallback for this session. "
                    f"Error details: {exc}"
                )
                raw = load_local_data(str(file_path))
            else:
                st.error(
                    "Unable to load data from Google Sheets and no local fallback file was found. "
                    "Check Streamlit secrets/service account permissions and workbook sharing."
                )
                st.exception(exc)
                st.stop()
        else:
            st.error(
                "Unable to load dashboard data. Verify Google Sheets secrets/service account access "
                "or switch data source to local_file."
            )
            st.exception(exc)
            st.stop()

    campaign_df = clean_campaign_data(raw.get(SHEET_CAMPAIGN, pd.DataFrame()))
    ga4_df = clean_ga4_data(raw.get(SHEET_GA4, pd.DataFrame()))
    lp_df = clean_lp_data(raw.get(SHEET_LP, pd.DataFrame()))

    # Determine global date limits across all date columns.
    date_series = pd.concat(
        [
            campaign_df.get("date", pd.Series(dtype="datetime64[ns]")).dropna(),
            ga4_df.get("date", pd.Series(dtype="datetime64[ns]")).dropna(),
            lp_df.get("week_start", pd.Series(dtype="datetime64[ns]")).dropna(),
        ],
        ignore_index=True,
    )
    min_date = date_series.min() if not date_series.empty else pd.Timestamp("today").normalize()
    max_date = date_series.max() if not date_series.empty else pd.Timestamp("today").normalize()

    with st.sidebar:
        if st.button("Reset filters"):
            for key in list(st.session_state.keys()):
                if key.startswith("flt_"):
                    del st.session_state[key]
            st.rerun()

        st.markdown("---")
        st.markdown("### Global Filters")

        default_start = pd.to_datetime(st.session_state.get("flt_start_date", min_date)).date()
        default_end = pd.to_datetime(st.session_state.get("flt_end_date", max_date)).date()
        date_range = st.date_input(
            "Date range",
            value=(default_start, default_end),
            min_value=min_date.date(),
            max_value=max_date.date(),
            key="flt_date_range",
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
        else:
            start_date = pd.to_datetime(default_start)
            end_date = pd.to_datetime(default_end)

        # Use combined options to support overview filters spanning datasets.
        year_opts = sorted(
            pd.concat(
                [
                    campaign_df.get("year", pd.Series(dtype=float)),
                    ga4_df.get("year", pd.Series(dtype=float)),
                ]
            )
            .dropna()
            .astype(int)
            .astype(str)
            .unique()
            .tolist()
        )
        month_opts = sorted(
            pd.concat([campaign_df.get("month", pd.Series(dtype=str)), ga4_df.get("month", pd.Series(dtype=str)), lp_df.get("month", pd.Series(dtype=str))])
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        show_campaign_filters = page in ["Executive Overview", "Paid Media Performance", "Campaign / Creative Drilldown", "Data Explorer"]
        show_ga4_filters = page in ["Executive Overview", "Website & GA4 Trends", "Data Explorer"]
        show_lp_filters = page in ["Executive Overview", "Landing Page Performance", "Data Explorer"]

        year = st.selectbox("Year", ["All", *year_opts], key="flt_year") if (show_campaign_filters or show_ga4_filters) else "All"
        month = st.selectbox("Month", ["All", *month_opts], key="flt_month") if (show_campaign_filters or show_ga4_filters or show_lp_filters) else "All"

        objective = (
            st.selectbox("Objective", build_filter_options(campaign_df, "objective"), key="flt_objective")
            if show_campaign_filters
            else "All"
        )
        platform = (
            st.selectbox("Platform", build_filter_options(campaign_df, "platform"), key="flt_platform")
            if show_campaign_filters
            else "All"
        )
        ad_topic = (
            st.selectbox("Ad Topic", build_filter_options(campaign_df, "ad_topic"), key="flt_ad_topic")
            if show_campaign_filters
            else "All"
        )

        landing_page = (
            st.selectbox("Landing Page", build_filter_options(lp_df, "landing_page"), key="flt_landing_page")
            if show_lp_filters
            else "All"
        )
        source = (
            st.selectbox("Source", build_filter_options(lp_df, "source"), key="flt_source")
            if show_lp_filters
            else "All"
        )
        medium = (
            st.selectbox("Medium", build_filter_options(lp_df, "medium"), key="flt_medium")
            if show_lp_filters
            else "All"
        )
        device = (
            st.selectbox("Device", build_filter_options(lp_df, "device"), key="flt_device")
            if show_lp_filters
            else "All"
        )

    # Apply filters by dataset.
    filt_campaign = apply_filters(
        campaign_df,
        date_col="date",
        start_date=start_date,
        end_date=end_date,
        year=year,
        month=month,
        objective=objective,
        platform=platform,
        ad_topic=ad_topic,
    )

    filt_ga4 = apply_filters(
        ga4_df,
        date_col="date",
        start_date=start_date,
        end_date=end_date,
        year=year,
        month=month,
    )

    filt_lp = apply_filters(
        lp_df,
        date_col="week_start",
        start_date=start_date,
        end_date=end_date,
        month=month,
        landing_page=landing_page,
        source=source,
        medium=medium,
        device=device,
    )

    active_filter_text = filter_summary_text(
        {
            "Date": f"{start_date.date()} to {end_date.date()}",
            "Year": year,
            "Month": month,
            "Objective": objective,
            "Platform": platform,
            "Ad Topic": ad_topic,
            "Landing page": landing_page,
            "Source": source,
            "Medium": medium,
            "Device": device,
        }
    )

    render_header(max_data_date=max_date, active_filter_text=active_filter_text)

    # Route to selected page.
    if page == "Executive Overview":
        render_executive_overview(filt_campaign, filt_ga4, filt_lp)
    elif page == "Paid Media Performance":
        render_paid_media_page(filt_campaign, start_date, end_date)
    elif page == "Website & GA4 Trends":
        render_ga4_page(filt_ga4)
    elif page == "Landing Page Performance":
        render_lp_page(filt_lp)
    elif page == "Campaign / Creative Drilldown":
        render_drilldown_page(filt_campaign)
    else:
        render_data_explorer(filt_campaign, filt_ga4, filt_lp)


if __name__ == "__main__":
    main()
