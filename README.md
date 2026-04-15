# HCZ Reporting Dashboard

Streamlit dashboard for Harlem Children’s Zone marketing reporting across paid media, GA4, and landing pages.

## Google Sheets configuration (primary data source)

The app defaults to loading all datasets directly from Google Sheets using a Google service account.

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure secrets

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
2. Fill in real service account credentials under `[google_service_account]`.
3. Set `google_sheet_id` to your workbook id.

### 3) Share the Google Sheet with the service account

In Google Sheets, share the workbook with the `client_email` from your service account credentials (Viewer access is enough for read-only reporting).

### 4) Run the app

```bash
streamlit run app.py
```

## Data source behavior

- Default source: `google_sheets`
- Optional fallback source: `local_file` (`HCZ - Master Data File.xlsx` beside `app.py`)
- If Google Sheets access fails and local fallback is enabled, the app shows a warning and uses local data when available.
