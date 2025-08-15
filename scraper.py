# scraper.py
import pandas as pd
import requests
from io import BytesIO

def load_data_from_url(url):
    """Load data from URL dynamically. Returns a dict if multiple tables found."""
    response = requests.get(url)
    response.raise_for_status()
    content_type = response.headers.get('Content-Type', '').lower()

    if 'html' in content_type or url.lower().endswith(('.htm', '.html')):
        tables = pd.read_html(response.text)
        if tables:
            # Return all tables as a dict {table_1: df, table_2: df, ...}
            return {f"table_{i+1}": table for i, table in enumerate(tables)}
        else:
            return {}
    elif 'csv' in content_type or url.lower().endswith('.csv'):
        return pd.read_csv(BytesIO(response.content))
    elif 'excel' in content_type or url.lower().endswith(('.xls', '.xlsx')):
        return pd.read_excel(BytesIO(response.content))
    else:
        raise ValueError(f"Unsupported URL data type: {content_type}")

def load_data_from_file(file):
    """Load data from uploaded file (CSV, Excel, JSON)."""
    filename = file.filename.lower()
    if filename.endswith('.csv'):
        return pd.read_csv(file.file)
    elif filename.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file.file)
    elif filename.endswith('.json'):
        return pd.read_json(file.file)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
