import pandas as pd
import requests
from io import BytesIO

def load_data_from_url(url):
    """Load data from URL dynamically."""
    response = requests.get(url)
    response.raise_for_status()
    content_type = response.headers.get('Content-Type', '')

    if 'html' in content_type:
        tables = pd.read_html(response.text)
        # return first table by default
        return tables[0] if tables else pd.DataFrame()
    elif 'csv' in content_type or url.endswith('.csv'):
        return pd.read_csv(BytesIO(response.content))
    elif 'excel' in content_type or url.endswith(('.xls', '.xlsx')):
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
