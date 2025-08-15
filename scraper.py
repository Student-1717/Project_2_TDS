# scraper.py
import pandas as pd
import requests
from io import BytesIO

def scrape_table_from_url(url):
    """Scrape all HTML tables from a given URL and return as dict or merged DataFrame."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")

    # If HTML, try reading all tables
    if "html" in content_type or url.endswith(".html") or url.endswith("/"):
        tables = pd.read_html(response.text)
        if not tables:
            return pd.DataFrame()
        if len(tables) == 1:
            return tables[0]
        else:
            # Return merged DataFrame for analysis
            merged = pd.concat(tables, ignore_index=True)
            return merged

    # CSV
    elif "csv" in content_type or url.endswith(".csv"):
        return pd.read_csv(BytesIO(response.content))

    # Excel
    elif "excel" in content_type or url.endswith((".xls", ".xlsx")):
        return pd.read_excel(BytesIO(response.content))

    else:
        raise ValueError(f"Unsupported URL data type: {content_type}")
