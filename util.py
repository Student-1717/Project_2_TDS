import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np
from urllib.parse import urlparse

# ---------------- Scraper ----------------
def scrape_table_from_url(url):
    """Fetch tables from any URL and return the first non-empty DataFrame."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        for table in tables:
            if not table.empty:
                return table
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
    return pd.DataFrame()  # fallback empty

# ---------------- Key Extractor ----------------
def extract_keys_from_url_data(url, df):
    """
    Generate possible evaluator keys from:
      1. URL path
      2. Table column names
      3. 'Rank' / 'Title' style column headers
    """
    keys = set()

    # 1. From URL
    try:
        parsed = urlparse(url)
        if parsed.path:
            path_parts = [p for p in parsed.path.split("/") if p]
            for part in path_parts:
                keys.add(part.lower().replace(" ", "_"))
    except Exception:
        pass

    # 2. From table columns
    if isinstance(df, pd.DataFrame) and not df.empty:
        for col in df.columns:
            keys.add(str(col).lower().strip().replace(" ", "_"))

    # 3. From specific known useful headers
    for col in df.columns:
        if "rank" in str(col).lower() or "title" in str(col).lower():
            keys.add(str(col).lower().replace(" ", "_"))

    return list(keys)

# ---------------- Collect All Keys ----------------
def collect_all_keys(dataframes):
    """
    Collect keys for all DataFrames in the format:
      {url: [keys]}
    """
    all_keys = {}
    for url, df in dataframes.items():
        keys = extract_keys_from_url_data(url, df)

        # Add standard numeric keys
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            keys.add(f"{col}_count")
            keys.add(f"{col}_sum")
            keys.add(f"{col}_mean")
        # Add correlation keys if 2+ numeric columns
        if len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    keys.add(f"{numeric_cols[i]}_{numeric_cols[j]}_correlation")

        all_keys[url] = list(keys)
    return all_keys

# ---------------- Plotting ----------------
def plot_scatter_base64(df, x_col, y_col, regression=True):
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=x_col, y=y_col, data=df)
    if regression:
        sns.regplot(x=x_col, y=y_col, data=df, scatter=False, color='red')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80)
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

# ---------------- Question Parser ----------------
def parse_questions(content):
    questions = []
    urls = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("http"):  # treat as URL
            urls.append(line)
        else:
            questions.append(line)
    return questions, urls

# ---------------- Analyzer (Updated) ----------------
def analyze_question(question, dataframes, all_keys=None):
    """
    Analyze a question using available DataFrames and collected keys.
    - question: string
    - dataframes: {url: df}
    - all_keys: {url: [keys]}
    """
    question_lower = question.lower()

    # Try each DataFrame in order
    for url, df in dataframes.items():
        if df.empty:
            continue

        numeric_cols = df.select_dtypes(include=np.number).columns
        keys = set(all_keys.get(url, [])) if all_keys else set()

        # Map question keywords to numeric operations
        if "how many" in question_lower:
            for col in numeric_cols:
                key = f"{col}_count"
                if key in keys:
                    return int(df[col].count())

        elif "sum" in question_lower:
            for col in numeric_cols:
                key = f"{col}_sum"
                if key in keys:
                    return float(df[col].sum())

        elif "average" in question_lower or "mean" in question_lower:
            for col in numeric_cols:
                key = f"{col}_mean"
                if key in keys:
                    return float(df[col].mean())

        elif "correlation" in question_lower:
            if len(numeric_cols) >= 2:
                col1, col2 = numeric_cols[:2]
                key = f"{col1}_{col2}_correlation"
                if key in keys:
                    return float(df[col1].corr(df[col2]))
            return "Not enough numeric columns"

        elif "plot" in question_lower:
            if len(numeric_cols) >= 2:
                col1, col2 = numeric_cols[:2]
                return plot_scatter_base64(df, col1, col2)
            return "Not enough numeric columns"

        else:
            # Try matching any known key from all_keys
            for key in keys:
                if key in question_lower:
                    if key.endswith("_count"):
                        col = key[:-6]
                        return int(df[col].count())
                    elif key.endswith("_sum"):
                        col = key[:-4]
                        return float(df[col].sum())
                    elif key.endswith("_mean"):
                        col = key[:-5]
                        return float(df[col].mean())
                    elif "_correlation" in key:
                        col1, col2 = key.split("_")[:2]
                        return float(df[col1].corr(df[col2]))

    return "No matching data found"
