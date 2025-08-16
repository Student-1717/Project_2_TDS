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

# ---------------- Plotting ----------------
def plot_scatter_base64(df, x_col, y_col, regression=True, save_path=None):
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=x_col, y=y_col, data=df)
    if regression:
        sns.regplot(x=x_col, y=y_col, data=df, scatter=False, color='red')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80)
    if save_path:
        plt.savefig(save_path, dpi=80)  # save to disk
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

# ---------------- Analyzer ----------------
def analyze_question(question, dataframes, uploaded_data=None):
    question_lower = question.lower()
    df = None

    # Choose first numeric dataframe
    for df_candidate in dataframes.values():
        if not df_candidate.empty and any(np.issubdtype(dt, np.number) for dt in df_candidate.dtypes):
            df = df_candidate
            break

    if df is None:
        return "No data"

    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        return "No numeric data"

    # Basic computations
    if "how many" in question_lower:
        return int(df[numeric_cols[0]].count())

    elif "sum" in question_lower:
        return float(df[numeric_cols[0]].sum())

    elif "average" in question_lower or "mean" in question_lower:
        return float(df[numeric_cols[0]].mean())

    elif "correlation" in question_lower:
        if len(numeric_cols) >= 2:
            return float(df[numeric_cols[0]].corr(df[numeric_cols[1]]))
        else:
            return "Not enough numeric columns"

    elif "plot" in question_lower or "scatter" in question_lower:
        if len(numeric_cols) >= 2:
            # Return Base64 image
            return plot_scatter_base64(df, numeric_cols[0], numeric_cols[1])
        else:
            return "Not enough numeric columns"

    else:
        return "Not implemented"

# ---------------- Utility to collect all keys ----------------
def collect_all_keys(urls, dataframes):
    """
    Collect keys from all URLs and DataFrames for evaluator mapping
    """
    all_keys = set()
    for url in urls:
        df = dataframes.get(url, pd.DataFrame())
        keys = extract_keys_from_url_data(url, df)
        all_keys.update(keys)
    return list(all_keys)
