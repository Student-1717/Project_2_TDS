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

        # Use StringIO to safely parse literal HTML
        tables = pd.read_html(io.StringIO(response.text))
        for table in tables:
            if not table.empty:
                # Convert numeric-like columns safely
                for col in table.columns:
                    table[col] = pd.to_numeric(table[col], errors='ignore')
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
      3. Known useful headers like 'Rank', 'Title', etc.
    """
    keys = set()

    # 1. From URL path
    try:
        parsed = urlparse(url)
        if parsed.path:
            path_parts = [p for p in parsed.path.split("/") if p]
            keys.update([p.lower().replace(" ", "_") for p in path_parts])
    except Exception:
        pass

    # 2. From table columns
    if isinstance(df, pd.DataFrame) and not df.empty:
        keys.update([str(col).lower().strip().replace(" ", "_") for col in df.columns])

    # 3. From specific useful headers
    for col in df.columns:
        col_lower = str(col).lower()
        if "rank" in col_lower or "title" in col_lower or "peak" in col_lower:
            keys.add(col_lower.replace(" ", "_"))

    return list(keys)

# ---------------- Plotting ----------------
def plot_scatter_base64(df, x_col, y_col, regression=True, save_path=None):
    """Create scatterplot with optional regression line and return base64 PNG."""
    # Ensure numeric
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna(subset=[x_col, y_col])

    if df.empty:
        return "No numeric data available for scatterplot"

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=x_col, y=y_col, data=df)
    if regression:
        sns.regplot(x=x_col, y=y_col, data=df, scatter=False, color='red', line_kws={"linestyle":"dotted"})
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80)
    if save_path:
        plt.savefig(save_path, dpi=80)
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

# ---------------- Question Parser ----------------
def parse_questions(content):
    """Separate questions and URLs from uploaded text."""
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
    """
    Analyze a single question based on numeric columns in dataframes.
    Handles counts, sums, averages, correlations, and scatterplots.
    """
    question_lower = question.lower()
    df = None

    # Pick first numeric dataframe
    for df_candidate in dataframes.values():
        if not df_candidate.empty and any(np.issubdtype(dt, np.number) for dt in df_candidate.dtypes):
            df = df_candidate
            break

    if df is None:
        return "No data available"

    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        return "No numeric columns available"

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
            return "Not enough numeric columns for correlation"

    elif "plot" in question_lower or "scatter" in question_lower:
        if len(numeric_cols) >= 2:
            return plot_scatter_base64(df, numeric_cols[0], numeric_cols[1])
        else:
            return "Not enough numeric columns for scatterplot"

    else:
        return "Question type not implemented"

# ---------------- Utility to collect all keys ----------------
def collect_all_keys(urls, dataframes):
    """
    Collect keys from all URLs and DataFrames for evaluator mapping.
    """
    all_keys = set()
    for url in urls:
        df = dataframes.get(url, pd.DataFrame())
        keys = extract_keys_from_url_data(url, df)
        all_keys.update(keys)
    return list(all_keys)
