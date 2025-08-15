# util.py
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np

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

# ---------------- Plotting ----------------
def plot_scatter_base64(df, x_col, y_col, regression=True):
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=x_col, y=y_col, data=df)
    if regression:
        # Remove linestyle; regplot will draw a default regression line
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
    """
    Parse raw text of questions.txt into (questions_list, urls_list)
    """
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
    Dynamic analysis:
      - Counts numeric columns for 'how many'
      - Correlation between numeric columns for 'correlation'
      - Scatterplot for 'plot'
    """
    question_lower = question.lower()

    # Choose first available dataframe with numeric data
    df = None
    for df_candidate in dataframes.values():
        if not df_candidate.empty and any(np.issubdtype(dt, np.number) for dt in df_candidate.dtypes):
            df = df_candidate
            break

    if df is None:
        return "No data"

    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        return "No numeric data"

    # Dynamic answers
    if "how many" in question_lower:
        # Return total rows of first numeric column
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

    elif "plot" in question_lower:
        if len(numeric_cols) >= 2:
            return plot_scatter_base64(df, numeric_cols[0], numeric_cols[1])
        else:
            return "Not enough numeric columns"

    else:
        return "Not implemented"
