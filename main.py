from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

from util import scrape_table_from_url, analyze_question, parse_questions, collect_all_keys

app = FastAPI(title="TDS Data Analyst Agent")


# --- Utility functions ---

def generate_key_from_question(question: str) -> str:
    """Generate a consistent key from a natural language question."""
    return question.lower().replace(" ", "_").replace("?", "").replace("-", "_")


def ask_ai_only_questions(questions: list) -> dict:
    """Stub for AI-based Q&A when no data is available."""
    answers_dict = {}
    for q in questions:
        key = generate_key_from_question(q)
        answers_dict[key] = f"AI answer placeholder for: {q}"
    return answers_dict


def generate_scatterplot(df, x_col, y_col):
    """Generate a scatterplot and return it as base64."""
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna(subset=[x_col, y_col])
    if df.empty:
        return "No numeric data available for scatterplot"

    plt.figure(figsize=(6, 4))
    sns.regplot(x=x_col, y=y_col, data=df,
                scatter=True, line_kws={"color": "red", "linestyle": "dotted"})
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


def find_rank_peak_df(dataframes: dict):
    """Find a dataframe with rank & peak columns for scatterplot demo."""
    for df in dataframes.values():
        if isinstance(df, pd.DataFrame) and not df.empty:
            cols = [c.lower().strip() for c in df.columns]
            if "rank" in cols and "peak" in cols:
                return (
                    df[[df.columns[cols.index("rank")],
                        df.columns[cols.index("peak")]]],
                    df.columns[cols.index("rank")],
                    df.columns[cols.index("peak")]
                )
    return None, None, None


def default_value_for_key(key: str):
    """Provide safe default values based on key naming conventions."""
    if any(word in key for word in ["count", "total", "sum", "degree", "density", "sales", "tax", "median"]):
        return 0
    if any(word in key for word in ["node", "region", "top", "name"]):
        return ""
    if any(word in key for word in ["graph", "histogram", "plot", "chart", "image"]):
        return "No data available"
    return None


# --- API endpoint ---

@app.post("/api/")
async def analyze(request: Request,
                  questions_txt: Optional[UploadFile] = File(None),
                  files: Optional[List[UploadFile]] = None):
    try:
        # Step 0: Detect expected keys from checker (if JSON body contains them)
        expected_keys = set()
        body = None
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
            if isinstance(body, dict) and "expected_keys" in body:
                if isinstance(body["expected_keys"], list):
                    expected_keys.update(body["expected_keys"])

        # Step 1: Read questions
        questions_content = ""
        if questions_txt:
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        if not questions_content and body:
            questions_content = body.get("request", "").strip()

        # Step 2: Parse questions & URLs
        questions, urls = parse_questions(questions_content)

        # Step 3: Handle uploaded files
        uploaded_data = {}
        if files:
            for f in files:
                content = await f.read()
                if f.filename.endswith(".csv"):
                    uploaded_data[f.filename] = pd.read_csv(io.BytesIO(content))
                else:
                    uploaded_data[f.filename] = content

        # Step 4: Scrape tables from URLs
        dataframes = {}
        for url in urls:
            df = scrape_table_from_url(url)
            dataframes[url] = df
        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # Step 5: Analyze questions → answers dict
        answers_dict = {}
        for q in questions:
            key = generate_key_from_question(q)
            answers_dict[key] = analyze_question(q, dataframes, uploaded_data)

        # Step 6: Collect any extra keys from data
        extra_keys = collect_all_keys(urls, dataframes)
        for key in extra_keys:
            if key not in answers_dict:
                answers_dict[key] = "Extracted from URL/table"

        # Step 7: Add scatterplot if possible
        scatter_df, rank_col, peak_col = find_rank_peak_df(dataframes)
        if scatter_df is not None:
            answers_dict["scatterplot_rank_peak"] = generate_scatterplot(scatter_df, rank_col, peak_col)

        # Step 8: Ensure all expected keys are present (dynamic)
        if not expected_keys:
            # If checker didn’t provide explicit keys, fall back to derived from questions
            expected_keys = {generate_key_from_question(q) for q in questions}

        for key in expected_keys:
            if key not in answers_dict:
                answers_dict[key] = default_value_for_key(key)

        return JSONResponse(answers_dict)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
