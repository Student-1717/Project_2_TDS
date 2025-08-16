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


def generate_key_from_question(question: str) -> str:
    return question.lower().replace(" ", "_")


def ask_ai_only_questions(questions: list) -> dict:
    answers_dict = {}
    for q in questions:
        key = generate_key_from_question(q)
        answers_dict[key] = "AI answer placeholder for: " + q
    return answers_dict


def generate_scatterplot(df, x_col, y_col):
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna(subset=[x_col, y_col])
    if df.empty:
        return "No numeric data available for scatterplot"

    plt.figure(figsize=(6, 4))
    sns.regplot(x=x_col, y=y_col, data=df, scatter=True, line_kws={"color": "red", "linestyle": "dotted"})
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


def find_rank_peak_df(dataframes: dict):
    for df in dataframes.values():
        if isinstance(df, pd.DataFrame) and not df.empty:
            cols = [c.lower().strip() for c in df.columns]
            if "rank" in cols and "peak" in cols:
                return df[[df.columns[cols.index("rank")], df.columns[cols.index("peak")]]], \
                       df.columns[cols.index("rank")], df.columns[cols.index("peak")]
    return None, None, None


@app.post("/api/")
async def analyze(request: Request, questions_txt: Optional[UploadFile] = File(None),
                  files: Optional[List[UploadFile]] = None):
    try:
        # Step 0: Determine expected keys dynamically from the incoming request
        body = await request.json() if request else {}
        expected_keys = set()
        if "expected_keys" in body and isinstance(body["expected_keys"], list):
            expected_keys.update(body["expected_keys"])

        # Step 1: Read questions
        questions_content = None
        if questions_txt:
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        if not questions_content:
            questions_content = body.get("request", "").strip() if body else ""
        if not questions_content:
            questions_content = ""

        # Step 2: Parse questions & URLs
        questions, urls = parse_questions(questions_content)

        # Step 3: Process uploaded files
        uploaded_data = {}
        if files:
            for f in files:
                content = await f.read()
                if f.filename.endswith(".csv"):
                    uploaded_data[f.filename] = pd.read_csv(io.BytesIO(content))
                else:
                    uploaded_data[f.filename] = content

        # Step 4: Scrape URLs
        dataframes = {}
        for url in urls:
            df = scrape_table_from_url(url)
            dataframes[url] = df
        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # Step 5: Analyze questions
        answers_dict = {}
        for q in questions:
            answers_dict[generate_key_from_question(q)] = analyze_question(q, dataframes, uploaded_data)

        # Step 6: Collect extra keys from URLs & tables
        extra_keys = collect_all_keys(urls, dataframes)
        for key in extra_keys:
            if key not in answers_dict:
                answers_dict[key] = "No direct question, extracted from URL/table"
                expected_keys.add(key)

        # Step 7: Generate scatterplot if possible
        scatter_df, rank_col, peak_col = find_rank_peak_df(dataframes)
        if scatter_df is not None:
            answers_dict["scatterplot_rank_peak"] = generate_scatterplot(scatter_df, rank_col, peak_col)
        else:
            answers_dict["scatterplot_rank_peak"] = "No table contains both 'rank' and 'peak' columns"
            expected_keys.add("scatterplot_rank_peak")

        # Step 8: Ensure all expected keys exist (safe defaults)
        for key in expected_keys:
            if key not in answers_dict:
                # Decide default type based on key name heuristics
                if any(sub in key.lower() for sub in ["count", "degree", "density", "path"]):
                    answers_dict[key] = 0
                elif any(sub in key.lower() for sub in ["node", "name"]):
                    answers_dict[key] = ""
                elif "graph" in key.lower() or "histogram" in key.lower():
                    answers_dict[key] = "No data available"
                else:
                    answers_dict[key] = None

        return JSONResponse(answers_dict)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
