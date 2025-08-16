from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import openai

from util import scrape_table_from_url, analyze_question, parse_questions

app = FastAPI(title="TDS Data Analyst Agent")

# =========================
# Local plotting utility
# =========================
def generate_scatterplot(df, x_col, y_col):
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna(subset=[x_col, y_col])
    if df.empty:
        return "data:image/png;base64,"

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

# =========================
# AI utilities
# =========================
def ai_infer_key(question: str) -> str:
    prompt = f"Suggest the most probable JSON key name (single string) for this question: \"{question}\""
    resp = openai.ChatCompletion.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10
    )
    return resp.choices[0].message['content'].strip()

def ai_analyze_question(question: str, dataframes: dict, raw_files: dict):
    return analyze_question(question, dataframes, raw_files)

# =========================
# API endpoint (array only)
# =========================
@app.post("/api/")
async def analyze(request: Request, questions_txt: Optional[UploadFile] = File(None),
                  files: Optional[List[UploadFile]] = None):
    try:
        # Step 1: Read questions
        questions_content = ""
        if questions_txt:
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        elif request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
            questions_content = body.get("request", "").strip() if "request" in body else ""

        questions, urls = parse_questions(questions_content)

        # Step 2: Process uploaded files
        uploaded_data = {}
        if files:
            for f in files:
                content = await f.read()
                if f.filename.endswith(".csv"):
                    uploaded_data[f.filename] = pd.read_csv(io.BytesIO(content))
                else:
                    uploaded_data[f.filename] = content

        # Step 3: Scrape URLs
        dataframes = {url: scrape_table_from_url(url) for url in urls}
        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # Step 4: AI-driven analysis
        answers_array = []
        for q in questions:
            _ = ai_infer_key(q)  # key is inferred but not returned
            value = ai_analyze_question(q, dataframes, uploaded_data)
            answers_array.append(value)

        # Step 5: Scatterplot
        scatter_df, rank_col, peak_col = find_rank_peak_df(dataframes)
        if scatter_df is not None:
            scatter_img = generate_scatterplot(scatter_df, rank_col, peak_col)
            answers_array.append(scatter_img)

        return JSONResponse(answers_array)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
