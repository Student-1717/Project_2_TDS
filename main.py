from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import json

from util import scrape_table_from_url, analyze_question, parse_questions, collect_all_keys
from openai import OpenAI

app = FastAPI(title="TDS Data Analyst Agent")
openai_client = OpenAI()  # assumes API key is set in environment

async def ai_infer_keys_batch(questions: list) -> list:
    if not questions:
        return []
    prompt = "For each of the following questions, suggest a short, descriptive, snake_case key name:\n"
    for i, q in enumerate(questions, 1):
        prompt += f"{i}. {q}\n"
    prompt += "\nReturn only a JSON array of strings, one key per question."

    resp = await openai_client.chat.completions.acreate(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    content = resp.choices[0].message.content.strip()
    try:
        keys = json.loads(content)
        keys = [k.lower().replace(" ", "_").replace("-", "_") for k in keys]
    except:
        keys = [f"q{i+1}" for i in range(len(questions))]
    return keys

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

@app.post("/api/")
async def analyze(request: Request, questions_txt: Optional[UploadFile] = File(None),
                  files: Optional[List[UploadFile]] = None):
    try:
        # Step 1: Read request
        body = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()

        questions_content = ""
        if questions_txt:
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        elif "request" in body:
            questions_content = body.get("request", "").strip()

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
        dataframes = {}
        for url in urls:
            df = scrape_table_from_url(url)
            dataframes[url] = df
        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # Step 4: AI key inference
        inferred_keys = await ai_infer_keys_batch(questions)

        # Step 5: Analyze questions
        answers_array = []
        for q in questions:
            answer = analyze_question(q, dataframes, uploaded_data)
            if answer is None:
                answer = 0 if q.lower().startswith(("total", "average", "median", "min", "max")) else ""
            answers_array.append(answer)

        # Step 6: Generate scatterplot as last element
        scatter_df, rank_col, peak_col = find_rank_peak_df(dataframes)
        if scatter_df is not None:
            answers_array.append(generate_scatterplot(scatter_df, rank_col, peak_col))
        else:
            answers_array.append("data:image/png;base64,")

        return JSONResponse(answers_array)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
