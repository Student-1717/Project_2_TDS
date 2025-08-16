import yaml
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import re

from util import scrape_table_from_url, parse_questions

app = FastAPI(title="TDS Data Analyst Agent")
client = OpenAI()  # ensure OPENAI_API_KEY is set

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

async def ai_generate_value_for_key(key: str, question: str, dataframes: dict):
    data_preview = {k: (df.head(5).to_dict(orient="records") if isinstance(df, pd.DataFrame) else str(df))
                    for k, df in dataframes.items()}
    prompt = f"""
You are a data analyst AI.
Key: "{key}"
User Question: "{question}"
Available dataframes (sample 5 rows each): {data_preview}
Return a JSON with:
{{"value": "computed_or_suggested_value"}}
If the key requires a plot, suggest 'scatterplot' or another type of plot.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        import json
        result = json.loads(content)
        return result.get("value", "N/A")
    except Exception:
        return "N/A"

def extract_keys_from_questions(txt):
    pattern = r"- `([^`]+)`"
    return re.findall(pattern, txt)

@app.post("/api/")
async def analyze(request: Request,
                  questions_txt: Optional[UploadFile] = File(None),
                  files: Optional[List[UploadFile]] = None):

    try:
        # --- Step 1: Read questions ---
        questions_content = ""
        body = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
            questions_content = body.get("request", "").strip()
        elif questions_txt:
            questions_content = (await questions_txt.read()).decode("utf-8").strip()

        # Extract questions and URLs
        questions, urls = parse_questions(questions_content)

        # --- Step 2: Process uploaded files ---
        uploaded_data = {}
        if files:
            for f in files:
                content = await f.read()
                if f.filename.endswith(".csv"):
                    try:
                        uploaded_data[f.filename] = pd.read_csv(io.BytesIO(content))
                    except Exception:
                        uploaded_data[f.filename] = None
                else:
                    uploaded_data[f.filename] = content

        # --- Step 3: Scrape URLs ---
        dataframes = {}
        for url in urls:
            try:
                df = scrape_table_from_url(url)
                dataframes[url] = df
            except Exception:
                dataframes[url] = None

        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # --- Step 4: Extract keys from questions.txt ---
        expected_keys = extract_keys_from_questions(questions_content)
        answers_dict = {key: "N/A" for key in expected_keys}

        # --- Step 5: AI-driven values for each key ---
        for key in expected_keys:
            value = await ai_generate_value_for_key(key, questions_content, dataframes)

            # Detect plot requests and auto-generate scatterplots
            if isinstance(value, str) and "plot" in value.lower():
                scatter_done = False
                for df in dataframes.values():
                    if isinstance(df, pd.DataFrame):
                        numeric_cols = df.select_dtypes(include="number").columns
                        if len(numeric_cols) >= 2:
                            value = generate_scatterplot(df, numeric_cols[0], numeric_cols[1])
                            scatter_done = True
                            break
                if not scatter_done:
                    value = "data:image/png;base64,"

            answers_dict[key] = value

        # --- Step 6: Return response in promtfoo-compatible scheme ---
        response_payload = {
            "dict": answers_dict,
            "array": [answers_dict[k] for k in expected_keys]
        }

        return JSONResponse(response_payload)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
