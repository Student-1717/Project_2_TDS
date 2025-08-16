import io
import json
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
from openai import OpenAI
from util import scrape_table_from_url, parse_questions

app = FastAPI(title="TDS Data Analyst Agent")
client = OpenAI()  # make sure OPENAI_API_KEY is set

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

def compute_local_value(key, dataframes):
    """
    Example local computations for small CSVs. 
    Add heuristics for other types dynamically if needed.
    """
    for df in dataframes.values():
        if isinstance(df, pd.DataFrame):
            # Example: numeric columns
            if key.lower() in df.columns.str.lower():
                return df[key].sum() if df[key].dtype in [int, float] else "N/A"
            # If key contains 'scatterplot', return plot
            if "plot" in key.lower() and len(df.columns) >= 2:
                return generate_scatterplot(df, df.columns[0], df.columns[1])
    return None  # fallback to AI

async def ai_generate_value_for_key(key: str, question: str, dataframes: dict):
    """
    Sends the question + available data to AI to generate a value for a given key.
    """
    data_preview = {k: (df.head(5).to_dict(orient="records") if isinstance(df, pd.DataFrame) else str(df))
                    for k, df in dataframes.items()}
    prompt = f"""
You are a data analyst AI.
Key: "{key}"
User Question: "{question}"
Available dataframes (sample 5 rows each): {data_preview}
Return a JSON with: {{"value": "computed_or_suggested_value"}}
If the key requires a plot, suggest 'scatterplot' or another type of plot.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        return result.get("value", "N/A")
    except Exception:
        return "N/A"

def extract_keys_from_questions(txt):
    """
    Extract all keys from questions YAML or text.
    Keys are enclosed in backticks, e.g. - `key_name`: description
    """
    import re
    pattern = r"- `([^`]+)`"
    return re.findall(pattern, txt)

@app.post("/api/")
async def analyze(request: Request,
                  questions_txt: Optional[UploadFile] = File(None),
                  files: Optional[List[UploadFile]] = None):
    try:
        # --- Step 1: Read questions ---
        questions_content = ""
        if questions_txt:
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        else:
            body = await request.json()
            questions_content = body.get("request", "").strip()

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

        # --- Step 4: Extract keys dynamically ---
        expected_keys = extract_keys_from_questions(questions_content)
        answers_dict = {key: "N/A" for key in expected_keys}

        # --- Step 5: Generate values for each key ---
        for key in expected_keys:
            # First try local computation
            local_val = compute_local_value(key, dataframes)
            if local_val is not None:
                answers_dict[key] = local_val
            else:
                # fallback to AI
                ai_val = await ai_generate_value_for_key(key, questions_content, dataframes)
                answers_dict[key] = ai_val

        return JSONResponse({"dict": answers_dict, "array": list(answers_dict.values())})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
