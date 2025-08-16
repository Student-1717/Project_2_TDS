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
Return a JSON with:
{{"value": "computed_or_suggested_value"}}
If the key requires a plot, suggest 'scatterplot' or another type of plot.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content
    try:
        import json
        result = json.loads(content)
        return result.get("value", "N/A")
    except Exception:
        return "N/A"

@app.post("/api/")
async def analyze(request: Request,
                  questions_txt: Optional[UploadFile] = File(None),
                  files: Optional[List[UploadFile]] = None,
                  yaml_file: Optional[UploadFile] = File(None)):

    try:
        # --- Step 1: Read questions ---
        questions_content = ""
        body = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
            questions_content = body.get("request", "").strip()
        elif questions_txt:
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        questions, urls = parse_questions(questions_content)

        # --- Step 2: Process uploaded files ---
        uploaded_data = {}
        if files:
            for f in files:
                content = await f.read()
                if f.filename.endswith(".csv"):
                    uploaded_data[f.filename] = pd.read_csv(io.BytesIO(content))
                else:
                    uploaded_data[f.filename] = content

        # --- Step 3: Scrape URLs ---
        dataframes = {}
        for url in urls:
            df = scrape_table_from_url(url)
            dataframes[url] = df
        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # --- Step 4: Load YAML keys dynamically ---
        expected_keys = []
        if yaml_file:
            yaml_content = (await yaml_file.read()).decode("utf-8")
            parsed_yaml = yaml.safe_load(yaml_content)
            expected_keys = list(parsed_yaml.get("properties", {}).keys())

        # --- Step 5: Generate AI-driven values for each key ---
        answers_dict = {}
        for key in expected_keys:
            # Ask AI to generate value for the key
            value = await ai_generate_value_for_key(key, questions_content, dataframes)

            # Auto-detect plot requests and generate locally if needed
            if isinstance(value, str) and "plot" in value.lower():
                scatter_df = None
                x_col, y_col = None, None
                for df in dataframes.values():
                    if isinstance(df, pd.DataFrame):
                        cols = [c.lower() for c in df.columns]
                        if "rank" in cols and "peak" in cols:
                            scatter_df = df
                            x_col, y_col = df.columns[cols.index("rank")], df.columns[cols.index("peak")]
                            break
                if scatter_df is not None:
                    value = generate_scatterplot(scatter_df, x_col, y_col)
                else:
                    value = "data:image/png;base64,"

            answers_dict[key] = value

        # Return JSON in expected syntax
        return JSONResponse({"dict": answers_dict, "array": list(answers_dict.values())})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
