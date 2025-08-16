from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

from util import scrape_table_from_url, parse_questions, collect_all_keys

app = FastAPI(title="TDS Data Analyst Agent")
client = OpenAI()  # make sure OPENAI_API_KEY is set in your environment

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

async def ai_generate_key_and_value(question: str, dataframes: dict):
    """
    Sends the question + available data to OpenAI to infer:
    1) key name
    2) suggested value / computation
    """
    data_preview = {k: (df.head(5).to_dict(orient="records") if isinstance(df, pd.DataFrame) else str(df))
                    for k, df in dataframes.items()}
    prompt = f"""
You are a data analyst AI.
Given this question: "{question}"
And the following dataframes (sample 5 rows each): {data_preview}
Return a JSON with:
{{
  "key": "suggested_key_name_based_on_question",
  "value": "computed_or_suggested_value"
}}
Do not hardcode any key names.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content
    try:
        import json
        result = json.loads(content)
        return result.get("key"), result.get("value")
    except Exception:
        # fallback if AI fails
        return question.lower().replace(" ", "_"), "N/A"

@app.post("/api/")
async def analyze(request: Request, questions_txt: Optional[UploadFile] = File(None),
                  files: Optional[List[UploadFile]] = None):
    try:
        # Step 1: Read questions
        questions_content = ""
        body = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
            questions_content = body.get("request", "").strip()
        elif questions_txt:
            questions_content = (await questions_txt.read()).decode("utf-8").strip()

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

        # Step 4: AI-driven key & value generation
        answers_dict = {}
        answers_array = []
        for q in questions:
            key, value = await ai_generate_key_and_value(q, dataframes)
            answers_dict[key] = value
            answers_array.append(value)

        # Step 5: Extra keys from URL/table
        extra_keys = collect_all_keys(urls, dataframes)
        for key in extra_keys:
            if key not in answers_dict:
                answers_dict[key] = "No direct question, extracted from URL/table"

        # Step 6: Scatterplot
        scatter_df, rank_col, peak_col = find_rank_peak_df(dataframes)
        if scatter_df is not None:
            img = generate_scatterplot(scatter_df, rank_col, peak_col)
            answers_dict["scatterplot_rank_peak"] = img
            answers_array.append(img)

        return JSONResponse({"dict": answers_dict, "array": answers_array})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
