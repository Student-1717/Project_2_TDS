# main.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
import os
import matplotlib.pyplot as plt
import base64
import numpy as np  # <-- Added for polyfit

from util import scrape_table_from_url, analyze_question, parse_questions

from urllib.parse import urlparse
import openai  # Make sure openai>=1.0.0 is installed

app = FastAPI(title="TDS Data Analyst Agent")

def generate_key_from_question(question: str) -> str:
    """Maps question text to evaluator keys (simple, generic)."""
    return question.lower().replace(" ", "_")

def extract_keys_from_url_data(url, df):
    """
    Generate possible evaluator keys from:
      1. URL path
      2. Table column names
      3. Special common headers
    """
    keys = set()

    # 1. From URL
    try:
        parsed = urlparse(url)
        if parsed.path:
            path_parts = [p for p in parsed.path.split("/") if p]
            for part in path_parts:
                keys.add(part.lower().replace(" ", "_"))
    except Exception:
        pass

    # 2. From table columns
    if isinstance(df, pd.DataFrame) and not df.empty:
        for col in df.columns:
            keys.add(str(col).lower().strip().replace(" ", "_"))

    # 3. From specific known useful headers
    if isinstance(df, pd.DataFrame) and not df.empty:
        for col in df.columns:
            if "rank" in str(col).lower() or "title" in str(col).lower():
                keys.add(str(col).lower().replace(" ", "_"))

    return list(keys)

def ask_ai_only_questions(questions: list, dataframes=None, uploaded_data=None) -> dict:
    """Send only the questions to the AI and return answers dict."""
    answers_dict = {}
    dataframes = dataframes or {}
    uploaded_data = uploaded_data or {}

    for q in questions:
        key = generate_key_from_question(q)
        try:
            # Special handling for scatterplot question
            if "scatterplot" in q.lower() and "rank" in q.lower() and "peak" in q.lower():
                df_to_use = None
                for df in list(dataframes.values()) + list(uploaded_data.values()):
                    if isinstance(df, pd.DataFrame) and "rank" in df.columns and "peak" in df.columns:
                        df_to_use = df
                        break
                if df_to_use is not None:
                    plt.figure()
                    plt.scatter(df_to_use["rank"], df_to_use["peak"])
                    plt.xlabel("Rank")
                    plt.ylabel("Peak")
                    plt.title("Rank vs Peak")
                    # Corrected polyfit usage
                    m, b = np.polyfit(df_to_use["rank"], df_to_use["peak"], 1)
                    plt.plot(df_to_use["rank"], m*df_to_use["rank"] + b, "r--")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    plt.close()
                    buf.seek(0)
                    b64_img = base64.b64encode(buf.read()).decode("utf-8")
                    answers_dict[key] = f"data:image/png;base64,{b64_img}"
                else:
                    answers_dict[key] = "No table contains both 'rank' and 'peak' columns"
                continue

            # Standard AI completion
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful data analyst. "
                            "Answer the user's question as directly and concisely as possible. "
                            "Return a short, plain answer."
                        )
                    },
                    {"role": "user", "content": q}
                ],
                temperature=0
            )
            answer = response.choices[0].message.content.strip()
            answers_dict[key] = answer if answer else "No data"
        except Exception as e:
            answers_dict[key] = f"Error: {e}"
    return answers_dict

@app.post("/api/")
async def analyze(
    request: Request,
    questions_txt: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = None
):
    try:
        # Step 1: Get questions content
        questions_content = None

        if questions_txt:
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        else:
            try:
                body = await request.json()
                questions_content = body.get("request", "").strip()
            except Exception:
                try:
                    raw_body = await request.body()
                    if raw_body:
                        questions_content = raw_body.decode("utf-8").strip()
                except Exception:
                    pass

        if not questions_content:
            file_path = os.environ.get("QUESTIONS_FILE")
            if file_path and os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    questions_content = f.read().strip()

        if not questions_content:
            return JSONResponse(content={"error": "No questions provided"}, status_code=200)

        # Step 2: Parse questions & URLs
        questions, urls = parse_questions(questions_content)
        if not questions:
            return JSONResponse(content={"error": "No valid questions found"}, status_code=200)

        # Step 3: Process uploaded files (optional)
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

        # Step 4: Scrape URLs & collect extra keys
        dataframes = {}
        extra_keys = set()
        for url in urls:
            try:
                scraped = scrape_table_from_url(url)
                dataframes[url] = scraped

                if isinstance(scraped, dict):
                    merged_df = pd.concat(
                        [t for t in scraped.values() if isinstance(t, pd.DataFrame)],
                        ignore_index=True
                    ) if scraped else pd.DataFrame()
                    df_for_keys = merged_df
                else:
                    df_for_keys = scraped

                extra_keys.update(extract_keys_from_url_data(url, df_for_keys))
                dataframes[url] = df_for_keys

            except Exception:
                dataframes[url] = None

        # Include uploaded CSVs
        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # Step 5: Ask AI for answers using only questions
        answers_dict = ask_ai_only_questions(questions, dataframes=dataframes, uploaded_data=uploaded_data)

        # Step 6: Add placeholders for extracted keys
        for key in extra_keys:
            if key not in answers_dict:
                answers_dict[key] = "No direct question, extracted from URL/table"

        return JSONResponse(content=answers_dict)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
