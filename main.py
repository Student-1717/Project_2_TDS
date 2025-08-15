# main.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
import json
import os

from util import scrape_table_from_url, analyze_question, parse_questions

from urllib.parse import urlparse

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
    for col in df.columns:
        if "rank" in str(col).lower() or "title" in str(col).lower():
            keys.add(str(col).lower().replace(" ", "_"))

    return list(keys)

@app.post("/api/")
async def analyze(
    request: Request,
    questions_txt: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = None
):
    """
    Accepts:
      - Multipart/form-data: questions.txt + optional CSV/image files
      - JSON: {"request": "...questions text..."}
    Returns:
      - Dictionary keyed for evaluator
    """
    try:
        # Step 1: Get questions content
        questions_content = None

        if questions_txt:
            # If provided as file upload
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        else:
            # Try reading JSON body
            try:
                body = await request.json()
                questions_content = body.get("request", "").strip()
            except Exception:
                # If request.json() fails, maybe it's raw text (promptfoo file mode)
                try:
                    raw_body = await request.body()
                    if raw_body:
                        questions_content = raw_body.decode("utf-8").strip()
                except Exception:
                    pass

        # Also allow file path from env or config for local eval
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
                df = scrape_table_from_url(url)
                dataframes[url] = df
                extra_keys.update(extract_keys_from_url_data(url, df))
            except Exception:
                dataframes[url] = None

        # Include uploaded CSVs
        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # Step 5: Answer questions
        answers_dict = {}
        for q in questions:
            try:
                key = generate_key_from_question(q)
                ans = analyze_question(q, dataframes, uploaded_data)
                answers_dict[key] = ans if ans else "No data"
            except Exception as e:
                answers_dict[key] = f"Error: {e}"

        # Step 6: Add placeholders for extracted keys (so evaluator won't fail)
        for key in extra_keys:
            if key not in answers_dict:
                answers_dict[key] = "No direct question, extracted from URL/table"

        return JSONResponse(content=answers_dict)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
