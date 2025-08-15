# main.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
import json

from util import scrape_table_from_url, analyze_question, parse_questions

app = FastAPI(title="TDS Data Analyst Agent")

def generate_key_from_question(question: str) -> str:
    """
    Placeholder function to map question text to evaluator keys.
    Replace this with your actual mapping logic.
    """
    return question.lower().replace(" ", "_")  # simple example

@app.post("/api/")
async def analyze(
    request: Optional[Request] = None,
    questions_txt: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = None
):
    """
    Accepts:
      - questions.txt file (multipart/form-data)
      - JSON body with {"request": "...questions text..."}
      - Optional additional files (CSV/images)
    Returns:
      - Dictionary keyed for evaluator
    """
    try:
        # Step 1: Read questions content
        questions_content = None

        if questions_txt:
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        elif request:
            try:
                body = await request.json()
                questions_content = body.get("request", "").strip()
            except json.JSONDecodeError:
                pass

        if not questions_content:
            return JSONResponse(content={"error": "No questions provided"}, status_code=200)

        # Step 2: Parse questions and URLs
        questions, urls = parse_questions(questions_content)
        if not questions:
            return JSONResponse(content={"error": "No valid questions found"}, status_code=200)

        # Step 3: Optional uploaded files
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
                    uploaded_data[f.filename] = content  # raw bytes for images, etc.

        # Step 4: Scrape URLs
        dataframes = {}
        for url in urls:
            try:
                dataframes[url] = scrape_table_from_url(url)
            except Exception:
                dataframes[url] = None

        # Include uploaded CSVs
        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # Step 5: Compute answers in evaluator-compatible dict
        answers_dict = {}
        for q in questions:
            try:
                key = generate_key_from_question(q)
                ans = analyze_question(q, dataframes, uploaded_data)
                answers_dict[key] = ans if ans else "No data"
            except Exception as e:
                answers_dict[key] = f"Error: {e}"

        return JSONResponse(content=answers_dict)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
