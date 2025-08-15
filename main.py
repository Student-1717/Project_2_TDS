# main.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
import json
import os
import openai

from util import scrape_table_from_url, parse_questions

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
    if isinstance(df, pd.DataFrame) and not df.empty:
        for col in df.columns:
            if "rank" in str(col).lower() or "title" in str(col).lower():
                keys.add(str(col).lower().replace(" ", "_"))

    return list(keys)

def ask_ai_only_questions(questions: list) -> dict:
    """
    Send only the questions list to the AI assistant and return a dictionary keyed
    for the evaluator.
    """
    answers_dict = {}
    for q in questions:
        key = generate_key_from_question(q)
        try:
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
            answer = response.choices[0].message["content"].strip()
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
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        else:
            try:
                body = await request.json()
                questions_content = body.get("request", "").strip()
            except Exception:
                raw_body = await request.body()
                if raw_body:
                    questions_content = raw_body.decode("utf-8").strip()

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

        # Step 3: Ask AI only questions
        answers_dict = ask_ai_only_questions(questions)

        return JSONResponse(content=answers_dict)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
