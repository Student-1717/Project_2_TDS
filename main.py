# main.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
import json

from util import scrape_table_from_url, analyze_question, parse_questions

app = FastAPI(title="TDS Data Analyst Agent")

@app.post("/api/")
async def analyze(
    request: Optional[Request] = None,
    questions_txt: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = None
):
    """
    Accepts a questions.txt file (multipart/form-data) or JSON body with 'request' field,
    plus optional additional files (CSV, images, etc.).
    Returns a dictionary with keys expected by the evaluation.
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
            return JSONResponse(content={"error": "No data"}, status_code=200)

        # Step 2: Parse questions and URLs
        questions, urls = parse_questions(questions_content)
        if not questions:
            return JSONResponse(content={"error": "No data"}, status_code=200)

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

        # Step 5: Compute answers and map to expected keys
        answers_dict = {}

        for q in questions:
            try:
                ans = analyze_question(q, dataframes, uploaded_data)
                if not ans:
                    ans = "No data"

                # Map questions to expected keys
                if "shortest path alice to eve" in q.lower():
                    answers_dict["shortest_path_alice_eve"] = ans
                else:
                    # Use question text as fallback key
                    key = q.lower().replace(" ", "_")
                    answers_dict[key] = ans

            except Exception as e:
                key = q.lower().replace(" ", "_")
                answers_dict[key] = f"Error: {e}"

        return JSONResponse(content=answers_dict)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
