# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import io
import os

# Import your util functions
from util import scrape_table_from_url, analyze_question, parse_questions

app = FastAPI(title="TDS Data Analyst Agent")

class AnalyzeRequest(BaseModel):
    questions_txt: str  # can be raw text or "file://<path>"

@app.post("/api/")
async def analyze(
    request: AnalyzeRequest,
    files: Optional[List[UploadFile]] = File(None)
):
    try:
        questions_content = request.questions_txt.strip()

        # If questions_txt is a file path, read the file
        if questions_content.startswith("file://"):
            file_path = questions_content[len("file://"):]
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    questions_content = f.read().strip()
            else:
                return JSONResponse(content=["File not found"], status_code=400)

        if not questions_content:
            return JSONResponse(content=["No data"], status_code=200)

        # Parse questions and URLs
        questions, urls = parse_questions(questions_content)
        if not questions:
            return JSONResponse(content=["No data"], status_code=200)

        # Optional: read uploaded files into dict
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
                    uploaded_data[f.filename] = content  # keep raw bytes for images etc.

        # Scrape each URL into DataFrames
        dataframes = {}
        for url in urls:
            try:
                dataframes[url] = scrape_table_from_url(url)
            except Exception:
                dataframes[url] = None

        # Include uploaded CSVs in dataframes
        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # Compute answers for each question
        answers = []
        for q in questions:
            try:
                ans = analyze_question(q, dataframes, uploaded_data)
                if not ans:
                    ans = "No data"
                answers.append(ans)
            except Exception as e:
                answers.append(f"Error: {e}")

        return JSONResponse(content=answers)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
