# main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import io
import base64

# Import your util functions
from util import scrape_table_from_url, analyze_question, parse_questions

app = FastAPI(title="TDS Data Analyst Agent")

class FileItem(BaseModel):
    filename: str
    content_base64: str  # Base64-encoded file content

class QuestionRequest(BaseModel):
    questions_txt: str
    files: Optional[List[FileItem]] = None

@app.post("/api/")
async def analyze(request: QuestionRequest):
    try:
        questions_content = request.questions_txt.strip()
        if not questions_content:
            return JSONResponse(content=["No data"], status_code=200)

        # Parse questions and URLs
        questions, urls = parse_questions(questions_content)
        if not questions:
            return JSONResponse(content=["No data"], status_code=200)

        # Optional: read uploaded files
        uploaded_data: Dict[str, Optional[pd.DataFrame]] = {}
        if request.files:
            for f in request.files:
                content_bytes = base64.b64decode(f.content_base64)
                if f.filename.endswith(".csv"):
                    try:
                        uploaded_data[f.filename] = pd.read_csv(io.BytesIO(content_bytes))
                    except Exception:
                        uploaded_data[f.filename] = None
                else:
                    uploaded_data[f.filename] = content_bytes  # keep raw bytes for images, etc.

        # Scrape each URL into DataFrames
        dataframes: Dict[str, Optional[pd.DataFrame]] = {}
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
