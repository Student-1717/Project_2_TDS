# main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel  # <-- Add this import
from typing import List, Optional
import pandas as pd
import io
import base64

# Import your util functions
from util import scrape_table_from_url, analyze_question, parse_questions

app = FastAPI(title="TDS Data Analyst Agent")

class APIRequestModel(BaseModel):
    questions_txt: str  # the content of questions.txt as string
    files: Optional[List[dict]] = None  
    # each dict: { "filename": str, "content_base64": str }

@app.post("/api/")
async def analyze(request: APIRequestModel):
    """
    Accepts JSON input instead of file upload:
    {
        "questions_txt": "contents of questions.txt",
        "files": [
            { "filename": "data.csv", "content_base64": "<base64 encoded file>" }
        ]
    }
    Returns JSON array of answers.
    """
    try:
        questions_content = request.questions_txt.strip()
        if not questions_content:
            return JSONResponse(content=["No data"], status_code=200)

        # Parse questions and URLs
        questions, urls = parse_questions(questions_content)
        if not questions:
            return JSONResponse(content=["No data"], status_code=200)

        # Optional: read other uploaded files into dict for later use
        uploaded_data = {}
        if request.files:
            for f in request.files:
                content = base64.b64decode(f["content_base64"])
                if f["filename"].endswith(".csv"):
                    try:
                        uploaded_data[f["filename"]] = pd.read_csv(io.BytesIO(content))
                    except Exception:
                        uploaded_data[f["filename"]] = None
                else:
                    uploaded_data[f["filename"]] = content  # raw bytes for images, etc.

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
