# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io

# Import your util functions
from util import scrape_table_from_url, analyze_question, parse_questions

app = FastAPI(title="TDS Data Analyst Agent")

@app.post("/api/")
async def analyze(
    questions_txt: UploadFile = File(...),
    files: Optional[List[UploadFile]] = None
):
    """
    Accepts a questions.txt file and optional additional files (CSV, images, etc.)
    Returns JSON array of answers.
    """
    try:
        # Read questions.txt
        questions_content = (await questions_txt.read()).decode("utf-8").strip()
        if not questions_content:
            return JSONResponse(content=["No data"], status_code=200)

        # Parse questions and URLs
        questions, urls = parse_questions(questions_content)
        if not questions:
            return JSONResponse(content=["No data"], status_code=200)

        # Optional: read other uploaded files into dict for later use
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
                    uploaded_data[f.filename] = content  # keep raw bytes for images, etc.

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
