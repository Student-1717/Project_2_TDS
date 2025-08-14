from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import re
from typing import List, Optional

from scraper import scrape_and_analyze
from analyzer import analyze_data_files

app = FastAPI()

DUMMY_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="

def process_request(question_path, file_dict):
    with open(question_path, "r", encoding="utf-8") as f:
        question = f.read()

    url_match = re.search(r'https?://[^\s)]+', question)
    if url_match:
        url = url_match.group(0)
        answers = scrape_and_analyze(url, question)
    else:
        answers = analyze_data_files(question, file_dict)

    # Ensure list of exactly 4 items
    if not (isinstance(answers, (list, tuple)) and len(answers) == 4):
        answers = ["Task failed", "", "", DUMMY_IMAGE]

    # Ensure image slot is not empty
    if not answers[3] or answers[3] == "?":
        answers = list(answers)
        answers[3] = DUMMY_IMAGE

    return [str(a) for a in answers]

@app.post("/api/", response_class=JSONResponse)
async def api_endpoint(
    questions: UploadFile = File(...),
    files: Optional[List[UploadFile]] = File(None)
):
    tmp_dir = "/tmp/tds_agent"
    os.makedirs(tmp_dir, exist_ok=True)

    # Save questions.txt
    question_path = os.path.join(tmp_dir, questions.filename)
    with open(question_path, "wb") as f:
        f.write(await questions.read())

    # Save other files
    file_dict = {}
    if files:
        for file in files:
            file_path = os.path.join(tmp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            file_dict[file.filename] = file_path

    try:
        answers = process_request(question_path, file_dict)
    finally:
        try:
            os.remove(question_path)
            for path in file_dict.values():
                os.remove(path)
        except Exception:
            pass

    return JSONResponse(content=answers)
