from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse
import os
import re
from typing import List

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

    # Ensure answers is a list/tuple of length 4
    if not (isinstance(answers, (list, tuple)) and len(answers) == 4):
        answers = ["Task failed", "", "", DUMMY_IMAGE]

    # Replace any empty image answers with dummy image
    if not answers[3] or answers[3] == "?":
        answers = list(answers)
        answers[3] = DUMMY_IMAGE

    # Convert all answers to strings
    answers = [str(a) for a in answers]

    # Return exactly 4 lines, joined by newline (plain text)
    return "\n".join(answers)

@app.post("/api/", response_class=PlainTextResponse)
async def api_endpoint(
    questions: UploadFile = File(...),
    files: List[UploadFile] = File(None)
):
    tmp_dir = "/tmp/tds_agent"
    os.makedirs(tmp_dir, exist_ok=True)

    question_path = os.path.join(tmp_dir, "questions.txt")
    with open(question_path, "wb") as f:
        f.write(await questions.read())

    file_dict = {}
    if files:
        for file in files:
            file_path = os.path.join(tmp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            file_dict[file.filename] = file_path

    try:
        response_text = process_request(question_path, file_dict)
    finally:
        # Optional: Cleanup all files after processing
        try:
            os.remove(question_path)
            for path in file_dict.values():
                os.remove(path)
        except Exception:
            pass

    return response_text
