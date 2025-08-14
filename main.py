from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse
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

    # Ensure exactly 4 answers
    if not (isinstance(answers, (list, tuple)) and len(answers) == 4):
        answers = ["Task failed", "", "", DUMMY_IMAGE]

    # Replace missing image
    if not answers[3] or answers[3] == "?":
        answers = list(answers)
        answers[3] = DUMMY_IMAGE

    # Convert all to string
    return [str(a) for a in answers]


@app.post("/api/", response_class=PlainTextResponse)
async def api_endpoint(
    questions: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    all_files: Optional[List[UploadFile]] = File(None, alias="files")  # Fallback for single-field uploads
):
    tmp_dir = "/tmp/tds_agent"
    os.makedirs(tmp_dir, exist_ok=True)

    file_dict = {}
    question_path = None

    # Case 1: old format (questions + files)
    if questions is not None:
        question_path = os.path.join(tmp_dir, "questions.txt")
        with open(question_path, "wb") as f:
            f.write(await questions.read())
        if files:
            for file in files:
                file_path = os.path.join(tmp_dir, file.filename)
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                file_dict[file.filename] = file_path

    # Case 2: new format (all files in one list)
    elif all_files:
        for file in all_files:
            file_path = os.path.join(tmp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            if file.filename.lower() == "questions.txt":
                question_path = file_path
            else:
                file_dict[file.filename] = file_path

    # If still no questions.txt, fail
    if not question_path:
        return "\n".join(["Task failed", "", "", DUMMY_IMAGE])

    try:
        answers = process_request(question_path, file_dict)
    finally:
        try:
            os.remove(question_path)
            for path in file_dict.values():
                os.remove(path)
        except Exception:
            pass

    return "\n".join(answers)
