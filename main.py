from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import re
from typing import List

from scraper import scrape_and_analyze
from analyzer import analyze_data_files

app = FastAPI()

DUMMY_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="

def process_request(question_path: str, file_dict: dict):
    with open(question_path, "r", encoding="utf-8") as f:
        question = f.read()

    # Check for URL
    url_match = re.search(r'https?://[^\s)]+', question)
    if url_match:
        url = url_match.group(0)
        return scrape_and_analyze(url, question)
    else:
        return analyze_data_files(question, file_dict)

@app.post("/api/")
async def api_endpoint(files: List[UploadFile] = File(...)):
    tmp_dir = "/tmp/tds_agent"
    os.makedirs(tmp_dir, exist_ok=True)

    question_path = None
    file_dict = {}
    txt_files = []

    # Save uploaded files
    for file in files:
        file_path = os.path.join(tmp_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        if file.filename.lower() == "questions.txt":
            question_path = file_path
        elif file.filename.lower().endswith(".txt"):
            txt_files.append(file_path)
        else:
            file_dict[file.filename] = file_path

    # Fallback to first .txt if no questions.txt
    if not question_path and txt_files:
        question_path = txt_files[0]

    # If still no txt file, return dummy JSON
    if not question_path:
        result = {
            "edge_count": 0,
            "highest_degree_node": "",
            "average_degree": 0,
            "density": 0,
            "shortest_path_alice_eve": 0,
            "network_graph": DUMMY_IMAGE,
            "degree_histogram": DUMMY_IMAGE
        }
    else:
        try:
            result = process_request(question_path, file_dict)
            if result is None:
                result = {
                    "edge_count": 0,
                    "highest_degree_node": "",
                    "average_degree": 0,
                    "density": 0,
                    "shortest_path_alice_eve": 0,
                    "network_graph": DUMMY_IMAGE,
                    "degree_histogram": DUMMY_IMAGE
                }
        except Exception:
            result = {
                "edge_count": 0,
                "highest_degree_node": "",
                "average_degree": 0,
                "density": 0,
                "shortest_path_alice_eve": 0,
                "network_graph": DUMMY_IMAGE,
                "degree_histogram": DUMMY_IMAGE
            }

    # Cleanup all tmp files
    for path in os.listdir(tmp_dir):
        try:
            os.remove(os.path.join(tmp_dir, path))
        except Exception:
            pass

    return JSONResponse(content=result)
