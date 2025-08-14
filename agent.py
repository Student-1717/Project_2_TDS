import re
import json
from scraper import scrape_and_analyze
from analyzer import analyze_data_files

def process_request(question_path, file_dict):
    with open(question_path, "r", encoding="utf-8") as f:
        question = f.read()

    url_match = re.search(r'https?://[^\s)]+', question)
    if url_match:
        url = url_match.group(0)
        result = scrape_and_analyze(url, question)
    else:
        result = analyze_data_files(question, file_dict)

    # Guarantee JSON-serializable dict
    if not isinstance(result, dict):
        result = {"error": "Invalid return type"}

    return json.dumps(result)
