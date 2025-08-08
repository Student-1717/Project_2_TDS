
import re
from scraper import scrape_and_analyze
from analyzer import analyze_data_files

def process_request(question_path, file_dict):
    with open(question_path, "r") as f:
        question = f.read()

    url_match = re.search(r'https?://[^\s)]+', question)
    if url_match:
        url = url_match.group(0)
        return scrape_and_analyze(url, question)

    return analyze_data_files(question, file_dict)
