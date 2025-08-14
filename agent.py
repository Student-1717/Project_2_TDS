import re
from typing import Dict, Any, List, Union

from scraper import scrape_and_analyze
from analyzer import analyze_data_files

JsonType = Union[Dict[str, Any], List[Any]]

def _wants_json_array(question_text: str) -> bool:
    # heuristic based on spec phrasing
    return bool(re.search(r"\bjson\s+array\b", question_text, re.I))

def _declares_json_object_keys(question_text: str) -> bool:
    return bool(re.search(r"\breturn\s+a\s+json\s+object\b", question_text, re.I))

def process_request(question_path: str, file_dict: Dict[str, str]) -> JsonType:
    with open(question_path, "r", encoding="utf-8") as f:
        question = f.read().strip()

    # if a URL exists → scraping task
    url_match = re.search(r'https?://[^\s)]+', question)
    if url_match:
        url = url_match.group(0)
        wants_array = _wants_json_array(question)
        return scrape_and_analyze(url, question, force_array=wants_array)

    # otherwise → local data analysis (csv/parquet/networks, etc.)
    declares_object = _declares_json_object_keys(question)
    return analyze_data_files(question, file_dict, force_declared_object=declares_object)
