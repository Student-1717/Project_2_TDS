import re
from typing import List, Dict

def parse_questions(questions_text: str) -> List[Dict]:
    """
    Parse questions.txt and return a list of dicts:
    [
        {"operation": "count", "column": "Worldwide gross", "threshold": 2000000000},
        {"operation": "correlation", "columns": ["Rank", "Peak"]},
        {"operation": "regression_slope", "columns": ["date_of_registration", "decision_date"]},
        {"operation": "scatterplot", "columns": ["Rank", "Peak"], "regression": True}
    ]
    """
    parsed = []
    lines = [line.strip() for line in questions_text.splitlines() if line.strip()]
    
    for line in lines:
        line_lower = line.lower()
        if "how many" in line_lower or "count" in line_lower:
            # extract number if exists
            m = re.search(r'\$?([\d,.]+)\s*(m|bn)?', line_lower)
            threshold = None
            if m:
                num = float(m.group(1).replace(",", ""))
                if m.group(2) == "bn":
                    num *= 1_000_000_000
                elif m.group(2) == "m":
                    num *= 1_000_000
                threshold = num
            # extract column name (naive: word after "in" or "of")
            col_match = re.search(r'in (\w+)|of (\w+)', line_lower)
            column = col_match.group(1) or col_match.group(2) if col_match else None
            parsed.append({"operation": "count", "column": column, "threshold": threshold})
        
        elif "correlation" in line_lower:
            cols = re.findall(r'between (\w+) and (\w+)', line_lower)
            if cols:
                parsed.append({"operation": "correlation", "columns": list(cols[0])})
        
        elif "regression" in line_lower or "slope" in line_lower:
            cols = re.findall(r'(\w+) .* (\w+)', line_lower)
            if cols:
                parsed.append({"operation": "regression_slope", "columns": list(cols[0])})
        
        elif "scatterplot" in line_lower or "plot" in line_lower:
            cols = re.findall(r'(\w+) .* (\w+)', line_lower)
            parsed.append({"operation": "scatterplot", "columns": list(cols[0]), "regression": "regression line" in line_lower})
    
    return parsed
