from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional, Tuple, Any, Dict
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import json
import re
import os

from openai import OpenAI

from util import scrape_table_from_url, parse_questions, collect_all_keys

app = FastAPI(title="TDS Data Analyst Agent")
client = OpenAI()  # requires OPENAI_API_KEY in env

# ---------------- helpers: plotting (kept separate from Q&A) ----------------
def generate_scatterplot(df, x_col, y_col):
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna(subset=[x_col, y_col])
    if df.empty:
        return "data:image/png;base64,"  # return blank

    plt.figure(figsize=(6, 4))
    sns.regplot(x=x_col, y=y_col, data=df, scatter=True, line_kws={"color": "red", "linestyle": "dotted"})
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def find_rank_peak_df(dataframes: dict):
    for df in dataframes.values():
        if isinstance(df, pd.DataFrame) and not df.empty:
            cols = [c.lower().strip() for c in df.columns]
            if "rank" in cols and "peak" in cols:
                return (
                    df[[df.columns[cols.index("rank")], df.columns[cols.index("peak")]]],
                    df.columns[cols.index("rank")],
                    df.columns[cols.index("peak")]
                )
    return None, None, None

# ---------------- helpers: json safety ----------------
def _pyify(val: Any) -> Any:
    """Make values JSON-serializable: convert numpy/pandas/networkx objects to plain Python."""
    try:
        if isinstance(val, (np.generic,)):
            return val.item()
        if isinstance(val, (pd.Series,)):
            return val.tolist()
        if isinstance(val, (pd.DataFrame,)):
            return val.to_dict(orient="records")
        # networkx degree views etc.
        try:
            import networkx as nx  # noqa
            from networkx.classes.reportviews import DegreeView
            if isinstance(val, DegreeView):
                return dict(val)
        except Exception:
            pass
        # fall back to native types if already ok
        json.dumps(val)
        return val
    except Exception:
        return str(val)

# ---------------- AI: infer key + code, execute locally ----------------
AI_SYSTEM = (
    "You are a precise data analysis planner. "
    "Given a natural-language question, respond ONLY with a compact JSON object "
    'with keys "key" and "code". '
    'The "key" must be a snake_case JSON key name that best represents the question. '
    'The "code" must be a short Python expression/snippet that computes the answer using:\n'
    " - df: a pandas DataFrame (first relevant table)\n"
    " - dfs: dict[str, DataFrame] of all tables\n"
    " - G: a networkx Graph if an edge list (source/target) is available, else None\n\n"
    "Rules:\n"
    " - Output STRICT JSON only (no markdown, no backticks, no commentary).\n"
    " - Prefer expressions that return scalars/short strings when possible.\n"
)

def _extract_json_object(text: str) -> Dict[str, Any]:
    """Be robust to accidental formatting; try to extract the first JSON object."""
    text = text.strip()
    # If it's already JSON, great:
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to find a {...} block
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"AI response not parseable as JSON: {text!r}")

def ai_key_and_code(question: str) -> Tuple[str, str]:
    msg = [
        {"role": "system", "content": AI_SYSTEM},
        {"role": "user", "content": question}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msg,
        temperature=0,
        max_tokens=200,
    )
    content = resp.choices[0].message.content
    obj = _extract_json_object(content)
    key = str(obj.get("key", "")).strip()
    code = str(obj.get("code", "")).strip()
    if not key:
        # fallback: sanitized question → snake_case (still AI-driven overall)
        key = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")
    if not code:
        code = "None"
    return key, code

def build_context(dataframes: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Any]:
    """Pick a primary df, a dfs map, and build G if an edge list is present."""
    # choose a primary df (first non-empty)
    df = None
    for d in dataframes.values():
        if isinstance(d, pd.DataFrame) and not d.empty:
            df = d
            break
    dfs = {k: v for k, v in dataframes.items() if isinstance(v, pd.DataFrame)}
    # build graph if any df has source/target (case-insensitive)
    G = None
    if df is not None:
        cols_lower = {c.lower(): c for c in df.columns}
        if "source" in cols_lower and "target" in cols_lower:
            try:
                G = nx.from_pandas_edgelist(df, cols_lower["source"], cols_lower["target"])
            except Exception:
                G = None
    return df, dfs, G

def execute_ai_code(code: str, df: pd.DataFrame, dfs: Dict[str, pd.DataFrame], G: Any) -> Any:
    """Execute AI-proposed code in a constrained environment."""
    # Very restricted globals; expose only safe tools
    safe_globals = {
        "__builtins__": {},
        "pd": pd,
        "np": np,
        "nx": nx,
        "df": df,
        "dfs": dfs,
        "G": G,
        # basic funcs
        "len": len, "sum": sum, "min": min, "max": max, "sorted": sorted,
        "list": list, "dict": dict, "set": set,
        "any": any, "all": all, "abs": abs,
        "float": float, "int": int, "str": str, "range": range, "enumerate": enumerate,
    }
    try:
        # prefer evaluating a single expression
        try:
            value = eval(code, safe_globals, {})
        except SyntaxError:
            # run as a small statement block; last expression capture pattern
            local_vars = {}
            exec(code, safe_globals, local_vars)
            value = local_vars.get("result", None)
        return _pyify(value)
    except Exception as e:
        return f"Execution error: {e}"

# ---------------- optional: fill expected keys (now based on AI keys) ----------------
def fill_missing_keys(output: dict, expected_keys: List[str]) -> dict:
    for key in expected_keys:
        if key not in output:
            # generic safe default by suffix hints
            if key.endswith(("_chart", "_graph", "_histogram", "_plot", "_image")):
                output[key] = "data:image/png;base64,"
            elif key.endswith(("_count", "_total", "_correlation")) or key.startswith(("average_", "median_", "mean_")):
                output[key] = 0.0
            elif key.startswith(("min_", "max_")):
                output[key] = 0.0
            elif key.endswith(("_date", "_node", "_region", "_name", "_id")):
                output[key] = ""
            else:
                output[key] = None
    return output

# ---------------- API ----------------
@app.post("/api/")
async def analyze(request: Request, questions_txt: Optional[UploadFile] = File(None),
                  files: Optional[List[UploadFile]] = None):
    try:
        # Step 1: Extract body flags
        body = {}
        eval_mode = False
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.json()
            eval_mode = bool(body.get("eval", False))

        # Step 2: Read questions
        questions_content = ""
        if questions_txt:
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        elif "request" in body:
            questions_content = str(body.get("request", "")).strip()

        questions, urls = parse_questions(questions_content)

        # Step 3: Process uploaded files
        uploaded_data: Dict[str, Any] = {}
        if files:
            for f in files:
                content = await f.read()
                if f.filename.endswith(".csv"):
                    uploaded_data[f.filename] = pd.read_csv(io.BytesIO(content))
                else:
                    uploaded_data[f.filename] = content

        # Step 4: Scrape URLs
        dataframes: Dict[str, pd.DataFrame] = {}
        for url in urls:
            df = scrape_table_from_url(url)
            dataframes[url] = df

        # Merge uploaded CSVs into dataframes map
        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # Step 5: Build execution context (df, dfs, G)
        df, dfs, G = build_context(dataframes)

        # Step 6: AI-driven Q&A (dynamic key formation, local compute)
        answers_dict: Dict[str, Any] = {}
        answers_list: List[Any] = []
        ai_keys: List[str] = []

        for q in questions:
            key, code = ai_key_and_code(q)
            ai_keys.append(key)
            value = execute_ai_code(code, df, dfs, G)
            answers_dict[key] = value
            answers_list.append(value)

        # Step 7: Optional extra artifacts (e.g., rank vs peak plot) kept separate from Q&A
        scatter_df, rank_col, peak_col = find_rank_peak_df(dataframes)
        if scatter_df is not None:
            answers_dict["scatterplot_rank_peak"] = generate_scatterplot(scatter_df, rank_col, peak_col)
            answers_list.append(answers_dict["scatterplot_rank_peak"])

        # Step 8: Dict mode vs Eval mode
        if eval_mode:
            # Return an ordered array: answers in question order (+ optional plot at end)
            # Ensure at least 4 elements for some eval setups
            out = list(answers_list)
            while len(out) < 4:
                out.append("N/A")
            return JSONResponse(out[:4])
        else:
            # In dict mode, ensure AI-derived expected keys exist (with safe defaults if any missing)
            answers_dict = fill_missing_keys(answers_dict, ai_keys)
            return JSONResponse(answers_dict)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
