import os
import io
import math
import base64
import time
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from openai import OpenAI, RateLimitError

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("main")

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

last_ai_call = 0
AI_COOLDOWN = 3  # seconds

# ---------- Helpers ----------

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return "N/A"
        return obj
    elif obj is None:
        return "N/A"
    return obj

def df_to_json(dfs):
    return {f"file_{i}": df.to_dict(orient="records") for i, df in enumerate(dfs)}

def make_chart(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ---------- Local Compute ----------

def compute_local_value(key, question, dfs):
    try:
        df = dfs[0] if dfs else None
        if df is None or df.empty:
            logger.debug(f"[{key}] No dataframe available.")
            return "N/A"

        text = (key + " " + question).lower()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
        logger.debug(f"[{key}] Numeric cols: {numeric_cols}, Categorical cols: {categorical_cols}")
        if not numeric_cols:
            return "N/A"

        num_col = numeric_cols[0]
        logger.debug(f"[{key}] Using numeric column '{num_col}'")

        if "total" in text or "sum" in text:
            return float(df[num_col].sum())
        if "average" in text or "mean" in text:
            return float(df[num_col].mean())
        if "median" in text:
            return float(df[num_col].median())
        if any(w in text for w in ["max", "highest", "top"]):
            return float(df[num_col].max())
        if any(w in text for w in ["min", "lowest", "bottom"]):
            return float(df[num_col].min())
        if "correlation" in text and "date" in df.columns:
            df["day"] = pd.to_datetime(df["date"]).dt.day
            return float(df["day"].corr(df[num_col]))
        if "bar" in text and categorical_cols:
            fig, ax = plt.subplots()
            df.groupby(categorical_cols[0])[num_col].sum().plot(kind="bar", ax=ax)
            return make_chart(fig)
        if "cumulative" in text and "chart" in text:
            df_sorted = df.sort_values("date") if "date" in df.columns else df.copy()
            df_sorted["cumulative_val"] = df_sorted[num_col].cumsum()
            fig, ax = plt.subplots()
            ax.plot(df_sorted.index, df_sorted["cumulative_val"])
            return make_chart(fig)

        return "N/A"

    except Exception as e:
        logger.warning(f"Local compute failed for {key}: {e}")
        return "N/A"

# ---------- AI Fallback ----------

def safe_ai_call(fn, *args, **kwargs):
    global last_ai_call
    now = time.time()
    if now - last_ai_call < AI_COOLDOWN:
        logger.warning("Skipping AI call to respect cooldown")
        return None
    last_ai_call = now
    return fn(*args, **kwargs)

def ask_ai_batch(keys, questions, files_json):
    try:
        q_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data assistant. Always respond in valid JSON matching the keys provided."},
                {"role": "user", "content": f"Keys: {keys}\n\nQuestions:\n{q_text}\n\nData:\n{files_json}"}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content
    except RateLimitError as e:
        logger.error(f"Rate limit reached, skipping AI fallback: {e}")
        return None
    except Exception as e:
        logger.error(f"AI fallback failed: {e}")
        return None

# ---------- API Endpoint ----------

@app.post("/api/")
async def analyze(request: Request, questions_txt: UploadFile = File(None), files: list[UploadFile] = File(default=[], alias="files[]")):
    try:
        # Detect if JSON body or file upload
        if questions_txt is None:
            try:
                body = await request.json()
                questions_text = body.get("questions_txt", "")
                files_list = body.get("files", [])
            except Exception:
                return JSONResponse({"error": "Invalid JSON input"}, status_code=400)
        else:
            questions_text = (await questions_txt.read()).decode("utf-8")
            files_list = files

        # ---------- Extract keys and questions ----------
        lines = [l.strip() for l in questions_text.splitlines() if l.strip()]
        keys, questions = [], []
        in_answer = False

        for line in lines:
            if line.lower().startswith("return a json object with keys:"):
                continue
            if line.lower().startswith("answer:"):
                in_answer = True
                continue
            if in_answer:
                if "." in line:
                    q = line.split(".", 1)[1].strip()
                    questions.append(q)
            elif "-" in line:
                k = line.strip("- ").split(":")[0].strip()
                keys.append(k)

        # Initialize output dictionary with default empty/zero values
        answers_dict = {}
        for k in keys:
            # Set numeric-like keys to 0, otherwise empty string
            answers_dict[k] = 0 if any(w in k.lower() for w in ["sales", "tax", "correlation", "median"]) else ""

        logger.info(f"Extracted keys: {keys}")
        logger.info(f"Extracted questions: {questions}")
        logger.info(f"Initialized answers dict: {answers_dict}")

        # Load CSVs
        dfs = []
        for f in files_list:
            if isinstance(f, dict):
                # JSON input: assume dict contains CSV content as string
                df = pd.read_csv(io.StringIO(f.get("content", "")))
                dfs.append(df)
            else:
                # UploadFile
                if f.filename.endswith(".csv"):
                    content = await f.read()
                    df = pd.read_csv(io.BytesIO(content))
                    dfs.append(df)

        files_json = df_to_json(dfs)

        # Local computation
        unanswered_keys, unanswered_qs = [], []
        for key, q in zip(keys, questions):
            val = compute_local_value(key, q, dfs)
            if val == "N/A":
                unanswered_keys.append(key)
                unanswered_qs.append(q)
            answers_dict[key] = val

        # AI fallback
        if unanswered_keys:
            ai_raw = safe_ai_call(ask_ai_batch, unanswered_keys, unanswered_qs, files_json)
            if ai_raw:
                try:
                    ai_answers = json.loads(ai_raw)
                    for k in unanswered_keys:
                        answers_dict[k] = ai_answers.get(k, "N/A")
                except Exception as e:
                    logger.error(f"Failed to parse AI JSON: {e}")

        returned_keys = list(answers_dict.keys())
        if set(keys) != set(returned_keys):
            logger.warning(f"⚠️ Key mismatch!\nExpected: {keys}\nReturned: {returned_keys}")

        result = sanitize_for_json(answers_dict)
        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Error in /api/: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)
