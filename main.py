import os
import time
import json
import base64
import logging
from io import BytesIO
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI, RateLimitError

# -------------------------
# Setup
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

last_ai_call = 0
AI_COOLDOWN = 3  # seconds between AI calls


# -------------------------
# Helpers
# -------------------------
def safe_ai_call(fn, *args, **kwargs):
    global last_ai_call
    now = time.time()
    if now - last_ai_call < AI_COOLDOWN:
        logger.warning("Skipping AI call to respect cooldown")
        return None
    last_ai_call = now
    return fn(*args, **kwargs)


def extract_keys_and_questions(text: str):
    keys, questions = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            # candidate key line
            k = line.strip("- ").split(":")[0].strip()
            k = k.replace("`", "")
            if not k.lower().startswith("analyze "):
                keys.append(k)
        elif line[0].isdigit() and "." in line:
            q = line.split(".", 1)[1].strip()
            questions.append(q)
    return keys, questions


def df_to_json_for_ai(df: pd.DataFrame):
    # For small CSVs we can just pass rows to AI
    return df.to_dict(orient="records")


def compute_locally(df: pd.DataFrame, key: str):
    """Try to compute common metrics locally instead of using AI"""
    try:
        k = key.lower()
        if "total_sales" in k:
            return float(df["sales"].sum())
        if "median_sales" in k:
            return float(df["sales"].median())
        if "top_region" in k:
            return str(df.groupby("region")["sales"].sum().idxmax())
        if "total_sales_tax" in k:
            return float(df["sales"].sum() * 0.10)
        if "bar_chart" in k:
            plt.figure()
            df.groupby("region")["sales"].sum().plot(kind="bar", color="blue")
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        if "cumulative_sales_chart" in k:
            plt.figure()
            df = df.sort_values("date")
            df["cumulative"] = df["sales"].cumsum()
            plt.plot(df["date"], df["cumulative"], color="red")
            plt.xticks(rotation=45)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        if "correlation" in k:
            df["day"] = pd.to_datetime(df["date"]).dt.day
            return float(df["day"].corr(df["sales"]))
    except Exception as e:
        logger.warning(f"Local compute failed for {key}: {e}")
    return "N/A"


def ask_ai_batch(keys: List[str], questions: List[str], files_json: Dict):
    try:
        q_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data assistant. Always respond in valid JSON matching the keys provided."},
                {"role": "user", "content": f"Keys: {keys}\n\nQuestions:\n{q_text}\n\nData:\n{files_json}"}
            ],
            temperature=0,
            response_format={"type": "json_object"}  # enforce JSON
        )
        return resp.choices[0].message.content
    except RateLimitError as e:
        logger.error(f"Rate limit reached, skipping AI fallback: {e}")
        return None
    except Exception as e:
        logger.error(f"AI fallback failed: {e}")
        return None


# -------------------------
# API Endpoint
# -------------------------
@app.post("/api/")
async def analyze(questions_txt: UploadFile, files: List[UploadFile] = []):
    try:
        # read questions
        q_text = (await questions_txt.read()).decode("utf-8")
        keys, questions = extract_keys_and_questions(q_text)
        logger.info(f"Extracted keys: {keys}")
        logger.info(f"Extracted questions: {questions}")

        # load CSVs
        dfs = {}
        for f in files:
            df = pd.read_csv(f.file)
            dfs[f.filename] = df

        # compute answers
        answers_dict = {k: "N/A" for k in keys}
        for f, df in dfs.items():
            for k in keys:
                val = compute_locally(df, k)
                if val != "N/A":
                    answers_dict[k] = val

        # unanswered → AI
        unanswered_keys = [k for k, v in answers_dict.items() if v == "N/A"]
        unanswered_qs = [
            questions[keys.index(k)]
            for k in unanswered_keys if k in keys
        ]

        if unanswered_keys:
            files_json = {name: df_to_json_for_ai(df) for name, df in dfs.items()}
            ai_raw = safe_ai_call(ask_ai_batch, unanswered_keys, unanswered_qs, files_json)
            if ai_raw:
                try:
                    ai_answers = json.loads(ai_raw)
                    for k in unanswered_keys:
                        answers_dict[k] = ai_answers.get(k, "N/A")
                except Exception as e:
                    logger.error(f"Failed to parse AI JSON: {e}, raw: {ai_raw[:200]}")

        return JSONResponse(content=answers_dict)

    except Exception as e:
        logger.error(f"Error in /api/: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)
