# main.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
import os
from urllib.parse import urlparse
import openai  # make sure openai>=1.0.0 is installed
import base64
import matplotlib.pyplot as plt
import seaborn as sns

from util import scrape_table_from_url, analyze_question, parse_questions

app = FastAPI(title="TDS Data Analyst Agent")


def generate_key_from_question(question: str) -> str:
    return question.lower().replace(" ", "_")


def extract_keys_from_url_data(url, df):
    keys = set()
    # From URL
    try:
        parsed = urlparse(url)
        if parsed.path:
            path_parts = [p for p in parsed.path.split("/") if p]
            for part in path_parts:
                keys.add(part.lower().replace(" ", "_"))
    except Exception:
        pass

    # From table columns
    if isinstance(df, pd.DataFrame) and not df.empty:
        for col in df.columns:
            keys.add(str(col).lower().strip().replace(" ", "_"))

    # From known headers
    if isinstance(df, pd.DataFrame) and not df.empty:
        for col in df.columns:
            if "rank" in str(col).lower() or "title" in str(col).lower():
                keys.add(str(col).lower().replace(" ", "_"))

    return list(keys)


def ask_ai_only_questions(questions: list) -> dict:
    answers_dict = {}
    for q in questions:
        key = generate_key_from_question(q)
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful data analyst. "
                            "Answer the user's question as directly and concisely as possible. "
                            "Return a short, plain answer."
                        )
                    },
                    {"role": "user", "content": q}
                ],
                temperature=0
            )
            answer = response.choices[0].message.content.strip()
            answers_dict[key] = answer if answer else "No data"
        except Exception as e:
            answers_dict[key] = f"Error: {e}"
    return answers_dict


# --- New functions for scatterplot ---
def find_rank_peak_df(dataframes: dict):
    for name, df in dataframes.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            cols = [c.lower().strip() for c in df.columns]
            if "rank" in cols and "peak" in cols:
                rank_col = df.columns[cols.index("rank")]
                peak_col = df.columns[cols.index("peak")]
                return df[[rank_col, peak_col]], rank_col, peak_col
    return None, None, None


def generate_scatterplot(df, rank_col, peak_col):
    plt.figure(figsize=(6, 4))
    sns.regplot(
        x=rank_col, y=peak_col, data=df,
        scatter=True, line_kws={"color": "red", "linestyle": "dotted"}
    )
    plt.xlabel(rank_col)
    plt.ylabel(peak_col)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    buf.seek(0)
    img_bytes = buf.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


@app.post("/api/")
async def analyze(
    request: Request,
    questions_txt: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = None
):
    try:
        # Step 1: Get questions content
        questions_content = None
        if questions_txt:
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        else:
            try:
                body = await request.json()
                questions_content = body.get("request", "").strip()
            except Exception:
                try:
                    raw_body = await request.body()
                    if raw_body:
                        questions_content = raw_body.decode("utf-8").strip()
                except Exception:
                    pass

        if not questions_content:
            file_path = os.environ.get("QUESTIONS_FILE")
            if file_path and os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    questions_content = f.read().strip()

        if not questions_content:
            return JSONResponse(content={"error": "No questions provided"}, status_code=200)

        # Step 2: Parse questions & URLs
        questions, urls = parse_questions(questions_content)
        if not questions:
            return JSONResponse(content={"error": "No valid questions found"}, status_code=200)

        # Step 3: Process uploaded files
        uploaded_data = {}
        if files:
            for f in files:
                content = await f.read()
                if f.filename.endswith(".csv"):
                    try:
                        uploaded_data[f.filename] = pd.read_csv(io.BytesIO(content))
                    except Exception:
                        uploaded_data[f.filename] = None
                else:
                    uploaded_data[f.filename] = content

        # Step 4: Scrape URLs & collect extra keys
        dataframes = {}
        extra_keys = set()
        for url in urls:
            try:
                scraped = scrape_table_from_url(url)
                dataframes[url] = scraped

                if isinstance(scraped, dict):
                    merged_df = pd.concat(
                        [t for t in scraped.values() if isinstance(t, pd.DataFrame)],
                        ignore_index=True
                    ) if scraped else pd.DataFrame()
                    df_for_keys = merged_df
                else:
                    df_for_keys = scraped

                extra_keys.update(extract_keys_from_url_data(url, df_for_keys))
                dataframes[url] = df_for_keys

            except Exception:
                dataframes[url] = None

        # Include uploaded CSVs
        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # Step 5: Ask AI questions
        answers_dict = ask_ai_only_questions(questions)

        # Step 6: Add placeholders for extracted keys
        for key in extra_keys:
            if key not in answers_dict:
                answers_dict[key] = "No direct question, extracted from URL/table"

        # Step 7: Generate scatterplot if possible
        scatter_df, rank_col, peak_col = find_rank_peak_df(dataframes)
        if scatter_df is not None:
            answers_dict["scatterplot_rank_peak"] = generate_scatterplot(scatter_df, rank_col, peak_col)
        else:
            answers_dict["scatterplot_rank_peak"] = "No table contains both 'rank' and 'peak' columns"

        return JSONResponse(content=answers_dict)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
