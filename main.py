import io
import json
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
from openai import OpenAI
from util import scrape_table_from_url, parse_questions
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TDS_API")

app = FastAPI(title="TDS Data Analyst Agent")
client = OpenAI()  # make sure OPENAI_API_KEY is set

def to_base64_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def generate_scatterplot(df, x_col, y_col):
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna(subset=[x_col, y_col])
    if df.empty:
        return "N/A"
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(x=x_col, y=y_col, data=df, scatter=True, line_kws={"color": "red", "linestyle": "dotted"}, ax=ax)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    return to_base64_plot(fig)

def generate_barplot(df, col):
    counts = df[col].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", color="green", ax=ax)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    plt.tight_layout()
    return to_base64_plot(fig)

def generate_network_plot(edges_df):
    G = nx.from_pandas_edgelist(edges_df)
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", ax=ax)
    plt.tight_layout()
    return to_base64_plot(fig), G

def compute_local_value(key, dataframes):
    """
    Compute values dynamically from uploaded CSVs/dataframes.
    Handles numeric, non-numeric, and network data.
    """
    key_lower = key.lower()
    for df_name, df in dataframes.items():
        if isinstance(df, pd.DataFrame):
            # Numeric column exact match
            for col in df.columns:
                if col.lower() == key_lower:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        return df[col].sum()
                    else:
                        return df[col].astype(str).mode()[0]

            # Aggregate numeric stats
            if any(k in key_lower for k in ["sum", "total"]):
                numeric_cols = df.select_dtypes(include="number").columns
                if len(numeric_cols) > 0:
                    return df[numeric_cols].sum().to_dict()
            if any(k in key_lower for k in ["mean", "average"]):
                numeric_cols = df.select_dtypes(include="number").columns
                if len(numeric_cols) > 0:
                    return df[numeric_cols].mean().to_dict()

            # Scatterplot request
            if "scatterplot" in key_lower or "plot" in key_lower:
                numeric_cols = df.select_dtypes(include="number").columns
                if len(numeric_cols) >= 2:
                    return generate_scatterplot(df, numeric_cols[0], numeric_cols[1])

            # Network CSV (2-column edge list)
            if df.shape[1] == 2:
                plot_key = ["network_graph", "graph", "degree_histogram"]
                if any(k in key_lower for k in plot_key):
                    network_plot, G = generate_network_plot(df)
                    if "degree_histogram" in key_lower:
                        deg = pd.Series(dict(G.degree()))
                        fig, ax = plt.subplots(figsize=(6, 4))
                        deg.plot(kind="bar", color="green", ax=ax)
                        plt.tight_layout()
                        return to_base64_plot(fig)
                    return network_plot

            # Non-numeric fallback: return first non-empty value
            return df[df.columns[0]].astype(str).iloc[0]

    return None  # fallback to AI if not computable locally

async def ai_generate_value_for_key(key: str, question: str, dataframes: dict):
    data_preview = {k: (df.head(5).to_dict(orient="records") if isinstance(df, pd.DataFrame) else str(df))
                    for k, df in dataframes.items()}
    prompt = f"""
You are a data analyst AI.
Key: "{key}"
User Question: "{question}"
Available dataframes (sample 5 rows each): {data_preview}
Return a JSON with: {{"value": "computed_or_suggested_value"}}
If the key requires a plot, suggest 'scatterplot' or another type of plot.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        return result.get("value", "N/A")
    except Exception:
        return "N/A"

def extract_keys_from_questions(txt):
    pattern = r"- `([^`]+)`"
    return re.findall(pattern, txt)

@app.post("/api/")
async def analyze(request: Request,
                  questions_txt: Optional[UploadFile] = File(None),
                  files: Optional[List[UploadFile]] = None):
    try:
        # --- Read request body safely once ---
        body_bytes = await request.body()
        body_str = body_bytes.decode("utf-8").strip()
        try:
            body_json = json.loads(body_str)
        except json.JSONDecodeError:
            body_json = {}

        # --- Read questions ---
        questions_content = ""
        if questions_txt:
            await questions_txt.seek(0)
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        else:
            questions_content = body_json.get("request", "").strip()

        if not questions_content:
            logger.warning("No questions content provided!")

        # --- Parse questions ---
        questions, urls = parse_questions(questions_content)
        logger.info(f"Parsed questions: {questions}")
        logger.info(f"Parsed URLs: {urls}")

        # --- Process uploaded files ---
        uploaded_data = {}
        if files:
            for f in files:
                await f.seek(0)
                content = await f.read()
                if f.filename.endswith(".csv"):
                    try:
                        uploaded_data[f.filename] = pd.read_csv(io.BytesIO(content))
                        logger.info(f"CSV loaded: {f.filename} with shape {uploaded_data[f.filename].shape}")
                    except Exception as e:
                        uploaded_data[f.filename] = None
                        logger.warning(f"Failed to read CSV {f.filename}: {e}")
                else:
                    uploaded_data[f.filename] = content

        # --- Scrape URLs ---
        dataframes = {}
        for url in urls:
            try:
                df = scrape_table_from_url(url)
                dataframes[url] = df
                logger.info(f"Scraped URL {url} with shape {df.shape}")
            except Exception as e:
                dataframes[url] = None
                logger.warning(f"Failed to scrape URL {url}: {e}")

        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # --- Extract keys dynamically ---
        expected_keys = extract_keys_from_questions(questions_content)
        logger.info(f"Extracted keys: {expected_keys}")
        answers_dict = {key: "N/A" for key in expected_keys}

        # --- Compute values for each key ---
        for key in expected_keys:
            local_val = compute_local_value(key, dataframes)
            if local_val is not None:
                answers_dict[key] = local_val
                logger.info(f"Local computation for '{key}': {local_val}")
            else:
                ai_val = await ai_generate_value_for_key(key, questions_content, dataframes)
                answers_dict[key] = ai_val
                logger.info(f"AI computation for '{key}': {ai_val}")

        return JSONResponse({"dict": answers_dict, "array": list(answers_dict.values())})

    except Exception as e:
        logger.error(f"Error in /api/: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)
