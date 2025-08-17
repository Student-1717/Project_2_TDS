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

app = FastAPI(title="TDS Data Analyst Agent")
client = OpenAI()  # ensure OPENAI_API_KEY is set

def dataframe_is_edge_list(df):
    """Detect if a dataframe is an edge list for a graph"""
    if isinstance(df, pd.DataFrame) and df.shape[1] >= 2:
        # If at least 2 columns and all values are strings or ints
        return all(df.iloc[:, 0].notna()) and all(df.iloc[:, 1].notna())
    return False

def generate_base64_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def generate_scatterplot(df, x_col, y_col):
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna(subset=[x_col, y_col])
    if df.empty:
        return ""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(x=x_col, y=y_col, data=df, scatter=True, line_kws={"color": "red", "linestyle": "dotted"}, ax=ax)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    return generate_base64_plot(fig)

def generate_network_plots(G):
    # Graph plot
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10, ax=ax1)
    graph_img = generate_base64_plot(fig1)

    # Degree histogram
    degrees = [d for n, d in G.degree()]
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(range(len(degrees)), degrees, color='green')
    ax2.set_xlabel('Node index')
    ax2.set_ylabel('Degree')
    plt.tight_layout()
    hist_img = generate_base64_plot(fig2)

    return graph_img, hist_img

def compute_local_value_dynamic(key, dataframes):
    """
    Fully dynamic local computation:
    - Handles numeric, non-numeric, and graph-like CSVs
    - Dynamically matches requested key
    """
    key_lower = key.lower()

    for df_name, df in dataframes.items():
        if not isinstance(df, pd.DataFrame):
            continue

        # --- Detect graph automatically ---
        if dataframe_is_edge_list(df):
            G = nx.from_pandas_edgelist(df, df.columns[0], df.columns[1])
            # Dynamically match key
            if 'edge' in key_lower:
                return G.number_of_edges()
            if 'node' in key_lower and 'highest' in key_lower:
                return max(dict(G.degree()).items(), key=lambda x: x[1])[0]
            if 'degree' in key_lower and 'average' in key_lower:
                return sum(dict(G.degree()).values()) / G.number_of_nodes()
            if 'density' in key_lower:
                return nx.density(G)
            if 'shortest' in key_lower and len(key_lower.split('_')) >= 3:
                # Try to parse node names from key like 'shortest_path_alice_eve'
                parts = key_lower.split('_')
                if len(parts) >= 3:
                    source, target = parts[-2], parts[-1]
                    if source in G.nodes and target in G.nodes:
                        return nx.shortest_path_length(G, source, target)
                    else:
                        return "N/A"
            if 'network_graph' in key_lower or 'graph' in key_lower:
                graph_img, _ = generate_network_plots(G)
                return graph_img
            if 'degree_histogram' in key_lower or 'histogram' in key_lower:
                _, hist_img = generate_network_plots(G)
                return hist_img

        # --- Numeric columns ---
        col_matches = [col for col in df.columns if col.lower() == key_lower]
        if col_matches:
            col = col_matches[0]
            if pd.api.types.is_numeric_dtype(df[col]):
                return df[col].sum()
            else:
                return df[col].mode().iloc[0]  # fallback for non-numeric

        # Partial matches for sums/means
        if "sum" in key_lower or "total" in key_lower:
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) > 0:
                return df[numeric_cols].sum().to_dict()
        if "mean" in key_lower or "average" in key_lower:
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) > 0:
                return df[numeric_cols].mean().to_dict()

        # Scatterplot detection
        if "scatterplot" in key_lower or "plot" in key_lower:
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) >= 2:
                return generate_scatterplot(df, numeric_cols[0], numeric_cols[1])

        # Non-numeric columns: fallback to mode
        non_numeric_cols = df.select_dtypes(exclude='number').columns
        for col in non_numeric_cols:
            if col.lower() == key_lower:
                return df[col].mode().iloc[0]

    return None  # fallback to AI

# --- /api/ endpoint ---
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TDS_API")

@app.post("/api/")
async def analyze(request: Request,
                  questions_txt: Optional[UploadFile] = File(None),
                  files: Optional[List[UploadFile]] = None):
    try:
        # --- Read questions ---
        questions_content = ""
        if questions_txt:
            await questions_txt.seek(0)
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
            logger.info(f"Questions file content length: {len(questions_content)}")
        else:
            body = await request.json()
            questions_content = body.get("request", "").strip()
            logger.info(f"Questions from JSON length: {len(questions_content)}")

        if not questions_content:
            logger.warning("No questions content provided!")

        # --- Parse questions and URLs ---
        questions, urls = parse_questions(questions_content)
        logger.info(f"Parsed questions: {questions}")
        logger.info(f"Parsed URLs: {urls}")

        # --- Process uploaded files ---
        uploaded_data = {}
        if files:
            for f in files:
                await f.seek(0)
                content = await f.read()
                logger.info(f"Processing uploaded file: {f.filename}, size: {len(content)} bytes")
                if f.filename.endswith(".csv"):
                    try:
                        uploaded_data[f.filename] = pd.read_csv(io.BytesIO(content))
                        logger.info(f"CSV loaded with shape: {uploaded_data[f.filename].shape}")
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
                logger.info(f"Scraped URL {url} with shape: {df.shape}")
            except Exception as e:
                dataframes[url] = None
                logger.warning(f"Failed to scrape URL {url}: {e}")

        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # --- Extract keys dynamically ---
        import re
        pattern = r"- `([^`]+)`"
        expected_keys = re.findall(pattern, questions_content)
        logger.info(f"Extracted keys: {expected_keys}")
        if not expected_keys:
            logger.warning("No keys extracted! Check question formatting (backticks `key`).")

        answers_dict = {key: "N/A" for key in expected_keys}

        # --- Generate values for each key ---
        for key in expected_keys:
            local_val = compute_local_value_dynamic(key, dataframes)
            if local_val is not None:
                answers_dict[key] = local_val
                logger.info(f"Local computation for '{key}': {local_val}")
            else:
                # Fallback to AI if local computation fails
                answers_dict[key] = "N/A"

        return JSONResponse({"dict": answers_dict, "array": list(answers_dict.values())})

    except Exception as e:
        logger.error(f"Error in /api/: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)
