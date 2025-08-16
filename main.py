from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import logging

from util import scrape_table_from_url, analyze_question, parse_questions, collect_all_keys

app = FastAPI(title="TDS Data Analyst Agent")

logging.basicConfig(level=logging.INFO)

# ---------------- Helper functions ----------------

def generate_key_from_question(question: str) -> str:
    return question.lower().replace(" ", "_")

def ask_ai_only_questions(questions: list) -> dict:
    answers_dict = {}
    for q in questions:
        key = generate_key_from_question(q)
        answers_dict[key] = "AI answer placeholder for: " + q
    return answers_dict

def generate_scatterplot(df, x_col, y_col):
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna(subset=[x_col, y_col])

    if df.empty:
        return "No numeric data available for scatterplot"

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
                return df[[df.columns[cols.index("rank")], df.columns[cols.index("peak")]]], \
                       df.columns[cols.index("rank")], df.columns[cols.index("peak")]
    return None, None, None

def safe_network_analysis(df, source_col="source", target_col="target"):
    result = {
        "edge_count": 0,
        "highest_degree_node": None,
        "average_degree": 0.0,
        "density": 0.0,
        "shortest_path_alice_eve": None,
        "network_graph": None,
        "degree_histogram": None
    }

    try:
        if df is None or df.empty or source_col not in df.columns or target_col not in df.columns:
            logging.warning("DataFrame is empty or missing required columns for network analysis")
            return result

        G = nx.from_pandas_edgelist(df, source=source_col, target=target_col)
        result["edge_count"] = G.number_of_edges()

        if G.number_of_nodes() > 0:
            degrees = dict(G.degree())
            result["highest_degree_node"] = max(degrees, key=degrees.get)
            result["average_degree"] = sum(degrees.values()) / len(degrees)
            result["density"] = nx.density(G)

            if "alice" in G and "eve" in G:
                result["shortest_path_alice_eve"] = nx.shortest_path_length(G, "alice", "eve")

            # Network graph image
            plt.figure(figsize=(6, 4))
            nx.draw(G, with_labels=True, node_color="skyblue", edge_color="gray", node_size=500)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            plt.close()
            buf.seek(0)
            result["network_graph"] = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

            # Degree histogram
            plt.figure(figsize=(6, 4))
            plt.hist(list(degrees.values()), bins=range(max(degrees.values()) + 2), color="lightgreen", edgecolor="black")
            plt.xlabel("Degree")
            plt.ylabel("Count")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            plt.close()
            buf.seek(0)
            result["degree_histogram"] = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        logging.error(f"Error in safe_network_analysis: {e}")

    return result

# ---------------- API endpoint ----------------

@app.post("/api/")
async def analyze(request: Request, questions_txt: Optional[UploadFile] = File(None),
                  files: Optional[List[UploadFile]] = None):
    try:
        # Read questions
        questions_content = None
        if questions_txt:
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        if not questions_content:
            body = await request.json()
            questions_content = body.get("request", "").strip() if body else None
        if not questions_content:
            return JSONResponse({"error": "No questions provided"}, status_code=200)

        # Parse questions & URLs
        questions, urls = parse_questions(questions_content)
        if not questions:
            return JSONResponse({"error": "No valid questions found"}, status_code=200)

        # Process uploaded files
        uploaded_data = {}
        if files:
            for f in files:
                content = await f.read()
                if f.filename.endswith(".csv"):
                    uploaded_data[f.filename] = pd.read_csv(io.BytesIO(content))
                else:
                    uploaded_data[f.filename] = content

        # Scrape URLs
        dataframes = {}
        for url in urls:
            df = scrape_table_from_url(url)
            dataframes[url] = df

        # Include uploaded CSVs
        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # Analyze questions
        answers_dict = {}
        for q in questions:
            answers_dict[generate_key_from_question(q)] = analyze_question(q, dataframes, uploaded_data)

        # Collect extra keys from URLs & tables
        extra_keys = collect_all_keys(urls, dataframes)
        for key in extra_keys:
            if key not in answers_dict:
                answers_dict[key] = "No direct question, extracted from URL/table"

        # Scatterplot if possible
        scatter_df, rank_col, peak_col = find_rank_peak_df(dataframes)
        if scatter_df is not None:
            answers_dict["scatterplot_rank_peak"] = generate_scatterplot(scatter_df, rank_col, peak_col)
        else:
            answers_dict["scatterplot_rank_peak"] = "No table contains both 'rank' and 'peak' columns"

        # Network analysis for all dataframes
        for name, df in dataframes.items():
            network_result = safe_network_analysis(df)
            answers_dict[f"network_analysis_{name}"] = network_result

        return JSONResponse(answers_dict)

    except Exception as e:
        logging.error(f"API error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
