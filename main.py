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
from util import parse_questions

app = FastAPI(title="TDS Data Analyst Agent")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TDS_API")


def generate_network_plot(G):
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


def generate_degree_histogram(G):
    degrees = [d for n, d in G.degree()]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(range(len(degrees))), y=degrees, color='green')
    plt.xlabel("Node index")
    plt.ylabel("Degree")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


def compute_graph_metrics(df):
    G = nx.from_pandas_edgelist(df, source='source', target='target')
    metrics = {
        "edge_count": G.number_of_edges(),
        "highest_degree_node": max(G.degree, key=lambda x: x[1])[0],
        "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "density": nx.density(G)
    }
    # Optional: compute shortest path between Alice and Eve if nodes exist
    if "Alice" in G and "Eve" in G:
        try:
            metrics["shortest_path_alice_eve"] = nx.shortest_path_length(G, "Alice", "Eve")
        except nx.NetworkXNoPath:
            metrics["shortest_path_alice_eve"] = None
    else:
        metrics["shortest_path_alice_eve"] = None

    # Generate plots
    metrics["network_graph"] = generate_network_plot(G)
    metrics["degree_histogram"] = generate_degree_histogram(G)
    return metrics


def extract_keys_from_questions(txt):
    import re
    pattern = r"- `([^`]+)`"
    return re.findall(pattern, txt)


@app.post("/api/")
async def analyze(request: Request,
                  questions_txt: Optional[UploadFile] = File(None),
                  files: Optional[List[UploadFile]] = None):
    try:
        # --- Step 1: Read questions ---
        questions_content = ""
        if questions_txt:
            await questions_txt.seek(0)
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
            logger.info(f"Questions file content length: {len(questions_content)}")
        else:
            body = await request.json()
            questions_content = body.get("request", "").strip()
            logger.info(f"Questions from JSON length: {len(questions_content)}")

        # --- Step 2: Process uploaded files ---
        dataframes = {}
        if files:
            for f in files:
                await f.seek(0)
                content = await f.read()
                logger.info(f"Processing uploaded file: {f.filename}, size: {len(content)} bytes")
                if f.filename.endswith(".csv"):
                    try:
                        dataframes[f.filename] = pd.read_csv(io.BytesIO(content))
                        logger.info(f"CSV loaded with shape: {dataframes[f.filename].shape}")
                    except Exception as e:
                        dataframes[f.filename] = None
                        logger.warning(f"Failed to read CSV {f.filename}: {e}")
                else:
                    dataframes[f.filename] = content

        # --- Step 3: Extract keys ---
        expected_keys = extract_keys_from_questions(questions_content)
        logger.info(f"Extracted keys: {expected_keys}")

        answers_dict = {key: "N/A" for key in expected_keys}

        # --- Step 4: Compute graph metrics if edges.csv is present ---
        if "edges.csv" in dataframes and isinstance(dataframes["edges.csv"], pd.DataFrame):
            metrics = compute_graph_metrics(dataframes["edges.csv"])
            for key in expected_keys:
                if key in metrics:
                    answers_dict[key] = metrics[key]

        return JSONResponse({"dict": answers_dict, "array": list(answers_dict.values())})

    except Exception as e:
        logger.error(f"Error in /api/: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)
