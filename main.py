from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from util import scrape_table_from_url, analyze_question, parse_questions, collect_all_keys

app = FastAPI(title="TDS Data Analyst Agent")


def generate_key_from_question(question: str) -> str:
    """Generate a dictionary key from the question text."""
    return question.lower().replace(" ", "_")


def ask_ai_only_questions(questions: list) -> dict:
    """Placeholder for AI integration. Returns question text mapped to keys."""
    answers_dict = {}
    for q in questions:
        key = generate_key_from_question(q)
        answers_dict[key] = "AI answer placeholder for: " + q
    return answers_dict


def generate_scatterplot(df, x_col, y_col):
    """Generate a scatterplot with regression and return base64 string."""
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
    """Find the first DataFrame containing both 'rank' and 'peak' columns."""
    for df in dataframes.values():
        if isinstance(df, pd.DataFrame) and not df.empty:
            cols = [c.lower().strip() for c in df.columns]
            if "rank" in cols and "peak" in cols:
                return df[[df.columns[cols.index("rank")], df.columns[cols.index("peak")]]], \
                       df.columns[cols.index("rank")], df.columns[cols.index("peak")]
    return None, None, None


def build_network_graph(dataframes: dict):
    """Combine all tables with 'source' and 'target' columns into a NetworkX graph and compute metrics."""
    combined_df = pd.DataFrame()
    for df in dataframes.values():
        if isinstance(df, pd.DataFrame):
            cols = [c.lower().strip() for c in df.columns]
            if "source" in cols and "target" in cols:
                combined_df = pd.concat([combined_df, df[[df.columns[cols.index("source")], df.columns[cols.index("target")]]]])

    if combined_df.empty:
        return None

    combined_df = combined_df.dropna()
    G = nx.from_pandas_edgelist(combined_df, source=combined_df.columns[0], target=combined_df.columns[1])
    
    # Metrics
    metrics = {
        "edge_count": G.number_of_edges(),
        "highest_degree_node": max(dict(G.degree()).items(), key=lambda x: x[1])[0] if G.number_of_nodes() > 0 else None,
        "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else None,
        "density": nx.density(G),
        "shortest_path_alice_eve": nx.shortest_path_length(G, source="Alice", target="Eve") if "Alice" in G.nodes and "Eve" in G.nodes else None
    }

    # Graph visualization
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    buf.seek(0)
    metrics["network_graph"] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

    # Degree histogram
    degrees = [d for n, d in G.degree()]
    plt.figure(figsize=(6, 4))
    plt.hist(degrees, bins=range(1, max(degrees) + 2), color='skyblue', edgecolor='black')
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    buf.seek(0)
    metrics["degree_histogram"] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

    return metrics


@app.post("/api/")
async def analyze(request: Request, questions_txt: Optional[UploadFile] = File(None),
                  files: Optional[List[UploadFile]] = None):
    try:
        # ---------------- Step 1: Read questions ----------------
        questions_content = None
        if questions_txt:
            questions_content = (await questions_txt.read()).decode("utf-8").strip()
        if not questions_content:
            body = await request.json()
            questions_content = body.get("request", "").strip() if body else None
        if not questions_content:
            return JSONResponse({"error": "No questions provided"}, status_code=200)

        # ---------------- Step 2: Parse questions & URLs ----------------
        questions, urls = parse_questions(questions_content)
        if not questions:
            return JSONResponse({"error": "No valid questions found"}, status_code=200)

        # ---------------- Step 3: Process uploaded files ----------------
        uploaded_data = {}
        if files:
            for f in files:
                content = await f.read()
                if f.filename.endswith(".csv"):
                    uploaded_data[f.filename] = pd.read_csv(io.BytesIO(content))
                else:
                    uploaded_data[f.filename] = content

        # ---------------- Step 4: Scrape URLs ----------------
        dataframes = {}
        for url in urls:
            df = scrape_table_from_url(url)
            dataframes[url] = df

        # Include uploaded CSVs
        for filename, df in uploaded_data.items():
            if isinstance(df, pd.DataFrame):
                dataframes[filename] = df

        # ---------------- Step 5: Analyze questions ----------------
        answers_dict = {}
        for q in questions:
            answers_dict[generate_key_from_question(q)] = analyze_question(q, dataframes, uploaded_data)

        # ---------------- Step 6: Collect extra keys from URLs & tables ----------------
        extra_keys = collect_all_keys(urls, dataframes)
        for key in extra_keys:
            if key not in answers_dict:
                answers_dict[key] = "No direct question, extracted from URL/table"

        # ---------------- Step 7: Generate scatterplot if possible ----------------
        scatter_df, rank_col, peak_col = find_rank_peak_df(dataframes)
        if scatter_df is not None:
            answers_dict["scatterplot_rank_peak"] = generate_scatterplot(scatter_df, rank_col, peak_col)
        else:
            answers_dict["scatterplot_rank_peak"] = "No table contains both 'rank' and 'peak' columns"

        # ---------------- Step 8: Build network graph ----------------
        network_metrics = build_network_graph(dataframes)
        if network_metrics:
            answers_dict.update(network_metrics)
        else:
            answers_dict.update({
                "edge_count": 0,
                "highest_degree_node": None,
                "average_degree": None,
                "density": 0,
                "shortest_path_alice_eve": None,
                "network_graph": "No network graph data available",
                "degree_histogram": "No network graph data available"
            })

        return JSONResponse(answers_dict)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
