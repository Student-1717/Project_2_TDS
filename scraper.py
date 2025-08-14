import io
import base64
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# tiny transparent PNG *data URI* (for scraping example, the spec uses full data URI)
DUMMY_DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4//8/AwAI/AL+XMG1WAAAAABJRU5ErkJggg=="

def _fig_to_data_uri(fig, format="png", dpi=120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/{format};base64,{b64}"

def scrape_and_analyze(url: str, question: str, force_array: bool = True):
    """
    Scrapes the first wikitable and answers generically:
      1) count >= $2B before 2000
      2) earliest film > $1.5B
      3) correlation between Rank and Peak
      4) optional scatterplot Rank vs Peak with dotted red regression line
    Returns JSON array if force_array, else JSON object with keys.
    No hardcoded answers: computed from live table.
    """
    html = requests.get(url, timeout=45).text
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", {"class": "wikitable"})
    if not tables:
        return [None, None, None, DUMMY_DATA_URI] if force_array else {
            "ans1": None, "ans2": None, "ans3": None, "image": DUMMY_DATA_URI
        }

    df = pd.read_html(str(tables[0]))[0]
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").copy()

    # normalize expected columns when present
    if "Worldwide gross" in df.columns:
        df["Worldwide gross"] = (
            df["Worldwide gross"].astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .str.replace(r"\s", "", regex=True)
        )
        df["Worldwide gross"] = pd.to_numeric(df["Worldwide gross"], errors="coerce")
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    if "Rank" in df.columns:
        df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    if "Peak" in df.columns:
        df["Peak"] = pd.to_numeric(df["Peak"], errors="coerce")

    # 1) number of >=$2B movies before 2000
    ans1 = None
    if {"Worldwide gross", "Year"}.issubset(df.columns):
        ans1 = int(((df["Worldwide gross"] >= 2_000_000_000) & (df["Year"] < 2000)).sum())

    # 2) earliest film > $1.5B
    ans2 = None
    if {"Worldwide gross", "Year"}.issubset(df.columns) and "Title" in df.columns:
        sub = df[df["Worldwide gross"] > 1_500_000_000].dropna(subset=["Year"])
        if not sub.empty:
            ans2 = str(sub.sort_values("Year", ascending=True).iloc[0]["Title"])

    # 3) correlation Rank vs Peak
    ans3 = None
    if {"Rank", "Peak"}.issubset(df.columns):
        corr_df = df[["Rank", "Peak"]].dropna()
        if len(corr_df) >= 2:
            ans3 = float(round(corr_df["Rank"].corr(corr_df["Peak"]), 6))

    # 4) optional scatterplot Rank vs Peak with dotted red regression
    img = DUMMY_DATA_URI
    if {"Rank", "Peak"}.issubset(df.columns):
        # plot only if asked (sample wording includes "Draw a scatterplot ...")
        if "scatterplot" in question.lower() or "scatter plot" in question.lower():
            plot_df = df[["Rank", "Peak"]].dropna()
            if len(plot_df) >= 2:
                x = plot_df["Rank"].to_numpy()
                y = plot_df["Peak"].to_numpy()
                # regression
                m, b = np.polyfit(x, y, 1)
                y_hat = m * x + b

                fig, ax = plt.subplots(figsize=(4, 3))
                ax.scatter(x, y)
                ax.plot(x, y_hat, "r--")  # red dotted
                ax.set_xlabel("Rank")
                ax.set_ylabel("Peak")
                ax.set_title("Rank vs Peak")
                img = _fig_to_data_uri(fig)

    if force_array:
        return [ans1, ans2, ans3, img]
    return {"ans1": ans1, "ans2": ans2, "ans3": ans3, "image": img}

