import requests
import pandas as pd
from bs4 import BeautifulSoup
from visualizer import plot_scatter_with_regression

# Dummy 1x1 transparent PNG base64 to return when no plot required
DUMMY_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4//8/AwAI/AL+XMG1WAAAAABJRU5ErkJggg=="

def scrape_and_analyze(url, question):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", {"class": "wikitable"})
    if not tables:
        return {
            "ans1": None,
            "ans2": None,
            "ans3": None,
            "image": DUMMY_IMAGE
        }

    df = pd.read_html(str(tables[0]))[0]
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")

    try:
        df['Worldwide gross'] = df['Worldwide gross'].replace('[\$,]', '', regex=True).astype(float)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Rank'] = pd.to_numeric(df.get('Rank', range(1, len(df)+1)), errors='coerce')
        df['Peak'] = pd.to_numeric(df.get('Peak'), errors='coerce')
    except Exception as e:
        print(f"Cleaning error: {e}")

    ans1 = None
    if '2 bn' in question:
        ans1 = int(df[(df['Worldwide gross'] >= 2e9) & (df['Year'] < 2000)].shape[0])

    ans2 = None
    if '1.5 bn' in question:
        df_15 = df[df['Worldwide gross'] > 1.5e9]
        if not df_15.empty:
            ans2 = str(df_15.sort_values('Year').iloc[0]['Title'])

    ans3 = None
    if 'correlation' in question and 'Rank' in df.columns and 'Peak' in df.columns:
        ans3 = round(df[['Rank', 'Peak']].dropna().corr().iloc[0, 1], 6)

    img = DUMMY_IMAGE
    if 'scatterplot' in question and 'Rank' in df.columns and 'Peak' in df.columns:
        img = plot_scatter_with_regression(df, 'Rank', 'Peak')

    return {
        "ans1": ans1,
        "ans2": ans2,
        "ans3": ans3,
        "image": img
    }
