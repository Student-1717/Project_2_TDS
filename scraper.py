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
        # Return 4 lines: first line error, rest empty
        return "No tables found\n\n\n" + DUMMY_IMAGE

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

    ans1 = str(df[(df['Worldwide gross'] >= 2e9) & (df['Year'] < 2000)].shape[0]) if '2 bn' in question else "?"
    ans2 = "?"
    if '1.5 bn' in question:
        df_15 = df[df['Worldwide gross'] > 1.5e9]
        if not df_15.empty:
            ans2 = str(df_15.sort_values('Year').iloc[0]['Title'])
    ans3 = "?"
    if 'correlation' in question and 'Rank' in df.columns and 'Peak' in df.columns:
        ans3 = str(round(df[['Rank', 'Peak']].dropna().corr().iloc[0, 1], 6))

    img = DUMMY_IMAGE
    if 'scatterplot' in question and 'Rank' in df.columns and 'Peak' in df.columns:
        img = plot_scatter_with_regression(df, 'Rank', 'Peak')

    # Return answers as a single string with 4 lines
    return f"{ans1}\n{ans2}\n{ans3}\n{img}"
