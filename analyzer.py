import pandas as pd
from visualizer import plot_scatter_with_regression

DUMMY_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4//8/AwAI/AL+XMG1WAAAAABJRU5ErkJggg=="

def analyze_data_files(question, file_dict):
    df = None
    for fname, path in file_dict.items():
        if fname.endswith(".csv"):
            df = pd.read_csv(path)
        elif fname.endswith(".parquet"):
            df = pd.read_parquet(path)

    if df is None:
        return {
            "slope": None,
            "image": DUMMY_IMAGE
        }

    if "delay" in question.lower():
        df['date_of_registration'] = pd.to_datetime(df['date_of_registration'], errors='coerce')
        df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')
        df['delay'] = (df['decision_date'] - df['date_of_registration']).dt.days
        avg_delay = df.groupby("year")['delay'].mean().dropna()

        slope = avg_delay.reset_index().corr().iloc[0, 1]
        slope = round(slope, 6)

        plot_uri = plot_scatter_with_regression(avg_delay.reset_index(), "year", "delay")

        return {
            "slope": slope,
            "image": plot_uri
        }

    return {
        "message": "Task not recognized",
        "image": DUMMY_IMAGE
    }
