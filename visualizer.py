
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

def plot_scatter_with_regression(df, x, y):
    plt.figure(figsize=(6, 4))
    sns.regplot(data=df, x=x, y=y, line_kws={"color": "red", "linestyle": "dotted"})
    plt.xlabel(x)
    plt.ylabel(y)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
