from pathlib import Path
import json
from flask import Flask, render_template, send_from_directory, abort
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

APP = Flask(__name__)

EXPORT_DIR = Path("data/exports")
IMG_DIR = Path("img")


def find_available_tickers(export_dir: Path):
    export_dir.mkdir(parents=True, exist_ok=True)
    files = list(export_dir.glob("*.csv"))
    tickers = set()
    for f in files:
        parts = f.name.split("_")
        if len(parts) > 1:
            tickers.add(parts[0])
    return sorted(tickers)


def latest_file_for(pattern: str, directory: Path):
    files = list(directory.glob(pattern))
    if not files:
        return None
    return sorted(files, key=lambda p: p.stat().st_mtime)[-1]


@APP.route("/")
def index():
    tickers = find_available_tickers(EXPORT_DIR)
    return render_template("index.html", tickers=tickers)


@APP.route("/ticker/<ticker>")
def ticker_page(ticker):
    ticker = ticker.upper()
    surface_path = latest_file_for(f"{ticker}_gex_surface_*.csv", EXPORT_DIR)
    strike_path = latest_file_for(f"{ticker}_gex_by_strike_*.csv", EXPORT_DIR)
    exp_path = latest_file_for(f"{ticker}_gex_by_expiration_*.csv", EXPORT_DIR)

    imgs = []
    if IMG_DIR.exists():
        imgs = sorted(IMG_DIR.glob(f"{ticker}_*.*"), key=lambda p: p.stat().st_mtime, reverse=True)

    heatmap_json = None
    scatter3d_json = None

    if surface_path:
        df = pd.read_csv(surface_path)
        if "expiration" in df.columns:
            df["expiration"] = pd.to_datetime(df["expiration"]).dt.date

        if not df.empty:
            # Pivot to create heatmap
            pivot = df.pivot(index="expiration", columns="strike", values="GEX").fillna(0)
            heatmap = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=[str(d) for d in pivot.index],
                colorscale="RdYlBu",
                reversescale=True,
            ))
            heatmap.update_layout(title=f"{ticker} GEX Surface (heatmap)")
            heatmap_json = json.dumps(heatmap, cls=PlotlyJSONEncoder)

            # 3D scatter
            scatter3d = go.Figure()
            scatter3d.add_trace(go.Scatter3d(
                x=df["strike"].astype(float),
                y=df["expiration"].astype(str),
                z=df["GEX"].astype(float),
                mode="markers",
                marker=dict(size=4, color=df["GEX"].astype(float), colorscale="Viridis", showscale=True)
            ))
            scatter3d.update_layout(title=f"{ticker} GEX Surface (3D scatter)")
            scatter3d_json = json.dumps(scatter3d, cls=PlotlyJSONEncoder)

    return render_template(
        "ticker.html",
        ticker=ticker,
        imgs=imgs,
        heatmap_json=heatmap_json,
        scatter3d_json=scatter3d_json,
        strike_csv=str(strike_path) if strike_path else None,
        exp_csv=str(exp_path) if exp_path else None,
    )


@APP.route("/img/<path:filename>")
def img_file(filename):
    path = IMG_DIR
    if not path.exists():
        abort(404)
    return send_from_directory(path, filename)


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=8501, debug=True)
