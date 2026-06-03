from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, render_template, send_from_directory
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

APP = Flask(__name__)
app = APP

EXPORT_DIR = Path("data/exports")
IMG_DIR = Path("img")

EXPORT_FILE_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9]+)_(?P<kind>gex_by_strike|gex_by_expiration|gex_surface|cumulative_gex|summary)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{6})\.(?P<ext>csv|json)$"
)


def fmt_num(value, digits=2):
    if value is None:
        return "N/A"
    return f"{value:,.{digits}f}"


def find_available_tickers(export_dir: Path):
    export_dir.mkdir(parents=True, exist_ok=True)
    tickers = set()
    for file in export_dir.iterdir():
        match = EXPORT_FILE_RE.match(file.name)
        if match:
            tickers.add(match.group("ticker"))
    return sorted(tickers)


def latest_export_map(ticker: str, export_dir: Path):
    latest = {}
    for file in export_dir.glob(f"{ticker}_*"):
        match = EXPORT_FILE_RE.match(file.name)
        if not match:
            continue
        kind = match.group("kind")
        current = latest.get(kind)
        if current is None or file.stat().st_mtime > current.stat().st_mtime:
            latest[kind] = file
    return latest


def list_img_snapshots(ticker: str, img_dir: Path):
    if not img_dir.exists():
        return []
    return sorted(img_dir.glob(f"{ticker}_*.*"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_series_csv(path: Path, value_col: str):
    df = pd.read_csv(path)
    if df.empty:
        return pd.Series(dtype=float)
    index_col = df.columns[0]
    series = pd.Series(df[value_col].astype(float).values, index=df[index_col])
    if value_col == "gex_bn_per_pct":
        try:
            series.index = series.index.astype(float)
        except Exception:
            pass
    return series


def load_surface_csv(path: Path):
    df = pd.read_csv(path)
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
    if "strike" in df.columns:
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    if "GEX" in df.columns:
        df["GEX"] = pd.to_numeric(df["GEX"], errors="coerce")
    return df.dropna(subset=["expiration", "strike", "GEX"])


def estimate_gamma_flip_from_cumulative(cumulative: pd.Series):
    if cumulative.empty:
        return None

    values = cumulative.astype(float).values
    strikes = cumulative.index.astype(float)
    signs = pd.Series(values).apply(lambda x: -1 if x < 0 else (1 if x > 0 else 0)).to_numpy()
    crossings = [i for i in range(len(signs) - 1) if signs[i] != signs[i + 1]]
    if not crossings:
        return None

    idx = crossings[0]
    x0, x1 = float(strikes[idx]), float(strikes[idx + 1])
    y0, y1 = float(values[idx]), float(values[idx + 1])
    if y1 == y0:
        return x0
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def build_summary_from_exports(ticker: str, strike: pd.Series, expiration: pd.Series, cumulative: pd.Series):
    strike = strike.sort_index()
    total_gex = float(strike.sum()) if not strike.empty else 0.0
    regime = "LONG gamma" if total_gex >= 0 else "SHORT gamma"

    call_wall_strike = float(strike.idxmax()) if not strike.empty else None
    call_wall_gex = float(strike.max()) if not strike.empty else None
    put_wall_strike = float(strike.idxmin()) if not strike.empty else None
    put_wall_gex = float(strike.min()) if not strike.empty else None

    pos = strike[strike > 0].sort_values(ascending=False).head(5)
    neg = strike[strike < 0].sort_values().head(5)

    flip = estimate_gamma_flip_from_cumulative(cumulative)

    summary = {
        "ticker": ticker,
        "generated_at_utc": None,
        "spot_price": None,
        "option_count": None,
        "total_gex_bn_per_pct": total_gex,
        "net_gamma_regime": regime,
        "call_wall": {"strike": call_wall_strike, "gex_bn_per_pct": call_wall_gex},
        "put_wall": {"strike": put_wall_strike, "gex_bn_per_pct": put_wall_gex},
        "gamma_flip": {
            "flip_strike": flip,
            "confidence": "estimated from cumulative exports" if flip is not None else "none",
            "message": "derived",
        },
        "top_positive_gex_strikes": [
            {"strike": float(k), "gex_bn_per_pct": float(v)} for k, v in pos.items()
        ],
        "top_negative_gex_strikes": [
            {"strike": float(k), "gex_bn_per_pct": float(v)} for k, v in neg.items()
        ],
        "nearest_expiration": None,
        "largest_expiration_gex": None,
    }

    if not expiration.empty:
        exp = expiration.copy()
        exp.index = pd.to_datetime(exp.index, errors="coerce")
        exp = exp.dropna()
        if not exp.empty:
            near_dt = exp.sort_index().index[0]
            max_dt = exp.abs().idxmax()
            summary["nearest_expiration"] = {
                "expiration": near_dt.date().isoformat(),
                "gex_bn_per_pct": float(exp.loc[near_dt]),
            }
            summary["largest_expiration_gex"] = {
                "expiration": max_dt.date().isoformat(),
                "gex_bn_per_pct": float(exp.loc[max_dt]),
            }

    return summary


def load_ticker_bundle(ticker: str):
    ticker = ticker.upper()
    files = latest_export_map(ticker, EXPORT_DIR)

    strike = pd.Series(dtype=float)
    expiration = pd.Series(dtype=float)
    cumulative = pd.Series(dtype=float)
    surface = pd.DataFrame(columns=["expiration", "strike", "GEX"])

    if "gex_by_strike" in files:
        strike = load_series_csv(files["gex_by_strike"], "gex_bn_per_pct")
    if "gex_by_expiration" in files:
        expiration = load_series_csv(files["gex_by_expiration"], "gex_bn_per_pct")
    if "cumulative_gex" in files:
        cumulative = load_series_csv(files["cumulative_gex"], "cumulative_gex_bn_per_pct")
    if "gex_surface" in files:
        surface = load_surface_csv(files["gex_surface"])

    summary = None
    if "summary" in files:
        with files["summary"].open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    if summary is None:
        summary = build_summary_from_exports(ticker, strike, expiration, cumulative)

    update_candidates = [f.stat().st_mtime for f in files.values()]
    updated_at = datetime.fromtimestamp(max(update_candidates)) if update_candidates else None

    return {
        "ticker": ticker,
        "files": files,
        "strike": strike,
        "expiration": expiration,
        "cumulative": cumulative,
        "surface": surface,
        "summary": summary,
        "images": list_img_snapshots(ticker, IMG_DIR),
        "updated_at": updated_at,
    }


def make_heatmap(surface: pd.DataFrame, ticker: str):
    if surface.empty:
        return None
    pivot = surface.pivot_table(index="expiration", columns="strike", values="GEX", aggfunc="sum").sort_index()
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.fillna(0).values,
            x=[float(c) for c in pivot.columns],
            y=[d.strftime("%Y-%m-%d") for d in pd.to_datetime(pivot.index)],
            colorscale="RdYlBu_r",
            colorbar=dict(title="GEX (M$ / %)"),
        )
    )
    fig.update_layout(title=f"{ticker} GEX Surface Heatmap", margin=dict(l=20, r=20, t=55, b=20), height=520)
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_surface_scatter(surface: pd.DataFrame, ticker: str):
    if surface.empty:
        return None
    fig = go.Figure(
        data=go.Scatter3d(
            x=surface["strike"],
            y=surface["expiration"].dt.strftime("%Y-%m-%d"),
            z=surface["GEX"],
            mode="markers",
            marker=dict(size=4, color=surface["GEX"], colorscale="Viridis", showscale=True),
        )
    )
    fig.update_layout(
        title=f"{ticker} 3D GEX Surface",
        scene=dict(xaxis_title="Strike", yaxis_title="Expiration", zaxis_title="GEX (M$ / %)"),
        margin=dict(l=20, r=20, t=55, b=20),
        height=520,
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_strike_chart(strike: pd.Series, ticker: str):
    if strike.empty:
        return None
    strike = strike.sort_index()
    colors = ["#22c55e" if v >= 0 else "#f43f5e" for v in strike.values]
    fig = go.Figure(
        data=go.Bar(
            x=[float(s) for s in strike.index],
            y=strike.values,
            marker_color=colors,
            name="GEX by strike",
        )
    )
    fig.update_layout(
        title=f"{ticker} GEX by Strike",
        xaxis_title="Strike",
        yaxis_title="GEX (Bn$ / %)",
        margin=dict(l=20, r=20, t=55, b=20),
        height=420,
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_exp_chart(expiration: pd.Series, ticker: str):
    if expiration.empty:
        return None
    exp = expiration.copy()
    exp.index = pd.to_datetime(exp.index, errors="coerce")
    exp = exp.dropna().sort_index()
    fig = go.Figure(
        data=go.Bar(
            x=[d.strftime("%Y-%m-%d") for d in exp.index],
            y=exp.values,
            marker_color="#60a5fa",
            name="GEX by expiration",
        )
    )
    fig.update_layout(
        title=f"{ticker} GEX by Expiration",
        xaxis_title="Expiration",
        yaxis_title="GEX (Bn$ / %)",
        margin=dict(l=20, r=20, t=55, b=20),
        height=420,
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_cumulative_chart(cumulative: pd.Series, ticker: str):
    if cumulative.empty:
        return None
    cumulative = cumulative.copy()
    cumulative.index = cumulative.index.astype(float)
    cumulative = cumulative.sort_index()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            mode="lines",
            line=dict(color="#f59e0b", width=2.5),
            name="Cumulative GEX",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(
        title=f"{ticker} Cumulative GEX by Strike",
        xaxis_title="Strike",
        yaxis_title="Cumulative GEX (Bn$ / %)",
        margin=dict(l=20, r=20, t=55, b=20),
        height=420,
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


@APP.route("/")
def index():
    cards = []
    for ticker in find_available_tickers(EXPORT_DIR):
        bundle = load_ticker_bundle(ticker)
        summary = bundle["summary"]
        cards.append(
            {
                "ticker": ticker,
                "regime": summary.get("net_gamma_regime", "N/A"),
                "total_gex": fmt_num(summary.get("total_gex_bn_per_pct"), 3),
                "spot": fmt_num(summary.get("spot_price"), 2),
                "flip": fmt_num((summary.get("gamma_flip") or {}).get("flip_strike"), 2),
                "call_wall": fmt_num((summary.get("call_wall") or {}).get("strike"), 2),
                "put_wall": fmt_num((summary.get("put_wall") or {}).get("strike"), 2),
                "updated_at": bundle["updated_at"].strftime("%Y-%m-%d %H:%M:%S") if bundle["updated_at"] else "N/A",
            }
        )

    return render_template("index.html", cards=cards)


@APP.route("/ticker/<ticker>")
def ticker_page(ticker):
    ticker = ticker.upper()
    bundle = load_ticker_bundle(ticker)
    if not bundle["files"]:
        abort(404)

    summary = bundle["summary"]

    chart_data = {
        "heatmap_json": make_heatmap(bundle["surface"], ticker),
        "scatter3d_json": make_surface_scatter(bundle["surface"], ticker),
        "strike_json": make_strike_chart(bundle["strike"], ticker),
        "exp_json": make_exp_chart(bundle["expiration"], ticker),
        "cumulative_json": make_cumulative_chart(bundle["cumulative"], ticker),
    }

    downloads = []
    for kind in ["summary", "gex_by_strike", "cumulative_gex", "gex_by_expiration", "gex_surface"]:
        file = bundle["files"].get(kind)
        if file is not None:
            downloads.append({"label": kind.replace("_", " "), "filename": file.name})

    return render_template(
        "ticker.html",
        ticker=ticker,
        summary=summary,
        chart_data=chart_data,
        downloads=downloads,
        imgs=bundle["images"],
        updated_at=bundle["updated_at"].strftime("%Y-%m-%d %H:%M:%S") if bundle["updated_at"] else "N/A",
    )


@APP.route("/exports/<path:filename>")
def export_file(filename):
    return send_from_directory(EXPORT_DIR, filename)


@APP.route("/img/<path:filename>")
def img_file(filename):
    path = IMG_DIR
    if not path.exists():
        abort(404)
    return send_from_directory(path, filename)


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=8501, debug=True)
