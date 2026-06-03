from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from flask import Flask, abort, jsonify, render_template, send_from_directory
from plotly.utils import PlotlyJSONEncoder

APP = Flask(__name__)
app = APP

EXPORT_DIR = Path("data/exports")
IMG_DIR = Path("img")

EXPORT_FILE_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9]+)_(?P<kind>gex_by_strike|gex_by_expiration|gex_surface|cumulative_gex|summary)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{6})\.(?P<ext>csv|json)$"
)


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.{digits}f}"


def _latest_file_for(pattern: str, directory: Path) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def find_available_tickers(export_dir: Path) -> list[str]:
    export_dir.mkdir(parents=True, exist_ok=True)
    tickers = set()
    for file in export_dir.iterdir():
        match = EXPORT_FILE_RE.match(file.name)
        if match:
            tickers.add(match.group("ticker"))
    return sorted(tickers)


def latest_exports_for_ticker(ticker: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for file in EXPORT_DIR.glob(f"{ticker}_*"):
        match = EXPORT_FILE_RE.match(file.name)
        if not match:
            continue
        kind = match.group("kind")
        current = files.get(kind)
        if current is None or file.stat().st_mtime > current.stat().st_mtime:
            files[kind] = file
    return files


def _load_series(path: Path, value_col: str) -> pd.Series:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.Series(dtype=float)
    index_col = frame.columns[0]
    values = pd.to_numeric(frame[value_col], errors="coerce")
    series = pd.Series(values.values, index=frame[index_col])
    return series.dropna()


def _load_surface(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["expiration", "strike", "GEX"])
    frame["expiration"] = pd.to_datetime(frame["expiration"], errors="coerce")
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce")
    frame["GEX"] = pd.to_numeric(frame["GEX"], errors="coerce")
    return frame.dropna(subset=["expiration", "strike", "GEX"])


def _estimate_gamma_flip(cumulative: pd.Series) -> dict:
    if cumulative.empty:
        return {"flip_strike": None, "confidence": "none", "message": "no cumulative data"}

    ordered = cumulative.sort_index().astype(float)
    signs = ordered.apply(lambda value: -1 if value < 0 else (1 if value > 0 else 0)).to_numpy()
    crossing_idx = [i for i in range(len(signs) - 1) if signs[i] != signs[i + 1]]

    if not crossing_idx:
        return {"flip_strike": None, "confidence": "none", "message": "no zero crossing"}

    idx = crossing_idx[0]
    x0 = float(ordered.index[idx])
    x1 = float(ordered.index[idx + 1])
    y0 = float(ordered.iloc[idx])
    y1 = float(ordered.iloc[idx + 1])

    if y1 == y0:
        flip = x0
    else:
        flip = x0 - y0 * (x1 - x0) / (y1 - y0)

    slope = abs(y1 - y0) / max(abs(x1 - x0), 1e-9)
    if slope >= 0.10:
        confidence = "high"
    elif slope >= 0.03:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "flip_strike": float(flip),
        "confidence": confidence,
        "message": "derived from cumulative curve",
    }


def _derive_summary(ticker: str, strike: pd.Series, expiration: pd.Series, cumulative: pd.Series, surface: pd.DataFrame) -> dict:
    strike = strike.sort_index().astype(float) if not strike.empty else strike
    total_gex = float(strike.sum()) if not strike.empty else 0.0

    if not strike.empty:
        call_wall_strike = float(strike.idxmax())
        call_wall_gex = float(strike.max())
        put_wall_strike = float(strike.idxmin())
        put_wall_gex = float(strike.min())
    else:
        call_wall_strike = None
        call_wall_gex = None
        put_wall_strike = None
        put_wall_gex = None

    positive = strike[strike > 0].sort_values(ascending=False).head(5) if not strike.empty else pd.Series(dtype=float)
    negative = strike[strike < 0].sort_values().head(5) if not strike.empty else pd.Series(dtype=float)

    nearest_exp = None
    largest_exp = None
    if not expiration.empty:
        exp = expiration.copy()
        exp.index = pd.to_datetime(exp.index, errors="coerce")
        exp = exp.dropna().sort_index().astype(float)
        if not exp.empty:
            nearest_key = exp.index[0]
            largest_key = exp.abs().idxmax()
            nearest_exp = {
                "expiration": nearest_key.date().isoformat(),
                "gex_bn_per_pct": float(exp.loc[nearest_key]),
            }
            largest_exp = {
                "expiration": largest_key.date().isoformat(),
                "gex_bn_per_pct": float(exp.loc[largest_key]),
            }

    spot_guess = None
    if not surface.empty:
        latest_exp = surface.sort_values("expiration").iloc[0]
        spot_guess = float(latest_exp["strike"])

    summary = {
        "ticker": ticker,
        "generated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "spot_price": spot_guess,
        "option_count": None,
        "total_gex_bn_per_pct": total_gex,
        "net_gamma_regime": "LONG gamma" if total_gex >= 0 else "SHORT gamma",
        "call_wall": {"strike": call_wall_strike, "gex_bn_per_pct": call_wall_gex},
        "put_wall": {"strike": put_wall_strike, "gex_bn_per_pct": put_wall_gex},
        "gamma_flip": _estimate_gamma_flip(cumulative),
        "top_positive_gex_strikes": [
            {"strike": float(k), "gex_bn_per_pct": float(v)} for k, v in positive.items()
        ],
        "top_negative_gex_strikes": [
            {"strike": float(k), "gex_bn_per_pct": float(v)} for k, v in negative.items()
        ],
        "nearest_expiration": nearest_exp,
        "largest_expiration_gex": largest_exp,
    }
    return summary


def _build_narrative(summary: dict) -> list[str]:
    lines = []
    total = float(summary.get("total_gex_bn_per_pct") or 0.0)
    regime = summary.get("net_gamma_regime") or "N/A"
    flip = (summary.get("gamma_flip") or {}).get("flip_strike")

    if abs(total) < 2:
        lines.append("Gamma profile is close to neutral; hedge flows may be less sticky intraday.")
    elif total > 0:
        lines.append("Long-gamma regime suggests mean-reversion pressure around key walls.")
    else:
        lines.append("Short-gamma regime can amplify trend moves and volatility pockets.")

    call_wall = (summary.get("call_wall") or {}).get("strike")
    put_wall = (summary.get("put_wall") or {}).get("strike")
    if call_wall is not None and put_wall is not None:
        width = abs(call_wall - put_wall)
        lines.append(f"Wall corridor width is {width:,.0f} points between put/call walls.")

    if flip is not None:
        lines.append(f"Estimated gamma flip sits near {flip:,.2f}; monitor regime transition around this zone.")

    lines.append(f"Current regime label: {regime}.")
    return lines


def _build_concepts(summary: dict) -> dict:
    total_gex = float(summary.get("total_gex_bn_per_pct") or 0.0)
    flip = (summary.get("gamma_flip") or {}).get("flip_strike")
    spot = summary.get("spot_price")
    call_wall = (summary.get("call_wall") or {}).get("strike")
    put_wall = (summary.get("put_wall") or {}).get("strike")

    regime_score = max(-100.0, min(100.0, total_gex * 3.5))

    nearest_wall_distance = None
    if spot is not None and call_wall is not None and put_wall is not None:
        nearest_wall_distance = min(abs(spot - call_wall), abs(spot - put_wall))

    wall_tension = None
    if nearest_wall_distance is not None:
        wall_tension = max(0.0, 100.0 - nearest_wall_distance / max(spot, 1.0) * 10000.0)

    return {
        "base_total_gex": total_gex,
        "base_flip": flip,
        "base_spot": spot,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "regime_score": regime_score,
        "wall_tension": wall_tension,
    }


def _plotly_theme(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,10,22,0.55)",
        font=dict(color="#e2e8f0"),
        margin=dict(l=20, r=20, t=55, b=20),
        height=height,
    )
    return fig


def _make_heatmap(surface: pd.DataFrame, ticker: str) -> str | None:
    if surface.empty:
        return None

    pivot = (
        surface.pivot_table(index="expiration", columns="strike", values="GEX", aggfunc="sum")
        .sort_index()
        .fillna(0)
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[float(val) for val in pivot.columns],
            y=[idx.strftime("%Y-%m-%d") for idx in pd.to_datetime(pivot.index)],
            colorscale="RdYlBu_r",
            colorbar=dict(title="GEX (M$ / %)"),
        )
    )
    fig = _plotly_theme(fig, f"{ticker} Gamma Surface Heatmap", height=500)
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def _make_3d(surface: pd.DataFrame, ticker: str) -> str | None:
    if surface.empty:
        return None

    fig = go.Figure(
        data=go.Scatter3d(
            x=surface["strike"],
            y=surface["expiration"].dt.strftime("%Y-%m-%d"),
            z=surface["GEX"],
            mode="markers",
            marker=dict(size=4, color=surface["GEX"], colorscale="Turbo", showscale=True),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Expiration",
            zaxis_title="GEX (M$ / %)",
            bgcolor="rgba(0,0,0,0)",
        )
    )
    fig = _plotly_theme(fig, f"{ticker} Surface Orbit (3D)", height=500)
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def _make_strike_bar(strike: pd.Series, ticker: str) -> str | None:
    if strike.empty:
        return None
    ordered = strike.sort_index().astype(float)
    colors = ["#34d399" if val >= 0 else "#fb7185" for val in ordered.values]
    fig = go.Figure(data=go.Bar(x=[float(v) for v in ordered.index], y=ordered.values, marker_color=colors))
    fig = _plotly_theme(fig, f"{ticker} GEX by Strike")
    fig.update_xaxes(title="Strike")
    fig.update_yaxes(title="GEX (Bn$ / %)")
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def _make_exp_bar(expiration: pd.Series, ticker: str) -> str | None:
    if expiration.empty:
        return None
    exp = expiration.copy()
    exp.index = pd.to_datetime(exp.index, errors="coerce")
    exp = exp.dropna().sort_index().astype(float)
    fig = go.Figure(data=go.Bar(x=[v.strftime("%Y-%m-%d") for v in exp.index], y=exp.values, marker_color="#60a5fa"))
    fig = _plotly_theme(fig, f"{ticker} GEX by Expiration")
    fig.update_xaxes(title="Expiration")
    fig.update_yaxes(title="GEX (Bn$ / %)")
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def _make_cumulative(cumulative: pd.Series, ticker: str) -> str | None:
    if cumulative.empty:
        return None
    curve = cumulative.copy()
    curve.index = pd.to_numeric(curve.index, errors="coerce")
    curve = curve.dropna().sort_index().astype(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", line=dict(color="#f59e0b", width=2.5)))
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    fig = _plotly_theme(fig, f"{ticker} Cumulative GEX")
    fig.update_xaxes(title="Strike")
    fig.update_yaxes(title="Cumulative (Bn$ / %)")
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def load_ticker_bundle(ticker: str) -> dict:
    ticker = ticker.upper()
    files = latest_exports_for_ticker(ticker)

    strike = _load_series(files["gex_by_strike"], "gex_bn_per_pct") if "gex_by_strike" in files else pd.Series(dtype=float)
    expiration = _load_series(files["gex_by_expiration"], "gex_bn_per_pct") if "gex_by_expiration" in files else pd.Series(dtype=float)
    cumulative = _load_series(files["cumulative_gex"], "cumulative_gex_bn_per_pct") if "cumulative_gex" in files else pd.Series(dtype=float)
    surface = _load_surface(files["gex_surface"]) if "gex_surface" in files else pd.DataFrame(columns=["expiration", "strike", "GEX"])

    summary = None
    if "summary" in files:
        with files["summary"].open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    if summary is None:
        summary = _derive_summary(ticker, strike, expiration, cumulative, surface)

    images = sorted(IMG_DIR.glob(f"{ticker}_*.*"), key=lambda p: p.stat().st_mtime, reverse=True) if IMG_DIR.exists() else []
    updated_at = datetime.fromtimestamp(max(file.stat().st_mtime for file in files.values())) if files else None

    concepts = _build_concepts(summary)
    narrative = _build_narrative(summary)

    metrics = {
        "ticker": ticker,
        "regime": summary.get("net_gamma_regime", "N/A"),
        "total_gex": _fmt(summary.get("total_gex_bn_per_pct"), 3),
        "spot": _fmt(summary.get("spot_price"), 2),
        "flip": _fmt((summary.get("gamma_flip") or {}).get("flip_strike"), 2),
        "call_wall": _fmt((summary.get("call_wall") or {}).get("strike"), 2),
        "put_wall": _fmt((summary.get("put_wall") or {}).get("strike"), 2),
        "updated_at": updated_at.strftime("%Y-%m-%d %H:%M:%S") if updated_at else "N/A",
    }

    charts = {
        "heatmap": _make_heatmap(surface, ticker),
        "surface3d": _make_3d(surface, ticker),
        "strike": _make_strike_bar(strike, ticker),
        "expiration": _make_exp_bar(expiration, ticker),
        "cumulative": _make_cumulative(cumulative, ticker),
    }

    return {
        "ticker": ticker,
        "files": files,
        "summary": summary,
        "metrics": metrics,
        "concepts": concepts,
        "narrative": narrative,
        "charts": charts,
        "images": images,
    }


@APP.route("/")
def index():
    tickers = find_available_tickers(EXPORT_DIR)
    cards = [load_ticker_bundle(ticker)["metrics"] for ticker in tickers]
    return render_template("index.html", cards=cards)


@APP.route("/ticker/<ticker>")
def ticker_page(ticker: str):
    bundle = load_ticker_bundle(ticker)
    if not bundle["files"]:
        abort(404)

    downloads = []
    for kind in ["summary", "gex_by_strike", "cumulative_gex", "gex_by_expiration", "gex_surface"]:
        file = bundle["files"].get(kind)
        if file:
            downloads.append({"label": kind.replace("_", " "), "filename": file.name})

    return render_template(
        "ticker.html",
        ticker=bundle["ticker"],
        metrics=bundle["metrics"],
        summary=bundle["summary"],
        concepts=bundle["concepts"],
        narrative=bundle["narrative"],
        charts=bundle["charts"],
        downloads=downloads,
        imgs=bundle["images"],
    )


@APP.route("/api/ticker/<ticker>/concepts")
def ticker_concepts(ticker: str):
    bundle = load_ticker_bundle(ticker)
    if not bundle["files"]:
        abort(404)
    return jsonify(bundle["concepts"])


@APP.route("/exports/<path:filename>")
def export_file(filename: str):
    return send_from_directory(EXPORT_DIR, filename)


@APP.route("/img/<path:filename>")
def img_file(filename: str):
    if not IMG_DIR.exists():
        abort(404)
    return send_from_directory(IMG_DIR, filename)


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=8501, debug=True)
