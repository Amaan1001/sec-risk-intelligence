"""
server.py — FastAPI web server for the SEC Risk Intelligence System
Serves the React frontend and exposes the pipeline as a REST API.

Usage:
    uv add fastapi uvicorn python-multipart
    uv run uvicorn server:app --reload --port 8000

Then open http://localhost:8000 in your browser.
"""

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils import load_env
load_env()

import agent1_fetcher
import agent2_analyzer
import agent3_synthesizer
import agent4_predictor

app = FastAPI(title="SEC Risk Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    ticker: str
    run_predictions: bool = True


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    ticker = req.ticker.strip().upper()
    if not ticker:
        raise HTTPException(400, "Ticker is required")
    try:
        a1 = agent1_fetcher.run(ticker)
        a2 = agent2_analyzer.run(a1)
        a4 = None
        if req.run_predictions:
            a4 = agent4_predictor.run(a2, run_backtest=True)
        report = agent3_synthesizer.run(a2, output_dir="./reports", agent4_output=a4)

        # Merge everything the frontend needs into one response
        return JSONResponse({
            "ticker":          a2["ticker"],
            "cik":             a2.get("cik", ""),
            "older_date":      a2["older_date"],
            "newer_date":      a2["newer_date"],
            "summary_counts":  a2["summary_counts"],
            "changes":         a2["changes"],
            "predictions":     a4["predictions"] if a4 else [],
            "track_record":    a4.get("track_record_global", {}) if a4 else {},
            "report":          report,
        })
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ── Serve the React frontend ───────────────────────────────────────────────
# After you build the frontend (see README), static files live in ./frontend/dist
_DIST = Path(__file__).parent / "frontend" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", response_class=HTMLResponse)
    def serve_spa(full_path: str):
        return (_DIST / "index.html").read_text()
else:
    @app.get("/", response_class=HTMLResponse)
    def root():
        return """
        <h2>Frontend not built yet.</h2>
        <p>Run: <code>cd frontend && npm install && npm run build</code></p>
        <p>API is live at <a href='/api/health'>/api/health</a></p>
        """
