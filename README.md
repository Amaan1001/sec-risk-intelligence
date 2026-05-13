# SEC Risk Intelligence System
FE 524 — Prompt Engineering 

---

## Overview

A 4-agent Python pipeline that automatically retrieves two consecutive SEC 10-K filings for any US public company, identifies and classifies meaningful changes in the Risk Factors section, predicts which risks are likely to materialize within 12 months, and exports a plain-English risk intelligence report as both JSON and a formatted PDF.

Optionally, a React web frontend lets you run the pipeline and view results in a browser instead of the terminal.

```
Ticker
  └─▶ Agent 1 · Document Fetcher   — EDGAR REST API, no key required
        └─▶ Agent 2 · Risk Analyzer  — GPT-4.1-mini, 4-step prompt chain
              └─▶ Agent 4 · Predictor  — tiered probability forecast + backtest
                    └─▶ Agent 3 · Synthesizer — PDF + JSON report
```

The system uses only the free SEC EDGAR REST API for document retrieval and a single OpenAI API key for analysis. No data subscriptions or paid feeds are required.

---

## File Structure

```
agent1_fetcher.py       Agent 1: EDGAR queries, pagination, section extraction
agent2_analyzer.py      Agent 2: 4-step risk change classification via GPT-4.1-mini
agent3_synthesizer.py   Agent 3: report synthesis, PDF and JSON export
agent4_predictor.py     Agent 4: 12-month risk materialization forecast + backtest
main.py                 Orchestrator — wires all four agents together
server.py               FastAPI web server — exposes pipeline as REST API for the frontend
utils.py                Shared helpers — .env loading, LLM JSON parsing, truncation repair
diagnose.py             Quick diagnostic to inspect Agent 1 extraction output
evaluate.py             Accuracy and qualitative evaluation suite
requirements.txt        Python dependencies
.env                    Your OpenAI API key — never commit this
README.md
frontend/               React web frontend (optional)
  package.json
  vite.config.js
  index.html
  src/
    main.jsx
    App.jsx
```

---

## Prerequisites

### Python
- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) **or** standard pip

Install uv if you don't have it:
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Node.js and npm (only needed for the web frontend)
- Node.js 18 or higher
- npm (comes bundled with Node.js)

Download from https://nodejs.org — choose the LTS version. Verify after installing:
```bash
node --version   # should print v18.x.x or higher
npm --version    # should print 9.x.x or higher
```

---

## Python Setup

### 1. Create and activate a virtual environment

```bash
uv venv
```

Activate it:
```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\activate

# Windows (CMD)
.venv\Scripts\activate.bat
```

### 2. Install all Python dependencies

```bash
uv pip install -r requirements.txt
```

This installs everything — OpenAI SDK, EDGAR request handling, PDF generation, FastAPI server, and all shared utilities.

### 3. Create your .env file

Create a file called `.env` in the project root (same folder as `main.py`):

```
OPENAI_API_KEY=sk-...your-key-here...
```

Get your key from https://platform.openai.com/api-keys. Never commit this file — it is already in `.gitignore`.

---

## Running the Pipeline (Terminal Only)

No frontend required. Just run:

```bash
uv run main.py AAPL
```

Other examples:

```bash
uv run main.py MSFT --output-dir ./reports
uv run main.py TSLA --save-intermediate    # saves agent1/2/4 JSON at each step for debugging
uv run main.py NVDA --no-predict           # skip Agent 4, runs faster
```

Output files are written to `./reports/` (or `--output-dir`):
- `AAPL_risk_report_<date>.json` — structured report data
- `AAPL_risk_report_<date>.pdf` — formatted PDF report

---

## Running the Web Frontend

The frontend lets you type a ticker in a browser and see the full report rendered as an interactive dashboard. It requires two terminals running simultaneously.

### Step 1 — Install frontend dependencies (first time only)

```bash
cd frontend
npm install
cd ..
```

### Step 2 — Start the Python backend server (Terminal 1)

```bash
uv run uvicorn server:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

Keep this terminal open.

### Step 3 — Start the React frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in ...ms
  ➜  Local:   http://localhost:5173/
```

### Step 4 — Open the browser

Go to **http://localhost:5173**

Type any ticker (e.g. `AAPL`, `MSFT`, `TSLA`) and click **Run Pipeline**. The full pipeline runs on the backend and results render in the browser. Reports are still saved to `./reports/` as PDF and JSON.

---

## Diagnostics

If a ticker returns empty or suspiciously short Risk Factors, run the diagnostic before anything else:

```bash
uv run diagnose.py MSFT
```

Prints the first and last characters of each extracted section, vocabulary overlap between the two filings, and saves a full `MSFT_agent1_diagnostic.json` for inspection.

---

## Evaluation

### Mode 1 — Generate predictions and annotation templates

```bash
uv run evaluate.py generate \
    --tickers AAPL MSFT TSLA NVDA AMZN GOOGL META JPM \
    --output-dir ./eval_data
```

### Mode 2 — Score predictions against ground truth

Compile filled-in annotation templates into a single `ground_truth.json`:

```json
{
  "AAPL": [
    {"title": "Macroeconomic conditions", "ground_truth_change_type": "Escalating Risk", "ground_truth_severity": 4},
    {"title": "Regulatory risk", "ground_truth_change_type": "Stable", "ground_truth_severity": 3}
  ]
}
```

Then score:

```bash
uv run evaluate.py score \
    --predictions ./eval_data \
    --ground-truth ./ground_truth.json
```

Reports Precision / Recall / F1 per category, macro averages, overall accuracy, and severity MAE. Results saved to `./eval_data/classification_metrics.json`.

### Mode 3 — Qualitative coherence scoring

```bash
uv run evaluate.py qualitative --scores ./qualitative_scores.json
```

---

## Agent Details

### Agent 1 — Document Fetcher (`agent1_fetcher.py`)
Resolves ticker → CIK via `company_tickers.json`, fetches two consecutive 10-K filings from the EDGAR submissions API, downloads the primary `.htm` document for each, and extracts **Item 1A (Risk Factors)** and **Item 7 (MD&A)** using a three-pass regex strategy:

- **Pass 1** — Classic inline format: `ITEM 1A. RISK FACTORS` on one line. Works for AAPL, NVDA, TSLA, AMZN, and most smaller filers.
- **Pass 2** — Split label format: `ITEM 1A.` and `RISK FACTORS` on separate lines (iXBRL two-column layout). Works for MSFT, GOOGL, JPM, META post-2023.
- **Pass 3** — Standalone heading fallback for bare section titles.

Also resolves EDGAR pagination — the `recent` block caps at ~1000 entries. For large filers like JPM, Agent 1 follows `data["filings"]["files"]` until it collects 2 qualifying 10-K filings. The resolved CIK is passed forward through the entire pipeline so Agent 4 never needs to re-fetch it.

### Agent 2 — Risk Change Analyzer (`agent2_analyzer.py`)
4-step GPT-4.1-mini prompt chain:

1. **Extract** structured risk topics from the older filing
2. **Extract** structured risk topics from the newer filing
3. **Compare and classify** each change as `New Risk`, `Escalating Risk`, `Stable`, or `Resolved` with severity score 1–5
4. **Semantic consolidation pass** — checks every `New Risk` against the older filing's list and reclassifies anything substantively the same under different wording (prevents MSFT's 18 spurious new risks)

Includes a retry mechanism: if the LLM response is truncated or malformed JSON, Agent 2 sends one follow-up API call asking the model to re-emit valid JSON before raising an error.

### Agent 3 — Report Synthesizer (`agent3_synthesizer.py`)
Generates:
- **Overall Risk Rating**: `Escalating`, `Stable`, or `Improving`
- **Rating justification** (2–3 sentences)
- **Key investor takeaways** (3–5 bullets)
- **Detailed findings** per risk with severity scores
- **Analyst note** (forward-looking commentary)

Exports as formatted JSON and PDF (via `reportlab`). Falls back to `.txt` if `reportlab` is unavailable.

### Agent 4 — Predictor & Backtester (`agent4_predictor.py`)
Two-stage 12-month forecast:

1. **Tier ranking** — GPT-4.1-mini assigns each risk to tier 1 (most likely), 2, or 3
2. **Constrained probabilities** — Tier 1: 55–80% · Tier 2: 30–54% · Tier 3: 10–29%

Forces real spread across predictions instead of clustering around 50%.

Backtesting compares prior predictions against subsequent 10-Q filings. MISS rule: risks absent from both MD&A and risk sections of the follow-up 10-Q score as `MISS`. Results stored in `predictions.db` with a `UNIQUE (ticker, filing_date, risk_title)` constraint — running the pipeline twice for the same ticker never creates duplicate rows.

### `utils.py` — Shared Utilities
- `load_env()` — loads `.env` using auto-search from cwd, falls back to the project directory. Works regardless of which directory you run the script from.
- `parse_llm_json(raw)` — strips markdown fences (` ```json `, ` ``` `, including variants with spaces) using a proper regex before parsing. Used by all agents.
- `repair_truncated_json_array(raw)` — attempts to recover a valid JSON array from a truncated LLM response before triggering a retry.

### `server.py` — FastAPI Web Server
Wraps the pipeline as a REST API:
- `POST /api/analyze` — runs the full pipeline for a given ticker, returns combined JSON
- `GET /api/health` — health check

In production (after `npm run build`), also serves the compiled React frontend as a single-page app from `./frontend/dist/`.

---

## Known Issues and Limitations

- **iXBRL two-column filers**: MSFT, GOOGL, JPM, and META render section headings as two separate layout elements. The three-pass extractor handles these but may still yield shorter sections than classic filers. Run `diagnose.py` first if output looks short.
- **Section extraction is regex-based**: unusual filing structures may still yield empty sections. Agent 1 emits a `WARNING` when final content is below 500 chars.
- **EDGAR rate limit**: SEC enforces a polite crawl policy. Agent 1 includes 0.4s delays between requests. Do not remove these.
- **Backtest requires time**: `predictions.db` accumulates signal as subsequent 10-Qs are filed. A fresh install shows an empty track record on first run — this fills in naturally over time.
- **Model**: configured for `gpt-4.1-mini` by default. Switch to `gpt-4o` in `agent2_analyzer.py` and `agent3_synthesizer.py` for higher quality at increased cost.
- **Frontend requires Node.js**: the Python pipeline (`main.py`) works completely independently without Node. Node is only needed if you want the browser UI.

---

## Data Sources

| Source | Description | Cost |
|--------|-------------|------|
| SEC EDGAR REST API | All 10-K/10-Q filings for US public companies | Free |
| `company_tickers.json` | Ticker → CIK mapping maintained by SEC | Free |
| OpenAI API | LLM for risk analysis and report synthesis | API key required |

---

## Quick Reference — All Commands

```bash
# ── Setup ──────────────────────────────────────────────────────────────────
uv venv                                   # create virtual environment
source .venv/bin/activate                 # activate (macOS/Linux)
.venv\Scripts\activate                    # activate (Windows)
uv pip install -r requirements.txt        # install all Python dependencies
cd frontend && npm install && cd ..       # install frontend dependencies (optional)

# ── Run pipeline in terminal ───────────────────────────────────────────────
uv run main.py AAPL
uv run main.py MSFT --output-dir ./reports
uv run main.py TSLA --save-intermediate
uv run main.py NVDA --no-predict

# ── Run with web frontend (two terminals) ─────────────────────────────────
uv run uvicorn server:app --reload --port 8000   # Terminal 1: backend
cd frontend && npm run dev                        # Terminal 2: frontend
# Then open http://localhost:5173

# ── Diagnostics & evaluation ──────────────────────────────────────────────
uv run diagnose.py MSFT
uv run evaluate.py generate --tickers AAPL MSFT TSLA --output-dir ./eval_data
uv run evaluate.py score --predictions ./eval_data --ground-truth ./ground_truth.json
uv run evaluate.py qualitative --scores ./qualitative_scores.json
```

---

## Notes

- `predictions.db` is also excluded — it accumulates locally and is fully compatible across versions.
- Generated reports (PDFs, JSONs) are excluded from git by default.
- The frontend dev server proxies all `/api` requests to `localhost:8000` automatically — no CORS configuration needed during development.
