# SEC Risk Intelligence System
**FE 524 — Prompt Engineering Lab for Business Applications**  
Group Atlanta: Amaan Shaikh · Sakshi Kadam · Tanishka Dighe · Abhilash Athili

---

## Overview

A 4-agent Python pipeline that automatically retrieves two consecutive SEC 10-K filings for any US public company, identifies and classifies meaningful changes in the Risk Factors section, predicts which risks are likely to materialize within 12 months, and exports a plain-English risk intelligence report as both JSON and a formatted PDF.

```
Ticker
  └─▶ Agent 1 · Document Fetcher   — EDGAR REST API, no key required
        └─▶ Agent 2 · Risk Analyzer  — GPT-4.1-mini, 4-step prompt chain
              └─▶ Agent 4 · Predictor  — tiered probability forecast + backtest
                    └─▶ Agent 3 · Synthesizer — PDF + JSON report
```

The system uses only the free SEC EDGAR REST API for document retrieval and a single OpenAI API key for analysis. No data subscriptions or paid feeds are required.

---

## Quickstart

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

Run the full pipeline:
```bash
python main.py AAPL
python main.py MSFT --output-dir ./reports
python main.py TSLA --save-intermediate   # saves agent JSON at each step
python main.py NVDA --no-predict          # skip Agent 4, faster
```

Output files are written to `./reports/` (or `--output-dir`):
- `AAPL_risk_report_<date>.json` — structured report data
- `AAPL_risk_report_<date>.pdf` — formatted PDF report

---

## File Structure

```
agent1_fetcher.py       Agent 1: EDGAR queries, pagination, section extraction
agent2_analyzer.py      Agent 2: 4-step risk change classification via GPT-4.1-mini
agent3_synthesizer.py   Agent 3: report synthesis, PDF and JSON export
agent4_predictor.py     Agent 4: 12-month risk materialization forecast + backtest
main.py                 Orchestrator — wires all four agents
diagnose.py             Quick diagnostic to inspect Agent 1 extraction output
evaluate.py             Accuracy and qualitative evaluation suite
requirements.txt
.env                    Your OpenAI API key — never commit this
README.md
```

---

## Agent Details

### Agent 1 — Document Fetcher
Resolves ticker → CIK via `company_tickers.json`, then fetches two consecutive 10-K filings from the EDGAR submissions API. Downloads the primary `.htm` document for each filing and extracts **Item 1A (Risk Factors)** and **Item 7 (MD&A)** using a three-pass regex strategy:

- **Pass 1** — Classic inline format: `ITEM 1A. RISK FACTORS` on one line. Works for AAPL, NVDA, TSLA, AMZN, and most smaller filers.
- **Pass 2** — Split label format: `ITEM 1A.` and `RISK FACTORS` appear on separate lines (iXBRL two-column layout). Works for MSFT, GOOGL, JPM, META post-2023.
- **Pass 3** — Standalone heading fallback: finds `RISK FACTORS` / `MANAGEMENT'S DISCUSSION` as a bare section title.

**EDGAR pagination fix**: the `recent` block caps at ~1000 entries. For large filers like JPM whose older filings live in paginated sub-files, Agent 1 now follows `data["filings"]["files"]` until it collects 2 qualifying 10-K filings.

### Agent 2 — Risk Change Analyzer
4-step GPT-4.1-mini prompt chain:

1. **Extract** structured risk topics from the older filing
2. **Extract** structured risk topics from the newer filing
3. **Compare and classify** each change as `New Risk`, `Escalating Risk`, `Stable`, or `Resolved`, with a severity score 1–5
4. **Semantic consolidation pass** — checks every `New Risk` against the older filing's list and reclassifies anything that is substantively the same under different wording. This prevents reformatting and cosmetic rewording from inflating the "New Risk" count (previously caused MSFT to show 18 spurious new risks).

### Agent 3 — Report Synthesizer
Receives Agent 2 and Agent 4 output and generates:
- **Overall Risk Rating**: `Escalating`, `Stable`, or `Improving`
- **Rating justification** (2–3 sentences)
- **Key investor takeaways** (3–5 bullets)
- **Detailed findings** by category with severity scores
- **Analyst note** (forward-looking commentary)

Exports as formatted JSON and PDF (via `reportlab`). Falls back to `.txt` if `reportlab` is unavailable.

### Agent 4 — Predictor & Backtester
Generates a 12-month risk materialization forecast for each flagged risk using a two-stage process:

1. **Tier ranking** — GPT-4.1-mini assigns each risk to tier 1 (most likely), 2, or 3 based on filing language and historical patterns
2. **Probability assignment** within constrained ranges: tier 1 → 55–80%, tier 2 → 30–54%, tier 3 → 10–29%

This forces real spread across predictions rather than clustering in the 30–65% band.

Backtesting compares prior predictions against the subsequent 10-Q filing. The MISS rule was tightened: near/medium-term risks absent from both the MD&A and risk sections of the follow-up 10-Q score as `MISS` (not `UNCLEAR`), improving the signal quality of the hit rate metric. Results are stored in `predictions.db` and surfaced in the report.

---

## Evaluation

The `evaluate.py` script provides three evaluation modes.

### Mode 1 — Generate predictions and annotation templates
```bash
python evaluate.py generate \
    --tickers AAPL MSFT TSLA NVDA AMZN GOOGL META JPM \
    --output-dir ./eval_data
```
Runs the full pipeline for each ticker and saves `<TICKER>_agent2.json` plus `<TICKER>_annotation_template.json` to `./eval_data/`. The annotation template has one entry per detected risk with `ground_truth_change_type` and `ground_truth_severity` fields set to `null` — fill these in manually by comparing against the actual filings.

### Mode 2 — Score predictions against ground truth
First, compile your filled-in annotation templates into a single `ground_truth.json`:
```json
{
  "AAPL": [
    {"title": "Macroeconomic conditions", "ground_truth_change_type": "Escalating Risk", "ground_truth_severity": 4},
    {"title": "Regulatory risk", "ground_truth_change_type": "Stable", "ground_truth_severity": 3}
  ],
  "MSFT": [...]
}
```

Then score:
```bash
python evaluate.py score \
    --predictions ./eval_data \
    --ground-truth ./ground_truth.json
```

Reports **Precision / Recall / F1** per category (`New Risk`, `Escalating Risk`, `Stable`, `Resolved`), macro averages, overall accuracy, and severity MAE. Title matching uses `difflib.SequenceMatcher` with a 0.80 threshold so minor LLM rewording of risk titles does not cause spurious misses. Results are saved to `./eval_data/classification_metrics.json`.

### Mode 3 — Qualitative coherence scoring
Each team member independently scores each generated report on a 1–5 rubric across three dimensions (Clarity, Factual Consistency, Plain-English Quality). Compile into `qualitative_scores.json`:
```json
{
  "raters": ["Alice", "Bob", "Carol"],
  "rubric_dimensions": ["Clarity", "Factual Consistency", "Plain-English Quality"],
  "scores": {
    "AAPL": {
      "Alice": {"Clarity": 4, "Factual Consistency": 5, "Plain-English Quality": 4},
      "Bob":   {"Clarity": 3, "Factual Consistency": 4, "Plain-English Quality": 4},
      "Carol": {"Clarity": 4, "Factual Consistency": 4, "Plain-English Quality": 5}
    }
  }
}
```

Then run:
```bash
python evaluate.py qualitative --scores ./qualitative_scores.json
```

Reports per-dimension averages and pairwise inter-rater agreement (within ±1 point).

---

## Diagnostics

If a ticker returns empty or suspiciously short Risk Factors, run the diagnostic before anything else:

```bash
python diagnose.py MSFT
```

Prints first/last characters of each extracted section, vocabulary overlap between filings, and saves a full `MSFT_agent1_diagnostic.json` for inspection. This is the fastest way to confirm whether Agent 1's three-pass extraction is working for a given filer.

---

## Known Issues and Limitations

- **iXBRL two-column filers**: MSFT, GOOGL, JPM, and META render section headings as two separate layout elements. The three-pass extractor handles these but may still yield shorter sections than classic filers. If Risk Factors looks short, run `diagnose.py` first.
- **Section extraction is regex-based**: unusual filing structures (e.g. inline XBRL with no visible headings) may still yield empty sections. Agent 1 emits a `WARNING` when final content is below 500 chars.
- **EDGAR rate limit**: SEC enforces a polite crawl policy. Agent 1 includes 0.4s delays between requests. Do not remove these.
- **Backtest requires time**: `predictions.db` accumulates signal as subsequent 10-Qs are filed. A fresh install will show an empty track record on first run — this fills in naturally over time.
- **Model**: configured for `gpt-4.1-mini` by default. Switch to `gpt-4o` in `agent2_analyzer.py` and `agent3_synthesizer.py` for higher analytical quality at increased cost.

---

## Data Sources

| Source | Description | Cost |
|--------|-------------|------|
| SEC EDGAR REST API | All 10-K/10-Q filings for US public companies | Free |
| `company_tickers.json` | Ticker → CIK mapping maintained by SEC | Free |
| OpenAI API (GPT-4.1-mini) | LLM for risk analysis and report synthesis | API key required |

---

## Notes

- Never commit your `.env` file. It is already excluded by `.gitignore`.
- `predictions.db` is also excluded — it accumulates locally and is fully compatible across versions.
- Generated reports (PDFs, JSONs) are excluded from git by default. Add `reports/` to `.gitignore` overrides if you want to track specific outputs.
- `python-dotenv` must be installed (`pip install python-dotenv`). The `.env` file must be in the same directory as `main.py`.
