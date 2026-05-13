"""
Agent 4 — Predictive Risk Materializer
FE 524 Final Project · Group Atlanta

Takes Agent 2's classified risk changes and:
  1. PREDICT  — For each New/Escalating risk, generates a structured prediction:
                  - materialization_probability (0.0–1.0)
                  - predicted_impact_type (earnings_miss | regulatory_action |
                    stock_drawdown | litigation | supply_disruption | guidance_cut | other)
                  - predicted_timeframe (near_term: <6mo | medium_term: 6–12mo | long_term: >12mo)
                  - confidence (high | medium | low)
                  - reasoning (1–2 sentences)

  2. BACKTEST — For a prior prediction set, fetch the next 1–2 10-Q filings from
                EDGAR and parse their Risk Factors + MD&A for materialization signals.
                Scores each prediction as HIT | PARTIAL | MISS | UNCLEAR.
                Stores everything in predictions.db (SQLite).

  3. TRACK RECORD — Queries the DB to compute running accuracy metrics that
                    Agent 3 can embed in the final report.

FIX LOG (this version):
- TWO-STAGE PREDICTION: predictions now use a ranking pass first, then probability
  assignment per tier. This forces spread across the 0–1 range instead of
  everything clustering at 0.30–0.65.
- BACKTEST MISS FIX: UNCLEAR is no longer the default for absent topics. If a risk
  topic is genuinely absent from the 10-Q AND the timeframe is near_term or
  medium_term (<=12 months), score as MISS. UNCLEAR is now reserved for long_term
  predictions or when the 10-Q is demonstrably too short/early.
- Raised 10-Q extraction minimum from 1_000 → 15_000 chars so the broad-slice
  fallback actually fires when Item 1A is boilerplate ("no material changes").
- Added BACKTEST_MIN_AGE_DAYS = 60: only backtest predictions from filings that
  are at least 60 days old.
- Fixed double updated += 1 bug.
- Tightened combined_10q minimum check from 1_000 → 10_000.
"""

import json
import re
import sqlite3
import time
from datetime import datetime, date, timedelta
from pathlib import Path

from openai import OpenAI
from utils import load_env, parse_llm_json, repair_truncated_json_array
load_env()

CLIENT = OpenAI()
MODEL  = "gpt-4.1-mini"

# ── paths ─────────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "predictions.db"

# ── backtest timing guard ─────────────────────────────────────────────────────
BACKTEST_MIN_AGE_DAYS = 60

# ── EDGAR constants ───────────────────────────────────────────────────────────
HEADERS    = {"User-Agent": "FE524-RiskIntel research@stevens.edu",
              "Accept-Encoding": "gzip, deflate"}
EDGAR_BASE = "https://data.sec.gov"
SEC_BASE   = "https://www.sec.gov"


# ══════════════════════════════════════════════════════════════════════════════
# SQLite schema
# ══════════════════════════════════════════════════════════════════════════════

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                  TEXT    NOT NULL,
    filing_date             TEXT    NOT NULL,
    risk_title              TEXT    NOT NULL,
    change_type             TEXT    NOT NULL,
    severity                INTEGER,
    impact_type             TEXT,
    probability             REAL,
    timeframe               TEXT,
    confidence              TEXT,
    reasoning               TEXT,
    created_at              TEXT    NOT NULL,
    backtest_status         TEXT    DEFAULT 'PENDING',
    backtest_evidence       TEXT,
    backtest_date           TEXT,
    backtest_filing_date    TEXT,
    UNIQUE (ticker, filing_date, risk_title)
);

CREATE TABLE IF NOT EXISTS track_record (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT    NOT NULL,
    filing_date         TEXT    NOT NULL,
    total_predictions   INTEGER,
    hits                INTEGER,
    partials            INTEGER,
    misses              INTEGER,
    unclear             INTEGER,
    hit_rate            REAL,
    computed_at         TEXT    NOT NULL
);
"""


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Two-stage calibrated predictions
# ══════════════════════════════════════════════════════════════════════════════

RANK_SYSTEM = """You are a quantitative equity research analyst.
You are given a list of New Risk and Escalating Risk items from a company's 10-K filing.

Your first task is to RANK them by materialization likelihood — which risks are most
likely to have a material financial impact on the company within the next 12 months?

Assign each risk to one of these tiers:
  "tier_1": Top 20% — highest likelihood, clearest near-term pathway to impact
  "tier_2": Next 30% — elevated risk, some near-term indicators
  "tier_3": Bottom 50% — disclosed but lower near-term probability

Respond ONLY with a JSON array. Each element:
  "risk_title": exact title from input
  "tier": "tier_1" | "tier_2" | "tier_3"
  "tier_rationale": one sentence explaining the tier assignment

No markdown fences, no preamble."""

PREDICT_SYSTEM = """You are a quantitative equity research analyst.
You are given a ranked list of risk items from a company's 10-K filing, with tier assignments.

Your task is to assign calibrated materialization probabilities using these tier ranges:
  tier_1: 0.55 – 0.80  (high near-term risk, use the full range — don't cluster at 0.65)
  tier_2: 0.30 – 0.54  (moderate risk)
  tier_3: 0.10 – 0.29  (lower near-term risk)

Within each tier, differentiate further based on:
  - Severity (4–5 = push toward upper end of tier range)
  - Specificity of language (vague = lower, specific with numbers/names = higher)
  - Whether it's a New Risk vs Escalating (new unknown = slight reduction)
  - Industry base rates: supply chain/tax risks for large-caps materialize more often
    than regulatory or litigation risks

For each risk output:
  "risk_title":                 exact title from input
  "materialization_probability": float within the tier's range (use full range, not midpoint)
  "predicted_impact_type": one of:
        "earnings_miss" | "regulatory_action" | "litigation" |
        "supply_disruption" | "guidance_cut" | "stock_drawdown" | "other"
  "predicted_timeframe":   "near_term" (<6 months) | "medium_term" (6–12 months) | "long_term" (>12 months)
  "confidence":            "high" | "medium" | "low"
  "reasoning":             1–2 sentences explaining your prediction

Respond ONLY with a JSON array. No markdown fences, no preamble."""


def generate_predictions(analyzer_output: dict) -> list[dict]:
    """
    Step 1: Two-stage LLM prediction.
    Stage A: rank risks by tier.
    Stage B: assign calibrated probabilities within each tier's range.
    This forces spread across the probability range instead of clustering.
    """
    changes = analyzer_output["changes"]
    ticker  = analyzer_output["ticker"]

    actionable = [
        c for c in changes
        if c.get("change_type") in ("New Risk", "Escalating Risk")
    ]
    if not actionable:
        print(f"[Agent 4] No New or Escalating risks found for {ticker} — skipping prediction.")
        return []

    print(f"[Agent 4] Stage A — Ranking {len(actionable)} risks by tier...")

    # Stage A: ranking
    rank_payload = json.dumps({
        "ticker":      ticker,
        "filing_date": analyzer_output["newer_date"],
        "risks":       actionable,
    }, indent=2)

    rank_resp = CLIENT.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": RANK_SYSTEM},
            {"role": "user",   "content": rank_payload},
        ],
    )
    rank_raw = rank_resp.choices[0].message.content
    ranked   = parse_llm_json(rank_raw)

    # Merge tier info back into actionable list
    tier_map = {r["risk_title"]: r for r in ranked}
    enriched = []
    for item in actionable:
        tier_info = tier_map.get(item.get("title", ""), {})
        enriched.append({**item, "tier": tier_info.get("tier", "tier_3"),
                         "tier_rationale": tier_info.get("tier_rationale", "")})

    tier_counts = {}
    for r in enriched:
        tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1
    print(f"[Agent 4]   Tier distribution: {tier_counts}")

    # Stage B: probability assignment
    print(f"[Agent 4] Stage B — Assigning calibrated probabilities...")
    pred_payload = json.dumps({
        "ticker":      ticker,
        "filing_date": analyzer_output["newer_date"],
        "ranked_risks": enriched,
    }, indent=2)

    pred_resp = CLIENT.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": PREDICT_SYSTEM},
            {"role": "user",   "content": pred_payload},
        ],
    )
    pred_raw    = pred_resp.choices[0].message.content
    predictions = parse_llm_json(pred_raw)
    print(f"[Agent 4] Generated {len(predictions)} predictions.")

    # Log probability spread for diagnostics
    probs = sorted([p.get("materialization_probability", 0) for p in predictions])
    if probs:
        print(f"[Agent 4]   Probability range: {min(probs):.2f} – {max(probs):.2f} "
              f"(mean: {sum(probs)/len(probs):.2f})")

    return predictions


def save_predictions(ticker: str, filing_date: str, predictions: list[dict],
                     analyzer_output: dict) -> None:
    """Persist predictions to SQLite, avoiding duplicates."""
    conn = get_db()
    now  = datetime.utcnow().isoformat()

    sev_lookup = {c["title"]: c.get("severity", 0) for c in analyzer_output["changes"]}
    ct_lookup  = {c["title"]: c.get("change_type", "") for c in analyzer_output["changes"]}

    inserted = 0
    for p in predictions:
        title = p.get("risk_title", "")
        result = conn.execute("""
            INSERT OR IGNORE INTO predictions
              (ticker, filing_date, risk_title, change_type, severity,
               impact_type, probability, timeframe, confidence, reasoning, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ticker,
            filing_date,
            title,
            ct_lookup.get(title, ""),
            sev_lookup.get(title, 0),
            p.get("predicted_impact_type", "other"),
            p.get("materialization_probability", 0.0),
            p.get("predicted_timeframe", "medium_term"),
            p.get("confidence", "medium"),
            p.get("reasoning", ""),
            now,
        ))
        inserted += result.rowcount

    conn.commit()
    conn.close()
    print(f"[Agent 4] Saved {inserted} new predictions to {DB_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Fetch 10-Q filings for backtesting
# ══════════════════════════════════════════════════════════════════════════════

def _get_cik(ticker: str) -> str:
    import requests
    resp = requests.get(f"{SEC_BASE}/files/company_tickers.json",
                        headers=HEADERS, timeout=15)
    resp.raise_for_status()
    for entry in resp.json().values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR.")


def get_10q_filings_after(cik: str, after_date: str, n: int = 2) -> list[dict]:
    """Fetch up to n 10-Q filings filed AFTER after_date. Follows EDGAR pagination."""
    import requests as req

    all_10q  = []
    url      = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    is_root  = True
    next_url = None

    while url and len(all_10q) < n:
        time.sleep(0.4)
        resp = req.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if is_root:
            block        = data["filings"]["recent"]
            pages        = data["filings"].get("files", [])
            primary_name = f"CIK{cik}.json"
            next_url     = None
            for page in pages:
                name = page.get("name", "")
                if name and name != primary_name:
                    next_url = f"{EDGAR_BASE}/submissions/{name}"
                    break
        else:
            block    = data
            next_url = None

        forms    = block.get("form",            [])
        dates    = block.get("filingDate",      [])
        accnums  = block.get("accessionNumber", [])
        pri_docs = block.get("primaryDocument", [""] * len(forms))

        for form, fdate, acc, pdoc in zip(forms, dates, accnums, pri_docs):
            if form == "10-Q" and fdate > after_date:
                acc_clean = acc.replace("-", "")
                all_10q.append({
                    "accession":   acc,
                    "acc_clean":   acc_clean,
                    "date":        fdate,
                    "primary_doc": pdoc,
                    "cik":         cik,
                    "cik_int":     int(cik),
                })
            if len(all_10q) >= n:
                break

        url     = next_url
        is_root = False

    return all_10q[:n]


def _fetch_url(url: str) -> str:
    time.sleep(0.4)
    import requests as req
    r = req.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.text


def _strip_and_extract_10q(html: str) -> str:
    """Strip HTML and extract MD&A (Item 2) + Risk Factors (Item 1A) from a 10-Q."""
    import agent1_fetcher as a1
    plain = a1.strip_html(html)

    if len(plain) < 2000:
        return plain

    mda_text = a1.find_section(plain, "mda",          max_chars=40_000)
    rf_text  = a1.find_section(plain, "risk_factors", max_chars=20_000)

    combined = ""
    if mda_text and len(mda_text) >= 500:
        combined += f"=== MD&A (Item 2) ===\n{mda_text}\n\n"
    if rf_text and len(rf_text) >= 500:
        combined += f"=== RISK FACTORS (Item 1A) ===\n{rf_text}\n\n"

    if len(combined) < 15_000:
        print("[Agent 4]   Note: section extraction found little — using broad doc slice")
        combined = plain[8_000:78_000]

    return combined.strip()[:70_000]


def fetch_10q_text(meta: dict) -> str:
    """Download a 10-Q and return stripped plain text focused on MD&A + Risk Factors."""
    cik_int   = meta["cik_int"]
    acc_clean = meta["acc_clean"]
    base_url  = f"{SEC_BASE}/Archives/edgar/data/{cik_int}/{acc_clean}/"
    pdoc      = meta.get("primary_doc", "")

    urls_to_try = []
    if pdoc:
        urls_to_try.append(base_url + pdoc)

    try:
        idx_url = f"{base_url}{meta['accession']}-index.json"
        time.sleep(0.2)
        import requests as req
        r = req.get(idx_url, headers=HEADERS, timeout=12)
        if r.status_code == 200:
            items = r.json().get("directory", {}).get("item", [])
            items_sorted = sorted(items, key=lambda x: int(x.get("size", 0) or 0), reverse=True)
            for item in items_sorted:
                name = item.get("name", "")
                nl = name.lower()
                if name.endswith((".htm", ".html")) and "ex" not in nl and "exhibit" not in nl:
                    candidate = base_url + name
                    if candidate not in urls_to_try:
                        urls_to_try.append(candidate)
    except Exception:
        pass

    for url in urls_to_try[:4]:
        try:
            html = _fetch_url(url)
            if len(html) < 5000:
                continue
            extracted = _strip_and_extract_10q(html)
            print(f"[Agent 4]   Extracted {len(extracted):,} chars from {url.split('/')[-1]}")
            if len(extracted) >= 1000:
                return extracted
        except Exception as e:
            print(f"[Agent 4]   ✗ 10-Q probe failed: {url.split('/')[-1]} — {e}")
            continue

    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — LLM-based backtesting
# ══════════════════════════════════════════════════════════════════════════════

BACKTEST_SYSTEM = """You are a financial analyst performing a post-hoc evaluation of risk predictions.

You are given:
  1. A set of risk predictions made at the time of a company's 10-K annual filing
  2. Text from subsequent 10-Q quarterly filings (the next 1–2 quarters)

Your task: for each prediction, determine whether the predicted risk SHOWED SIGNS OF
MATERIALIZING in the period covered by the 10-Q filings.

SCORING GUIDE — read carefully:

  "HIT" — The risk showed clear signs of materializing. Use this when:
    • The 10-Q mentions a financial impact consistent with the prediction
      (e.g. gross margin declined, an investigation was disclosed, guidance was cut,
       a legal charge was taken, FX caused a revenue headwind, tax provisions increased)
    • Management commentary acknowledges the risk is actively affecting results
    • A quantitative metric moved in the predicted direction
    NOTE: The 10-Q does NOT need to say "this risk materialized." Look for
    FINANCIAL EVIDENCE — charges, impairments, revised guidance, disclosed investigations,
    quantified FX impacts, increased effective tax rate, margin compression, etc.

  "PARTIAL" — Some early movement toward materialization but not a full impact yet:
    • The risk is re-disclosed with stronger/more urgent language
    • A related metric moved slightly in the predicted direction
    • Management flagged increasing concern

  "MISS" — Use MISS when ANY of these apply:
    • The 10-Q contains affirmative counter-evidence (margins expanded, risk resolved,
      guidance raised, tax settled favorably)
    • The risk topic is completely absent from both the Risk Factors AND MD&A sections
      of the 10-Q AND the timeframe was "near_term" or "medium_term"
    A risk that simply is not discussed in the 10-Q but had a near/medium timeframe
    should be MISS — if it were materializing, management would have disclosed it.

  "UNCLEAR" — Reserve for:
    • Predictions with timeframe "long_term" (>12 months) where it's too early
    • Cases where the 10-Q text is clearly too short (<5,000 chars) to make a judgment
    • Macro/geopolitical risks that genuinely require longer observation
    Do NOT use UNCLEAR just because the topic isn't mentioned — that is a MISS for
    near_term and medium_term predictions.

For each prediction output:
  "risk_title":   exact title from input
  "status":       HIT | PARTIAL | MISS | UNCLEAR
  "evidence":     2–3 sentences citing SPECIFIC language or numbers from the 10-Q.
                  For MISS due to absence, state: "Topic not mentioned in 10-Q MD&A
                  or Risk Factors despite near/medium-term timeframe."
                  For UNCLEAR, state why the judgment cannot be made yet.

Respond ONLY with a JSON array. No markdown fences, no preamble."""


def backtest_predictions(ticker: str, filing_date: str,
                         q10_texts: list[tuple[str, str]]) -> None:
    """
    For all PENDING predictions for ticker+filing_date, run LLM backtest
    using the provided 10-Q texts.
    """
    conn = get_db()
    pending = conn.execute("""
        SELECT id, risk_title, impact_type, probability, timeframe, reasoning
        FROM predictions
        WHERE ticker=? AND filing_date=? AND backtest_status='PENDING'
    """, (ticker, filing_date)).fetchall()

    if not pending:
        print(f"[Agent 4] No pending predictions to backtest for {ticker} {filing_date}.")
        conn.close()
        return

    print(f"[Agent 4] Backtesting {len(pending)} predictions using "
          f"{len(q10_texts)} 10-Q filing(s)...")

    combined_10q = ""
    for qdate, qtext in q10_texts:
        combined_10q += f"\n\n=== 10-Q Filed {qdate} ===\n"
        combined_10q += qtext[:35_000]

    combined_10q = combined_10q[:70_000]

    if len(combined_10q.strip()) < 10_000:
        print(f"[Agent 4] Combined 10-Q text only {len(combined_10q):,} chars — too short to backtest reliably. Deferring.")
        conn.close()
        return

    BATCH_SIZE   = 7
    pending_list = list(pending)
    now          = datetime.utcnow().isoformat()
    backtest_filing_date = q10_texts[-1][0] if q10_texts else ""
    updated = 0

    for batch_start in range(0, len(pending_list), BATCH_SIZE):
        batch = pending_list[batch_start : batch_start + BATCH_SIZE]

        payload = json.dumps({
            "ticker":      ticker,
            "filing_date": filing_date,
            "predictions": [
                {
                    "risk_title":  row["risk_title"],
                    "impact_type": row["impact_type"],
                    "probability": row["probability"],
                    "timeframe":   row["timeframe"],
                    "reasoning":   row["reasoning"],
                }
                for row in batch
            ],
            "subsequent_10q_text": combined_10q,
        }, indent=2)

        resp = CLIENT.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": BACKTEST_SYSTEM},
                {"role": "user",   "content": payload},
            ],
        )
        raw = resp.choices[0].message.content.strip().strip("```json").strip("```").strip()
        results = json.loads(raw)

        result_map = {}
        for r in results:
            result_map[r["risk_title"]] = r
            result_map[r["risk_title"].lower().strip()] = r

        for row in batch:
            title = row["risk_title"]
            res = result_map.get(title) or result_map.get(title.lower().strip())
            if not res:
                conn.execute("""
                    UPDATE predictions
                    SET backtest_status='UNCLEAR',
                        backtest_evidence='Title not matched in LLM response.',
                        backtest_date=?, backtest_filing_date=?
                    WHERE id=?
                """, (now, backtest_filing_date, row["id"]))
            else:
                conn.execute("""
                    UPDATE predictions
                    SET backtest_status=?, backtest_evidence=?, backtest_date=?,
                        backtest_filing_date=?
                    WHERE id=?
                """, (
                    res.get("status", "UNCLEAR"),
                    res.get("evidence", ""),
                    now,
                    backtest_filing_date,
                    row["id"],
                ))
            updated += 1

    conn.commit()
    conn.close()
    print(f"[Agent 4] Backtest complete — {updated} predictions scored.")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Track record computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_track_record(ticker: str = None) -> dict:
    """Compute running accuracy metrics across all tickers or a specific one."""
    conn  = get_db()
    where  = "WHERE backtest_status != 'PENDING'"
    params: tuple = ()
    if ticker:
        where  += " AND ticker=?"
        params  = (ticker,)

    rows = conn.execute(f"""
        SELECT backtest_status, COUNT(*) as cnt
        FROM predictions {where}
        GROUP BY backtest_status
    """, params).fetchall()
    conn.close()

    counts = {"HIT": 0, "PARTIAL": 0, "MISS": 0, "UNCLEAR": 0}
    for row in rows:
        counts[row["backtest_status"]] = row["cnt"]

    total = sum(counts.values())
    if total == 0:
        return {
            "total_evaluated": 0,
            "message": "No backtested predictions yet — run with prior filing data to build track record.",
        }

    effective_hits = counts["HIT"] + 0.5 * counts["PARTIAL"]
    definitive     = counts["HIT"] + counts["PARTIAL"] + counts["MISS"]

    return {
        "total_evaluated":       total,
        "hits":                  counts["HIT"],
        "partials":              counts["PARTIAL"],
        "misses":                counts["MISS"],
        "unclear":               counts["UNCLEAR"],
        "hit_rate":              round(counts["HIT"] / total, 3),
        "adjusted_hit_rate":     round(effective_hits / total, 3),
        "definitive_hit_rate":   round(effective_hits / definitive, 3) if definitive else None,
        "scope":                 ticker or "all_tickers",
    }


def get_predictions_for_report(ticker: str, filing_date: str) -> list[dict]:
    """Return all predictions for a given ticker+filing as dicts for Agent 3."""
    conn  = get_db()
    rows  = conn.execute("""
        SELECT risk_title, change_type, severity, impact_type, probability,
               timeframe, confidence, reasoning, backtest_status, backtest_evidence
        FROM predictions
        WHERE ticker=? AND filing_date=?
        ORDER BY probability DESC
    """, (ticker, filing_date)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# Backtest attempt helper
# ══════════════════════════════════════════════════════════════════════════════

def _attempt_backtest(ticker: str, cik: str = None) -> None:
    """
    Find ALL PENDING predictions for this ticker and attempt to backtest them.
    Skips filings less than BACKTEST_MIN_AGE_DAYS old.

    Fix 1: accepts a pre-resolved CIK from Agent 1 output so we don't re-fetch
    company_tickers.json on every run.
    """
    import requests

    conn = get_db()
    pending_filings = conn.execute("""
        SELECT DISTINCT filing_date
        FROM predictions
        WHERE ticker=? AND backtest_status='PENDING'
        ORDER BY filing_date ASC
    """, (ticker,)).fetchall()
    conn.close()

    if not pending_filings:
        print(f"[Agent 4] No pending predictions to backtest for {ticker}.")
        return

    today     = date.today()
    min_age   = timedelta(days=BACKTEST_MIN_AGE_DAYS)
    to_process = []

    for row in pending_filings:
        filing_date = row["filing_date"]
        try:
            fd = date.fromisoformat(filing_date)
        except ValueError:
            continue
        age = today - fd
        if age < min_age:
            print(f"[Agent 4] Skipping backtest for {filing_date} filing "
                  f"(only {age.days} days old, need ≥{BACKTEST_MIN_AGE_DAYS}). "
                  f"Re-run in {(min_age - age).days} days.")
        else:
            to_process.append(filing_date)

    if not to_process:
        print(f"[Agent 4] All pending filings are too recent to backtest.")
        return

    print(f"[Agent 4] Found {len(to_process)} filing(s) eligible for backtesting.")

    try:
        if not cik:
            # CIK not passed through — fall back to fetching (one extra network call)
            cik_resp = requests.get(f"{SEC_BASE}/files/company_tickers.json",
                                    headers=HEADERS, timeout=15)
            cik_resp.raise_for_status()
            cik = None
            for entry in cik_resp.json().values():
                if entry["ticker"].upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    break
            if not cik:
                print(f"[Agent 4] Could not resolve CIK for {ticker} — skipping backtest.")
                return

        for filing_date in to_process:
            print(f"[Agent 4] Fetching 10-Qs after {filing_date} for {ticker}...")

            q10_metas = get_10q_filings_after(cik, filing_date, n=2)
            if not q10_metas:
                print(f"[Agent 4]   No 10-Qs found after {filing_date} — backtest deferred.")
                continue

            q10_texts = []
            for meta in q10_metas:
                print(f"[Agent 4]   Fetching 10-Q filed {meta['date']}...")
                text = fetch_10q_text(meta)
                if text and len(text) >= 1000:
                    q10_texts.append((meta["date"], text))
                else:
                    print(f"[Agent 4]   ✗ Too short or empty — skipping this 10-Q.")

            if q10_texts:
                backtest_predictions(ticker, filing_date, q10_texts)
            else:
                print(f"[Agent 4]   No usable 10-Q text — backtest deferred.")

    except Exception as e:
        print(f"[Agent 4] Backtest attempt failed: {e} — predictions remain PENDING.")
        import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# Main agent entry point
# ══════════════════════════════════════════════════════════════════════════════

def run(analyzer_output: dict, run_backtest: bool = True) -> dict:
    """
    Full Agent 4 pipeline.
    1. Generate two-stage calibrated predictions
    2. Save to SQLite
    3. Optionally backtest prior predictions using 10-Qs
    4. Return predictions + track record for Agent 3
    """
    ticker       = analyzer_output["ticker"]
    filing_date  = analyzer_output["newer_date"]
    cik          = analyzer_output.get("cik", "")  # Fix 1: passed from Agent 1 via Agent 2

    print(f"\n[Agent 4] === Prediction & Backtesting for {ticker} ({filing_date}) ===\n")

    predictions = generate_predictions(analyzer_output)
    if predictions:
        save_predictions(ticker, filing_date, predictions, analyzer_output)

    if run_backtest:
        _attempt_backtest(ticker, cik=cik)

    saved_preds  = get_predictions_for_report(ticker, filing_date)
    track        = compute_track_record()
    track_ticker = compute_track_record(ticker)

    print(f"[Agent 4] Track record (all tickers): {track}")
    print(f"[Agent 4] Track record ({ticker}):     {track_ticker}")
    print("[Agent 4] Done.\n")

    return {
        "ticker":              ticker,
        "filing_date":         filing_date,
        "predictions":         saved_preds,
        "track_record_global": track,
        "track_record_ticker": track_ticker,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agent4_predictor.py <agent2_output.json>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        a2_out = json.load(f)
    result = run(a2_out)
    print(json.dumps(result, indent=2, default=str))