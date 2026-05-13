"""
Agent 2 — Risk Change Analyzer
Takes two versions of a company's Risk Factors section and performs a
structured multi-step prompt chain to classify changes as:
  New Risk | Escalating Risk | Stable | Resolved
Each change is assigned a severity score (1–5).

FIX LOG:
- Raised MAX_SECTION_CHARS from 40_000 to 80_000 so large filings (MSFT, AAPL)
  aren't truncated before real risk content begins.
- Added chunked extraction: if a section is >80K chars it's split into 2 chunks,
  each extracted separately, then merged+deduped. This prevents the "0 risks"
  failure where the LLM only saw a ToC or boilerplate intro.
- extract_risk_list now logs char count so truncation is visible in output.
- NEW (this version): Added Step 4 — semantic consolidation pass.
  After comparison+dedup, a 4th LLM call re-examines every "New Risk" entry
  and checks whether it semantically overlaps with any older-filing risk (even
  under different wording). Near-matches are reclassified to "Escalating Risk"
  or "Stable". This prevents filing reformats and rewording from inflating new
  risk counts (e.g. MSFT getting 18 "new" risks that are mostly renamed old ones).
"""

import json
from openai import OpenAI

from utils import load_env, parse_llm_json, repair_truncated_json_array
load_env()

CLIENT = OpenAI()
MODEL = "gpt-4.1-mini"

# ── token budget ──────────────────────────────────────────────────────────────
MAX_SECTION_CHARS = 80_000
CHUNK_SIZE        = 75_000


def _truncate(text: str, max_chars: int = MAX_SECTION_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated for length ...]"


# ── prompt chain ──────────────────────────────────────────────────────────────

STEP1_SYSTEM = """You are a senior financial analyst specializing in SEC filings.
Your task is to extract a structured list of all distinct risk topics from a single
10-K Risk Factors section. For each risk, output a short title (≤10 words) and a
one-sentence summary of the core concern.

Respond ONLY with a JSON array. Each element must have:
  "id": integer (1-based),
  "title": string,
  "summary": string

No markdown fences, no preamble."""

STEP2_SYSTEM = """You are a senior financial analyst specializing in SEC filings.
You will be given two JSON arrays of risk topics extracted from consecutive 10-K filings
for the same company — one from the OLDER filing and one from the NEWER filing.

Your task: compare them and classify every risk change. For each item produce:
  "change_type": one of "New Risk" | "Escalating Risk" | "Stable" | "Resolved"
  "severity": integer 1 (minimal) to 5 (critical)
  "title": short label for the risk
  "older_summary": summary from older filing (or null if new)
  "newer_summary": summary from newer filing (or null if resolved)
  "rationale": 1–2 sentences explaining your classification

Rules:
- New Risk: appears in newer but not older filing
- Resolved: appears in older but not newer filing
- Escalating Risk: present in both, but language sharpened / stakes raised / severity worsened
- Stable: present in both, no material change in language or severity

IMPORTANT: Be conservative with "New Risk". Companies often reword or reorganise
existing risk disclosures each year. If a risk in the newer filing is substantively
the same concern as an older one — even if the title or framing changed — classify it
as "Escalating Risk" or "Stable", not "New Risk". Only use "New Risk" when the underlying
business concern genuinely did not appear in the prior year.

Respond ONLY with a JSON array. No markdown fences, no preamble."""

# ── NEW: Step 4 system prompt ─────────────────────────────────────────────────
STEP4_SYSTEM = """You are a senior financial analyst reviewing risk classification results
from an automated SEC filing comparison system.

You are given:
  1. A list of risks from the OLDER 10-K filing
  2. A list of classified risk changes, some of which are tagged "New Risk"

Your task: for each "New Risk" entry, check whether it is SEMANTICALLY EQUIVALENT
to any risk in the older filing — even if the title, wording, or framing changed.

Two risks are semantically equivalent if they describe the SAME underlying business
concern. For example:
  - "Supply chain concentration risk" (older) ≈ "Supplier and component dependencies" (newer) → NOT new
  - "Income tax rate uncertainty" (older) ≈ "Effective tax rate variability" (newer) → NOT new
  - "Cyberattack risk" does NOT overlap with "AI model bias risk" → both genuinely new

For each "New Risk" item, you must decide:
  - If it genuinely has NO equivalent in the older filing → keep as "New Risk"
  - If it maps to an existing older risk with stronger language → reclassify as "Escalating Risk"
  - If it maps to an existing older risk with similar language → reclassify as "Stable"

Output the COMPLETE list of changes (not just the New Risk entries), with any
reclassifications applied. All other fields (title, severity, summaries, rationale)
must be preserved exactly. Only "change_type" may be modified, and only for items
that were "New Risk".

Respond ONLY with a JSON array of the complete updated changes list. No markdown fences, no preamble."""


def _extract_chunk(chunk_text: str, filing_label: str, chunk_num: int, total_chunks: int) -> list[dict]:
    """Extract risks from a single chunk of text."""
    resp = CLIENT.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": STEP1_SYSTEM},
            {
                "role": "user",
                "content": f"Here is part {chunk_num} of {total_chunks} of the Risk Factors section from the {filing_label} 10-K filing:\n\n{chunk_text}",
            },
        ],
    )
    raw = resp.choices[0].message.content
    risks = parse_llm_json(raw)
    print(f"[Agent 2]     Chunk {chunk_num}: found {len(risks)} risks.")
    return risks


def extract_risk_list(section_text: str, filing_label: str) -> list[dict]:
    """
    Step 1: Extract a structured list of risks from one filing's text.

    If the section is longer than CHUNK_SIZE, splits into chunks and extracts
    each independently, then merges via deduplication.
    """
    print(f"[Agent 2] Step 1 — Extracting risk list from {filing_label} filing "
          f"({len(section_text):,} chars)...")

    if len(section_text) <= CHUNK_SIZE:
        resp = CLIENT.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": STEP1_SYSTEM},
                {
                    "role": "user",
                    "content": f"Here is the Risk Factors section from the {filing_label} 10-K filing:\n\n{section_text}",
                },
            ],
        )
        raw = resp.choices[0].message.content
        risks = parse_llm_json(raw)
        print(f"[Agent 2]   Found {len(risks)} risks in {filing_label} filing.")
        return risks

    # Multi-chunk path
    print(f"[Agent 2]   Section too large — splitting into chunks of {CHUNK_SIZE:,} chars...")
    chunks = []
    for i in range(0, len(section_text), CHUNK_SIZE):
        chunks.append(section_text[i : i + CHUNK_SIZE])

    all_risks = []
    for idx, chunk in enumerate(chunks, 1):
        chunk_risks = _extract_chunk(chunk, filing_label, idx, len(chunks))
        all_risks.extend(chunk_risks)

    for i, r in enumerate(all_risks, 1):
        r["id"] = i

    print(f"[Agent 2]   Raw total across {len(chunks)} chunks: {len(all_risks)} risks — deduplicating...")
    deduped = _dedup_risk_list(all_risks, filing_label)
    print(f"[Agent 2]   Found {len(deduped)} risks in {filing_label} filing after dedup.")
    return deduped


def _dedup_risk_list(risks: list[dict], filing_label: str) -> list[dict]:
    """Merge near-duplicate risk entries across chunks using LLM."""
    if len(risks) <= 5:
        return risks

    DEDUP_LIST_SYSTEM = """You are a financial analyst. You are given a JSON array of risk topics
extracted from different parts of the same 10-K filing. Some entries are near-duplicates.

Merge near-duplicates into a single entry. Keep the entry with the richer/more specific summary.
Re-number "id" fields sequentially starting from 1 after merging.

Two entries are duplicates if they describe the SAME underlying risk.
Different aspects of a broad topic (e.g. "FX on revenue" vs "FX on costs") are NOT duplicates.

Respond ONLY with a JSON array. No markdown fences, no preamble."""

    try:
        resp = CLIENT.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": DEDUP_LIST_SYSTEM},
                {"role": "user", "content": json.dumps(risks, indent=2)},
            ],
        )
        raw = resp.choices[0].message.content
        return parse_llm_json(raw)
    except Exception as e:
        print(f"[Agent 2]   Warning: risk-list dedup failed ({e}) — using raw list.")
        return risks


def compare_risk_lists(older_risks: list[dict], newer_risks: list[dict]) -> list[dict]:
    """Step 2: Compare old vs new risk lists and classify each change."""
    print("[Agent 2] Step 2 — Comparing risk lists across filings...")
    payload = json.dumps({
        "older_filing_risks": older_risks,
        "newer_filing_risks": newer_risks,
    }, indent=2)

    resp = CLIENT.chat.completions.create(
        model=MODEL,
        max_tokens=16000,
        messages=[
            {"role": "system", "content": STEP2_SYSTEM},
            {"role": "user", "content": payload},
        ],
    )
    raw = resp.choices[0].message.content

    # Fix 2: robust truncation repair with one retry instead of brittle string append
    changes = repair_truncated_json_array(raw)
    if not changes:
        print("[Agent 2]   Warning: initial JSON parse failed — retrying with repair prompt...")
        retry_resp = CLIENT.chat.completions.create(
            model=MODEL,
            max_tokens=16000,
            messages=[
                {"role": "system", "content": STEP2_SYSTEM},
                {"role": "user", "content": payload},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    "Your response appears to have been cut off or contains invalid JSON. "
                    "Please output ONLY the complete, valid JSON array from the beginning. "
                    "No markdown fences, no preamble."
                )},
            ],
        )
        changes = repair_truncated_json_array(retry_resp.choices[0].message.content)
        if not changes:
            raise ValueError("Agent 2 compare_risk_lists: could not obtain valid JSON after retry.")

    print(f"[Agent 2]   Classified {len(changes)} risk entries.")
    return changes


DEDUP_SYSTEM = """You are a financial analyst. You are given a list of risk change entries
extracted from SEC 10-K filings. Some entries are near-duplicates — they describe the
same underlying risk using slightly different titles or phrasings.

Your task: merge near-duplicates into a single entry. Keep the entry with the HIGHER
severity score. If severities are equal, keep the one with the richer description.

Rules:
- Two entries are duplicates if they describe the SAME risk topic.
- Different aspects of a broad topic (e.g. "FX risk on revenue" vs "FX risk on costs")
  are NOT duplicates — keep both.
- Preserve all fields exactly as-is from the entry you keep.
- Do NOT merge entries that are genuinely distinct risks.

Respond ONLY with a JSON array of the deduplicated entries (same schema as input).
No markdown fences, no preamble."""


def deduplicate_changes(changes: list[dict]) -> list[dict]:
    """Step 3: Use LLM to merge near-duplicate risk entries."""
    if len(changes) <= 3:
        return changes
    print(f"[Agent 2] Step 3 — Deduplicating {len(changes)} entries...")
    resp = CLIENT.chat.completions.create(
        model=MODEL,
        max_tokens=8192,
        messages=[
            {"role": "system", "content": DEDUP_SYSTEM},
            {"role": "user", "content": json.dumps(changes, indent=2)},
        ],
    )
    raw = resp.choices[0].message.content
    deduped = parse_llm_json(raw)
    print(f"[Agent 2]   Reduced to {len(deduped)} entries after deduplication.")
    return deduped


def consolidate_new_risks(changes: list[dict], older_risks: list[dict]) -> list[dict]:
    """
    Step 4 (NEW): Semantic consolidation pass.

    Re-examines every "New Risk" entry and checks whether it genuinely
    didn't exist in the older filing, or whether it's a reworded/reorganised
    version of an existing risk. Reclassifies false-new-risks to
    "Escalating Risk" or "Stable" as appropriate.

    Skips this step if there are fewer than 3 New Risk entries (not worth
    an extra API call) or if older_risks is empty.
    """
    new_risk_count = sum(1 for c in changes if c.get("change_type") == "New Risk")
    if new_risk_count < 3 or not older_risks:
        return changes

    print(f"[Agent 2] Step 4 — Semantic consolidation of {new_risk_count} 'New Risk' entries...")

    payload = json.dumps({
        "older_filing_risks": older_risks,
        "classified_changes":  changes,
    }, indent=2)

    try:
        resp = CLIENT.chat.completions.create(
            model=MODEL,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": STEP4_SYSTEM},
                {"role": "user",   "content": payload},
            ],
        )
        raw = resp.choices[0].message.content
        consolidated = parse_llm_json(raw)

        # Count reclassifications for logging
        reclassified = sum(
            1 for old, new in zip(changes, consolidated)
            if old.get("change_type") == "New Risk"
            and new.get("change_type") != "New Risk"
        )
        new_remaining = sum(1 for c in consolidated if c.get("change_type") == "New Risk")
        print(f"[Agent 2]   Reclassified {reclassified} false-new risks. "
              f"{new_remaining} genuine new risks remain.")
        return consolidated

    except Exception as e:
        print(f"[Agent 2]   Warning: consolidation step failed ({e}) — keeping original classifications.")
        return changes


def summarize_counts(changes: list[dict]) -> dict:
    """Tally counts and average severity by change type."""
    summary = {}
    for ch in changes:
        ct = ch.get("change_type", "Unknown")
        summary.setdefault(ct, {"count": 0, "total_severity": 0})
        summary[ct]["count"] += 1
        summary[ct]["total_severity"] += ch.get("severity", 0)
    for ct in summary:
        n = summary[ct]["count"]
        summary[ct]["avg_severity"] = round(summary[ct]["total_severity"] / n, 2) if n else 0
        del summary[ct]["total_severity"]
    return summary


# ── main agent function ───────────────────────────────────────────────────────

def run(fetcher_output: dict) -> dict:
    """
    Full Agent 2 pipeline.
    Expects fetcher_output from agent1_fetcher.run():
      {ticker, filings: [{date, risk_factors, ...}, ...]}  (newest first)

    Returns:
      {ticker, newer_date, older_date, changes: [...], summary_counts: {...}}
    """
    ticker = fetcher_output["ticker"]
    filings = fetcher_output["filings"]
    newer, older = filings[0], filings[1]

    print(f"[Agent 2] Analyzing {ticker}: {older['date']} → {newer['date']}\n")

    older_risks = extract_risk_list(older["risk_factors"], f"older ({older['date']})")
    newer_risks = extract_risk_list(newer["risk_factors"], f"newer ({newer['date']})")
    changes     = compare_risk_lists(older_risks, newer_risks)
    changes     = deduplicate_changes(changes)
    changes     = consolidate_new_risks(changes, older_risks)   # NEW step 4
    counts      = summarize_counts(changes)

    print(f"[Agent 2] Change summary: {counts}")
    print("[Agent 2] Done.\n")

    return {
        "ticker":        ticker,
        "cik":           fetcher_output.get("cik", ""),
        "newer_date":    newer["date"],
        "older_date":    older["date"],
        "older_risks":   older_risks,
        "newer_risks":   newer_risks,
        "changes":       changes,
        "summary_counts": counts,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agent2_analyzer.py <agent1_output.json>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        fetcher_out = json.load(f)
    result = run(fetcher_out)
    print(json.dumps(result, indent=2))