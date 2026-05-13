"""
Agent 1 — Document Fetcher
Accepts a stock ticker, queries SEC EDGAR REST API for the two most recent
10-K filings, and extracts Item 1A (Risk Factors) and Item 7 (MD&A) sections.

Root causes fixed in this version
───────────────────────────────────────────────────────────────────────────────
1. iXBRL heading format (MSFT, GOOGL, JPM, META post-2023)
   Modern filers render section headings as two separate layout elements:
     - A left-column item label:  "ITEM 1A."   (its own <div>/<td>)
     - A right-column title:      "RISK FACTORS" (another <div>/<td>)
   After strip_html these appear on separate lines with no inline relationship.
   Our old "item\s+1a ... risk factors" regex requires them on the same line,
   so it only matched the TOC entry (which does have them together) and never
   the real body.

   Fix: multi-pass extraction —
     Pass 1: try "item 1a ... risk factors" (classic single-line format)
     Pass 2: if pass 1 only yields TOC-length content, try "item 1a" alone
             and look for substantial content after it
     Pass 3: if still nothing, search for the standalone heading
             "RISK FACTORS" / "MANAGEMENT'S DISCUSSION" with content after it

2. Document selection still correct — the 8 MB msft-20250630.htm IS the right
   file (only HTM in the accession besides exhibits). Probe loop keeps it.

FIX (this version):
3. EDGAR pagination for large filers (JPM, BAC, WFC etc.)
   The EDGAR submissions API only returns ~1000 filings in the "recent" block.
   For companies with long filing histories, older 10-Ks (and for some large
   filers even the most recent ones) fall into paginated sub-files listed in
   data["filings"]["files"]. Without following those pages, get_recent_10k_filings
   crashes with "Found only 0 10-K filings" for tickers like JPM.

   Fix: after exhausting the recent block, iterate data["filings"]["files"]
   and fetch each page until we have n 10-K results.
"""

import re
import time
import requests

HEADERS = {
    "User-Agent": "FE524-RiskIntel research@stevens.edu",
    "Accept-Encoding": "gzip, deflate",
}

EDGAR_BASE = "https://data.sec.gov"
SEC_BASE   = "https://www.sec.gov"

MIN_RF_CHARS = 2_000   # minimum chars to consider a match the real body
MIN_RF_WARN  = 500     # warn if final result is still below this


# ══════════════════════════════════════════════════════════════════════════════
# CIK + filing metadata
# ══════════════════════════════════════════════════════════════════════════════

def get_cik(ticker: str) -> str:
    resp = requests.get(
        f"{SEC_BASE}/files/company_tickers.json", headers=HEADERS, timeout=15
    )
    resp.raise_for_status()
    for entry in resp.json().values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR.")


def _extract_10k_from_block(block: dict, results: list, n: int) -> None:
    """
    Helper: scan a filings block dict (recent or paginated) and append
    up to (n - len(results)) 10-K entries into results in place.
    """
    forms    = block.get("form", [])
    dates    = block.get("filingDate", [])
    accnums  = block.get("accessionNumber", [])
    pri_docs = block.get("primaryDocument", [""] * len(forms))

    for form, date, acc, primary_doc in zip(forms, dates, accnums, pri_docs):
        if len(results) >= n:
            break
        if form == "10-K":
            acc_clean = acc.replace("-", "")
            results.append({
                "accession":   acc,
                "acc_clean":   acc_clean,
                "date":        date,
                "primary_doc": primary_doc,
            })


def get_recent_10k_filings(cik: str, n: int = 2) -> list[dict]:
    """
    Fetch metadata for the n most recent 10-K filings for a given CIK.

    FIX: Follows EDGAR pagination (data["filings"]["files"]) so large filers
    like JPM whose older filings aren't in the "recent" block no longer crash.
    """
    resp = requests.get(
        f"{EDGAR_BASE}/submissions/CIK{cik}.json", headers=HEADERS, timeout=15
    )
    resp.raise_for_status()
    data = resp.json()

    results: list[dict] = []

    # ── Step 1: scan the "recent" block ───────────────────────────────────────
    _extract_10k_from_block(data["filings"]["recent"], results, n)

    # ── Step 2: follow paginated sub-files if we still need more ──────────────
    if len(results) < n:
        for page_meta in data["filings"].get("files", []):
            if len(results) >= n:
                break
            page_name = page_meta.get("name", "")
            if not page_name:
                continue
            time.sleep(0.3)
            try:
                page_resp = requests.get(
                    f"{EDGAR_BASE}/submissions/{page_name}",
                    headers=HEADERS, timeout=15
                )
                page_resp.raise_for_status()
                _extract_10k_from_block(page_resp.json(), results, n)
            except Exception as e:
                print(f"[Agent 1]   Warning: failed to fetch paginated filings "
                      f"({page_name}): {e}")

    if len(results) < n:
        raise ValueError(
            f"Found only {len(results)} 10-K filing(s) for CIK {cik} "
            f"(need {n}). The company may not have enough annual filings on EDGAR."
        )

    # Attach CIK fields needed downstream
    for r in results:
        r["cik"]     = cik
        r["cik_int"] = int(cik)

    return results[:n]


# ══════════════════════════════════════════════════════════════════════════════
# Filing index — enumerate HTM candidates
# ══════════════════════════════════════════════════════════════════════════════

def _safe_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


_EXCLUDE_NAME = re.compile(
    r"exhibit|\-ex\d|_ex\d|(?<!\w)ex\d{1,3}(?!\w)|^r\d+\.html?$|cover|certific|consent|^ex\d",
    re.IGNORECASE,
)


def _is_excluded(filename: str) -> bool:
    fn = filename.lower()
    if not (fn.endswith(".htm") or fn.endswith(".html")):
        return True
    return bool(_EXCLUDE_NAME.search(fn))


def _get_htm_candidates(cik_int: int, acc_clean: str, accession: str,
                        ticker_lower: str) -> list[str]:
    base      = f"{SEC_BASE}/Archives/edgar/data/{cik_int}/{acc_clean}/"
    file_list = []

    # Try JSON index first
    time.sleep(0.2)
    try:
        r = requests.get(f"{base}{accession}-index.json", headers=HEADERS, timeout=12)
        if r.status_code == 200:
            for item in r.json().get("directory", {}).get("item", []):
                if isinstance(item, dict) and "name" in item:
                    file_list.append({
                        "name": item["name"],
                        "type": item.get("type", ""),
                        "size": _safe_int(item.get("size", 0)),
                    })
    except Exception:
        pass

    # Fall back to HTML index
    if not file_list:
        time.sleep(0.2)
        try:
            r = requests.get(f"{base}{accession}-index.htm", headers=HEADERS, timeout=12)
            r.raise_for_status()
            for row in re.findall(r"<tr[^>]*>.*?</tr>", r.text, re.DOTALL | re.IGNORECASE):
                href = re.search(r'href="[^"]*?/([^"/]+\.html?)"', row, re.IGNORECASE)
                if not href:
                    continue
                tds      = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL | re.IGNORECASE)
                doc_type = re.sub(r"<[^>]+>", "", tds[0]).strip() if tds else ""
                file_list.append({"name": href.group(1), "type": doc_type, "size": 0})
        except Exception:
            pass

    typed_10k, ticker_hits, others = [], [], []
    for f in file_list:
        if _is_excluded(f["name"]):
            continue
        url = base + f["name"]
        if f["type"].strip().upper() == "10-K":
            typed_10k.append((f["size"], url))
        elif ticker_lower and ticker_lower in f["name"].lower():
            ticker_hits.append((f["size"], url))
        else:
            others.append((f["size"], url))

    for lst in (typed_10k, ticker_hits, others):
        lst.sort(reverse=True)

    return (
        [u for _, u in typed_10k]
        + [u for _, u in ticker_hits]
        + [u for _, u in others]
    )


# ══════════════════════════════════════════════════════════════════════════════
# HTML → plain text
# ══════════════════════════════════════════════════════════════════════════════

_HTML_ENTITIES = [
    ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " "), ("&quot;", '"'),
    ("&ldquo;", "\u201c"), ("&rdquo;", "\u201d"), ("&lsquo;", "\u2018"),
    ("&rsquo;", "\u2019"), ("&ndash;", "\u2013"), ("&mdash;", "\u2014"),
    ("&hellip;", "\u2026"),
]


def strip_html(html: str) -> str:
    # Drop script/style
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html,
                  flags=re.DOTALL | re.IGNORECASE)
    # Block elements → newline
    text = re.sub(r"<(p|div|br|tr|li|h[1-6])\b[^>]*>", "\n", text,
                  flags=re.IGNORECASE)
    # Strip all remaining tags (includes ix:nonNumeric etc.)
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode numeric entities
    text = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), text)
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    for ent, ch in _HTML_ENTITIES:
        text = text.replace(ent, ch)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# Section extraction — three-pass strategy
# ══════════════════════════════════════════════════════════════════════════════
#
# Pass 1 — Classic format:  "ITEM 1A.  RISK FACTORS" on one line
#           Works for: AAPL, NVDA, TSLA, AMZN, most smaller filers
#
# Pass 2 — Split format:    "ITEM 1A." label and "RISK FACTORS" title in
#           separate layout cells, appear on separate lines after strip_html
#           Works for: MSFT, GOOGL, JPM, META (iXBRL two-column layout)
#           Strategy: find "item 1a" then scan forward for substantial content
#
# Pass 3 — Standalone heading: just "RISK FACTORS" / "MANAGEMENT'S DISCUSSION"
#           as a section heading with no item-number prefix visible in text
#           Fallback for unusual layouts
#
# For each pass, we scan matches last-to-first so the real body (which comes
# after the TOC) wins over the TOC entry.
# ══════════════════════════════════════════════════════════════════════════════

_END_PATTERNS = {
    "risk_factors": [
        r"item\s+1b[\.\:]",
        r"item\s+2[\.\:]",
        r"unresolved\s+staff\s+comments",
    ],
    "mda": [
        r"item\s+7a[\.\:]",
        r"item\s+8[\.\:]",
        r"quantitative\s+and\s+qualitative\s+disclosures",
    ],
}

# Pass 1: classic inline "ITEM 1A. RISK FACTORS" format
_PASS1_START = {
    "risk_factors": [
        r"item\s+1a[\.\:]*[\s\S]{0,120}?risk\s+factors",
        r"item\s+1a[\.\:]*\s*[\u2013\u2014\-]?\s*risk\s+factors",
    ],
    "mda": [
        r"item\s+7[\.\:]*[\s\S]{0,120}?management['\u2019]?s?\s+discussion",
        r"item\s+7[\.\:]*\s*[\u2013\u2014\-]?\s*management['\u2019]?s?\s+discussion",
    ],
}

# Pass 2: item number only — look for content following it
_PASS2_START = {
    "risk_factors": [r"item\s+1a\b"],
    "mda":          [r"item\s+7\b"],
}

# Pass 3: standalone section title only (no item number required)
_PASS3_START = {
    "risk_factors": [r"(?:^|\n)\s*risk\s+factors\s*\n"],
    "mda":          [r"(?:^|\n)\s*management['\u2019]?s?\s+discussion\s+and\s+analysis\s*\n"],
}


def _find_end(text_lower: str, after: int, section: str) -> int:
    end = len(text_lower)
    region = text_lower[after:]
    for pat in _END_PATTERNS[section]:
        m = re.search(pat, region)
        if m:
            end = min(end, after + m.start())
    return end


def _best_match(matches: list[re.Match], text: str, text_lower: str,
                section: str, max_chars: int) -> str:
    """
    Given a list of regex matches, return the extracted section from the match
    that yields the most content, preferring matches with >= MIN_RF_CHARS.
    Scans last-to-first so real body (after TOC) wins over TOC entry.
    """
    best_text, best_len = "", 0
    for m in reversed(matches):
        content_start = m.end()
        end_idx       = _find_end(text_lower, content_start, section)
        content_len   = end_idx - content_start
        if content_len > best_len:
            best_len  = content_len
            best_text = text[m.start():end_idx].strip()[:max_chars]
        if content_len >= MIN_RF_CHARS:
            break
    return best_text


def find_section(text: str, section: str, max_chars: int = 120_000) -> str:
    """
    Three-pass section extraction.
    Returns the best result across all passes, preferring >= MIN_RF_CHARS.
    """
    text_lower = text.lower()
    result     = ""
    result_len = 0

    # ── Pass 1: classic inline format ─────────────────────────────────────
    for pat in _PASS1_START[section]:
        hits = list(re.finditer(pat, text_lower, re.DOTALL))
        if hits:
            candidate = _best_match(hits, text, text_lower, section, max_chars)
            if len(candidate) > result_len:
                result     = candidate
                result_len = len(candidate)
            if result_len >= MIN_RF_CHARS:
                return result
            break

    # ── Pass 2: item-number-only label, scan for content after ────────────
    for pat in _PASS2_START[section]:
        hits = list(re.finditer(pat, text_lower))
        if hits:
            candidate = _best_match(hits, text, text_lower, section, max_chars)
            if len(candidate) > result_len:
                result     = candidate
                result_len = len(candidate)
            if result_len >= MIN_RF_CHARS:
                return result
            break

    # ── Pass 3: standalone heading (iXBRL split-layout fallback) ──────────
    for pat in _PASS3_START[section]:
        hits = list(re.finditer(pat, text_lower, re.MULTILINE))
        if hits:
            candidate = _best_match(hits, text, text_lower, section, max_chars)
            if len(candidate) > result_len:
                result     = candidate
                result_len = len(candidate)
            if result_len >= MIN_RF_CHARS:
                return result
            break

    return result  # best we could do across all passes


def extract_sections(html: str) -> dict[str, str]:
    plain = strip_html(html)
    return {
        "risk_factors": find_section(plain, "risk_factors"),
        "mda":          find_section(plain, "mda"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HTTP + probe loop
# ══════════════════════════════════════════════════════════════════════════════

def _fetch(url: str) -> str:
    time.sleep(0.4)
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.text


def fetch_filing_sections(meta: dict) -> tuple[str, dict]:
    """
    Download and extract sections from the best available document.
    Probes candidates in priority order; stops when Risk Factors >= MIN_RF_CHARS.
    Falls back to the candidate with the most content if none clears the bar.
    """
    cik_int      = meta["cik_int"]
    acc_clean    = meta["acc_clean"]
    accession    = meta["accession"]
    ticker_lower = meta.get("ticker_lower", "")
    base_url     = f"{SEC_BASE}/Archives/edgar/data/{cik_int}/{acc_clean}/"

    candidates = _get_htm_candidates(cik_int, acc_clean, accession, ticker_lower)

    primary_url = base_url + meta["primary_doc"]
    if primary_url not in candidates:
        candidates.append(primary_url)

    best_url      = candidates[0]
    best_sections = {"risk_factors": "", "mda": ""}

    for url in candidates:
        print(f"[Agent 1]   Probing: {url.split('/')[-1]}")
        try:
            html     = _fetch(url)
            sections = extract_sections(html)
            rf_len   = len(sections["risk_factors"])

            if rf_len > len(best_sections["risk_factors"]):
                best_sections = sections
                best_url      = url

            if rf_len >= MIN_RF_CHARS:
                print(f"[Agent 1]   ✓ Extracted {rf_len:,} chars of Risk Factors")
                break

        except Exception as e:
            print(f"[Agent 1]   ✗ {url.split('/')[-1]}: {e}")
            continue

    return best_url, best_sections


# ══════════════════════════════════════════════════════════════════════════════
# Main agent entry point
# ══════════════════════════════════════════════════════════════════════════════

def run(ticker: str) -> dict:
    print(f"[Agent 1] Resolving ticker '{ticker}' to CIK...")
    cik = get_cik(ticker)
    print(f"[Agent 1] CIK = {cik}")

    print("[Agent 1] Fetching two most recent 10-K filing metadata...")
    filings_meta = get_recent_10k_filings(cik, n=2)
    for meta in filings_meta:
        meta["ticker_lower"] = ticker.lower()

    results = []
    for meta in filings_meta:
        print(f"[Agent 1] Processing 10-K filed {meta['date']} "
              f"(acc: {meta['accession']})...")

        doc_url, sections = fetch_filing_sections(meta)
        rf_len  = len(sections["risk_factors"])
        mda_len = len(sections["mda"])

        print(f"[Agent 1]   Final document: {doc_url.split('/')[-1]}")
        print(f"[Agent 1]   Risk Factors: {rf_len:,} chars | MD&A: {mda_len:,} chars")

        if rf_len < MIN_RF_WARN:
            print(f"[Agent 1]   WARNING: Risk Factors only {rf_len} chars — "
                  f"filing may use an unusual structure.")

        results.append({
            "date":         meta["date"],
            "accession":    meta["accession"],
            "doc_url":      doc_url,
            "risk_factors": sections["risk_factors"],
            "mda":          sections["mda"],
        })

    print("[Agent 1] Done.\n")
    return {"ticker": ticker.upper(), "cik": cik, "filings": results}


if __name__ == "__main__":
    import json, sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    output = run(ticker)
    print(json.dumps({
        "ticker": output["ticker"],
        "filings": [
            {
                "date":               f["date"],
                "accession":          f["accession"],
                "doc_url":            f["doc_url"],
                "risk_factors_chars": len(f["risk_factors"]),
                "mda_chars":          len(f["mda"]),
            }
            for f in output["filings"]
        ]
    }, indent=2))