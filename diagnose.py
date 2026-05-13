"""
diagnose.py — Quick diagnostic to inspect Agent 1 extraction output.
Run this to see exactly what text was pulled from each filing before
passing it to Agent 2.

Usage:
    python diagnose.py AAPL
"""

import sys
import json
from utils import load_env
load_env()

import agent1_fetcher

ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

print(f"Running Agent 1 for {ticker}...\n")
output = agent1_fetcher.run(ticker)

for i, filing in enumerate(output["filings"]):
    label = "NEWER" if i == 0 else "OLDER"
    print("=" * 70)
    print(f"{label} FILING — {filing['date']}")
    print(f"URL: {filing['doc_url']}")
    print(f"Risk Factors length: {len(filing['risk_factors']):,} chars")
    print(f"MD&A length:         {len(filing['mda']):,} chars")
    print()

    rf = filing["risk_factors"]
    if rf:
        print("--- Risk Factors FIRST 800 chars ---")
        print(rf[:800])
        print()
        print("--- Risk Factors LAST 400 chars ---")
        print(rf[-400:])
    else:
        print("!!! Risk Factors section is EMPTY !!!")
    print()

# Also check if both filings pulled the same text
rf0 = output["filings"][0]["risk_factors"]
rf1 = output["filings"][1]["risk_factors"]
if rf0 == rf1:
    print("!!! WARNING: Both filings returned IDENTICAL Risk Factors text !!!")
    print("    This means section extraction is pulling the same content for both.")
else:
    overlap = len(set(rf0.split()) & set(rf1.split()))
    total   = max(len(set(rf0.split())), len(set(rf1.split())))
    sim     = overlap / total if total else 0
    print(f"Vocabulary overlap between filings: {sim:.1%}")
    if sim > 0.95:
        print("WARNING: Filings are >95% similar — may explain the Stable rating.")

# Save full output for inspection
with open(f"{ticker}_agent1_diagnostic.json", "w") as f:
    json.dump({
        "ticker": output["ticker"],
        "filings": [
            {
                "date":               fl["date"],
                "doc_url":            fl["doc_url"],
                "risk_factors_chars": len(fl["risk_factors"]),
                "mda_chars":          len(fl["mda"]),
                "risk_factors_first_2000": fl["risk_factors"][:2000],
            }
            for fl in output["filings"]
        ]
    }, f, indent=2)
print(f"\nFull diagnostic saved to {ticker}_agent1_diagnostic.json")