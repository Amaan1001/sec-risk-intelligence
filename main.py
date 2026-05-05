"""
main.py — Orchestrator for the Agentic SEC Filing Risk Intelligence System
FE 524 Final Project · Group Atlanta

Usage:
    python main.py AAPL
    python main.py MSFT --output-dir ./reports
    python main.py TSLA --save-intermediate       # saves agent1/2/4 JSON for debugging
    python main.py NVDA --no-predict              # skip Agent 4 (faster, no predictions)

Pipeline:
    Ticker → Agent 1 (Fetch 10-Ks) → Agent 2 (Classify Risk Changes)
          → Agent 4 (Predict Materialization + Backtest) → Agent 3 (Synthesize Report)
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv("../.env", override=True)

import agent1_fetcher
import agent2_analyzer
import agent3_synthesizer
import agent4_predictor


def run_pipeline(
    ticker: str,
    output_dir: str = "./reports",
    save_intermediate: bool = False,
    run_predictions: bool = True,
) -> dict:
    start = time.time()

    print(f"\n{'='*60}")
    print(f"  SEC Risk Intelligence System — {ticker.upper()}")
    print(f"{'='*60}\n")

    # ── Agent 1: Fetch 10-K documents ────────────────────────────────────────
    a1_output = agent1_fetcher.run(ticker)

    if save_intermediate:
        p = Path(output_dir) / f"{ticker.upper()}_agent1.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        slim = {
            "ticker":  a1_output["ticker"],
            "filings": [
                {
                    "date":         f["date"],
                    "accession":    f["accession"],
                    "doc_url":      f["doc_url"],
                    "risk_factors": f["risk_factors"],
                    "mda":          f["mda"],
                }
                for f in a1_output["filings"]
            ],
        }
        p.write_text(json.dumps(slim, indent=2), encoding="utf-8")
        print(f"[main] Agent 1 output saved → {p}\n")

    # ── Agent 2: Analyze risk changes ─────────────────────────────────────────
    a2_output = agent2_analyzer.run(a1_output)

    if save_intermediate:
        p = Path(output_dir) / f"{ticker.upper()}_agent2.json"
        p.write_text(json.dumps(a2_output, indent=2), encoding="utf-8")
        print(f"[main] Agent 2 output saved → {p}\n")

    # ── Agent 4: Predict & Backtest ───────────────────────────────────────────
    a4_output = None
    if run_predictions:
        a4_output = agent4_predictor.run(a2_output, run_backtest=True)

        if save_intermediate:
            p = Path(output_dir) / f"{ticker.upper()}_agent4.json"
            p.write_text(json.dumps(a4_output, indent=2, default=str), encoding="utf-8")
            print(f"[main] Agent 4 output saved → {p}\n")
    else:
        print("[main] Skipping Agent 4 (--no-predict flag set).\n")

    # ── Agent 3: Synthesize report ────────────────────────────────────────────
    report = agent3_synthesizer.run(
        a2_output,
        output_dir=output_dir,
        agent4_output=a4_output,
    )

    elapsed = round(time.time() - start, 1)
    print("=" * 60)
    print(f"  Pipeline complete in {elapsed}s")
    print(f"  Overall Rating: {report.get('overall_rating', 'N/A')}")

    if a4_output and a4_output.get("predictions"):
        n_preds = len(a4_output["predictions"])
        tr = a4_output.get("track_record_global", {})
        total_bt = tr.get("total_evaluated", 0)
        print(f"  Predictions generated: {n_preds}")
        if total_bt > 0:
            print(f"  Model track record: {round(tr.get('hit_rate',0)*100)}% hit rate "
                  f"({total_bt} backtested predictions)")
        else:
            print(f"  Track record: building — run again as 10-Qs become available")

    print(f"  Reports saved to: {output_dir}/")
    print("=" * 60)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Agentic SEC Filing Risk Intelligence System",
        epilog="Example: python main.py AAPL --output-dir ./reports",
    )
    parser.add_argument("ticker",             help="Stock ticker symbol (e.g. AAPL, MSFT, TSLA)")
    parser.add_argument("--output-dir",       default="./reports", help="Directory for output files")
    parser.add_argument("--save-intermediate",action="store_true",
                        help="Save agent1/2/4 JSON outputs for debugging")
    parser.add_argument("--no-predict",       action="store_true",
                        help="Skip Agent 4 predictions and backtesting (faster)")
    args = parser.parse_args()

    try:
        run_pipeline(
            ticker=args.ticker,
            output_dir=args.output_dir,
            save_intermediate=args.save_intermediate,
            run_predictions=not args.no_predict,
        )
    except ValueError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected failure: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
