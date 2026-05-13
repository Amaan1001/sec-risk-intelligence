"""
evaluate.py — Accuracy Evaluation for Agent 2 (Risk Change Classifier)
FE 524 Final Project · Group Atlanta

FIX (this version):
- score_predictions now uses fuzzy title matching (difflib.SequenceMatcher,
  threshold 0.80) instead of exact string equality. The LLM sometimes slightly
  rewrites risk titles between the annotation template and final output, causing
  spurious "Unknown" predictions and artificially low accuracy scores.

Usage:
    python evaluate.py generate --tickers AAPL MSFT TSLA NVDA AMZN --output-dir ./eval_data
    python evaluate.py score --predictions ./eval_data --ground-truth ./ground_truth.json
    python evaluate.py qualitative --scores ./qualitative_scores.json
"""

import argparse
import json
import math
import sys
from difflib import SequenceMatcher
from itertools import zip_longest
from pathlib import Path
from collections import defaultdict

import agent1_fetcher
import agent2_analyzer
import agent3_synthesizer

from utils import load_env
load_env()


# ── 1. Generate predictions ───────────────────────────────────────────────────

def generate_predictions(tickers: list[str], output_dir: str) -> None:
    """Run the full pipeline for each ticker and save agent2 output for annotation."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Processing {ticker}...")
        print(f"{'='*50}")
        try:
            a1 = agent1_fetcher.run(ticker)
            a2 = agent2_analyzer.run(a1)
            path = out / f"{ticker}_agent2.json"
            path.write_text(json.dumps(a2, indent=2), encoding="utf-8")
            print(f"Saved → {path}")

            annotation_template = []
            for ch in a2["changes"]:
                annotation_template.append({
                    "title": ch.get("title"),
                    "model_prediction": ch.get("change_type"),
                    "model_severity": ch.get("severity"),
                    "ground_truth_change_type": None,
                    "ground_truth_severity": None,
                    "older_summary": ch.get("older_summary"),
                    "newer_summary": ch.get("newer_summary"),
                    "rationale": ch.get("rationale"),
                })
            ann_path = out / f"{ticker}_annotation_template.json"
            ann_path.write_text(json.dumps(annotation_template, indent=2), encoding="utf-8")
            print(f"Annotation template → {ann_path}")

        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")


# ── 2. Classification accuracy ────────────────────────────────────────────────

CATEGORIES = ["New Risk", "Escalating Risk", "Stable", "Resolved"]

FUZZY_THRESHOLD = 0.80  # titles with similarity >= this are treated as the same risk


def _fuzzy_match(query: str, candidates: dict, threshold: float = FUZZY_THRESHOLD):
    """
    Find the best fuzzy match for query in the candidates dict (title → item).
    Returns the matched item or None if no match exceeds the threshold.
    """
    best_score = 0.0
    best_item  = None
    q = query.lower().strip()
    for key, item in candidates.items():
        score = SequenceMatcher(None, q, key.lower().strip()).ratio()
        if score > best_score:
            best_score = score
            best_item  = item
    return best_item if best_score >= threshold else None


def compute_classification_metrics(predictions: list[str], ground_truth: list[str]) -> dict:
    """Compute per-class Precision, Recall, F1 and macro averages."""
    _SENTINEL = object()
    if len(predictions) != len(ground_truth):
        mismatched = [
            (i, p, g)
            for i, (p, g) in enumerate(zip_longest(predictions, ground_truth, fillvalue=_SENTINEL))
            if p is _SENTINEL or g is _SENTINEL
        ]
        raise AssertionError(
            f"predictions ({len(predictions)}) and ground_truth ({len(ground_truth)}) "
            f"have different lengths. First mismatched index: {mismatched[0][0] if mismatched else '?'}. "
            "This usually means a fuzzy-match failure left one list longer — check 'not_found' count."
        )

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for pred, gt in zip(predictions, ground_truth):
        if pred == gt:
            tp[gt] += 1
        else:
            fp[pred] += 1
            fn[gt] += 1

    results = {}
    for cat in CATEGORIES:
        p  = tp[cat] / (tp[cat] + fp[cat]) if (tp[cat] + fp[cat]) > 0 else 0.0
        r  = tp[cat] / (tp[cat] + fn[cat]) if (tp[cat] + fn[cat]) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        results[cat] = {"precision": round(p, 3), "recall": round(r, 3),
                        "f1": round(f1, 3), "support": tp[cat] + fn[cat]}

    macro_p  = sum(v["precision"] for v in results.values()) / len(CATEGORIES)
    macro_r  = sum(v["recall"]    for v in results.values()) / len(CATEGORIES)
    macro_f1 = sum(v["f1"]        for v in results.values()) / len(CATEGORIES)
    results["macro_avg"] = {
        "precision": round(macro_p, 3),
        "recall":    round(macro_r, 3),
        "f1":        round(macro_f1, 3),
    }

    total   = len(predictions)
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    results["overall_accuracy"] = round(correct / total, 3) if total else 0.0

    return results


def score_predictions(predictions_dir: str, ground_truth_path: str) -> None:
    """
    Load annotated ground-truth files and compute metrics.
    Uses fuzzy title matching so minor LLM rewrites don't cause missed lookups.

    ground_truth.json format:
    {
      "AAPL": [
        {"title": "...", "ground_truth_change_type": "New Risk", "ground_truth_severity": 4},
        ...
      ]
    }
    """
    gt_file = Path(ground_truth_path)
    if not gt_file.exists():
        print(f"[ERROR] Ground truth file not found: {gt_file}")
        sys.exit(1)

    ground_truth = json.loads(gt_file.read_text())
    pred_dir     = Path(predictions_dir)

    all_preds, all_gt = [], []
    severity_errors   = []
    fuzzy_matched     = 0
    exact_matched     = 0
    not_found         = 0

    for ticker, gt_items in ground_truth.items():
        pred_path = pred_dir / f"{ticker}_agent2.json"
        if not pred_path.exists():
            print(f"[WARN] No prediction file for {ticker}, skipping.")
            continue

        pred_data    = json.loads(pred_path.read_text())
        pred_changes = {ch["title"]: ch for ch in pred_data.get("changes", [])}

        for gt_item in gt_items:
            title    = gt_item["title"]
            gt_label = gt_item.get("ground_truth_change_type")
            gt_sev   = gt_item.get("ground_truth_severity")
            if not gt_label:
                continue

            # Try exact match first, then fuzzy
            pred_item = pred_changes.get(title)
            if pred_item:
                exact_matched += 1
            else:
                pred_item = _fuzzy_match(title, pred_changes)
                if pred_item:
                    fuzzy_matched += 1
                else:
                    not_found += 1

            if pred_item:
                pred_label = pred_item.get("change_type", "Unknown")
                pred_sev   = pred_item.get("severity", 0)
            else:
                pred_label = "Unknown"
                pred_sev   = 0

            all_preds.append(pred_label)
            all_gt.append(gt_label)
            if gt_sev is not None and pred_sev:
                severity_errors.append(abs(pred_sev - gt_sev))

    if not all_preds:
        print("[ERROR] No matching predictions found.")
        sys.exit(1)

    metrics = compute_classification_metrics(all_preds, all_gt)

    print("\n" + "="*60)
    print("  Classification Accuracy Report")
    print("="*60)
    print(f"  Total samples evaluated: {len(all_preds)}")
    print(f"  Title matching: {exact_matched} exact, {fuzzy_matched} fuzzy, {not_found} not found")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.1%}")
    print()
    print(f"  {'Category':<20} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Support':>9}")
    print(f"  {'-'*20} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for cat in CATEGORIES:
        m = metrics[cat]
        print(f"  {cat:<20} {m['precision']:>9.3f} {m['recall']:>9.3f} {m['f1']:>9.3f} {m['support']:>9}")
    print()
    m = metrics["macro_avg"]
    print(f"  {'Macro Average':<20} {m['precision']:>9.3f} {m['recall']:>9.3f} {m['f1']:>9.3f}")

    if severity_errors:
        mae = sum(severity_errors) / len(severity_errors)
        print(f"\n  Severity MAE: {mae:.3f} (across {len(severity_errors)} samples)")

    print()
    out_path = Path(predictions_dir) / "classification_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Metrics saved → {out_path}")


# ── 3. Qualitative coherence ──────────────────────────────────────────────────

def compute_qualitative(scores_path: str) -> None:
    """Compute inter-rater agreement and averages from qualitative rubric scores."""
    data   = json.loads(Path(scores_path).read_text())
    raters = data["raters"]
    dims   = data["rubric_dimensions"]
    all_scores = data["scores"]

    print("\n" + "="*60)
    print("  Qualitative Coherence Evaluation")
    print("="*60)

    overall_dim_scores = defaultdict(list)

    for ticker, rater_scores in all_scores.items():
        print(f"\n  {ticker}:")
        for dim in dims:
            vals = [rater_scores[r][dim] for r in raters if r in rater_scores]
            avg  = sum(vals) / len(vals) if vals else 0
            std  = math.sqrt(sum((v - avg)**2 for v in vals) / len(vals)) if len(vals) > 1 else 0
            overall_dim_scores[dim].extend(vals)
            print(f"    {dim:<30} avg={avg:.2f}  std={std:.2f}  scores={vals}")

    print("\n  Overall Averages by Dimension:")
    for dim in dims:
        vals     = overall_dim_scores[dim]
        grand_avg = sum(vals) / len(vals) if vals else 0
        print(f"    {dim:<30} {grand_avg:.2f} / 5.00")

    print("\n  Pairwise Agreement Rate (within ±1 point):")
    for dim in dims:
        agree = 0
        total = 0
        for ticker, rater_scores in all_scores.items():
            vals  = [rater_scores[r][dim] for r in raters if r in rater_scores]
            pairs = [(vals[i], vals[j]) for i in range(len(vals)) for j in range(i+1, len(vals))]
            for a, b in pairs:
                total += 1
                if abs(a - b) <= 1:
                    agree += 1
        rate = agree / total if total else 0
        print(f"    {dim:<30} {rate:.1%}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FE524 Project Evaluation Suite")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Run pipeline and generate annotation templates")
    gen.add_argument("--tickers", nargs="+", required=True)
    gen.add_argument("--output-dir", default="./eval_data")

    score = sub.add_parser("score", help="Score predictions against ground truth")
    score.add_argument("--predictions", required=True)
    score.add_argument("--ground-truth", required=True)

    qual = sub.add_parser("qualitative", help="Compute qualitative inter-rater metrics")
    qual.add_argument("--scores", required=True)

    args = parser.parse_args()

    if args.command == "generate":
        generate_predictions(args.tickers, args.output_dir)
    elif args.command == "score":
        score_predictions(args.predictions, args.ground_truth)
    elif args.command == "qualitative":
        compute_qualitative(args.scores)


if __name__ == "__main__":
    main()