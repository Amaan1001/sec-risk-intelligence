"""
Agent 3 — Report Synthesizer  (updated for Agent 4 integration)
Receives structured change analysis from Agent 2 + predictions from Agent 4
and produces:
  1. A plain-English risk intelligence report (as structured text)
  2. A JSON export
  3. A PDF export — now includes a Prediction Forecast + Model Track Record section
"""

import json
import textwrap
from openai import OpenAI
from pathlib import Path
from datetime import datetime

from utils import load_env, parse_llm_json
load_env()

CLIENT = OpenAI()
MODEL = "gpt-4.1-mini"


# ── synthesis prompt ──────────────────────────────────────────────────────────

SYNTH_SYSTEM = """You are a senior equity research analyst writing a risk intelligence
briefing for retail investors and small advisory firms. You have received structured
data about how a company's SEC-disclosed risk factors changed between two consecutive
annual filings, plus forward-looking predictions about which risks are most likely to
materially impact the company.

Your job is to write a plain-English Risk Intelligence Report. The report must include:

1. OVERALL RISK RATING — choose exactly one: "Escalating" | "Stable" | "Improving"
2. RATING JUSTIFICATION — 2–3 sentences explaining the rating based on the data
3. KEY INVESTOR TAKEAWAYS — 3–5 bullet points, each 1–2 sentences, highlighting the
   most important changes an investor should know about
4. DETAILED FINDINGS — for each change_type category (New Risk, Escalating Risk,
   Stable, Resolved), list the top entries with a 1-sentence description and severity
5. ANALYST NOTE — 1 paragraph of forward-looking commentary that incorporates the
   prediction probabilities where relevant

Respond ONLY with a JSON object with these exact keys:
  "overall_rating": string,
  "rating_justification": string,
  "key_takeaways": [string, ...],
  "detailed_findings": {
    "new_risks": [{"title": string, "description": string, "severity": int}, ...],
    "escalating_risks": [{"title": string, "description": string, "severity": int}, ...],
    "resolved_risks": [{"title": string, "description": string}, ...],
    "stable_risks": [{"title": string, "description": string}, ...]
  },
  "analyst_note": string

No markdown fences, no preamble."""


def synthesize(analyzer_output: dict, agent4_output: dict = None) -> dict:
    """Call the LLM to produce the structured report from Agent 2 + 4 output."""
    print("[Agent 3] Synthesizing risk intelligence report...")

    changes = analyzer_output["changes"]
    counts  = analyzer_output["summary_counts"]
    sorted_changes = sorted(changes, key=lambda x: x.get("severity", 0), reverse=True)

    payload = {
        "ticker": analyzer_output["ticker"],
        "comparison_period": {
            "older_filing": analyzer_output["older_date"],
            "newer_filing": analyzer_output["newer_date"],
        },
        "change_counts":  counts,
        "risk_changes":   sorted_changes[:60],
    }

    # Inject top predictions into prompt context if available
    if agent4_output and agent4_output.get("predictions"):
        top_preds = sorted(
            agent4_output["predictions"],
            key=lambda x: x.get("probability", 0),
            reverse=True
        )[:10]
        payload["top_materialization_predictions"] = top_preds

    resp = CLIENT.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYNTH_SYSTEM},
            {"role": "user",   "content": json.dumps(payload, indent=2)},
        ],
    )
    raw = resp.choices[0].message.content
    report = parse_llm_json(raw)
    print(f"[Agent 3] Overall Risk Rating: {report.get('overall_rating', 'N/A')}")
    return report


# ── PDF generation ────────────────────────────────────────────────────────────

def _wrap(text: str, width: int = 95) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def _probability_bar(prob: float, width: int = 20) -> str:
    """ASCII probability bar e.g. '████████░░░░░░░░░░░░ 42%'"""
    filled = round(prob * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"{bar}  {round(prob * 100)}%"


def generate_pdf(report: dict, analyzer_output: dict, out_path: Path,
                 agent4_output: dict = None) -> None:
    """Generate PDF. Now includes Prediction Forecast + Track Record sections."""
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )

        ticker = analyzer_output["ticker"]
        older  = analyzer_output["older_date"]
        newer  = analyzer_output["newer_date"]
        rating = report.get("overall_rating", "N/A")

        RATING_COLORS = {
            "Escalating": colors.HexColor("#C0392B"),
            "Stable":     colors.HexColor("#2471A3"),
            "Improving":  colors.HexColor("#1E8449"),
        }
        rating_color = RATING_COLORS.get(rating, colors.black)

        doc = SimpleDocTemplate(
            str(out_path), pagesize=LETTER,
            rightMargin=0.85*inch, leftMargin=0.85*inch,
            topMargin=0.85*inch,   bottomMargin=0.85*inch,
        )
        styles = getSampleStyleSheet()

        def style(name, **kw):
            s = styles[name].clone(name + "_custom_" + str(id(kw)))
            for k, v in kw.items():
                setattr(s, k, v)
            return s

        title_style   = style("Title",   fontSize=20, textColor=colors.HexColor("#1a1a2e"), spaceAfter=4)
        sub_style     = style("Normal",  fontSize=10, textColor=colors.grey, spaceAfter=12)
        h1_style      = style("Heading1",fontSize=13, textColor=colors.HexColor("#1a1a2e"), spaceBefore=14, spaceAfter=6)
        h2_style      = style("Heading2",fontSize=11, textColor=colors.HexColor("#2c3e50"), spaceBefore=10, spaceAfter=4)
        body_style    = style("Normal",  fontSize=9.5, leading=14, spaceAfter=6)
        bullet_style  = style("Normal",  fontSize=9.5, leading=14, leftIndent=16, spaceAfter=4)
        mono_style    = style("Normal",  fontSize=8.5, fontName="Courier", leading=12, spaceAfter=4)
        rating_style  = style("Normal",  fontSize=22, textColor=rating_color,
                               fontName="Helvetica-Bold", spaceAfter=6)

        story = []

        # ── Header ────────────────────────────────────────────────────────────
        story.append(Paragraph(f"{ticker} · SEC Risk Intelligence Report", title_style))
        story.append(Paragraph(
            f"Comparing 10-K filings: {older} → {newer} &nbsp;|&nbsp; "
            f"Generated {datetime.today().strftime('%B %d, %Y')}",
            sub_style,
        ))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#cccccc"), spaceAfter=12))

        # ── Overall rating ─────────────────────────────────────────────────────
        story.append(Paragraph("Overall Risk Rating", h1_style))
        story.append(Paragraph(rating, rating_style))
        story.append(Paragraph(_wrap(report.get("rating_justification", "")), body_style))
        story.append(Spacer(1, 8))

        # ── Key takeaways ──────────────────────────────────────────────────────
        story.append(Paragraph("Key Investor Takeaways", h1_style))
        for item in report.get("key_takeaways", []):
            story.append(Paragraph(f"• {item}", bullet_style))
        story.append(Spacer(1, 8))

        # ── Change counts table ────────────────────────────────────────────────
        counts = analyzer_output.get("summary_counts", {})
        if counts:
            story.append(Paragraph("Change Summary", h1_style))
            table_data = [["Change Type", "Count", "Avg Severity"]]
            for ct, v in counts.items():
                table_data.append([ct, str(v["count"]), str(v["avg_severity"])])
            tbl = Table(table_data, colWidths=[3.4*inch, 1.2*inch, 1.5*inch])
            tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
                ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",      (0,0), (-1,-1), 9),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#f7f9fc"), colors.white]),
                ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
                ("ALIGN",         (1,0), (-1,-1), "CENTER"),
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING",    (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                ("LEFTPADDING",   (0,0), (-1,-1), 7),
                ("RIGHTPADDING",  (0,0), (-1,-1), 7),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 12))

        # ── Prediction Forecast (NEW SECTION) ─────────────────────────────────
        if agent4_output and agent4_output.get("predictions"):
            preds = agent4_output["predictions"]
            # Only show New/Escalating with probability >= 0.15
            show_preds = [
                p for p in preds
                if p.get("change_type") in ("New Risk", "Escalating Risk")
                and p.get("probability", 0) >= 0.15
            ]
            show_preds.sort(key=lambda x: x.get("probability", 0), reverse=True)

            if show_preds:
                story.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor("#cccccc"), spaceAfter=10))
                story.append(Paragraph("🔮 Risk Materialization Forecast", h1_style))
                story.append(Paragraph(
                    "Probability that each flagged risk will materially impact the company "
                    "within 12 months, based on SEC filing language and historical patterns.",
                    body_style,
                ))
                story.append(Spacer(1, 6))

                # Dropped "Type" column — redundant (all rows are New/Escalating) and too wide.
                # Risk title wrapped in Paragraph so long names word-wrap instead of overflow.
                cell_style = style("Normal", fontSize=8.5, leading=11)
                hdr_style  = style("Normal", fontSize=8.5, fontName="Helvetica-Bold",
                                   textColor=colors.white)
                pred_table_data = [[
                    Paragraph("Risk", hdr_style),
                    Paragraph("Risk Type", hdr_style),
                    Paragraph("Impact Type", hdr_style),
                    Paragraph("Timeframe", hdr_style),
                    Paragraph("Probability", hdr_style),
                ]]
                for p in show_preds[:15]:
                    prob_pct = f"{round(p.get('probability', 0) * 100)}%"
                    pred_table_data.append([
                        Paragraph(p.get("risk_title", ""), cell_style),
                        Paragraph(p.get("change_type", ""), cell_style),
                        Paragraph(p.get("impact_type", "").replace("_", " ").title(), cell_style),
                        Paragraph(p.get("timeframe", "").replace("_", " ").title(), cell_style),
                        Paragraph(prob_pct, cell_style),
                    ])

                # Total = 6.8" usable; leave ~0.08" breathing room → 6.72" total
                col_w = [2.6*inch, 1.2*inch, 1.2*inch, 1.0*inch, 0.72*inch]
                pred_tbl = Table(pred_table_data, colWidths=col_w, repeatRows=1)

                # Color-code probability column
                prob_styles = [
                    ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#34495e")),
                    ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
                    ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
                    ("FONTSIZE",      (0,0), (-1,-1), 8.5),
                    ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#fdfefe"), colors.HexColor("#f4f6f7")]),
                    ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
                    ("ALIGN",         (1,0), (-1,-1), "CENTER"),
                    ("VALIGN",        (0,0), (-1,-1), "TOP"),
                    ("TOPPADDING",    (0,0), (-1,-1), 5),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                    ("LEFTPADDING",   (0,0), (-1,-1), 5),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 5),
                ]
                # Red-tint high probability rows
                for i, p in enumerate(show_preds[:15], start=1):
                    prob = p.get("probability", 0)
                    if prob >= 0.6:
                        prob_styles.append(("BACKGROUND", (4,i), (4,i), colors.HexColor("#FADBD8")))
                    elif prob >= 0.4:
                        prob_styles.append(("BACKGROUND", (4,i), (4,i), colors.HexColor("#FDEBD0")))
                    else:
                        prob_styles.append(("BACKGROUND", (4,i), (4,i), colors.HexColor("#EAFAF1")))

                pred_tbl.setStyle(TableStyle(prob_styles))
                story.append(pred_tbl)
                story.append(Spacer(1, 8))

                # Top 3 predictions with reasoning
                story.append(Paragraph("Top Predictions — Analyst Reasoning", h2_style))
                for p in show_preds[:3]:
                    conf = p.get("confidence", "").title()
                    prob_pct = round(p.get("probability", 0) * 100)
                    story.append(Paragraph(
                        f"<b>{p.get('risk_title', '')}</b> — "
                        f"{prob_pct}% probability ({conf} confidence)",
                        h2_style,
                    ))
                    story.append(Paragraph(p.get("reasoning", ""), body_style))
                story.append(Spacer(1, 8))

        # ── Track Record (NEW SECTION) ─────────────────────────────────────────
        if agent4_output:
            global_tr = agent4_output.get("track_record_global", {})
            ticker_tr = agent4_output.get("track_record_ticker", {})

            total_global = global_tr.get("total_evaluated", 0)
            total_ticker = ticker_tr.get("total_evaluated", 0)

            if total_global > 0 or total_ticker > 0:
                story.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor("#cccccc"), spaceAfter=10))
                story.append(Paragraph("📊 Model Track Record", h1_style))
                story.append(Paragraph(
                    "Historical accuracy of this system's prior predictions, "
                    "scored against subsequent 10-Q filings.",
                    body_style,
                ))

                def _tr_table(tr_data: dict, label: str):
                    if not tr_data or tr_data.get("total_evaluated", 0) == 0:
                        return
                    story.append(Paragraph(label, h2_style))
                    tr_rows = [["Metric", "Value"]]
                    tr_rows.append(["Total Backtested Predictions", str(tr_data.get("total_evaluated", 0))])
                    tr_rows.append(["Direct Hits", str(tr_data.get("hits", 0))])
                    tr_rows.append(["Partial Hits", str(tr_data.get("partials", 0))])
                    tr_rows.append(["Misses", str(tr_data.get("misses", 0))])
                    tr_rows.append(["Unclear", str(tr_data.get("unclear", 0))])
                    tr_rows.append(["Hit Rate (direct)", f"{round(tr_data.get('hit_rate', 0)*100)}%"])
                    adj = tr_data.get("adjusted_hit_rate")
                    if adj is not None:
                        tr_rows.append(["Adjusted Hit Rate (partial=0.5)", f"{round(adj*100)}%"])
                    def_rate = tr_data.get("definitive_hit_rate")
                    if def_rate is not None:
                        tr_rows.append(["Definitive Hit Rate (excl. Unclear)", f"{round(def_rate*100)}%"])

                    tr_tbl = Table(tr_rows, colWidths=[3.2*inch, 1.5*inch])
                    tr_tbl.setStyle(TableStyle([
                        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#34495e")),
                        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
                        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
                        ("FONTSIZE",      (0,0), (-1,-1), 9),
                        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#f7f9fc"), colors.white]),
                        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
                        ("ALIGN",         (1,0), (-1,-1), "CENTER"),
                        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                        ("TOPPADDING",    (0,0), (-1,-1), 5),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                        ("LEFTPADDING",   (0,0), (-1,-1), 7),
                        ("RIGHTPADDING",  (0,0), (-1,-1), 7),
                    ]))
                    story.append(tr_tbl)
                    story.append(Spacer(1, 8))

                _tr_table(ticker_tr, f"{ticker} — Historical Accuracy")
                _tr_table(global_tr, "All Tickers — Historical Accuracy")

                if total_global == 0:
                    story.append(Paragraph(
                        "ℹ️  No backtested predictions yet. Run the pipeline again in 90+ days "
                        "to begin building a track record as 10-Qs become available.",
                        style("Normal", fontSize=9, textColor=colors.grey, spaceAfter=6),
                    ))

        # ── Detailed findings ──────────────────────────────────────────────────
        findings = report.get("detailed_findings", {})

        def add_findings_section(heading, items, show_severity=True):
            if not items:
                return
            story.append(Paragraph(heading, h1_style))
            for item in items:
                sev = f"  [Severity: {item['severity']}/5]" if show_severity and "severity" in item else ""
                story.append(Paragraph(f"<b>{item['title']}</b>{sev}", h2_style))
                story.append(Paragraph(item.get("description", ""), body_style))

        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#cccccc"), spaceAfter=10))
        add_findings_section("🔴 New Risks",              findings.get("new_risks", []))
        add_findings_section("🟠 Escalating Risks",       findings.get("escalating_risks", []))
        add_findings_section("✅ Resolved Risks",          findings.get("resolved_risks", []),  show_severity=False)
        add_findings_section("🔵 Stable Risks (selected)", findings.get("stable_risks", [])[:5], show_severity=False)

        # ── Analyst note ───────────────────────────────────────────────────────
        story.append(Paragraph("Analyst Note", h1_style))
        story.append(Paragraph(_wrap(report.get("analyst_note", "")), body_style))

        # ── Footer ─────────────────────────────────────────────────────────────
        story.append(Spacer(1, 16))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#cccccc"), spaceAfter=6))
        story.append(Paragraph(
            "This report was generated by an AI system for educational purposes only. "
            "It does not constitute investment advice. Always verify information against "
            "original SEC filings at sec.gov. Prediction probabilities are model estimates "
            "and not guarantees of future outcomes.",
            style("Normal", fontSize=7.5, textColor=colors.grey),
        ))

        doc.build(story)
        print(f"[Agent 3] PDF saved → {out_path}")

    except ImportError:
        txt_path = out_path.with_suffix(".txt")
        print(f"[Agent 3] reportlab not installed — writing plain-text to {txt_path}")
        _write_text_report(report, analyzer_output, txt_path, agent4_output)


def _write_text_report(report: dict, analyzer_output: dict, out_path: Path,
                       agent4_output: dict = None) -> None:
    lines = []
    ticker = analyzer_output["ticker"]
    lines.append("=" * 80)
    lines.append(f"  {ticker} — SEC Risk Intelligence Report")
    lines.append(f"  {analyzer_output['older_date']} → {analyzer_output['newer_date']}")
    lines.append("=" * 80)
    lines.append(f"\nOVERALL RISK RATING: {report.get('overall_rating','N/A')}")
    lines.append("\n" + _wrap(report.get("rating_justification", "")))
    lines.append("\nKEY INVESTOR TAKEAWAYS")
    lines.append("-" * 40)
    for t in report.get("key_takeaways", []):
        lines.append(f"  • {_wrap(t, 90)}")

    if agent4_output and agent4_output.get("predictions"):
        lines.append("\nRISK MATERIALIZATION FORECAST")
        lines.append("-" * 40)
        for p in sorted(agent4_output["predictions"],
                        key=lambda x: x.get("probability", 0), reverse=True)[:10]:
            prob_pct = round(p.get("probability", 0) * 100)
            bar = _probability_bar(p.get("probability", 0))
            lines.append(f"  {p.get('risk_title','')[:60]}")
            lines.append(f"    {bar}  [{p.get('impact_type','').replace('_',' ')}]")

    if agent4_output:
        tr = agent4_output.get("track_record_global", {})
        if tr.get("total_evaluated", 0) > 0:
            lines.append("\nMODEL TRACK RECORD (all tickers)")
            lines.append("-" * 40)
            lines.append(f"  Total evaluated:  {tr['total_evaluated']}")
            lines.append(f"  Hit rate:         {round(tr.get('hit_rate',0)*100)}%")
            lines.append(f"  Adj hit rate:     {round(tr.get('adjusted_hit_rate',0)*100)}%")

    lines.append("\nANALYST NOTE")
    lines.append("-" * 40)
    lines.append(_wrap(report.get("analyst_note", "")))
    lines.append("\n" + "=" * 80)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Agent 3] Text report saved → {out_path}")


# ── main agent function ───────────────────────────────────────────────────────

def run(analyzer_output: dict, output_dir: str = ".",
        agent4_output: dict = None) -> dict:
    """
    Full Agent 3 pipeline.
    Now accepts optional agent4_output to embed predictions in report.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ticker     = analyzer_output["ticker"]
    newer_date = analyzer_output["newer_date"].replace("-", "")

    report = synthesize(analyzer_output, agent4_output)

    # Save JSON
    json_path = out / f"{ticker}_risk_report_{newer_date}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "ticker":        ticker,
                "older_filing":  analyzer_output["older_date"],
                "newer_filing":  analyzer_output["newer_date"],
                "generated":     datetime.today().isoformat(),
            },
            "report":        report,
            "change_counts": analyzer_output["summary_counts"],
            "predictions":   agent4_output.get("predictions", [])   if agent4_output else [],
            "track_record":  {
                "global": agent4_output.get("track_record_global", {}) if agent4_output else {},
                "ticker": agent4_output.get("track_record_ticker", {}) if agent4_output else {},
            },
        }, f, indent=2)
    print(f"[Agent 3] JSON saved → {json_path}")

    # Save PDF
    pdf_path = out / f"{ticker}_risk_report_{newer_date}.pdf"
    generate_pdf(report, analyzer_output, pdf_path, agent4_output)

    print("[Agent 3] Done.\n")
    return report


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python agent3_synthesizer.py <agent2_output.json> [agent4_output.json] [output_dir]")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        a2_out = json.load(f)
    a4_out = None
    if len(sys.argv) >= 3 and sys.argv[2].endswith(".json"):
        with open(sys.argv[2]) as f:
            a4_out = json.load(f)
    out_dir = sys.argv[-1] if not sys.argv[-1].endswith(".json") else "."
    result = run(a2_out, out_dir, a4_out)
    print(json.dumps(result, indent=2))