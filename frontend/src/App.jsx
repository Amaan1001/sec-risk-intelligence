import { useState, useEffect, useRef } from "react";

// ── Design tokens ──────────────────────────────────────────────────────────
const COLORS = {
  bg: "#0a0c10",
  surface: "#111419",
  border: "#1e2430",
  accent: "#e8c547",
  accentDim: "#b89c30",
  red: "#e05252",
  orange: "#e07a35",
  green: "#4caf80",
  blue: "#5b9bd5",
  muted: "#6b7280",
  text: "#e2e8f0",
  textDim: "#94a3b8",
};

const CHANGE_META = {
  "New Risk":       { color: COLORS.red,    icon: "⬆", label: "New Risk" },
  "Escalating Risk":{ color: COLORS.orange, icon: "⚠", label: "Escalating" },
  "Stable":         { color: COLORS.blue,   icon: "→",  label: "Stable" },
  "Resolved":       { color: COLORS.green,  icon: "✓",  label: "Resolved" },
};

// ── Typography (Google Fonts loaded via @import) ───────────────────────────
const fontStyle = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,300&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: ${COLORS.bg}; color: ${COLORS.text}; font-family: 'DM Mono', monospace; }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: ${COLORS.bg}; }
  ::-webkit-scrollbar-thumb { background: ${COLORS.border}; border-radius: 3px; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
  }
  @keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }
  @keyframes scanline {
    0%   { top: 0%; }
    100% { top: 100%; }
  }
`;

// ── Small helpers ──────────────────────────────────────────────────────────
function SeverityBar({ score }) {
  return (
    <div style={{ display: "flex", gap: 3, alignItems: "center" }}>
      {[1,2,3,4,5].map(i => (
        <div key={i} style={{
          width: 8, height: 8, borderRadius: 2,
          background: i <= score
            ? (score >= 4 ? COLORS.red : score >= 3 ? COLORS.orange : COLORS.blue)
            : COLORS.border,
          transition: "background 0.2s",
        }} />
      ))}
    </div>
  );
}

function ProbBar({ prob }) {
  const pct = Math.round(prob * 100);
  const col = pct >= 65 ? COLORS.red : pct >= 40 ? COLORS.orange : COLORS.green;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{
        flex: 1, height: 4, background: COLORS.border, borderRadius: 2, overflow: "hidden",
      }}>
        <div style={{
          width: `${pct}%`, height: "100%", background: col,
          borderRadius: 2, transition: "width 0.8s cubic-bezier(.4,0,.2,1)",
        }} />
      </div>
      <span style={{ fontSize: 11, color: col, minWidth: 30, textAlign: "right" }}>{pct}%</span>
    </div>
  );
}

function Tag({ label, color }) {
  return (
    <span style={{
      fontSize: 10, fontFamily: "'DM Mono', monospace", letterSpacing: "0.08em",
      padding: "2px 8px", borderRadius: 3,
      border: `1px solid ${color}44`, color, background: `${color}15`,
    }}>{label}</span>
  );
}

function Spinner() {
  return (
    <div style={{
      width: 18, height: 18, border: `2px solid ${COLORS.border}`,
      borderTop: `2px solid ${COLORS.accent}`,
      borderRadius: "50%", animation: "spin 0.8s linear infinite",
      display: "inline-block",
    }} />
  );
}

// ── Section cards ──────────────────────────────────────────────────────────
function Card({ children, style = {} }) {
  return (
    <div style={{
      background: COLORS.surface,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 8,
      padding: "20px 24px",
      animation: "fadeUp 0.4s ease both",
      ...style,
    }}>{children}</div>
  );
}

function SectionLabel({ text }) {
  return (
    <div style={{
      fontSize: 10, letterSpacing: "0.15em", color: COLORS.muted,
      textTransform: "uppercase", marginBottom: 12, fontWeight: 500,
    }}>{text}</div>
  );
}

// ── Rating badge ───────────────────────────────────────────────────────────
function RatingBadge({ rating }) {
  const map = {
    Escalating: { bg: `${COLORS.red}20`, border: COLORS.red, color: COLORS.red },
    Stable:     { bg: `${COLORS.blue}20`, border: COLORS.blue, color: COLORS.blue },
    Improving:  { bg: `${COLORS.green}20`, border: COLORS.green, color: COLORS.green },
  };
  const s = map[rating] || map.Stable;
  return (
    <span style={{
      fontFamily: "'Fraunces', serif", fontSize: 28, fontWeight: 600,
      color: s.color, padding: "4px 18px",
      border: `1px solid ${s.border}`, borderRadius: 6,
      background: s.bg, display: "inline-block",
    }}>{rating}</span>
  );
}

// ── Risk change row ────────────────────────────────────────────────────────
function RiskRow({ change, index }) {
  const [open, setOpen] = useState(false);
  const meta = CHANGE_META[change.change_type] || CHANGE_META["Stable"];

  return (
    <div style={{
      borderBottom: `1px solid ${COLORS.border}`,
      animation: `fadeUp 0.3s ease ${index * 0.04}s both`,
    }}>
      <div
        onClick={() => setOpen(o => !o)}
        style={{
          display: "flex", alignItems: "center", gap: 12,
          padding: "12px 0", cursor: "pointer",
          transition: "opacity 0.15s",
        }}
        onMouseEnter={e => e.currentTarget.style.opacity = "0.8"}
        onMouseLeave={e => e.currentTarget.style.opacity = "1"}
      >
        <span style={{ color: meta.color, fontSize: 14, width: 16, textAlign: "center" }}>
          {meta.icon}
        </span>
        <span style={{ flex: 1, fontSize: 13, color: COLORS.text }}>{change.title}</span>
        <Tag label={meta.label} color={meta.color} />
        <SeverityBar score={change.severity} />
        <span style={{ color: COLORS.muted, fontSize: 12, marginLeft: 4 }}>
          {open ? "▲" : "▼"}
        </span>
      </div>
      {open && (
        <div style={{
          paddingBottom: 14, paddingLeft: 28,
          animation: "fadeUp 0.2s ease both",
        }}>
          {change.rationale && (
            <p style={{ fontSize: 12, color: COLORS.textDim, lineHeight: 1.7, marginBottom: 8 }}>
              {change.rationale}
            </p>
          )}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 8 }}>
            {change.older_summary && (
              <div style={{ background: COLORS.bg, borderRadius: 6, padding: 12 }}>
                <div style={{ fontSize: 10, color: COLORS.muted, marginBottom: 4, letterSpacing: "0.1em" }}>OLDER FILING</div>
                <p style={{ fontSize: 11, color: COLORS.textDim, lineHeight: 1.6 }}>{change.older_summary}</p>
              </div>
            )}
            {change.newer_summary && (
              <div style={{ background: COLORS.bg, borderRadius: 6, padding: 12 }}>
                <div style={{ fontSize: 10, color: COLORS.muted, marginBottom: 4, letterSpacing: "0.1em" }}>NEWER FILING</div>
                <p style={{ fontSize: 11, color: COLORS.textDim, lineHeight: 1.6 }}>{change.newer_summary}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Prediction row ─────────────────────────────────────────────────────────
function PredRow({ pred, index }) {
  const statusColor = {
    HIT: COLORS.green, PARTIAL: COLORS.orange,
    MISS: COLORS.red, PENDING: COLORS.muted, UNCLEAR: COLORS.muted,
  };

  return (
    <div style={{
      borderBottom: `1px solid ${COLORS.border}`, padding: "12px 0",
      animation: `fadeUp 0.3s ease ${index * 0.04}s both`,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
        <span style={{ flex: 1, fontSize: 12, color: COLORS.text }}>{pred.risk_title}</span>
        <Tag label={pred.timeframe?.replace("_", " ")} color={COLORS.muted} />
        {pred.backtest_status && pred.backtest_status !== "PENDING" && (
          <Tag label={pred.backtest_status} color={statusColor[pred.backtest_status] || COLORS.muted} />
        )}
      </div>
      <ProbBar prob={pred.probability || 0} />
      {pred.reasoning && (
        <p style={{ fontSize: 11, color: COLORS.muted, marginTop: 6, lineHeight: 1.6 }}>
          {pred.reasoning}
        </p>
      )}
    </div>
  );
}

// ── Filter bar ─────────────────────────────────────────────────────────────
function FilterBar({ active, onChange, counts }) {
  const filters = ["All", "New Risk", "Escalating Risk", "Stable", "Resolved"];
  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      {filters.map(f => {
        const isActive = active === f;
        const meta = CHANGE_META[f];
        const col = meta ? meta.color : COLORS.accent;
        const count = f === "All"
          ? Object.values(counts).reduce((a,b) => a + (b.count || 0), 0)
          : counts[f]?.count || 0;
        return (
          <button key={f} onClick={() => onChange(f)} style={{
            fontFamily: "'DM Mono', monospace", fontSize: 11,
            letterSpacing: "0.06em", padding: "5px 12px",
            borderRadius: 4, cursor: "pointer",
            border: isActive ? `1px solid ${col}` : `1px solid ${COLORS.border}`,
            background: isActive ? `${col}20` : "transparent",
            color: isActive ? col : COLORS.muted,
            transition: "all 0.15s",
          }}>
            {f} {count > 0 && <span style={{ opacity: 0.7 }}>({count})</span>}
          </button>
        );
      })}
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  const [ticker, setTicker] = useState("");
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState("All");
  const [tab, setTab] = useState("risks"); // risks | predictions | meta
  const [streamLog, setStreamLog] = useState([]);
  const inputRef = useRef(null);

  const EXAMPLE_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META"];

  // Simulate streaming log updates during analysis
  const runPipeline = async (t) => {
    if (!t.trim()) return;
    setLoading(true);
    setError(null);
    setReport(null);
    setStreamLog([]);
    setFilter("All");
    setTab("risks");

    const steps = [
      { delay: 100,  msg: `[Agent 1] Resolving ticker '${t}' to CIK...` },
      { delay: 600,  msg: `[Agent 1] Fetching two most recent 10-K filing metadata...` },
      { delay: 1400, msg: `[Agent 1] Processing filings — extracting Risk Factors & MD&A...` },
      { delay: 2800, msg: `[Agent 2] Analyzing risk list from older filing...` },
      { delay: 3800, msg: `[Agent 2] Analyzing risk list from newer filing...` },
      { delay: 5200, msg: `[Agent 2] Comparing and classifying changes...` },
      { delay: 6600, msg: `[Agent 2] Deduplicating and consolidating new risks...` },
      { delay: 7800, msg: `[Agent 4] Stage A — Ranking risks by tier...` },
      { delay: 8800, msg: `[Agent 4] Stage B — Assigning calibrated probabilities...` },
      { delay: 9800, msg: `[Agent 3] Synthesizing risk intelligence report...` },
    ];

    steps.forEach(({ delay, msg }) => {
      setTimeout(() => setStreamLog(l => [...l, msg]), delay);
    });

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: t, run_predictions: true }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${response.status}`);
      }

      const parsed = await response.json();
      setReport(parsed);
      setStreamLog(l => [...l, `✓ Report generated — ${parsed.changes?.length || 0} risk entries analysed.`]);
    } catch (e) {
      setError("Failed to generate report: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  const changes = report?.changes || [];
  const filtered = filter === "All" ? changes : changes.filter(c => c.change_type === filter);
  const predictions = report?.predictions || [];

  const trackTotal = predictions.length;
  const tracked = predictions.filter(p => p.backtest_status !== "PENDING");
  const hits = predictions.filter(p => p.backtest_status === "HIT").length;

  return (
    <div style={{ minHeight: "100vh", background: COLORS.bg }}>
      <style>{fontStyle}</style>

      {/* Header */}
      <div style={{
        borderBottom: `1px solid ${COLORS.border}`,
        padding: "18px 32px",
        display: "flex", alignItems: "center", gap: 16,
      }}>
        <div style={{
          fontFamily: "'Fraunces', serif", fontSize: 20, fontStyle: "italic",
          color: COLORS.accent, letterSpacing: "-0.02em",
        }}>
          SEC Risk Intelligence
        </div>
        <div style={{
          flex: 1, height: 1, background: COLORS.border,
        }} />
        <div style={{ fontSize: 11, color: COLORS.muted, letterSpacing: "0.1em" }}>
          EDGAR · GPT-4.1 · AGENTIC
        </div>
      </div>

      {/* Main layout */}
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "40px 24px" }}>

        {/* Search */}
        <Card style={{ marginBottom: 28 }}>
          <SectionLabel text="Ticker Analysis" />
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <input
              ref={inputRef}
              value={ticker}
              onChange={e => setTicker(e.target.value.toUpperCase())}
              onKeyDown={e => e.key === "Enter" && runPipeline(ticker)}
              placeholder="AAPL"
              style={{
                background: COLORS.bg, border: `1px solid ${COLORS.border}`,
                borderRadius: 6, padding: "10px 16px",
                fontFamily: "'DM Mono', monospace", fontSize: 16,
                color: COLORS.text, outline: "none", width: 140,
                letterSpacing: "0.1em", textTransform: "uppercase",
              }}
              onFocus={e => e.target.style.borderColor = COLORS.accent}
              onBlur={e => e.target.style.borderColor = COLORS.border}
            />
            <button
              onClick={() => runPipeline(ticker)}
              disabled={loading || !ticker.trim()}
              style={{
                background: loading ? COLORS.border : COLORS.accent,
                color: COLORS.bg, border: "none", borderRadius: 6,
                padding: "10px 24px", fontFamily: "'DM Mono', monospace",
                fontSize: 13, fontWeight: 500, cursor: loading ? "not-allowed" : "pointer",
                transition: "all 0.15s", display: "flex", alignItems: "center", gap: 8,
                opacity: loading ? 0.7 : 1,
              }}
            >
              {loading ? <><Spinner /> Analysing…</> : "Run Pipeline →"}
            </button>
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
            {EXAMPLE_TICKERS.map(t => (
              <button key={t} onClick={() => { setTicker(t); runPipeline(t); }} style={{
                fontSize: 11, fontFamily: "'DM Mono', monospace",
                padding: "3px 10px", borderRadius: 4,
                border: `1px solid ${COLORS.border}`, background: "transparent",
                color: COLORS.muted, cursor: "pointer",
                transition: "all 0.15s",
              }}
              onMouseEnter={e => { e.target.style.borderColor = COLORS.accent; e.target.style.color = COLORS.accent; }}
              onMouseLeave={e => { e.target.style.borderColor = COLORS.border; e.target.style.color = COLORS.muted; }}
              >{t}</button>
            ))}
          </div>
        </Card>

        {/* Stream log */}
        {(loading || streamLog.length > 0) && !report && (
          <Card style={{ marginBottom: 28, fontFamily: "'DM Mono', monospace" }}>
            <SectionLabel text="Pipeline Output" />
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {streamLog.map((line, i) => (
                <div key={i} style={{
                  fontSize: 11, color: i === streamLog.length - 1 ? COLORS.accent : COLORS.muted,
                  animation: "fadeUp 0.2s ease both",
                }}>{line}</div>
              ))}
              {loading && (
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4 }}>
                  <div style={{
                    width: 6, height: 6, borderRadius: "50%",
                    background: COLORS.accent, animation: "pulse 1.2s infinite",
                  }} />
                  <span style={{ fontSize: 11, color: COLORS.muted }}>Processing…</span>
                </div>
              )}
            </div>
          </Card>
        )}

        {/* Error */}
        {error && (
          <Card style={{ marginBottom: 28, borderColor: `${COLORS.red}66` }}>
            <div style={{ color: COLORS.red, fontSize: 13 }}>{error}</div>
          </Card>
        )}

        {/* Report */}
        {report && (
          <div style={{ animation: "fadeUp 0.5s ease both" }}>
            {/* Header */}
            <div style={{ marginBottom: 24, display: "flex", alignItems: "flex-start", gap: 20 }}>
              <div style={{ flex: 1 }}>
                <div style={{
                  fontFamily: "'Fraunces', serif", fontSize: 36, fontWeight: 600,
                  color: COLORS.text, letterSpacing: "-0.03em", lineHeight: 1.1,
                }}>
                  {report.ticker}
                </div>
                <div style={{ fontSize: 11, color: COLORS.muted, marginTop: 4, letterSpacing: "0.08em" }}>
                  {report.older_date} → {report.newer_date} &nbsp;·&nbsp; 10-K Comparison
                </div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 10, color: COLORS.muted, marginBottom: 6, letterSpacing: "0.1em" }}>
                  OVERALL RISK RATING
                </div>
                <RatingBadge rating={report.report?.overall_rating} />
              </div>
            </div>

            {/* Summary strip */}
            <div style={{
              display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 20,
            }}>
              {Object.entries(report.summary_counts || {}).map(([k, v]) => {
                const meta = CHANGE_META[k];
                return (
                  <Card key={k} style={{ padding: "14px 18px", cursor: "pointer" }}
                    onClick={() => { setFilter(k); setTab("risks"); }}>
                    <div style={{ fontSize: 22, fontFamily: "'Fraunces', serif", fontWeight: 600, color: meta?.color || COLORS.text }}>
                      {v.count}
                    </div>
                    <div style={{ fontSize: 10, color: COLORS.muted, marginTop: 2, letterSpacing: "0.08em" }}>
                      {k.toUpperCase()}
                    </div>
                    {v.avg_severity != null && (
                      <div style={{ marginTop: 6 }}>
                        <SeverityBar score={Math.round(v.avg_severity)} />
                      </div>
                    )}
                  </Card>
                );
              })}
            </div>

            {/* Rating justification + takeaways */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 20 }}>
              <Card>
                <SectionLabel text="Rating Justification" />
                <p style={{ fontSize: 12, color: COLORS.textDim, lineHeight: 1.8 }}>
                  {report.report?.rating_justification}
                </p>
              </Card>
              <Card>
                <SectionLabel text="Key Investor Takeaways" />
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {(report.report?.key_takeaways || []).map((t, i) => (
                    <div key={i} style={{ display: "flex", gap: 10 }}>
                      <span style={{ color: COLORS.accent, fontSize: 10, marginTop: 3, flexShrink: 0 }}>◆</span>
                      <p style={{ fontSize: 11, color: COLORS.textDim, lineHeight: 1.7 }}>{t}</p>
                    </div>
                  ))}
                </div>
              </Card>
            </div>

            {/* Tab nav */}
            <div style={{ display: "flex", gap: 0, marginBottom: 16, borderBottom: `1px solid ${COLORS.border}` }}>
              {[["risks", "Risk Changes"], ["predictions", "12-Month Forecast"], ["meta", "Analyst Note"]].map(([key, label]) => (
                <button key={key} onClick={() => setTab(key)} style={{
                  fontFamily: "'DM Mono', monospace", fontSize: 12,
                  padding: "10px 20px", border: "none", background: "transparent",
                  borderBottom: tab === key ? `2px solid ${COLORS.accent}` : "2px solid transparent",
                  color: tab === key ? COLORS.accent : COLORS.muted,
                  cursor: "pointer", transition: "all 0.15s", letterSpacing: "0.05em",
                  marginBottom: -1,
                }}>{label}</button>
              ))}
            </div>

            {/* Tab: Risk Changes */}
            {tab === "risks" && (
              <Card>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                  <SectionLabel text={`${filtered.length} Entries`} />
                  <FilterBar active={filter} onChange={setFilter} counts={report.summary_counts || {}} />
                </div>
                {filtered.length === 0 && (
                  <div style={{ color: COLORS.muted, fontSize: 12, padding: "16px 0" }}>No entries in this category.</div>
                )}
                {filtered.map((c, i) => <RiskRow key={i} change={c} index={i} />)}
              </Card>
            )}

            {/* Tab: Predictions */}
            {tab === "predictions" && (
              <Card>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                  <SectionLabel text={`${predictions.length} Predictions`} />
                  {tracked.length > 0 && (
                    <div style={{ fontSize: 11, color: COLORS.muted }}>
                      Track record: <span style={{ color: COLORS.green }}>{hits}</span>/{tracked.length} hits
                    </div>
                  )}
                </div>
                {predictions.length === 0 && (
                  <div style={{ color: COLORS.muted, fontSize: 12, padding: "16px 0" }}>
                    No escalating or new risks found — nothing to predict.
                  </div>
                )}
                {[...predictions].sort((a, b) => (b.probability || 0) - (a.probability || 0))
                  .map((p, i) => <PredRow key={i} pred={p} index={i} />)}
              </Card>
            )}

            {/* Tab: Analyst Note */}
            {tab === "meta" && (
              <Card>
                <SectionLabel text="Analyst Note" />
                <p style={{ fontSize: 13, color: COLORS.textDim, lineHeight: 1.9 }}>
                  {report.report?.analyst_note}
                </p>
                <div style={{
                  marginTop: 24, padding: "16px 20px",
                  background: COLORS.bg, borderRadius: 6,
                  borderLeft: `3px solid ${COLORS.border}`,
                }}>
                  <div style={{ fontSize: 10, color: COLORS.muted, marginBottom: 6, letterSpacing: "0.1em" }}>
                    PIPELINE METADATA
                  </div>
                  {[
                    ["Ticker", report.ticker],
                    ["Older Filing", report.older_date],
                    ["Newer Filing", report.newer_date],
                    ["Total Changes", changes.length],
                    ["Predictions", predictions.length],
                  ].map(([k, v]) => (
                    <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                      <span style={{ fontSize: 11, color: COLORS.muted }}>{k}</span>
                      <span style={{ fontSize: 11, color: COLORS.text }}>{v}</span>
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {/* Disclaimer */}
            <div style={{ marginTop: 16, fontSize: 10, color: COLORS.border, textAlign: "center", lineHeight: 1.8 }}>
              AI-generated for educational purposes only · Not investment advice ·
              Verify against original filings at sec.gov
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
