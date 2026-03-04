import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  createSession,
  getReport,
  getRun,
  sendMessage,
  streamLogs,
} from "./api";
import "./style.css";

/* ── Error Boundary ─────────────────────────────────────────────── */

class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, message: "" };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, message: error?.message || "Render error" };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="page">
          <div className="card">
            <h2>UI Error</h2>
            <p className="muted">{this.state.message}</p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

/* ── Logo ───────────────────────────────────────────────────────── */

function Logo() {
  return (
    <svg
      className="logo"
      viewBox="0 0 40 40"
      width="40"
      height="40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="Plant Pathway Classification"
    >
      <path
        d="M20 4C12 4 6 12 6 20c0 6 4 12 14 16 0-8-6-14-6-20S16 6 20 4z"
        fill="#16a34a"
        opacity="0.75"
      />
      <path
        d="M20 4c8 0 14 8 14 16 0 6-4 12-14 16 0-8 6-14 6-20S24 6 20 4z"
        fill="#16a34a"
        opacity="0.4"
      />
      <line x1="20" y1="36" x2="20" y2="20" stroke="#16a34a" strokeWidth="1.5" />
    </svg>
  );
}

/* ── Pipeline stage config ──────────────────────────────────────── */

const PIPELINE_STAGES = [
  { key: "queue", label: "Data" },
  { key: "preprocess", label: "Preprocess" },
  { key: "train", label: "Train" },
  { key: "inference", label: "Inference" },
  { key: "evaluate", label: "Evaluation" },
  { key: "report", label: "Report" },
];

/* ── Checkmark / X icons ────────────────────────────────────────── */

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path d="M3 7l3 3 5-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function XIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path d="M4 4l6 6M10 4l-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path d="M8 2v8m0 0l-3-3m3 3l3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3 12v2h10v-2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}


function ChevronIcon({ expanded }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 12 12"
      style={{ transform: expanded ? "rotate(90deg)" : "rotate(0deg)", transition: "transform 0.2s" }}
    >
      <path d="M4 2l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );
}

/* ── Pipeline Stepper ───────────────────────────────────────────── */

function PipelineStepper({ stages }) {
  const stageMap = useMemo(() => {
    const map = {};
    (stages || []).forEach((s) => {
      map[s.name] = s;
    });
    return map;
  }, [stages]);

  return (
    <div className="stepper">
      {PIPELINE_STAGES.map((stage, idx) => {
        const stageData = stageMap[stage.key];
        const status = stageData?.status || "queued";
        const prevStatus = idx > 0 ? (stageMap[PIPELINE_STAGES[idx - 1].key]?.status || "queued") : null;
        const lineActive = prevStatus === "completed";

        return (
          <React.Fragment key={stage.key}>
            {idx > 0 && <div className={`stepper-line ${lineActive ? "stepper-line-done" : ""}`} />}
            <div className={`stepper-step stepper-step-${status}`}>
              <div className="stepper-circle">
                {status === "completed" ? (
                  <CheckIcon />
                ) : status === "error" ? (
                  <XIcon />
                ) : (
                  <span className="stepper-number">{idx + 1}</span>
                )}
              </div>
              <span className="stepper-label">{stage.label}</span>
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );
}

/* ── PDF Icon ───────────────────────────────────────────────────── */

function PdfIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <rect x="2" y="1" width="12" height="14" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
      <path d="M5 5h2.5a1 1 0 010 2H5V5z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
      <path d="M5 9h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M5 11.5h4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

/* ── Completion Banner + Deliverables ───────────────────────────── */

function CompletionBanner({ run, report, onShowHtml, reportHtmlRef }) {
  if (!run || run.status !== "completed") return null;

  const downloadBlob = (content, filename, type) => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadPdf = () => {
    // Also show inline report
    onShowHtml();

    // Grab the rendered HTML from the report viewer already in the DOM
    const reportEl = reportHtmlRef?.current;
    const reportHtml = reportEl ? reportEl.innerHTML : "<p>Report not available.</p>";

    const win = window.open("", "_blank");
    if (!win) return;
    win.document.write(`<!DOCTYPE html>
<html><head><meta charset="UTF-8"/>
<title>Classification Report</title>
<style>
  body { font-family: 'Inter', -apple-system, system-ui, sans-serif; color: #1a1d27;
         max-width: 800px; margin: 0 auto; padding: 40px 32px; line-height: 1.7; font-size: 14px; }
  h1 { font-size: 22px; margin-top: 24px; margin-bottom: 12px; }
  h2 { font-size: 18px; margin-top: 20px; margin-bottom: 10px; }
  h3 { font-size: 15px; margin-top: 16px; margin-bottom: 8px; }
  p { margin-bottom: 12px; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }
  th, td { border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; }
  th { background: #f7f8fa; font-weight: 600; }
  img { max-width: 100%; margin: 12px 0; border-radius: 8px; }
  code { background: #f7f8fa; padding: 2px 6px; border-radius: 4px; font-size: 13px; }
  pre { background: #f7f8fa; padding: 14px; border-radius: 8px; overflow-x: auto; border: 1px solid #e5e7eb; }
  @media print { body { padding: 0; } }
</style>
</head><body>${reportHtml}</body></html>`);
    win.document.close();
    // Wait for images to load then trigger print
    setTimeout(() => { win.print(); }, 500);
  };

  const downloadMetadata = () => {
    const meta = {
      run_id: run.id,
      session_id: run.session_id,
      prompt: run.prompt,
      status: run.status,
      stages: run.stages,
      created_at: run.created_at,
      updated_at: run.updated_at,
      run_dir: run.run_dir,
      report_path: run.report_path,
    };
    downloadBlob(JSON.stringify(meta, null, 2), "run_metadata.json", "application/json");
  };

  return (
    <div className="completion-banner">
      <div className="completion-header">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          <circle cx="10" cy="10" r="9" stroke="var(--accent)" strokeWidth="2" />
          <path d="M6 10l3 3 5-6" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span>Classification Complete</span>
      </div>
      <div className="deliverables">
        {report && (
          <button className="btn-deliverable" onClick={downloadPdf}>
            <PdfIcon />
            Report PDF
          </button>
        )}
        <button className="btn-deliverable" onClick={downloadMetadata}>
          <DownloadIcon />
          Metadata (.json)
        </button>
      </div>
    </div>
  );
}

/* ── Error Summary (Overview mode) ──────────────────────────────── */

function ErrorSummary({ run }) {
  if (!run || run.status !== "error") return null;
  const failedStage = run.stages?.find((s) => s.status === "error");

  return (
    <div className="error-banner">
      <div className="error-header">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          <circle cx="10" cy="10" r="9" stroke="var(--error)" strokeWidth="2" />
          <path d="M10 6v4m0 3h.01" stroke="var(--error)" strokeWidth="2" strokeLinecap="round" />
        </svg>
        <span>Pipeline Error{failedStage ? ` \u2014 ${failedStage.name} stage` : ""}</span>
      </div>
      {run.error && <p className="error-detail">{run.error}</p>}
      <p className="error-action">
        Switch to <strong>Operator</strong> mode for full diagnostics.
      </p>
    </div>
  );
}

/* ── Operator Details (collapsible logs) ────────────────────────── */

function OperatorDetails({ logs }) {
  const [expanded, setExpanded] = useState(true);
  const logEndRef = useRef(null);

  useEffect(() => {
    if (expanded) {
      logEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs.length, expanded]);

  return (
    <div className="operator-section">
      <button className="collapse-toggle" onClick={() => setExpanded(!expanded)}>
        <ChevronIcon expanded={expanded} />
        Operator Details
        <span className="log-count">{logs.length} entries</span>
      </button>
      {expanded && (
        <div className="log-panel">
          {logs.length === 0 ? (
            <div className="muted log-empty">Logs will appear here when the pipeline starts.</div>
          ) : (
            logs.map((l, idx) => (
              <div key={`${idx}-${l.timestamp}`} className={`log-line${l.stage === "error" ? " log-error" : ""}`}>
                <span className="log-time">{new Date(l.timestamp).toLocaleTimeString()}</span>
                <span className="log-stage">{l.stage || ""}</span>
                <span className="log-msg">{l.message}</span>
              </div>
            ))
          )}
          <div ref={logEndRef} />
        </div>
      )}
    </div>
  );
}

/* ── Report Viewer ──────────────────────────────────────────────── */

const ReportViewer = React.forwardRef(function ReportViewer({ report, transformImageUri }, ref) {
  if (!report) return null;
  return (
    <div className="report-viewer">
      <div className="report-header">Classification Report</div>
      <div className="report-content markdown" ref={ref}>
        <ReactMarkdown remarkPlugins={[remarkGfm]} urlTransform={transformImageUri}>
          {report}
        </ReactMarkdown>
      </div>
    </div>
  );
});

/* ── Main App ───────────────────────────────────────────────────── */

function App() {
  const [session, setSession] = useState(null);
  const [runId, setRunId] = useState(null);
  const [prompt, setPrompt] = useState("Classify C3 vs C4 using CO2S curves");
  const [logs, setLogs] = useState([]);
  const [status, setStatus] = useState(null);
  const [run, setRun] = useState(null);
  const [report, setReport] = useState("");
  const [mode, setMode] = useState("overview");
  const [showReport, setShowReport] = useState(false);
  const reportHtmlRef = useRef(null);

  useEffect(() => {
    createSession("C3/C4 Classification").then(setSession).catch(console.error);
  }, []);

  useEffect(() => {
    if (!runId) return;

    const source = streamLogs(runId, (evt) => {
      setLogs((prev) => [...prev, evt]);
    });

    let active = true;
    const poll = async () => {
      try {
        const nextRun = await getRun(runId);
        setStatus(nextRun);
        setRun(nextRun);
        if (nextRun.status === "completed" && nextRun.report_path && active) {
          const text = await getReport(runId);
          setReport(text);
          setShowReport(true);
        }
        if (nextRun.status === "completed" || nextRun.status === "error") return;
        setTimeout(poll, 1000);
      } catch (err) {
        console.error(err);
      }
    };
    poll();

    return () => {
      active = false;
      source.close();
    };
  }, [runId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!session) return;
    setLogs([]);
    setReport("");
    setShowReport(false);
    setRun(null);
    setStatus(null);
    const { run_id } = await sendMessage(session.id, prompt);
    setRunId(run_id);
  };

  const transformImageUri = (src) => {
    if (!src || src.startsWith("http") || src.includes("://")) return src;
    if (!src.startsWith("/experiments/") && src.includes("/experiments/")) {
      src = src.slice(src.indexOf("/experiments/"));
    }
    if (!src.startsWith("/")) src = "/" + src;
    return `/api/files${encodeURI(src)}`;
  };

  const isRunning = run && run.status === "running";

  return (
    <div className="page">
      {/* ── Header ──────────────────────────────────────────── */}
      <header className="header">
        <div className="header-left">
          <Logo />
          <div>
            <h1 className="title">Plant Pathway Classification</h1>
            <p className="subtitle">C3/C4 photosynthetic pathway analysis using A/Ci response curves</p>
          </div>
        </div>
        <div className="header-right">
          <div className="mode-toggle">
            <button
              className={`mode-btn${mode === "overview" ? " mode-active" : ""}`}
              onClick={() => setMode("overview")}
            >
              Overview
            </button>
            <button
              className={`mode-btn${mode === "operator" ? " mode-active" : ""}`}
              onClick={() => setMode("operator")}
            >
              Operator
            </button>
          </div>
        </div>
      </header>

      {/* ── Prompt Input ────────────────────────────────────── */}
      <section className="card prompt-card">
        <form onSubmit={handleSubmit} className="prompt-form">
          <div className="prompt-wrapper">
            <label className="prompt-label">Classification Prompt</label>
            <input
              className="prompt-input"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g. Classify C3 vs C4 using CO2S curves"
            />
          </div>
          <button type="submit" className="btn-primary" disabled={!session}>
            Run Pipeline
          </button>
        </form>
      </section>

      {/* ── Pipeline Stepper ────────────────────────────────── */}
      {status && (
        <section className="card stepper-card">
          <PipelineStepper stages={status.stages} />
          {isRunning && (
            <div className="running-indicator">
              <span className="pulse-dot" />
              Performing AI modeling for plant pathway classification
            </div>
          )}
        </section>
      )}

      {/* ── Overview Mode ───────────────────────────────────── */}
      {mode === "overview" && (
        <>
          <ErrorSummary run={run} />
          <CompletionBanner
            run={run}
            report={report}
            onShowHtml={() => setShowReport(true)}
            reportHtmlRef={reportHtmlRef}
          />
          {showReport && report && (
            <section className="card">
              <ReportViewer ref={reportHtmlRef} report={report} transformImageUri={transformImageUri} />
            </section>
          )}
        </>
      )}

      {/* ── Operator Mode ───────────────────────────────────── */}
      {mode === "operator" && runId && (
        <section className="card">
          <OperatorDetails logs={logs} />
          {run && run.status === "error" && run.error && (
            <div className="operator-error">
              <strong>Error:</strong> {run.error}
            </div>
          )}
        </section>
      )}

      {/* Report also visible in operator mode when complete */}
      {mode === "operator" && report && (
        <section className="card">
          <ReportViewer ref={reportHtmlRef} report={report} transformImageUri={transformImageUri} />
        </section>
      )}

      {/* ── Footer ──────────────────────────────────────────── */}
      <footer className="footer">
        <div className="footer-content">
          <svg className="ornl-logo" width="120" height="40" viewBox="0 0 120 40" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="ORNL Logo">
            {/* Oak leaf */}
            <path d="M16 4c-2 2-5 5-6 9-1 3-0.5 6 0.5 8 1.5 3 3 4 3 7 0 2-1 4-2.5 5.5C12.5 35 14 36 16 36s3.5-1 5-2.5C19.5 32 18.5 30 18.5 28c0-3 1.5-4 3-7 1-2 1.5-5 0.5-8-1-4-4-7-6-9z" fill="#16a34a" />
            <line x1="16" y1="16" x2="16" y2="36" stroke="#16a34a" strokeWidth="1" opacity="0.5" />
            <line x1="16" y1="20" x2="12" y2="16" stroke="#16a34a" strokeWidth="0.8" opacity="0.4" />
            <line x1="16" y1="20" x2="20" y2="16" stroke="#16a34a" strokeWidth="0.8" opacity="0.4" />
            <line x1="16" y1="24" x2="13" y2="21" stroke="#16a34a" strokeWidth="0.8" opacity="0.4" />
            <line x1="16" y1="24" x2="19" y2="21" stroke="#16a34a" strokeWidth="0.8" opacity="0.4" />
            {/* OAK RIDGE text */}
            <text x="34" y="17" fontFamily="Inter, -apple-system, system-ui, sans-serif" fontSize="10" fontWeight="700" fill="#1a1d27" letterSpacing="0.08em">OAK RIDGE</text>
            {/* National Laboratory text */}
            <text x="34" y="29" fontFamily="Inter, -apple-system, system-ui, sans-serif" fontSize="8" fontWeight="400" fill="#6b7280" letterSpacing="0.04em">National Laboratory</text>
          </svg>
          <span className="muted">ORNL GPTgp Project funded by DOE-BER</span>
        </div>
      </footer>
    </div>
  );
}

/* ── Mount ──────────────────────────────────────────────────────── */

ReactDOM.createRoot(document.getElementById("root")).render(
  <AppErrorBoundary>
    <App />
  </AppErrorBoundary>
);
