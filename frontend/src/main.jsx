import React, { useEffect, useMemo, useState } from "react";
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
          <div className="panel">
            <div className="panel-title">UI Error</div>
            <div className="muted">{this.state.message}</div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function MarkdownViewer({ content, transformImageUri }) {
  if (!content) return null;
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} urlTransform={transformImageUri}>
      {content}
    </ReactMarkdown>
  );
}

const stageOrder = ["queue", "preprocess", "train", "inference", "evaluate", "report"];

function StatusPills({ stages }) {
  return (
    <div className="pills">
      {stages.map((s) => (
        <div key={s.name} className={`pill pill-${s.status}`}>
          <span className="pill-name">{s.name}</span>
          <span className="pill-dot" />
        </div>
      ))}
    </div>
  );
}

function LogPanel({ logs }) {
  return (
    <div className="log-panel">
      {logs.map((l, idx) => (
        <div key={`${idx}-${l.timestamp}`} className="log-line">
          <span className="log-time">{new Date(l.timestamp).toLocaleTimeString()}</span>
          <span className="log-stage">{l.stage || ""}</span>
          <span className="log-msg">{l.message}</span>
        </div>
      ))}
    </div>
  );
}

function App() {
  const [session, setSession] = useState(null);
  const [runId, setRunId] = useState(null);
  const [prompt, setPrompt] = useState("Classify C3 vs C4 using CO2S curves");
  const [logs, setLogs] = useState([]);
  const [status, setStatus] = useState(null);
  const [run, setRun] = useState(null);
  const [report, setReport] = useState("");

  useEffect(() => {
    createSession("C3/C4 Classification Console").then(setSession).catch(console.error);
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

  const orderedStages = useMemo(() => {
    if (!status?.stages) return [];
    const stageMap = Object.fromEntries(status.stages.map((s) => [s.name, s]));
    return stageOrder.map((name) => stageMap[name] || { name, status: "queued" });
  }, [status]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!session) return;
    setLogs([]);
    setReport("");
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

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="eyebrow">C3/C4 Classification Agent</div>
          <h1>Experiment Console</h1>
          <p className="muted">Submit a prompt to classify C3 vs C4 pathways using A/Ci curves.</p>
        </div>
        <div className="session">Session: {session?.id || "..."}</div>
      </header>

      <section className="panel">
        <form onSubmit={handleSubmit} className="prompt-form">
          <input
            className="prompt-input"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Classify C3 vs C4 using CO2S curves"
          />
          <button type="submit" className="btn">Send</button>
        </form>
        {status && <StatusPills stages={orderedStages} />}
      </section>

      <section className="grid">
        <div className="panel">
          <div className="panel-title">Logs</div>
          <LogPanel logs={logs} />
        </div>
        <div className="panel">
          <div className="panel-title">Report</div>
          {report ? (
            <div className="report markdown">
              <MarkdownViewer content={report} transformImageUri={transformImageUri} />
            </div>
          ) : (
            <div className="muted">Report will appear here once completed.</div>
          )}
        </div>
      </section>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <AppErrorBoundary>
    <App />
  </AppErrorBoundary>
);
