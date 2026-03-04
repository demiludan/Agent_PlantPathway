const BASE = "/api";

export async function createSession(title) {
  const res = await fetch(`${BASE}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  return res.json();
}

export async function sendMessage(sessionId, prompt) {
  const res = await fetch(`${BASE}/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  return res.json();
}

export async function getRun(runId) {
  const res = await fetch(`${BASE}/runs/${runId}`);
  return res.json();
}

export async function getReport(runId) {
  const res = await fetch(`${BASE}/runs/${runId}/report`);
  return res.text();
}

export function streamLogs(runId, onEvent) {
  const source = new EventSource(`${BASE}/runs/${runId}/stream`);
  source.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data));
    } catch (err) {
      console.error("Parse error", err);
    }
  };
  return source;
}
