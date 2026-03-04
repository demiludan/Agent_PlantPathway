from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    CreateMessageRequest,
    CreateSessionRequest,
    LogEvent,
    Run,
    RunSummary,
    Session,
)
from .runner import RunnerService
from .state import InMemoryStore

"""HTTP surface area for orchestrating the CLI pipeline via FastAPI."""

app = FastAPI(title="Plant Pathway Classification Web API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = InMemoryStore()
runner = RunnerService(store=store)
app.mount(
    "/files",
    StaticFiles(directory=str(Path(__file__).resolve().parent.parent)),
    name="files",
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/sessions", response_model=Session)
async def create_session(payload: CreateSessionRequest) -> Session:
    return await store.create_session(payload)


@app.get("/sessions", response_model=list[Session])
async def list_sessions() -> list[Session]:
    return await store.list_sessions()


@app.post("/sessions/{session_id}/messages", response_model=dict)
async def create_message(session_id: str, payload: CreateMessageRequest) -> dict[str, str]:
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    run_id = await runner.start_run(session_id=session_id, payload=payload)
    return {"run_id": run_id}


@app.get("/sessions/{session_id}/runs", response_model=list[Run])
async def list_runs(session_id: str) -> list[Run]:
    return await store.list_runs_for_session(session_id)


@app.get("/runs/{run_id}", response_model=Run)
async def get_run(run_id: str) -> Run:
    run = await store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/runs/{run_id}/summary", response_model=RunSummary)
async def get_run_summary(run_id: str) -> RunSummary:
    run = await get_run(run_id)
    return RunSummary(
        id=run.id, status=run.status, run_dir=run.run_dir,
        report_path=run.report_path, error=run.error,
    )


@app.get("/runs/{run_id}/logs", response_model=list[LogEvent])
async def get_logs(
    run_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
) -> list[LogEvent]:
    return await store.get_logs(run_id, offset=offset, limit=limit)


@app.get("/runs/{run_id}/stream")
async def stream_logs(run_id: str) -> StreamingResponse:
    run = await store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    queue = await store.subscribe_logs(run_id)

    async def event_generator():
        try:
            while True:
                event = await queue.get()
                yield f"data: {event.json()}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            await store.unsubscribe_logs(run_id, queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/runs/{run_id}/report")
async def get_report(run_id: str) -> PlainTextResponse:
    run = await store.get_run(run_id)
    if not run or not run.report_path:
        raise HTTPException(status_code=404, detail="Report not available")
    path = Path(run.report_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report file missing")
    return PlainTextResponse(path.read_text())


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "message": "Plant Pathway Classification Web API",
        "endpoints": [
            "/sessions",
            "/sessions/{session_id}/messages",
            "/runs/{run_id}",
            "/runs/{run_id}/logs",
            "/runs/{run_id}/stream",
            "/runs/{run_id}/report",
        ],
    }
