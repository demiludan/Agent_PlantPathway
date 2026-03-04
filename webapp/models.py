from __future__ import annotations

import enum
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class StageName(str, enum.Enum):
    """Named pipeline milestones so the UI can visualize progress."""
    queue = "queue"
    preprocess = "preprocess"
    train = "train"
    inference = "inference"
    evaluate = "evaluate"
    report = "report"


class StageStatus(str, enum.Enum):
    """Lifecycle of a stage as the CLI progresses."""
    queued = "queued"
    running = "running"
    completed = "completed"
    error = "error"
    skipped = "skipped"


class Stage(BaseModel):
    """Single stage record with timing metadata."""
    name: StageName
    status: StageStatus = StageStatus.queued
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class RunStatus(str, enum.Enum):
    """Overall run lifecycle separate from per-stage status."""
    queued = "queued"
    running = "running"
    completed = "completed"
    error = "error"


class Run(BaseModel):
    """Full run state mirrored to the frontend for polling."""
    id: str
    session_id: str
    prompt: str
    status: RunStatus
    stages: List[Stage]
    created_at: datetime
    updated_at: datetime
    run_dir: Optional[str] = None
    report_path: Optional[str] = None
    error: Optional[str] = None


class Session(BaseModel):
    """Chat-like thread that groups runs triggered by a single user conversation."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class CreateMessageRequest(BaseModel):
    prompt: str = Field(..., description="Natural-language request such as 'Classify photosynthesis pathway'")


class RunSummary(BaseModel):
    id: str
    status: RunStatus
    run_dir: Optional[str]
    report_path: Optional[str]
    error: Optional[str]


class LogEvent(BaseModel):
    run_id: str
    timestamp: datetime
    message: str
    stage: Optional[StageName] = None
    level: str = "info"
