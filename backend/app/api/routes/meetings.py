# ──────────────────────────────────────────────
# Meeting Routes — Audio/transcript upload, analysis, PDF reports
# Supports: audio files (→ whisper → analysis) or .txt transcripts (→ analysis)
# ──────────────────────────────────────────────
import os
import uuid
import json
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from app.db.database import get_db
from app.db.repositories import (
    create_meeting_record,
    get_meeting_by_id,
    list_meetings,
    delete_meeting_record,
)
from app.api.dependencies import require_permission
from app.meeting.transcript_processor import clean_transcript, estimate_duration
from app.meeting.meeting_analyzer import analyze_transcript_full, analyze_transcript_detailed
from app.meeting.report_generator import generate_meeting_report
from app.models.user import User
from app.config import UPLOAD_DIR, ALLOWED_AUDIO_EXTENSIONS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/meetings", tags=["Meetings"])

TRANSCRIPTS_DIR = UPLOAD_DIR / "transcripts"
AUDIO_DIR = UPLOAD_DIR / "audio"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ─── Schemas ─────────────────────────────────

class ActionItemSchema(BaseModel):
    person: str
    task: str
    deadline: str


class MeetingAnalysisResponse(BaseModel):
    meeting_id: str
    title: str
    summary: str
    key_points: List[str]
    action_items: List[ActionItemSchema]
    decisions: List[str]
    follow_up: List[str]
    estimated_duration: Optional[str]
    report_path: Optional[str]
    source_type: str = "transcript"      # "transcript" or "audio"
    transcription_metadata: Optional[dict] = None


class MeetingListItem(BaseModel):
    id: str
    title: str
    transcript_filename: str
    summary: str
    estimated_duration: Optional[str]
    created_at: Optional[str]


def _get_file_extension(filename: str) -> str:
    """Get lowercase file extension."""
    return Path(filename).suffix.lower()


def _is_audio_file(filename: str) -> bool:
    """Check if a file is a supported audio format."""
    return _get_file_extension(filename) in ALLOWED_AUDIO_EXTENSIONS


def _is_transcript_file(filename: str) -> bool:
    """Check if a file is a text transcript."""
    return _get_file_extension(filename) == ".txt"


# ─── Endpoints ───────────────────────────────

@router.post("/analyze", response_model=MeetingAnalysisResponse)
async def analyze_meeting(
    file: UploadFile = File(...),
    title: str = Form("Untitled Meeting"),
    mode: str = Form("fast"),
    current_user: User = Depends(require_permission("upload_documents")),
    db: Session = Depends(get_db),
):
    """
    Upload an audio file or transcript and get full meeting analysis + PDF report.

    - **file**: Audio file (.mp3, .wav, .m4a, .ogg, .flac, .webm) OR text file (.txt)
    - **title**: Meeting title for the report
    - **mode**: "fast" (single LLM call) or "detailed" (multiple LLM passes)

    Audio files are transcribed with faster-whisper (offline), then analyzed.
    Text files are analyzed directly.
    """
    filename = file.filename or "unknown"
    meeting_id = str(uuid.uuid4())

    source_type = "transcript"
    transcription_metadata = None

    # ─── Route based on file type ────────────
    if _is_audio_file(filename):
        # AUDIO PATH: save → transcribe → analyze
        source_type = "audio"
        transcript_text, transcription_metadata = await _handle_audio_upload(
            file, meeting_id, filename
        )

    elif _is_transcript_file(filename):
        # TRANSCRIPT PATH: read → clean → analyze
        transcript_text = await _handle_transcript_upload(file)

    else:
        ext = _get_file_extension(filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type: '{ext}'. "
                f"Supported: .txt (transcript), .mp3, .wav, .m4a, .ogg, .flac, .webm (audio)"
            ),
        )

    # Validate transcript length
    if len(transcript_text.strip()) < 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcript is too short. Minimum 50 characters after cleaning.",
        )

    # Save transcript to disk
    transcript_path = str(TRANSCRIPTS_DIR / f"{meeting_id}.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    # Estimate duration from text (audio metadata may also have real duration)
    duration = estimate_duration(transcript_text)
    if transcription_metadata and transcription_metadata.get("audio_duration_sec"):
        audio_secs = transcription_metadata["audio_duration_sec"]
        if audio_secs >= 60:
            mins = int(audio_secs // 60)
            secs = int(audio_secs % 60)
            duration = f"{mins}m {secs}s"
        else:
            duration = f"{int(audio_secs)}s"

    # Analyze with LLM (mistral)
    try:
        if mode == "detailed":
            analysis = await analyze_transcript_detailed(transcript_text)
        else:
            analysis = await analyze_transcript_full(transcript_text)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )

    # Generate PDF report
    try:
        report_path = generate_meeting_report(
            analysis=analysis,
            meeting_id=meeting_id,
            title=title,
            duration=duration,
            filename=filename,
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        report_path = None

    # Save to database
    create_meeting_record(
        db=db,
        meeting_id=meeting_id,
        title=title,
        transcript_filename=filename,
        transcript_path=transcript_path,
        report_path=report_path or "",
        summary=analysis.get("summary", ""),
        analysis_json=json.dumps(analysis, default=str),
        estimated_duration=duration or "",
        uploaded_by=current_user.id,
    )

    # Build response
    action_items = []
    for item in analysis.get("action_items", []):
        if isinstance(item, dict):
            action_items.append(ActionItemSchema(
                person=item.get("person", "Unassigned"),
                task=item.get("task", ""),
                deadline=item.get("deadline", "No deadline"),
            ))

    return MeetingAnalysisResponse(
        meeting_id=meeting_id,
        title=title,
        summary=analysis.get("summary", ""),
        key_points=analysis.get("key_points", []),
        action_items=action_items,
        decisions=analysis.get("decisions", []),
        follow_up=analysis.get("follow_up", []),
        estimated_duration=duration,
        report_path=report_path,
        source_type=source_type,
        transcription_metadata=transcription_metadata,
    )


async def _handle_audio_upload(
    file: UploadFile, meeting_id: str, filename: str
) -> tuple:
    """
    Save audio file, transcribe with Whisper.
    Returns (transcript_text, metadata_dict).
    """
    from app.meeting.transcriber import transcribe_audio

    ext = _get_file_extension(filename)
    audio_path = str(AUDIO_DIR / f"{meeting_id}{ext}")

    # Save audio to disk
    contents = await file.read()
    with open(audio_path, "wb") as f:
        f.write(contents)

    logger.info(f"Audio saved: {audio_path} ({len(contents)} bytes)")

    # Transcribe
    try:
        transcript_text, metadata = await transcribe_audio(audio_path)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        # Clean up audio file on failure
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio transcription failed: {str(e)}",
        )

    transcript_text = clean_transcript(transcript_text)
    return transcript_text, metadata


async def _handle_transcript_upload(file: UploadFile) -> str:
    """Read and clean a transcript text file."""
    contents = await file.read()
    try:
        transcript_text = contents.decode("utf-8")
    except UnicodeDecodeError:
        transcript_text = contents.decode("latin-1")

    return clean_transcript(transcript_text)


@router.get("/", response_model=List[MeetingListItem])
def get_meetings(
    current_user: User = Depends(require_permission("list_documents")),
    db: Session = Depends(get_db),
):
    """List all meeting analyses."""
    meetings = list_meetings(db)
    return [
        MeetingListItem(
            id=m.id,
            title=m.title,
            transcript_filename=m.transcript_filename,
            summary=m.summary or "",
            estimated_duration=m.estimated_duration,
            created_at=str(m.created_at) if m.created_at else None,
        )
        for m in meetings
    ]


@router.get("/{meeting_id}")
def get_meeting(
    meeting_id: str,
    current_user: User = Depends(require_permission("list_documents")),
    db: Session = Depends(get_db),
):
    """Get full meeting analysis details."""
    meeting = get_meeting_by_id(db, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found.")

    analysis = {}
    if meeting.analysis_json:
        try:
            analysis = json.loads(meeting.analysis_json)
        except json.JSONDecodeError:
            pass

    return {
        "id": meeting.id,
        "title": meeting.title,
        "transcript_filename": meeting.transcript_filename,
        "summary": meeting.summary,
        "analysis": analysis,
        "estimated_duration": meeting.estimated_duration,
        "has_report": bool(meeting.report_path and os.path.exists(meeting.report_path)),
        "created_at": str(meeting.created_at) if meeting.created_at else None,
    }


@router.get("/{meeting_id}/download")
def download_report(
    meeting_id: str,
    current_user: User = Depends(require_permission("list_documents")),
    db: Session = Depends(get_db),
):
    """Download the PDF meeting report."""
    meeting = get_meeting_by_id(db, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found.")

    if not meeting.report_path or not os.path.exists(meeting.report_path):
        raise HTTPException(status_code=404, detail="PDF report not found.")

    return FileResponse(
        path=meeting.report_path,
        media_type="application/pdf",
        filename=f"meeting_report_{meeting.title}.pdf",
    )


@router.delete("/{meeting_id}")
def delete_meeting(
    meeting_id: str,
    current_user: User = Depends(require_permission("delete_documents")),
    db: Session = Depends(get_db),
):
    """Delete a meeting and its associated files."""
    meeting = get_meeting_by_id(db, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found.")

    # Remove files
    if meeting.transcript_path and os.path.exists(meeting.transcript_path):
        os.remove(meeting.transcript_path)
    if meeting.report_path and os.path.exists(meeting.report_path):
        os.remove(meeting.report_path)

    # Remove from database
    delete_meeting_record(db, meeting_id)

    return {"message": "Meeting deleted successfully.", "meeting_id": meeting_id}
