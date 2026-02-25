# ──────────────────────────────────────────────
# Meeting Analyzer — LLM-powered transcript analysis
# Uses mistral model with optimized settings from notebook
# ──────────────────────────────────────────────
import re
import json
from typing import Dict, List
from app.llm.ollama_client import get_ollama_client
from app.config import (
    OLLAMA_MEETING_MODEL,
    MEETING_LLM_NUM_PREDICT,
    MEETING_LLM_TEMPERATURE,
    MEETING_LLM_TOP_K,
    MEETING_TRANSCRIPT_MAX_CHARS,
)
from app.meeting.prompts import (
    SUMMARY_PROMPT,
    KEY_POINTS_PROMPT,
    ACTION_ITEMS_PROMPT,
    DECISIONS_PROMPT,
    FOLLOWUP_PROMPT,
    FULL_ANALYSIS_PROMPT,
)
from app.meeting.transcript_processor import chunk_transcript


def _parse_bullet_list(text: str) -> List[str]:
    """Parse a bullet-point list from LLM output."""
    items = []
    for line in text.strip().splitlines():
        line = line.strip()
        # Match lines starting with -, *, •, or numbered
        cleaned = re.sub(r"^[-*•]\s*", "", line)
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
        if cleaned:
            items.append(cleaned)
    return items


def _parse_action_items(text: str) -> List[Dict]:
    """
    Parse action items from LLM output.
    Expected format: [Person] — Task — Deadline
    """
    items = []
    for line in text.strip().splitlines():
        line = line.strip()
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if not line:
            continue

        # Try to parse structured format: [Person] — Task — Deadline
        # Also handle alternate separators: -, :, |
        parts = re.split(r"\s*[—–\|]\s*", line)

        if len(parts) >= 3:
            person = parts[0].strip("[] ")
            task = parts[1].strip()
            deadline = parts[2].strip()
        elif len(parts) == 2:
            person = parts[0].strip("[] ")
            task = parts[1].strip()
            deadline = "No deadline"
        else:
            person = "Unassigned"
            task = line
            deadline = "No deadline"

        items.append({
            "person": person,
            "task": task,
            "deadline": deadline,
        })

    return items


def _try_parse_json(text: str) -> Dict | None:
    """
    Robust JSON extraction from LLM output.
    Ported from teammate's parse_json() function.
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return None


def _parse_full_analysis(text: str) -> Dict:
    """
    Parse the structured full analysis response from LLM.
    Tries JSON first (mistral often returns JSON), falls back to ## section parsing.
    """
    result = {
        "summary": "",
        "key_points": [],
        "action_items": [],
        "decisions": [],
        "follow_up": [],
    }

    # Try JSON parsing first (teammate's approach for mistral)
    json_result = _try_parse_json(text)
    if json_result:
        result["summary"] = json_result.get("summary", "")
        result["key_points"] = json_result.get("key_points", json_result.get("key_discussion_points", []))
        result["decisions"] = json_result.get("decisions", [])
        result["follow_up"] = json_result.get("follow_up", json_result.get("follow_ups", []))

        # Normalize action items / tasks
        raw_tasks = json_result.get("action_items", json_result.get("tasks", []))
        for item in raw_tasks:
            if isinstance(item, dict):
                result["action_items"].append({
                    "person": item.get("person", item.get("assigned_to", "Unassigned")),
                    "task": item.get("task", ""),
                    "deadline": item.get("deadline", "No deadline") or "No deadline",
                })
            elif isinstance(item, str):
                result["action_items"].append({
                    "person": "Unassigned", "task": item, "deadline": "No deadline",
                })
        return result

    # Fallback: parse ## section headers
    sections = re.split(r"##\s*", text)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.split("\n", 1)
        header = lines[0].strip().upper()
        body = lines[1].strip() if len(lines) > 1 else ""

        if "SUMMARY" in header:
            result["summary"] = body.strip()
        elif "KEY" in header or "DISCUSSION" in header:
            result["key_points"] = _parse_bullet_list(body)
        elif "ACTION" in header:
            result["action_items"] = _parse_action_items(body)
        elif "DECISION" in header:
            result["decisions"] = _parse_bullet_list(body)
        elif "FOLLOW" in header:
            result["follow_up"] = _parse_bullet_list(body)

    return result


async def analyze_transcript_full(transcript: str) -> Dict:
    """
    Full meeting analysis in a single LLM call using mistral.
    Uses teammate's efficiency settings: capped tokens, low temp, restricted top_k.

    Args:
        transcript: The meeting transcript text.

    Returns:
        Dict with summary, key_points, action_items, decisions, follow_up.
    """
    llm = get_ollama_client()

    # Smart truncation: use configurable limit, split at sentence boundaries
    chunks = chunk_transcript(transcript, max_chars=MEETING_TRANSCRIPT_MAX_CHARS)
    text_to_analyze = chunks[0]

    if len(chunks) > 1:
        text_to_analyze += f"\n\n[Note: This is part 1 of {len(chunks)} parts of the transcript]"

    prompt = FULL_ANALYSIS_PROMPT.format(transcript=text_to_analyze)
    response = await llm.generate(
        prompt,
        temperature=MEETING_LLM_TEMPERATURE,
        max_tokens=MEETING_LLM_NUM_PREDICT,
        model=OLLAMA_MEETING_MODEL,
        top_k=MEETING_LLM_TOP_K,
    )

    result = _parse_full_analysis(response)

    # If summary is empty, the parsing might have failed — use raw response
    if not result["summary"] and response:
        result["summary"] = response[:500]

    return result


async def analyze_transcript_detailed(transcript: str) -> Dict:
    """
    Detailed meeting analysis using separate LLM calls per section.
    More accurate but slower. Uses mistral model.

    Args:
        transcript: The meeting transcript text.

    Returns:
        Dict with summary, key_points, action_items, decisions, follow_up.
    """
    llm = get_ollama_client()

    chunks = chunk_transcript(transcript, max_chars=MEETING_TRANSCRIPT_MAX_CHARS)
    text = chunks[0]

    common_kwargs = {
        "temperature": MEETING_LLM_TEMPERATURE,
        "max_tokens": MEETING_LLM_NUM_PREDICT,
        "model": OLLAMA_MEETING_MODEL,
        "top_k": MEETING_LLM_TOP_K,
    }

    # Run each analysis pass with mistral
    summary_raw = await llm.generate(
        SUMMARY_PROMPT.format(transcript=text), **common_kwargs
    )
    key_points_raw = await llm.generate(
        KEY_POINTS_PROMPT.format(transcript=text), **common_kwargs
    )
    actions_raw = await llm.generate(
        ACTION_ITEMS_PROMPT.format(transcript=text), **common_kwargs
    )
    decisions_raw = await llm.generate(
        DECISIONS_PROMPT.format(transcript=text), **common_kwargs
    )
    followup_raw = await llm.generate(
        FOLLOWUP_PROMPT.format(transcript=text), **common_kwargs
    )

    return {
        "summary": summary_raw.strip(),
        "key_points": _parse_bullet_list(key_points_raw),
        "action_items": _parse_action_items(actions_raw),
        "decisions": _parse_bullet_list(decisions_raw),
        "follow_up": _parse_bullet_list(followup_raw),
    }
