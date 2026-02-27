# ──────────────────────────────────────────────
# Meeting-Specific Prompt Templates
# ──────────────────────────────────────────────

TITLE_PROMPT = """Read this transcript and generate a short, descriptive title (5-10 words max).
The title should capture the main topic. Return ONLY the title, nothing else.

Transcript:
{transcript}

Title:"""

SUMMARY_PROMPT = """You are a professional meeting analyst. Read the following meeting transcript and provide a clear, concise summary in 3-5 sentences.
Focus on the main purpose, key outcomes, and overall tone of the meeting.

Transcript:
{transcript}

Summary:"""


KEY_POINTS_PROMPT = """You are a professional meeting analyst. Read the following meeting transcript and extract the key discussion points.
List each point as a short bullet. Include only the most important topics that were discussed.

Transcript:
{transcript}

Key Discussion Points (one per line, start each with "- "):"""


ACTION_ITEMS_PROMPT = """You are a professional meeting analyst. Read the following meeting transcript and extract all action items.
For each action item, identify WHO is responsible, WHAT they need to do, and WHEN (deadline if mentioned).
If the responsible person or deadline is not clear, write "Unassigned" or "No deadline".

Transcript:
{transcript}

Action Items (format each as "- [Person] — Task — Deadline"):"""


DECISIONS_PROMPT = """You are a professional meeting analyst. Read the following meeting transcript and extract all decisions that were made during the meeting.
Only include clear decisions that the group agreed upon.

Transcript:
{transcript}

Decisions Made (one per line, start each with "- "):"""


FOLLOWUP_PROMPT = """You are a professional meeting analyst. Read the following meeting transcript and identify any follow-up items, next meeting plans, or unresolved topics that need further discussion.

Transcript:
{transcript}

Follow-Up Items (one per line, start each with "- "):"""


FULL_ANALYSIS_PROMPT = """You are a professional meeting analyst. Analyze the following meeting transcript and provide a structured report.

Transcript:
{transcript}

Provide your analysis in EXACTLY this format (keep the section headers as-is):

## SUMMARY
(Write 3-5 sentences summarizing the meeting)

## KEY DISCUSSION POINTS
- (point 1)
- (point 2)
- (add more as needed)

## ACTION ITEMS
- [Person] — Task — Deadline
- (add more as needed, use "Unassigned" and "No deadline" if unclear)

## DECISIONS MADE
- (decision 1)
- (add more as needed, or write "No explicit decisions recorded" if none)

## FOLLOW-UP
- (follow-up item 1)
- (add more as needed, or write "No follow-up items identified" if none)
"""
