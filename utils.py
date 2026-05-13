"""
utils.py — Shared utilities for the SEC Risk Intelligence System.
"""

import re
import json
from pathlib import Path


# ── .env loading ──────────────────────────────────────────────────────────────

def load_env() -> None:
    """
    Load .env from the project root using auto-search, then fall back to the
    directory containing this file. Works regardless of where the user invokes
    the script from (fixes the '../.env' relative-path assumption).
    """
    from dotenv import load_dotenv
    # Auto-search walks up from cwd; if that misses, try the project dir
    loaded = load_dotenv()
    if not loaded:
        load_dotenv(Path(__file__).parent / ".env", override=True)


# ── LLM JSON parsing ──────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_llm_json(raw: str) -> object:
    """
    Robustly parse a JSON string that may be wrapped in markdown fences.

    Handles:
      - ```json ... ```
      - ``` ... ```
      - Leading/trailing whitespace
      - Fences with a space before the backticks (e.g. ' ```json')
    """
    cleaned = _FENCE_RE.sub("", raw).strip()
    return json.loads(cleaned)


def repair_truncated_json_array(raw: str) -> list:
    """
    Attempt to parse a JSON array that may have been truncated mid-stream.

    Strategy:
      1. Try normal parse after fence stripping.
      2. If that fails and the string contains at least one complete object,
         truncate to the last complete '}' and close the array.
      3. Return empty list if all repair attempts fail.
    """
    cleaned = _FENCE_RE.sub("", raw).strip()

    # Fast path: valid JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Repair: find last well-formed object boundary
    last_brace = cleaned.rfind("}")
    if last_brace == -1:
        return []

    truncated = cleaned[: last_brace + 1].rstrip().rstrip(",") + "\n]"

    # Make sure we have an opening bracket
    if not truncated.lstrip().startswith("["):
        truncated = "[" + truncated

    try:
        return json.loads(truncated)
    except json.JSONDecodeError:
        return []