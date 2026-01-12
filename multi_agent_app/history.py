"Chat history helpers shared between routes and the orchestrator."

from __future__ import annotations

import json
import os
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List
from .settings import load_memory_settings
from .memory_manager import MemoryManager, get_memory_llm

_PRIMARY_CHAT_HISTORY_PATH = Path("chat_history.json")
_FALLBACK_CHAT_HISTORY_PATH = Path("var/chat_history.json")
_LEGACY_CHAT_HISTORY_PATHS = [Path("instance/chat_history.json")]

# Consolidation cadence: short-term is updated every turn; long-term is only
# consolidated after a few short-term refreshes to keep roles distinct.
_SHORT_TO_LONG_THRESHOLD = 3
_short_updates_since_last_long = 0
_short_update_lock = threading.Lock()


def _load_chat_history(prefer_fallback: bool = True) -> tuple[List[Dict[str, Any]], Path]:
    """Return chat history and the path it was loaded from with permission-aware fallbacks."""

    candidates = (
        [_FALLBACK_CHAT_HISTORY_PATH, _PRIMARY_CHAT_HISTORY_PATH]
        if prefer_fallback
        else [_PRIMARY_CHAT_HISTORY_PATH, _FALLBACK_CHAT_HISTORY_PATH]
    )
    for legacy_path in _LEGACY_CHAT_HISTORY_PATHS:
        if legacy_path not in candidates:
            candidates.append(legacy_path)
    last_error: Exception | None = None

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data, path
            logging.warning("Chat history at %s was not a list. Resetting.", path)
            return [], path
        except FileNotFoundError:
            continue
        except json.JSONDecodeError:
            logging.warning("Chat history JSON invalid at %s; resetting file.", path)
            return [], path
        except PermissionError as exc:
            last_error = exc
            logging.warning("Chat history not readable at %s: %s", path, exc)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logging.warning("Unexpected error reading chat history at %s: %s", path, exc)

    if last_error:
        logging.warning("Falling back to empty chat history due to previous errors: %s", last_error)

    fallback_path = candidates[0] if candidates else _PRIMARY_CHAT_HISTORY_PATH
    return [], fallback_path


def _write_chat_history(history: List[Dict[str, Any]], preferred_path: Path | None = None) -> Path:
    """Persist chat history to the first writable path, preferring the provided path."""

    candidate_paths: list[Path] = []
    if preferred_path:
        # Avoid noisy failures when the current file exists but is not writable.
        if not (preferred_path.exists() and not os.access(preferred_path, os.W_OK)):
            candidate_paths.append(preferred_path)
    candidate_paths.extend([_FALLBACK_CHAT_HISTORY_PATH, _PRIMARY_CHAT_HISTORY_PATH])

    seen: set[Path] = set()
    last_error: Exception | None = None

    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            # Best-effort mirror to other known locations for compatibility.
            mirror_targets = [_PRIMARY_CHAT_HISTORY_PATH, _FALLBACK_CHAT_HISTORY_PATH]
            for mirror in mirror_targets:
                if mirror == path:
                    continue
                try:
                    if mirror.exists() and not os.access(mirror, os.W_OK):
                        continue
                    mirror.parent.mkdir(parents=True, exist_ok=True)
                    with open(mirror, "w", encoding="utf-8") as mf:
                        json.dump(history, mf, ensure_ascii=False, indent=2)
                except Exception as exc:  # noqa: BLE001
                    logging.debug("Skipping mirror write to %s: %s", mirror, exc)

            return path
        except PermissionError as exc:
            last_error = exc
            logging.error("Failed to write chat history to %s: %s", path, exc)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logging.error("Unexpected error writing chat history to %s: %s", path, exc)

    if last_error:
        raise last_error
    raise RuntimeError("Unable to write chat history to any candidate path.")


def _read_chat_history(limit: int | None = None) -> List[Dict[str, Any]]:
    """Public helper to read chat history with fallbacks."""

    history, _ = _load_chat_history()
    if not isinstance(history, list):
        return []
    if limit is None:
        return history
    return history[-limit:]


def _reset_chat_history() -> None:
    """Reset chat history, preferring the writable fallback path."""

    _write_chat_history([], preferred_path=_FALLBACK_CHAT_HISTORY_PATH)


def _get_memory_llm():
    """Delegate to the shared memory LLM factory."""

    return get_memory_llm()


def _refresh_memory(memory_kind: str, recent_history: List[Dict[str, str]]) -> None:
    """Update short- or long-term memory by reconciling recent history with the current store."""

    global _short_updates_since_last_long  # noqa: PLW0603

    settings = load_memory_settings()
    if not settings.get("enabled", True):
        return

    llm = _get_memory_llm()
    if llm is None:
        return

    normalized_history: List[Dict[str, str]] = [
        {"role": entry.get("role"), "content": entry.get("content")}
        for entry in recent_history
        if isinstance(entry, dict)
        and isinstance(entry.get("role"), str)
        and isinstance(entry.get("content"), str)
        and str(entry.get("content")).strip()
    ]

    if not normalized_history:
        return

    if memory_kind == "short":
        memory_path = "short_term_memory.json"
    else:
        memory_path = "long_term_memory.json"

    manager = MemoryManager(memory_path)
    try:
        snapshot = manager.consolidate_memory(
            normalized_history,
            memory_kind="short" if memory_kind == "short" else "long",
            llm=llm,
        )
        if memory_kind == "short":
            with _short_update_lock:
                _short_updates_since_last_long += 1
        else:
            with _short_update_lock:
                _short_updates_since_last_long = 0
        return snapshot
    except Exception as exc:  # noqa: BLE001
        logging.warning("Memory consolidation (%s) failed: %s", memory_kind, exc)
        return None


def _consolidate_short_into_long(recent_history: List[Dict[str, str]]) -> None:
    """Persist short-term highlights into long-term memory, then reset short memory."""

    global _short_updates_since_last_long  # noqa: PLW0603

    llm = _get_memory_llm()
    if llm is None:
        return

    short_manager = MemoryManager("short_term_memory.json")
    short_snapshot = short_manager.load_memory()

    long_manager = MemoryManager("long_term_memory.json")
    try:
        long_manager.consolidate_memory(
            recent_history,
            memory_kind="long",
            llm=llm,
            short_snapshot=short_snapshot,
        )
        short_manager.reset_short_memory(preserve_active_task=True)
        with _short_update_lock:
            _short_updates_since_last_long = 0
    except Exception as exc:  # noqa: BLE001
        logging.warning("Short->Long consolidation failed: %s", exc)


def _append_to_chat_history(
    role: str, content: str, *, broadcast: bool = True, metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Append a message to the chat history file."""

    extras = metadata if isinstance(metadata, dict) else None
    history, source_path = _load_chat_history()

    next_id = len(history) + 1
    entry: Dict[str, Any] = {"id": next_id, "role": role, "content": content}
    if extras:
        for key, value in extras.items():
            if key in {"id", "role", "content"}:
                continue
            entry[key] = value
    history.append(entry)

    try:
        _write_chat_history(history, preferred_path=source_path)
    except Exception as exc:  # noqa: BLE001
        logging.error("Chat history write failed; message may not persist: %s", exc)
        raise

    total_entries = len(history)
    if total_entries == 0:
        return

    # Short-term memory: refresh every turn using the latest few lines as context
    threading.Thread(target=_refresh_memory, args=("short", history[-6:])).start()

    # Long-term memory: consolidate only after several short updates to avoid homogenization
    with _short_update_lock:
        should_consolidate_long = _short_updates_since_last_long >= _SHORT_TO_LONG_THRESHOLD
    if should_consolidate_long:
        threading.Thread(target=_consolidate_short_into_long, args=(history[-20:],)).start()
