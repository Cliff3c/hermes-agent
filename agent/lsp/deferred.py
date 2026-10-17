"""LSP diagnostics for a write, delivered after the write's tool result instead of inside it.

``write_file`` / ``patch`` return as soon as the bytes are on disk and verified; the language server
check runs on the LSP loop (``agent.lsp.edit_check``).  Its verdict is queued here under the session
that made the edit (bound by ``tools.file_tools`` around the write) and handed to that session's agent
at its next tool result, or its next user turn when no tool runs again
(``agent.plugin_live_notes``: a new row not yet flushed, so no cached byte moves).

A later edit of the same file takes over an undelivered verdict: the newer check diffs against the
older baseline, so an error the first edit introduced is still reported unless the second fixed it.
"""
from __future__ import annotations

import contextlib
import threading
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional

_OWNER: ContextVar[Optional[str]] = ContextVar("hermes_lsp_deferred_owner", default=None)
_MAX_OWNERS = 64  # sessions that never drain (a one-shot caller) must not grow this forever
_MAX_PER_OWNER = 32
_LOCK = threading.Lock()


@dataclass(frozen=True)
class _Note:
    text: str
    baseline_text: Optional[str]  # pre-edit content the verdict was diffed against


# owner -> {abs path -> note}, oldest owner first
_QUEUED: OrderedDict[str, OrderedDict[str, _Note]] = OrderedDict()


@contextlib.contextmanager
def bind_owner(owner: Optional[str]) -> Iterator[None]:
    """Attribute writes in this block to ``owner`` (the session id, else the task id)."""
    token = _OWNER.set(owner or None)
    try:
        yield
    finally:
        _OWNER.reset(token)


def current_owner() -> Optional[str]:
    return _OWNER.get()


def post(owner: str, path: str, text: str, baseline_text: Optional[str]) -> None:
    """Queue ``text`` (a formatted diagnostics block) for ``owner``.  A note still queued for ``path``
    (an older check finished while a newer one was starting) is kept in front, with its baseline."""
    with _LOCK:
        notes = _QUEUED.pop(owner, None) or OrderedDict()
        older = notes.pop(path, None)
        if older is not None and older.text != text:
            text, baseline_text = f"{older.text}\n\n{text}", older.baseline_text
        notes[path] = _Note(text, baseline_text)
        while len(notes) > _MAX_PER_OWNER:
            notes.popitem(last=False)
        _QUEUED[owner] = notes
        while len(_QUEUED) > _MAX_OWNERS:
            _QUEUED.popitem(last=False)


def take(owner: Optional[str], path: str) -> Optional[_Note]:
    """Withdraw ``owner``'s undelivered note for ``path``: a newer edit of the file supersedes it and
    diffs against its ``baseline_text``.  Hand it back to :func:`restore` if that newer check yields
    no verdict, so the older errors are not lost."""
    if not owner:
        return None
    with _LOCK:
        notes = _QUEUED.get(owner)
        return notes.pop(path, None) if notes else None


def restore(owner: Optional[str], path: str, note: Optional[_Note]) -> None:
    if owner and note is not None:
        post(owner, path, note.text, note.baseline_text)


def pop_notes(*owners: Optional[str]) -> list[tuple[str, str]]:
    """Every queued ``(path, text)`` for ``owners``, oldest first, and clear them."""
    out: list[tuple[str, str]] = []
    with _LOCK:
        for owner in dict.fromkeys(o for o in owners if o):
            notes = _QUEUED.pop(owner, None)
            if notes:
                out.extend((path, note.text) for path, note in notes.items())
    return out


def forward_to_agent(agent: Any) -> None:
    """Move ``agent``'s finished verdicts onto its live-notes queue, which rides the next tool result
    (or the next user turn).  Called right before that tool result is built."""
    owners = (getattr(agent, "session_id", None), getattr(agent, "_current_task_id", None))
    notes = pop_notes(*owners)
    if not notes:
        return
    from agent.plugin_live_notes import queue_plugin_notice
    for path, text in notes:
        queue_plugin_notice(agent, f"lsp:{path}", text)


__all__ = ["bind_owner", "current_owner", "forward_to_agent", "pop_notes", "post", "restore", "take"]
