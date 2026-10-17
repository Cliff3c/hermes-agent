"""Post-write LSP check that never holds the write.

``write_file`` / ``patch`` used to block on two language-server round trips per edit: a pre-write
baseline (open + wait) and the post-write verdict (open + save + wait), each up to ``lsp.wait_timeout``.
A push-only server that stays silent on a clean edit (typescript-language-server publishes nothing
when a file's diagnostics stay empty) ran both waits to the deadline: 10s per clean TypeScript edit.

Here the check is one coroutine on the service loop: sync the written file and wait for its verdict;
only when that verdict has diagnostics, get the verdict on the pre-edit text the write already holds
(or reuse the one an earlier check left for exactly that text) and return the diagnostics the edit
introduced.
The caller waits for it only within its latency budget (:func:`inline_wait`); a slower verdict is delivered later through
:mod:`agent.lsp.deferred`.  A newer check of the same file runs after an unfinished one and takes it
over (its baseline, and its result when the newer check gets no verdict), so a superseded edit's
errors are still caught unless the newer edit fixed them.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from agent.lsp import eventlog
from agent.lsp.client import _diagnostic_key
from agent.lsp.servers import language_id_for

if TYPE_CHECKING:
    from agent.lsp.manager import LSPService

logger = logging.getLogger("agent.lsp.edit_check")

# A write waits for an in-flight verdict only while it stays inside WRITE_BUDGET (from write entry),
# and never longer than INLINE_GRACE: a warm pyright usually answers a small edit inside it.
WRITE_BUDGET = 0.08
INLINE_GRACE = 0.05
_VERDICT_CAP = 256  # per-file last verdicts kept for baseline reuse
_SPAWN_CEILING = 60.0  # outer bound on one check beyond its two waits (cold spawn + initialize)

_Diags = list[dict[str, Any]]


@dataclass
class CheckResult:
    diagnostics: _Diags  # introduced by the edit (baseline removed); [] = clean
    baseline_text: Optional[str]  # the pre-edit text this result was diffed against
    superseded: bool = False  # a newer check of the file took this one over and will report it


@dataclass
class _Inflight:
    task: asyncio.Task
    baseline_text: Optional[str]
    superseded: bool = False


class EditChecks:
    """Per-service state, touched only on the loop thread: the last verdict per file (baseline reuse)
    and the newest unfinished check per file."""

    def __init__(self) -> None:
        self.verdicts: OrderedDict[str, tuple[str, _Diags]] = OrderedDict()
        self.inflight: dict[str, _Inflight] = {}

    def remember(self, abs_path: str, text: str, diags: _Diags) -> None:
        self.verdicts.pop(abs_path, None)
        self.verdicts[abs_path] = (_digest(text), list(diags))
        while len(self.verdicts) > _VERDICT_CAP:
            self.verdicts.popitem(last=False)

    def recall(self, abs_path: str, text: str) -> Optional[_Diags]:
        hit = self.verdicts.get(abs_path)
        return list(hit[1]) if hit is not None and hit[0] == _digest(text) else None


def inline_wait(started: Optional[float]) -> float:
    """Seconds the write may still wait for its verdict inline."""
    if started is None:
        return INLINE_GRACE
    return max(0.0, min(INLINE_GRACE, started + WRITE_BUDGET - time.monotonic()))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()


def start(svc: LSPService, path: str, *, pre_text: Optional[str],
          inherited: tuple[bool, Optional[str]] = (False, None)) -> Optional[concurrent.futures.Future]:
    """Schedule the check of ``path`` (already written) on the service loop; the Future resolves to a
    :class:`CheckResult` (or None when no server answered).  ``pre_text`` None = the file is new.
    ``inherited`` = ``(True, text)`` when an undelivered verdict for this file is being superseded."""
    from agent.async_utils import safe_schedule_threadsafe
    loop = svc._loop._loop
    if loop is None:
        return None
    return safe_schedule_threadsafe(_serialized(svc, os.path.abspath(path), pre_text, inherited), loop)


async def _serialized(svc: LSPService, abs_path: str, pre_text: Optional[str],
                      inherited: tuple[bool, Optional[str]]) -> Optional[CheckResult]:
    """Run after any unfinished check of the same file (two checks syncing one document at once would
    satisfy each other's waits), taking it over: its baseline becomes ours, and its result is ours to
    report when we get no verdict.  Never cancelled: cancelling mid-spawn would restart a cold server
    on every rapid edit."""
    checks: EditChecks = svc._edit_checks
    baseline_text = inherited[1] if inherited[0] else pre_text
    prev = checks.inflight.get(abs_path)
    if prev is not None and not prev.task.done():
        prev.superseded = True
        baseline_text = prev.baseline_text
    else:
        prev = None
    current = asyncio.current_task()
    assert current is not None
    me = checks.inflight[abs_path] = _Inflight(current, baseline_text)
    try:
        prev_result = None
        if prev is not None:
            await asyncio.wait({prev.task})
            if not prev.task.cancelled() and prev.task.exception() is None:
                prev_result = prev.task.result()
            if me.superseded:
                # A newer edit queued behind us while we waited: it checks the newest bytes against
                # our baseline, so our own round trip would only delay it.  Hand it what we hold.
                return CheckResult(prev_result.diagnostics, prev_result.baseline_text, True) if prev_result else None
        budget = svc._wait_budget(abs_path)
        try:
            diags = await asyncio.wait_for(_check(svc, abs_path, baseline_text, budget),
                                           timeout=2 * budget + _SPAWN_CEILING)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # timeout / dead server: same treatment as the old blocking path
            logger.debug("LSP edit check failed for %s", abs_path, exc_info=True)
            await _mark_broken(svc, abs_path, e)
            diags = None
        if diags is not None:
            result: Optional[CheckResult] = CheckResult(diags, baseline_text)
        elif prev_result is not None:
            result = CheckResult(prev_result.diagnostics, prev_result.baseline_text)
        else:
            result = None
        if result is not None and me.superseded:
            result.superseded = True
        return result
    finally:
        if checks.inflight.get(abs_path) is me:
            del checks.inflight[abs_path]


async def _check(svc: LSPService, abs_path: str, baseline_text: Optional[str], budget: float) -> Optional[_Diags]:
    """Verdict on the written file first; only when it has diagnostics, the verdict on ``baseline_text``
    (cached from an earlier check when that text was checked before) to drop the pre-existing ones.
    A clean edit costs one round trip, and push-only servers answer the baseline fast in the common
    case (errors -> fewer errors is a change they publish)."""
    client = await svc._get_or_spawn(abs_path)
    if client is None:
        return None
    srv = svc._server_for(abs_path)
    server_id = srv.server_id if srv is not None else "lsp"
    language_id = language_id_for(abs_path, srv)
    checks: EditChecks = svc._edit_checks
    if client.seeds_first_push and client.document_text(abs_path) is None:
        # First sync of this document on a seeding server: its first push is a verdict on the didOpen
        # text that never counts as fresh.  Open the pre-edit text so that push becomes the baseline;
        # a new file is opened as written, so the seed is the verdict itself.
        seed = await client.wait_for_seed(abs_path, budget) if await client.open_file(
            abs_path, language_id=language_id, text=baseline_text) == 0 else None
        if baseline_text is None:
            svc._touch(client)
            return _report(server_id, abs_path, seed)
        if seed is not None:
            checks.remember(abs_path, baseline_text, seed)
    version = await client.open_file(abs_path, language_id=language_id)
    await client.save_file(abs_path)
    fresh = await client.wait_for_diagnostics(abs_path, version, mode=svc._wait_mode, timeout=budget)
    svc._touch(client)
    if not fresh:
        # Alive but no verdict in budget: "no data", never stale stores (ghost diagnostics).
        return _report(server_id, abs_path, None)
    diags = list(client.diagnostics_for(abs_path, fresh_only=True))
    synced = client.document_text(abs_path)
    if synced is not None:
        checks.remember(abs_path, synced, diags)
    if diags and baseline_text is not None and synced is not None:
        baseline = checks.recall(abs_path, baseline_text)
        if baseline is None:
            baseline = []
            base_version = await client.open_file(abs_path, language_id=language_id, text=baseline_text)
            # No verdict on the pre-edit text -> empty baseline: the filter removes less, never more.
            if await client.wait_for_diagnostics(abs_path, base_version, mode=svc._wait_mode, timeout=budget):
                baseline = list(client.diagnostics_for(abs_path, fresh_only=True))
                checks.remember(abs_path, baseline_text, baseline)
            # Leave the server holding the bytes on disk again (its verdict on them is already in hand).
            await client.open_file(abs_path, language_id=language_id)
        from agent.lsp.range_shift import build_line_shift, shift_baseline
        seen = {_diagnostic_key(d) for d in shift_baseline(baseline, build_line_shift(baseline_text, synced))}
        diags = [d for d in diags if _diagnostic_key(d) not in seen]
    return _report(server_id, abs_path, diags)


def _report(server_id: str, abs_path: str, diags: Optional[_Diags]) -> Optional[_Diags]:
    if diags is None:
        eventlog.log_timeout(server_id, abs_path, kind="fresh diagnostics")
    elif diags:
        eventlog.log_diagnostics(server_id, abs_path, len(diags))
    else:
        eventlog.log_clean(server_id, abs_path)
    return diags


async def _mark_broken(svc: LSPService, abs_path: str, exc: BaseException) -> None:
    """Loop-side twin of ``LSPService._mark_broken_for_file`` (that one blocks on the loop)."""
    srv = svc._server_for(abs_path)
    key = svc._broken_key(srv, abs_path) if srv is not None else None
    if srv is None or key is None:
        return
    already = svc._is_broken(key)
    svc._mark_broken(key)
    with svc._state_lock:
        ckey = svc._live_key(srv, key[1])
        client = svc._clients.pop(ckey, None)
        svc._last_used.pop(ckey, None)
    if client is not None:
        try:
            await asyncio.wait_for(client.shutdown(), timeout=1.0)
        except Exception:  # best effort on an already-failed server; the pair is marked broken either way
            logger.debug("shutdown of broken LSP client failed", exc_info=True)
    if not already:
        eventlog.log_spawn_failed(key[0], key[1], exc)


__all__ = ["INLINE_GRACE", "WRITE_BUDGET", "CheckResult", "EditChecks", "inline_wait", "start"]
