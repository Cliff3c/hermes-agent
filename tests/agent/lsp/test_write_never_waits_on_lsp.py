"""Invariant: a write never waits on the language server, and its diagnostics still reach the model.

The write path used to block on a pre-write baseline wait plus a post-write wait, each up to
``lsp.wait_timeout``; a server that stays silent (typescript-language-server on a clean edit) ran
both to the deadline on every edit.
"""
from __future__ import annotations

import json
import time

import pytest

from agent.lsp.manager import LSPService
from model_tools import handle_function_call
from tests.agent.lsp.test_service import _install_mock_server


@pytest.fixture
def mock_workspace(monkeypatch, tmp_path):
    def install(script, **env):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "pyproject.toml").write_text("", encoding="utf-8")
        monkeypatch.chdir(str(repo))
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        gen = _install_mock_server(monkeypatch, script, "pyright")
        next(gen)
        svc = LSPService(enabled=True, wait_mode="document", wait_timeout=3.0, install_strategy="manual",
                         trusted_workspaces=[str(repo)])
        monkeypatch.setattr("agent.lsp.get_service", lambda: svc)
        return repo, svc, gen
    created = []

    def _install(script, **env):
        created.append(install(script, **env))
        return created[-1][:2]
    yield _install
    for _repo, svc, gen in created:
        svc.shutdown()
        try:
            next(gen)
        except StopIteration:
            pass


def _patch(path, old, new, session="S-lsp"):
    handle_function_call("read_file", {"path": str(path)}, task_id="t-lsp", session_id=session)
    started = time.monotonic()
    out = handle_function_call("patch", {"path": str(path), "old_string": old, "new_string": new},
                               task_id="t-lsp", session_id=session)
    return json.loads(out), time.monotonic() - started


def test_silent_server_does_not_hold_the_write(mock_workspace):
    """A server that never answers costs the write nothing: two edits stay far below one wait budget."""
    repo, _svc = mock_workspace("silent")
    target = repo / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    for old, new in (("x = 1", "x = 2"), ("x = 2", "x = 3")):
        result, elapsed = _patch(target, old, new)
        assert result["success"] is True, result
        assert elapsed < 1.0, f"write blocked {elapsed:.2f}s on a silent language server"
    assert target.read_text(encoding="utf-8-sig") == "x = 3\n"


def test_late_diagnostics_reach_the_session(mock_workspace):
    """A verdict slower than the write is delivered to the editing session afterwards (attached to
    its next tool result), and only the error the edit introduced is reported."""
    repo, _svc = mock_workspace("by_content", MOCK_LSP_PUSH_DELAY="0.4")
    target = repo / "b.py"
    target.write_text("x = 1\n", encoding="utf-8")
    result, elapsed = _patch(target, "x = 1", "x = bad")
    assert result["success"] is True and elapsed < 0.4, (result, elapsed)
    assert "lsp_diagnostics" not in result
    from agent.lsp import deferred
    notes = []
    deadline = time.monotonic() + 10
    while not notes and time.monotonic() < deadline:
        notes = deferred.pop_notes("S-lsp")
        time.sleep(0.05)
    assert notes, "the late verdict never reached the session"
    (path, text), = notes
    assert path == str(target) and "MOCK001" in text and "earlier edit" in text


def test_late_verdict_reaches_the_model_before_the_turn_ends(mock_workspace, monkeypatch):
    """The turn's last edit gets a verdict after its tool result returned: the turn must not end on
    it (a one-shot run has no next turn).  The final-answer gate waits for it and hands it to the
    model as a new message, continuing the turn."""
    from types import SimpleNamespace

    from agent.turn_stop_gates import apply_stop_gates

    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    repo, _svc = mock_workspace("by_content", MOCK_LSP_PUSH_DELAY="0.4")
    target = repo / "c.py"
    target.write_text("x = 1\n", encoding="utf-8")
    result, _elapsed = _patch(target, "x = 1", "x = bad", session="S-end")
    assert result["success"] is True and "lsp_diagnostics" not in result, result
    agent = SimpleNamespace(session_id="S-end", _current_task_id="t-lsp",
                            _interim_content_was_streamed=lambda _text: False,
                            _emit_interim_assistant_message=lambda _msg: None,
                            _flush_messages_to_session_db=lambda *_a: None)
    messages = [{"role": "user", "content": "edit c.py"}]
    verdict = apply_stop_gates(agent, {"role": "assistant", "content": "done"}, final_response="done",
                               messages=messages, conversation_history=[],
                               pending_verification_response=None, pending_verification_response_previewed=False)
    assert verdict.continue_turn, "the turn ended with the edit's verdict still undelivered"
    assert messages[-1]["role"] == "user" and "MOCK001" in messages[-1]["content"], messages[-1]
