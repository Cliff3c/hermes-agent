"""A foreground command whose ONLY backgrounding is a trailing `&` runs as terminal(background=true).

Refusing it cost a full model turn to resend the same command with background=true (488 refusals in
30 days of one user's sessions). Ambiguous shapes (inline `a & b`, nohup/disown/setsid, long-lived
servers without `&`) keep the refusal: the caller may expect later commands to run after a check.
"""
import json
import time
from unittest.mock import patch

import pytest


def _env_config(cwd):
    return {
        "env_type": "local", "timeout": 180, "cwd": str(cwd), "host_cwd": None, "modal_mode": "auto",
        "docker_image": "", "singularity_image": "", "modal_image": "", "daytona_image": "",
    }


def test_trailing_amp_runs_as_tracked_background_through_the_same_guards(tmp_path, monkeypatch):
    """Real local backend + registry: one tracked session, the payload runs once, the approval
    guard sees the command that actually runs, and the result says what Hermes did."""
    from tools.process_registry import process_registry
    from tools.terminal_tool import terminal_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    marker = tmp_path / "ran"
    seen = []

    def guard(command, env_type, **kw):
        seen.append(command)
        return {"approved": True}

    with patch("tools.terminal_tool._get_env_config", return_value=_env_config(tmp_path)), \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._check_all_guards", side_effect=guard):
        result = json.loads(terminal_tool(command=f"echo x >> {marker} && sleep 0.2 &"))

    assert result.get("error") is None, result
    assert result["session_id"].startswith("proc_")
    assert "trailing & was converted to background=true" in result["auto_backgrounded"]
    assert "hint" not in result  # the silent-bg hint would tell it to re-launch a second copy
    assert seen == [f"echo x >> {marker} && sleep 0.2"]
    session = process_registry.get(result["session_id"])
    assert session.command == f"echo x >> {marker} && sleep 0.2"
    deadline = time.time() + 10
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert marker.read_text(encoding="utf-8-sig").count("x") == 1


@pytest.mark.parametrize("command, expected", [
    ("python3 -m http.server 8765 & curl -s localhost:8765", "'&' backgrounding"),
    ("sleep 1 & sleep 2 &", "'&' backgrounding"),
    ("nohup ./worker.sh > /dev/null 2>&1 &", "nohup/disown/setsid"),
    ("python3 -m http.server 8765", "long-lived server"),
    # Approval keys on the `&` here (gateway outside its service manager): no conversion.
    ("hermes gateway run &", "'&' backgrounding"),
])
def test_ambiguous_backgrounding_is_still_refused_unchanged(tmp_path, command, expected):
    from tools.terminal_tool import terminal_tool

    with patch("tools.terminal_tool._get_env_config", return_value=_env_config(tmp_path)), \
         patch("tools.terminal_tool._start_cleanup_thread"), \
         patch("tools.terminal_tool._acquire_env", side_effect=AssertionError("must not run")):
        result = json.loads(terminal_tool(command=command))
    assert expected in result["error"]
