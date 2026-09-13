from __future__ import annotations

import json

from specpilot.cli import main


def test_demo_command(capsys) -> None:
    main(["demo", "--steps", "8", "--k-max", "2", "--seed", "1"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["synthetic"] is True
    assert len(payload["static"]) == 3


def test_generate_and_replay_commands(tmp_path, capsys) -> None:
    profile = tmp_path / "profile.json"
    trace = tmp_path / "trace.jsonl"
    output = tmp_path / "result.json"
    main(["generate-profile", "--k-max", "2", "--output", str(profile)])
    main(
        [
            "generate-trace",
            "--steps",
            "10",
            "--k-max",
            "2",
            "--output",
            str(trace),
        ]
    )
    main(
        [
            "replay",
            "--profile",
            str(profile),
            "--trace",
            str(trace),
            "--output",
            str(output),
            "--k-max",
            "2",
            "--explore-interval",
            "0",
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["profile_metadata"]["synthetic"] is True
    assert payload["specpilot"]["steps"] == 10
    assert "wrote synthetic profile" in capsys.readouterr().out
