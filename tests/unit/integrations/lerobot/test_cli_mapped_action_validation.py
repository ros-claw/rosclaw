"""Real command-function regressions with an explicitly synthetic body loader.

Only body discovery is substituted. Parsing, resolution, mapping generation,
conversion, mapped validation and report formatting use the real product route.
Root runs these tests separately; authoring does not execute them.
"""


def _h35_cli_args(values, json_output=True):
    from argparse import Namespace

    return Namespace(
        body_id="synthetic-h35",
        representation="joint_position",
        names="b,a",
        units="rad,deg",
        reference_frame="",
        values=values,
        chunk_size=2,
        allow_partial=False,
        json=json_output,
    )


def _h35_install_synthetic_body_loader(monkeypatch):
    from types import SimpleNamespace

    from rosclaw.integrations.lerobot import cli

    body = SimpleNamespace(
        profile_id="synthetic-h35",
        joints=[
            {"name": "a", "type": "revolute"},
            {"name": "b", "type": "revolute"},
        ],
    )
    calls = []

    def synthetic_loader(body_id):
        calls.append(body_id)
        return body

    monkeypatch.setattr(cli, "_resolve_effective_body_for_mapping", synthetic_loader)
    return cli, calls


def _h35_strict_json(text):
    import json

    def reject_constant(token):
        raise AssertionError("Nonstandard JSON numeric constant: " + token)

    return json.loads(text, parse_constant=reject_constant)


def test_h35_cli_nonfinite_json_rejection_then_finite_recovery(monkeypatch, capsys):
    import math

    import pytest

    cli, calls = _h35_install_synthetic_body_loader(monkeypatch)
    for token in ("nan", "inf", "-inf"):
        # Nonfinite appears in the later flattened chunk after real conversion.
        rc = cli.cmd_lerobot_mapping_map_action(_h35_cli_args("0.1,90,0.2," + token))
        output = capsys.readouterr()
        assert rc != 0
        payload = _h35_strict_json(output.out)
        assert payload["blocked"] is True
        assert payload["compatibility"] == "convertible"
        assert payload["body_joint_names"] == ["a", "b"]
        assert payload["body_action_values"][2] is None
        validation = payload["validation"]
        assert validation["blocked"] is True
        assert any(
            "Invalid value" in s
            and "$.body_action_values[2]" in s
            and "joint a" in s
            and "chunk 1" in s
            for s in validation["issues"]
        )

        rc = cli.cmd_lerobot_mapping_map_action(_h35_cli_args("0.1,90,0.2,180"))
        output = capsys.readouterr()
        assert rc == 0
        clean = _h35_strict_json(output.out)
        assert clean["blocked"] is False
        assert clean["compatibility"] == "convertible"
        assert clean["chunk_size"] == 2
        assert clean["body_action_values"] == pytest.approx([math.pi / 2, 0.1, math.pi, 0.2])
        assert clean["validation"]["issues"] == []
        assert clean["block_reasons"] == []
    assert calls == ["synthetic-h35"] * 6


def test_h35_cli_text_rejection_and_recovery(monkeypatch, capsys):
    cli, calls = _h35_install_synthetic_body_loader(monkeypatch)
    rc = cli.cmd_lerobot_mapping_map_action(_h35_cli_args("0.1,90,0.2,nan", False))
    output = capsys.readouterr()
    assert rc != 0
    assert "Blocked:       True" in output.out
    assert "Invalid value" in output.out
    assert "$.body_action_values[2]" in output.out
    rc = cli.cmd_lerobot_mapping_map_action(_h35_cli_args("0.1,90,0.2,180", False))
    output = capsys.readouterr()
    assert rc == 0
    assert "Blocked:       False" in output.out
    assert "Invalid value" not in output.out
    assert calls == ["synthetic-h35", "synthetic-h35"]
