"""Unit tests for config helpers."""

import pytest

from emblase.config import resolve_tiled_path


@pytest.mark.parametrize(
    "flag, env, expected",
    [
        # No flag — return env as-is
        (None, "smi/sandbox/_inputs", "smi/sandbox/_inputs"),
        ("", "smi/sandbox/_inputs", "smi/sandbox/_inputs"),
        # No flag, empty env
        (None, "", ""),
        ("", "", ""),
        # Absolute flag — strip leading slash, ignore env
        ("/other/path", "smi/sandbox/_inputs", "other/path"),
        ("/other/path", "", "other/path"),
        # Single-segment absolute
        ("/root", "smi/sandbox/_inputs", "root"),
        # Relative with './' prefix — append to env
        ("./subdir", "smi/sandbox/_inputs", "smi/sandbox/_inputs/subdir"),
        ("./a/b", "base", "base/a/b"),
        # Bare relative (no special prefix) — append to env
        ("subdir", "smi/sandbox/_inputs", "smi/sandbox/_inputs/subdir"),
        ("a/b/c", "base", "base/a/b/c"),
        # Bare relative with empty env
        ("subdir", "", "subdir"),
        ("./subdir", "", "subdir"),
    ],
)
def test_resolve_tiled_path(flag, env, expected):
    assert resolve_tiled_path(flag, env) == expected


def test_settings_tiled_container_env_vars(monkeypatch):
    """EMBLASE_TILED_INPUT/OUTPUT_CONTAINER env vars are picked up by Settings."""
    monkeypatch.setenv("EMBLASE_TILED_INPUT_CONTAINER", "my/input/path")
    monkeypatch.setenv("EMBLASE_TILED_OUTPUT_CONTAINER", "my/output/path")

    # Re-instantiate settings to pick up the monkeypatched env vars.
    from emblase.config import Settings

    s = Settings()
    assert s.tiled_input_container == "my/input/path"
    assert s.tiled_output_container == "my/output/path"


def test_resolve_tiled_path_combined_with_settings(monkeypatch):
    """resolve_tiled_path + Settings.tiled_input_container work end-to-end."""
    monkeypatch.setenv("EMBLASE_TILED_INPUT_CONTAINER", "proposal/pi/project/_inputs")

    from emblase.config import Settings

    s = Settings()
    # Relative flag → appended to container
    assert (
        resolve_tiled_path("run_1086139", s.tiled_input_container)
        == "proposal/pi/project/_inputs/run_1086139"
    )
    # Empty flag → container returned as-is
    assert resolve_tiled_path("", s.tiled_input_container) == "proposal/pi/project/_inputs"
    # Absolute flag → overrides completely
    assert (
        resolve_tiled_path("/other/inputs/run_xyz", s.tiled_input_container)
        == "other/inputs/run_xyz"
    )
