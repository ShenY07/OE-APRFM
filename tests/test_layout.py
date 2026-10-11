"""Check experiment imports against the unchanged third-party baselines."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "runner", ["run_rfm_1d", "run_mm_aprfm_1d", "run_mm_aprfm_2d"]
)
def test_baseline_cli(runner):
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ, PYTHONPATH=str(root / "src") + os.pathsep + str(root)
    )
    result = subprocess.run(
        [sys.executable, "-m", "experiments." + runner, "--help"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--features" in result.stdout
