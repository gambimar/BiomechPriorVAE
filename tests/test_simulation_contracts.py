"""Exercise the batch runner without launching MATLAB or simulations."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _bash() -> str:
    if os.name == "nt":
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        if git_bash.exists():
            return str(git_bash)
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is required for batch-runner tests")
    return bash


def _run(tmp_path: Path, *arguments: str, statuses: str = "10") -> tuple[subprocess.CompletedProcess, str]:
    log = tmp_path / "matlab.log"
    environment = os.environ.copy()
    environment.update(MOCK_LOG=log.as_posix(), MOCK_STATUSES=statuses)
    # Export a shell function so PATH and MATLAB installations do not affect tests.
    command = r'''
matlab() {
    printf '%s\n' "$*" >> "$MOCK_LOG"
    local status="${MOCK_STATUSES%%,*}"
    if [[ "$MOCK_STATUSES" == *,* ]]; then
        MOCK_STATUSES="${MOCK_STATUSES#*,}"
    fi
    return "$status"
}
export -f matlab
source "$1" "${@:2}"
'''
    result = subprocess.run(
        [_bash(), "-c", command, "test-runner", (ROOT / "runGait.sh").as_posix(), *arguments],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    return result, log.read_text() if log.exists() else ""


@pytest.mark.parametrize("status", ["1", "2", "127"])
def test_matlab_failure_aborts_instead_of_reporting_convergence(tmp_path: Path, status: str) -> None:
    result, log = _run(tmp_path, "--model=test.osim", "--min=1.33", "--max=1.33", statuses=status)
    assert result.returncode == int(status)
    assert "Successfully reached" not in result.stdout
    assert "Aborting" in result.stdout
    assert len(log.splitlines()) == 1


def test_retry_then_completion_uses_distinct_status(tmp_path: Path) -> None:
    result, log = _run(tmp_path, "--model=test.osim", "--min=1.33", "--max=1.33", statuses="0,10")
    assert result.returncode == 0, result.stderr
    assert "Restarting MATLAB" in result.stdout
    assert "Successfully reached 10" in result.stdout
    assert len(log.splitlines()) == 2
    assert "exit(10 * double(done))" in log


def test_all_historical_speed_bins_run_from_another_directory(tmp_path: Path) -> None:
    result, log = _run(tmp_path, "--model=test.osim", "--min=0", "--max=7")
    assert result.returncode == 0, result.stderr
    commands = log.splitlines()
    assert len(commands) == 42
    assert "runGaitSingleStep(0.00," in commands[0]
    assert "runGaitSingleStep(3.53," in commands[29]
    assert "runGaitSingleStep(3.73," in commands[30]
    assert "runGaitSingleStep(5.53," in commands[39]
    assert "runGaitSingleStep(6.00," in commands[40]
    assert "runGaitSingleStep(7.00," in commands[41]


@pytest.mark.parametrize("bounds", [
    ("NaN", "1.33"), ("1oops", "2"), ("-1", "2"), ("2", "1"), ("0.74", "0.75"),
])
def test_invalid_or_empty_ranges_do_not_launch_matlab(tmp_path: Path, bounds: tuple[str, str]) -> None:
    result, log = _run(tmp_path, "--model=test.osim", f"--min={bounds[0]}", f"--max={bounds[1]}")
    assert result.returncode == 1
    assert not log


def test_decimal_bounds_are_not_rounded_into_extra_speed_bins(tmp_path: Path) -> None:
    result, log = _run(tmp_path, "--model=test.osim", "--min=1.331", "--max=1.43")
    assert result.returncode == 0, result.stderr
    assert len(log.splitlines()) == 1
    assert "runGaitSingleStep(1.43," in log


def test_model_paths_and_quotes_are_preserved_as_matlab_strings(tmp_path: Path) -> None:
    result, log = _run(
        tmp_path, "--model=data/model/runner's model.osim", "--metmodel=none", "--min=1.33", "--max=1.33"
    )
    assert result.returncode == 0, result.stderr
    assert "'runner''s_model_none'" in log
    assert "'data/model/runner''s model.osim'" in log


def test_windows_model_paths_use_the_filename_for_results(tmp_path: Path) -> None:
    result, log = _run(tmp_path, r"--model=C:\models\runner.osim", "--min=1.33", "--max=1.33")
    assert result.returncode == 0, result.stderr
    assert "'runner', 'C:\\models\\runner.osim'" in log
