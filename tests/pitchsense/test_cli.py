import subprocess
import sys
from importlib.metadata import version


def test_help_and_version():
    help_result = subprocess.run(
        [sys.executable, "-m", "pitchsense.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--help" in help_result.stdout
    assert "--version" in help_result.stdout
    assert "run" not in help_result.stdout
    version_result = subprocess.run(
        [sys.executable, "-m", "pitchsense.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert version("pitchsense") in version_result.stdout


def test_no_arguments_shows_help():
    result = subprocess.run(
        [sys.executable, "-m", "pitchsense.cli"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "usage: pitchsense" in result.stdout
