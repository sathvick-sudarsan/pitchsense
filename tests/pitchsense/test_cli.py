import subprocess
import sys
from importlib.metadata import version

from pitchsense.cli import main


def test_help_and_version():
    help_result = subprocess.run(
        [sys.executable, "-m", "pitchsense.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--help" in help_result.stdout
    assert "--version" in help_result.stdout
    assert "run" in help_result.stdout
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


def test_run_reports_missing_inference_extra(capsys, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "rfdetr", None)
    source = tmp_path / "input.avi"
    source.touch()
    assert main(["run", str(source), "--output", str(tmp_path / "run")]) == 2
    assert "uv sync --extra inference" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_import_and_help_do_not_load_inference_backends():
    code = (
        "import contextlib, sys\n"
        "from pitchsense.cli import main\n"
        "with contextlib.suppress(SystemExit):\n"
        "    main(['--help'])\n"
        "assert not {'torch', 'rfdetr', 'supervision'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
