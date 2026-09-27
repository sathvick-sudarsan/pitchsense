"""Input hashing and installed version metadata."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pitchsense.errors import InputVideoError, PitchSenseError


def input_sha256(path: Path, error: type[PitchSenseError] = InputVideoError) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise error(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def installed_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "numpy",
        "opencv-python-headless",
        "pydantic",
        "rfdetr",
        "supervision",
        "torch",
        "torchvision",
        "transformers",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    return versions
