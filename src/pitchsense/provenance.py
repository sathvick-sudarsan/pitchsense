"""Input hashing and installed version metadata."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pitchsense.errors import InputVideoError


def input_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InputVideoError(f"cannot hash input video: {exc}") from exc
    return digest.hexdigest()


def installed_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "opencv-python-headless", "pydantic"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    return versions
