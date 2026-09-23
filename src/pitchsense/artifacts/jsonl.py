"""Streaming frame results."""

from pathlib import Path

from pitchsense.contracts import FrameResult
from pitchsense.errors import OutputWriteError


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        try:
            self.stream = self.path.open("w", encoding="utf-8", newline="\n")
        except OSError as exc:
            raise OutputWriteError(f"cannot open results file: {exc}") from exc
        return self

    def write(self, result: FrameResult) -> None:
        try:
            self.stream.write(result.model_dump_json() + "\n")
        except OSError as exc:
            raise OutputWriteError(f"cannot write results file: {exc}") from exc

    def __exit__(self, *_):
        if self.stream is not None:
            self.stream.close()
