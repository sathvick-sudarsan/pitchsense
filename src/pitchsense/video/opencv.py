"""Streaming OpenCV video reader and writer."""

import math
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from pitchsense.contracts import VideoMetadata
from pitchsense.errors import EmptyVideoError, InputVideoError, OutputWriteError
from pitchsense.video import VideoFrame


class VideoReader:
    def __init__(self, path: Path, capture_factory: Callable = cv2.VideoCapture):
        self.path = Path(path)
        self.capture_factory = capture_factory
        self.capture = None
        self.metadata: VideoMetadata | None = None

    def __enter__(self):
        if not self.path.is_file():
            raise InputVideoError(f"input video does not exist: {self.path}")
        try:
            self.capture = self.capture_factory(str(self.path))
            if not self.capture.isOpened():
                raise InputVideoError(f"cannot open input video: {self.path}")
            width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if not all(math.isfinite(value) for value in (width, height, fps)):
                raise InputVideoError("invalid video metadata")
            self.metadata = VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                reported_frame_count=int(count)
                if isinstance(count, (int, float))
                and math.isfinite(count)
                and count >= 0
                else None,
            )
            return self
        except (cv2.error, ValidationError, TypeError, ValueError) as exc:
            raise InputVideoError(f"invalid input video: {exc}") from exc
        finally:
            if self.metadata is None and self.capture is not None:
                self.capture.release()

    def __iter__(self) -> Iterator[VideoFrame]:
        if self.capture is None or self.metadata is None:
            raise RuntimeError("video reader is not open")
        index = 0
        while True:
            try:
                ok, image = self.capture.read()
            except cv2.error as exc:
                raise InputVideoError(f"cannot read video frame: {exc}") from exc
            if not ok:
                if index == 0:
                    raise EmptyVideoError(
                        f"input video has no readable frames: {self.path}"
                    )
                break
            yield VideoFrame(
                index=index, timestamp_s=index / self.metadata.fps, image=image
            )
            index += 1

    def __exit__(self, *_):
        if self.capture is not None:
            self.capture.release()


class VideoWriter:
    def __init__(
        self,
        path: Path,
        metadata: VideoMetadata,
        codec: str,
        writer_factory: Callable = cv2.VideoWriter,
    ):
        self.path = Path(path)
        self.metadata = metadata
        self.codec = codec
        self.writer_factory = writer_factory
        self.writer = None

    def __enter__(self):
        try:
            self.writer = self.writer_factory(
                str(self.path),
                cv2.VideoWriter_fourcc(*self.codec),
                self.metadata.fps,
                (self.metadata.width, self.metadata.height),
            )
            if not self.writer.isOpened():
                raise OutputWriteError(f"cannot open output video: {self.path}")
            return self
        except cv2.error as exc:
            raise OutputWriteError(f"cannot open output video: {exc}") from exc
        finally:
            if self.writer is not None and not self.writer.isOpened():
                self.writer.release()

    def write(self, image: NDArray[np.uint8]) -> None:
        if (
            image.shape != (self.metadata.height, self.metadata.width, 3)
            or image.dtype != np.uint8
        ):
            raise OutputWriteError(
                "output frame dimensions or format do not match source video"
            )
        try:
            self.writer.write(image)
            if not self.writer.isOpened():
                raise OutputWriteError(
                    f"output video closed while writing: {self.path}"
                )
        except cv2.error as exc:
            raise OutputWriteError(f"cannot write output video: {exc}") from exc

    def __exit__(self, *_):
        if self.writer is not None:
            self.writer.release()
