from collections.abc import Sequence
from typing import Protocol

from pitchsense.contracts import Detection, Track
from pitchsense.video import VideoFrame


class Tracker(Protocol):
    def update(
        self, detections: Sequence[Detection], frame: VideoFrame
    ) -> Sequence[Track]: ...

    def reset(self) -> None: ...
