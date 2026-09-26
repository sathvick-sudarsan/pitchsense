from collections.abc import Sequence
from typing import Protocol

from pitchsense.contracts import Detection, Track, VideoMetadata
from pitchsense.video import VideoFrame


class Tracker(Protocol):
    def update(
        self, detections: Sequence[Detection], frame: VideoFrame
    ) -> Sequence[Track]: ...

    def reset(self, metadata: VideoMetadata) -> None:
        """Start a new video; called before its first update()."""
