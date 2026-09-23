from collections.abc import Sequence
from typing import Protocol

from pitchsense.contracts import Detection, ModelIdentity
from pitchsense.video import VideoFrame


class Detector(Protocol):
    @property
    def model_identity(self) -> ModelIdentity: ...

    def detect(self, frame: VideoFrame) -> Sequence[Detection]: ...
