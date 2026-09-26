"""Supervision ByteTrack adapter.

Temporary M1 adapter. The modern standalone ``trackers`` package would collide
with this repository's inherited top-level ``trackers`` package, so ByteTrack
comes from supervision==0.30.5 (removed in 0.31) until a later milestone isolates
that historical namespace.
"""

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from pitchsense.config import TrackerConfig
from pitchsense.contracts import Detection, Track, VideoMetadata
from pitchsense.errors import InferenceError
from pitchsense.video import VideoFrame


class ByteTrackTracker:
    def __init__(self, config: TrackerConfig, supervision: Any):
        self.config = config
        self.supervision = supervision
        self.tracker = None

    def reset(self, metadata: VideoMetadata) -> None:
        with warnings.catch_warnings():
            # The deprecation is known and pinned; see the module docstring.
            warnings.filterwarnings(
                "ignore", "The `ByteTrack` was deprecated", FutureWarning
            )
            self.tracker = self.supervision.ByteTrack(
                track_activation_threshold=self.config.track_activation_threshold,
                lost_track_buffer=self.config.lost_track_buffer,
                minimum_matching_threshold=self.config.minimum_matching_threshold,
                frame_rate=metadata.fps,
            )

    def update(
        self, detections: Sequence[Detection], frame: VideoFrame
    ) -> Sequence[Track]:
        if self.tracker is None:
            raise RuntimeError("reset(metadata) must be called before update()")
        kept = [d for d in detections if d.class_name in self.config.classes]
        boxes = [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in kept]
        try:
            tracked = self.tracker.update_with_detections(
                self.supervision.Detections(
                    xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
                    confidence=np.array([d.confidence for d in kept], dtype=np.float32),
                    data={"index": np.arange(len(kept))},
                )
            )
            # Tracked rows are a selection of the input rows; "index" maps them
            # back. Supervision returns data-less Detections.empty() when nothing
            # is tracked.
            return [
                Track(
                    track_id=int(track_id),
                    class_id=kept[index].class_id,
                    class_name=kept[index].class_name,
                    confidence=kept[index].confidence,
                    bbox=kept[index].bbox,
                )
                for index, track_id in zip(
                    tracked.data.get("index", ()), tracked.tracker_id, strict=True
                )
            ]
        except Exception as exc:
            raise InferenceError(
                f"ByteTrack update failed on frame {frame.index}: {exc}"
            ) from exc
