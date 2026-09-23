"""Simple OpenCV track annotations."""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from pitchsense.contracts import Track
from pitchsense.video import VideoFrame


def render(
    frame: VideoFrame, tracks: Sequence[Track], enabled: bool = True
) -> NDArray[np.uint8]:
    if not enabled:
        return frame.image
    image = frame.image.copy()
    for track in tracks:
        box = track.bbox
        cv2.rectangle(
            image,
            (round(box.x1), round(box.y1)),
            (round(box.x2), round(box.y2)),
            (0, 255, 0),
            1,
        )
        cv2.putText(
            image,
            f"{track.class_name} #{track.track_id}",
            (round(box.x1), max(10, round(box.y1) - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return image
