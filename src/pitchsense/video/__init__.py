from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class VideoFrame:
    index: int
    timestamp_s: float
    image: NDArray[np.uint8]
