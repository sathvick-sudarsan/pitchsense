"""Production RF-DETR + ByteTrack backend construction."""

from pitchsense.config import PipelineConfig
from pitchsense.detection.base import Detector
from pitchsense.detection.rfdetr import RFDETRDetector
from pitchsense.errors import BackendUnavailableError, ConfigurationError
from pitchsense.tracking.base import Tracker
from pitchsense.tracking.bytetrack import ByteTrackTracker


def create_backend(config: PipelineConfig) -> tuple[Detector, Tracker]:
    if (config.detector.backend, config.tracker.backend) != ("rfdetr", "bytetrack"):
        raise ConfigurationError(
            "supported backends are detector 'rfdetr' with tracker 'bytetrack'; got "
            f"{config.detector.backend!r} with {config.tracker.backend!r}"
        )
    # Imported here so `import pitchsense` never loads torch or downloads weights.
    try:
        import supervision
        import torch
        from rfdetr import RFDETRNano
    except ImportError as exc:
        raise BackendUnavailableError(
            "the RF-DETR/ByteTrack backend needs the optional inference extra "
            f"({exc}); install it with `uv sync --extra inference` or "
            "`pip install 'pitchsense[inference]'`"
        ) from exc
    return (
        RFDETRDetector(config.detector, RFDETRNano, torch.cuda.is_available),
        ByteTrackTracker(config.tracker, supervision),
    )
