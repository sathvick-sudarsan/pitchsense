"""RF-DETR Nano detector adapter.

``pitchsense.backends`` injects the RF-DETR model class and CUDA probe, so this
module never imports torch or rfdetr itself.
"""

import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2

from pitchsense.config import DetectorConfig
from pitchsense.contracts import BoundingBox, Detection, ModelIdentity
from pitchsense.errors import (
    BackendUnavailableError,
    ConfigurationError,
    IncompatibleModelClassesError,
    InferenceError,
    ModelLoadError,
)
from pitchsense.provenance import input_sha256
from pitchsense.video import VideoFrame

OFFICIAL_MODEL = "rfdetr-nano"


def resolve_device(requested: str, cuda_available: Callable[[], bool]) -> str:
    if requested == "cpu":
        return "cpu"
    if cuda_available():
        return "cuda"
    if requested == "cuda":
        raise BackendUnavailableError(
            "CUDA was requested but PyTorch reports no usable CUDA device; install "
            "a CUDA-enabled PyTorch build or use --device cpu"
        )
    return "cpu"


class RFDETRDetector:
    def __init__(
        self,
        config: DetectorConfig,
        model_factory: Callable[..., Any],
        cuda_available: Callable[[], bool],
    ):
        self.config = config
        checkpoint = None if config.model == OFFICIAL_MODEL else Path(config.model)
        if checkpoint is not None and not checkpoint.is_file():
            raise ConfigurationError(
                f"model must be {OFFICIAL_MODEL!r} or an existing local RF-DETR Nano "
                f"checkpoint file; got {config.model!r}"
            )
        device = resolve_device(config.device, cuda_available)
        options = {"device": device}
        sha256 = None
        if checkpoint is not None:
            sha256 = input_sha256(checkpoint, error=ModelLoadError)
        # RF-DETR may redownload official weights over a checkpoint that fails to
        # load when its basename names an official asset, so it only ever sees a
        # private copy under an unregistered name, never the user's own file.
        with tempfile.TemporaryDirectory(prefix="pitchsense-") as staging:
            staged = Path(staging) / f"pitchsense-checkpoint{Path(config.model).suffix}"
            try:
                if checkpoint is not None:
                    shutil.copyfile(checkpoint, staged)
                    # Loaded with torch weights_only=True; trust is never enabled.
                    options["pretrain_weights"] = str(staged)
                self.model = model_factory(**options)
                class_names = list(self.model.class_names)
            except Exception as exc:
                raise ModelLoadError(
                    f"cannot load RF-DETR model {config.model!r} on {device}: {exc}"
                ) from exc
            if checkpoint is not None:
                if input_sha256(staged, error=ModelLoadError) != sha256:
                    raise ModelLoadError(
                        f"the staged copy of {checkpoint} was modified while RF-DETR "
                        "loaded it, so the loaded weights may not match its SHA-256"
                    )
        missing = [name for name in config.classes if name not in class_names]
        if missing:
            raise IncompatibleModelClassesError(
                f"detector classes {missing} are not provided by model "
                f"{config.model!r}; available classes: {class_names}"
            )
        self.model_identity = ModelIdentity(
            backend="rfdetr",
            identifier=config.model,
            sha256=sha256,
            class_names=class_names,
            resolved_device=device,
        )

    def detect(self, frame: VideoFrame) -> Sequence[Detection]:
        try:
            result = self.model.predict(
                cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB),
                threshold=self.config.confidence_threshold,
                include_source_image=False,
            )
            # class_id is RF-DETR's raw label (sparse COCO category IDs for the
            # pretrained model), so names come from RF-DETR's data["class_name"].
            return [
                Detection(
                    class_id=int(class_id),
                    class_name=str(class_name),
                    confidence=float(confidence),
                    bbox=BoundingBox(
                        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)
                    ),
                )
                for (x1, y1, x2, y2), confidence, class_id, class_name in zip(
                    result.xyxy,
                    result.confidence,
                    result.class_id,
                    result.data["class_name"],
                    strict=True,
                )
                # Boxes are clamped to the image and can collapse to zero area there.
                if class_name in self.config.classes and x2 > x1 and y2 > y1
            ]
        except Exception as exc:
            raise InferenceError(
                f"RF-DETR {self.config.model!r} failed on frame {frame.index}: {exc}"
            ) from exc
