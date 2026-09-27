"""RF-DETR/ByteTrack adapters with injected fakes: no weights, network, or GPU."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pitchsense.backends import create_backend
from pitchsense.cli import main
from pitchsense.config import load_pipeline_config
from pitchsense.contracts import BoundingBox, Detection, Track, VideoMetadata
from pitchsense.detection.rfdetr import RFDETRDetector, resolve_device
from pitchsense.errors import (
    BackendUnavailableError,
    ConfigurationError,
    IncompatibleModelClassesError,
    InferenceError,
    ModelLoadError,
)
from pitchsense.tracking.bytetrack import ByteTrackTracker
from pitchsense.video import VideoFrame

FRAME = VideoFrame(
    index=0, timestamp_s=0.0, image=np.zeros((48, 64, 3), dtype=np.uint8)
)
METADATA = VideoMetadata(width=64, height=48, fps=12.5)


class FakeRFDETR:
    """Stands in for rfdetr.RFDETRNano and mimics its predict() output."""

    class_names = ["person", "bicycle", "sports ball"]

    def __init__(self, **options):
        self.options = options
        self.calls = []

    def predict(self, image, threshold, **kwargs):
        self.calls.append((image, threshold))
        return SimpleNamespace(
            xyxy=np.array(
                [[1, 2, 11, 22], [5, 5, 9, 9], [30, 3, 30, 8], [40, 4, 44, 9]],
                dtype=np.float32,
            ),
            confidence=np.array([0.75, 0.5, 0.625, 0.5], dtype=np.float32),
            # Sparse COCO category IDs, deliberately not indices into class_names.
            class_id=np.array([1, 2, 37, 37]),
            data={
                "class_name": np.array(
                    ["person", "bicycle", "sports ball", "sports ball"], dtype=object
                ),
            },
        )


def make_detector(model_factory=FakeRFDETR, cuda=False, **fields):
    config = load_pipeline_config().detector.model_copy(update=fields)
    return RFDETRDetector(config, model_factory, lambda: cuda)


def detection(name, x, confidence=0.5, class_id=1):
    return Detection(
        class_id=class_id,
        class_name=name,
        confidence=confidence,
        bbox=BoundingBox(x1=x, y1=0, x2=x + 10, y2=20),
    )


@pytest.mark.parametrize(
    ("requested", "cuda", "expected"),
    [
        ("auto", False, "cpu"),
        ("auto", True, "cuda"),
        ("cpu", True, "cpu"),
        ("cuda", True, "cuda"),
    ],
)
def test_resolve_device(requested, cuda, expected):
    assert resolve_device(requested, lambda: cuda) == expected


def test_explicit_cuda_without_gpu_fails_instead_of_falling_back():
    with pytest.raises(BackendUnavailableError, match="--device cpu"):
        make_detector(device="cuda")


def test_official_model_identity():
    detector = make_detector()
    assert detector.model.options == {"device": "cpu"}
    assert detector.model_identity.model_dump() == {
        "backend": "rfdetr",
        "identifier": "rfdetr-nano",
        "sha256": None,
        "class_names": ["person", "bicycle", "sports ball"],
        "resolved_device": "cpu",
    }


ORIGINAL = b"synthetic checkpoint bytes; never deserialized"


def official_named_checkpoint(tmp_path):
    # Named like an official RF-DETR asset: the case RF-DETR could overwrite.
    checkpoint = tmp_path / "user" / "rf-detr-nano.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(ORIGINAL)
    return checkpoint


class StagingRFDETR(FakeRFDETR):
    def __init__(self, **options):
        super().__init__(**options)
        self.staged_bytes = Path(options["pretrain_weights"]).read_bytes()


def test_local_checkpoint_is_loaded_from_a_cleaned_up_renamed_copy(tmp_path):
    checkpoint = official_named_checkpoint(tmp_path)
    detector = make_detector(StagingRFDETR, cuda=True, model=str(checkpoint))
    staged = Path(detector.model.options["pretrain_weights"])
    assert staged != checkpoint
    assert staged.name != checkpoint.name
    assert detector.model.staged_bytes == ORIGINAL
    assert not staged.parent.exists()
    identity = detector.model_identity
    assert identity.sha256 == hashlib.sha256(ORIGINAL).hexdigest()
    assert identity.identifier == str(checkpoint)
    assert identity.resolved_device == "cuda"
    assert staged.parent.name not in identity.model_dump_json()


@pytest.mark.parametrize(
    ("load_fails", "message"), [(True, "safe load failed"), (False, "was modified")]
)
def test_backend_overwriting_staged_checkpoint_never_reaches_original(
    tmp_path, load_fails, message
):
    checkpoint = official_named_checkpoint(tmp_path)
    staged = []

    def clobbering(**options):
        # What RF-DETR's redownload does to a file with an official basename.
        staged.append(Path(options["pretrain_weights"]))
        staged[0].write_bytes(b"official weights")
        if load_fails:
            raise RuntimeError("safe load failed")
        return FakeRFDETR(**options)

    with pytest.raises(ModelLoadError, match=message):
        make_detector(clobbering, model=str(checkpoint))
    assert checkpoint.read_bytes() == ORIGINAL
    assert not staged[0].parent.exists()


@pytest.mark.parametrize("model", ["rf-detr-base", "missing/nano.pth"])
def test_unsupported_model_or_missing_checkpoint_is_rejected(model):
    with pytest.raises(ConfigurationError, match="rfdetr-nano"):
        make_detector(model=model)


def test_model_load_failure_is_wrapped():
    def broken(**options):
        raise RuntimeError("weights download failed")

    with pytest.raises(ModelLoadError, match="weights download failed"):
        make_detector(model_factory=broken)


def test_requested_class_missing_from_model_is_reported():
    with pytest.raises(IncompatibleModelClassesError) as error:
        make_detector(classes=["person", "footbal"])
    assert "['footbal']" in str(error.value)
    assert "['person', 'bicycle', 'sports ball']" in str(error.value)


def test_detect_passes_rgb_copy_and_threshold():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[..., 0], image[..., 2] = 10, 200
    original = image.copy()
    detector = make_detector(confidence_threshold=0.4)
    detector.detect(VideoFrame(index=0, timestamp_s=0.0, image=image))
    received, threshold = detector.model.calls[0]
    assert np.array_equal(received, original[..., ::-1])
    assert np.array_equal(image, original)
    assert threshold == 0.4


def test_detect_keeps_requested_classes_with_rfdetr_class_mapping():
    # bicycle is not requested; the zero-width box is dropped.
    assert make_detector().detect(FRAME) == [
        Detection(
            class_id=1,
            class_name="person",
            confidence=0.75,
            bbox=BoundingBox(x1=1, y1=2, x2=11, y2=22),
        ),
        Detection(
            class_id=37,
            class_name="sports ball",
            confidence=0.5,
            bbox=BoundingBox(x1=40, y1=4, x2=44, y2=9),
        ),
    ]


def test_detect_handles_empty_result():
    class Empty(FakeRFDETR):
        def predict(self, image, threshold, **kwargs):
            return SimpleNamespace(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty(0, dtype=np.float32),
                class_id=np.empty(0, dtype=int),
                data={"class_name": np.array([], dtype=object)},
            )

    assert make_detector(model_factory=Empty).detect(FRAME) == []


def test_factory_rejects_unsupported_backends():
    config = load_pipeline_config()
    config.tracker.backend = "sort"
    with pytest.raises(ConfigurationError, match="bytetrack"):
        create_backend(config)


class FakeByteTrack:
    """Tracks every other input row, like ByteTrack dropping unmatched rows."""

    def __init__(self, **options):
        self.options = options
        self.inputs = []

    def update_with_detections(self, detections):
        self.inputs.append(detections)
        index = detections.data["index"][::2]
        if not len(index):
            return SimpleNamespace(data={}, tracker_id=np.array([], dtype=int))
        return SimpleNamespace(data={"index": index}, tracker_id=index + 100)


FAKE_SUPERVISION = SimpleNamespace(ByteTrack=FakeByteTrack, Detections=SimpleNamespace)


def test_bytetrack_reset_uses_source_fps_and_config():
    tracker = ByteTrackTracker(load_pipeline_config().tracker, FAKE_SUPERVISION)
    tracker.reset(METADATA)
    assert tracker.tracker.options == {
        "track_activation_threshold": 0.25,
        "lost_track_buffer": 30,
        "minimum_matching_threshold": 0.8,
        "frame_rate": 12.5,
    }


def test_bytetrack_filters_tracked_classes_and_converts_tracks():
    tracker = ByteTrackTracker(load_pipeline_config().tracker, FAKE_SUPERVISION)
    tracker.reset(METADATA)
    detections = [
        detection("person", 0),
        detection("sports ball", 5, class_id=37),
        detection("person", 20, confidence=0.75),
        detection("person", 40),
    ]
    tracks = tracker.update(detections, FRAME)
    sent = tracker.tracker.inputs[0]
    assert sent.xyxy.tolist() == [[0, 0, 10, 20], [20, 0, 30, 20], [40, 0, 50, 20]]
    assert sent.confidence.tolist() == [0.5, 0.75, 0.5]
    assert tracks == [
        Track(track_id=100, **detections[0].model_dump()),
        Track(track_id=102, **detections[3].model_dump()),
    ]
    assert tracker.update([], FRAME) == []
    assert tracker.tracker.inputs[1].xyxy.shape == (0, 4)


def test_real_supervision_bytetrack_assigns_stable_ids():
    supervision = pytest.importorskip("supervision")
    tracker = ByteTrackTracker(load_pipeline_config().tracker, supervision)
    tracker.reset(METADATA)
    assert tracker.tracker.max_time_lost == int(12.5 / 30 * 30)
    detections = [
        detection("person", 0, confidence=0.9),
        detection("person", 30, confidence=0.9),
        detection("sports ball", 15, confidence=0.9, class_id=37),
    ]
    frames = [tracker.update(detections, FRAME) for _ in range(3)]
    for tracks in frames:
        assert tracks == [
            Track(track_id=1, **detections[0].model_dump()),
            Track(track_id=2, **detections[1].model_dump()),
        ]
    assert tracker.update([], FRAME) == []


class FailingRFDETR(FakeRFDETR):
    def predict(self, image, threshold, **kwargs):
        raise RuntimeError("CUDA out of memory")


def test_detector_inference_failure_names_model_and_frame():
    frame = VideoFrame(index=7, timestamp_s=0.5, image=FRAME.image)
    with pytest.raises(InferenceError, match="frame 7: CUDA out of memory") as error:
        make_detector(FailingRFDETR).detect(frame)
    assert "RF-DETR 'rfdetr-nano'" in str(error.value)
    assert isinstance(error.value.__cause__, RuntimeError)


def test_tracker_update_failure_names_frame():
    class Failing(FakeByteTrack):
        def update_with_detections(self, detections):
            raise ValueError("bad tensors")

    supervision = SimpleNamespace(ByteTrack=Failing, Detections=SimpleNamespace)
    tracker = ByteTrackTracker(load_pipeline_config().tracker, supervision)
    tracker.reset(METADATA)
    frame = VideoFrame(index=4, timestamp_s=0.25, image=FRAME.image)
    with pytest.raises(InferenceError, match="frame 4: bad tensors") as error:
        tracker.update([detection("person", 0)], frame)
    assert "ByteTrack" in str(error.value)
    assert isinstance(error.value.__cause__, ValueError)


def test_interrupts_are_not_wrapped():
    class Interrupted(FakeRFDETR):
        def predict(self, image, threshold, **kwargs):
            raise KeyboardInterrupt

    class InterruptedTracker(FakeByteTrack):
        def update_with_detections(self, detections):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        make_detector(Interrupted).detect(FRAME)
    supervision = SimpleNamespace(
        ByteTrack=InterruptedTracker, Detections=SimpleNamespace
    )
    tracker = ByteTrackTracker(load_pipeline_config().tracker, supervision)
    tracker.reset(METADATA)
    with pytest.raises(KeyboardInterrupt):
        tracker.update([detection("person", 0)], FRAME)


def test_cli_reports_inference_failure_as_controlled_error(tmp_path, capsys):
    source = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 12, (64, 48))
    writer.write(FRAME.image)
    writer.release()
    detector = make_detector(FailingRFDETR)
    tracker = ByteTrackTracker(load_pipeline_config().tracker, FAKE_SUPERVISION)
    output = tmp_path / "run"
    code = main(
        ["run", str(source), "--output", str(output)],
        backend_factory=lambda config: (detector, tracker),
    )
    assert code == 2
    error = capsys.readouterr().err
    assert "pitchsense: RF-DETR 'rfdetr-nano' failed on frame 0" in error
    assert not (output / "manifest.json").exists()
