# PitchSense

This repository contains two separate things:

1. **The PitchSense package** (`src/pitchsense/`, `tests/pitchsense/`): new, independently engineered code. It is an installable, configurable detection/tracking video pipeline with versioned results and provenance artifacts. This is the actively developed part.
2. **The historical football_analysis baseline** (repository root: `main.py`, `trackers/`, `team_assigner/`, and the other root modules): frozen upstream code kept for reference and attribution. The PitchSense package does not import, wrap, or modify it.

## Attribution and project status

PitchSense was originally inspired by and reconstructed from [Abdullah Tarek's football_analysis](https://github.com/abdullahtarek/football_analysis). It is now being independently extended and engineered by Sathvick Sudarsan. The complete upstream Git history is preserved; the upstream implementation is Abdullah Tarek's work, not an original implementation by this repository's maintainer. This first reconstruction checkpoint adds environment documentation, small maintenance changes, and synthetic tests; it does not claim a new computer-vision method or improved model performance.

Upstream baseline: `e4799632cdf271cf57f4eb0c4d872cf1b7ab0f17`. No LICENSE or COPYING file exists in the fetched upstream history. No replacement license has been invented; existing files and attribution remain intact.

---

## Part 1: The PitchSense package

### What M1 provides

- A `pitchsense run` command that reads a video, detects objects in each frame, tracks them, and writes an annotated video, per-frame JSONL results, and a provenance manifest.
- Typed, validated contracts for detections, tracks, frame results, model identity, and the run manifest.
- `Detector` and `Tracker` interfaces with one real backend: RF-DETR Nano detection (`rfdetr` 1.11.0) and ByteTrack tracking from Supervision 0.30.5.
- A deterministic, offline test suite built on generated video and fake detector/tracker implementations.

### Installation

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). Run commands from the repository root.

**Default, lightweight install.** This installs pydantic, numpy, and opencv-python-headless (no PyTorch). It provides the `pitchsense` CLI and everything the test suite needs:

```powershell
uv sync --group dev
```

With only the default install, `pitchsense run` exits with an error explaining how to install the inference extra.

**Real inference backend.** The optional `inference` extra adds `rfdetr==1.11.0` and `supervision==0.30.5`, together with their PyTorch/Transformers dependencies:

```powershell
uv sync --extra inference --group dev
```

With pip, the equivalent is `pip install ".[inference]"`. The lock uses PyPI PyTorch wheels. On Windows that is a CPU-only build (verified: `torch 2.14.0+cpu`). Other platforms and GPU builds have not been verified.

### Usage

```text
uv run pitchsense run INPUT_VIDEO --output RUN_DIR [--config CONFIG.toml]
    [--model MODEL] [--device {auto,cpu,cuda}] [--confidence THRESHOLD]
    [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
```

- `--output` must not exist yet. Existing directories are refused, never overwritten.
- `--model` is `rfdetr-nano` (the default, official COCO-pretrained weights) or a path to a local RF-DETR Nano checkpoint. PitchSense records the original file's SHA-256 and gives RF-DETR only a temporary, renamed copy, which it loads with safe `weights_only` deserialization. RF-DETR therefore can never modify or replace your checkpoint.
- `--device auto` (the default) uses CUDA when PyTorch reports a usable device and otherwise CPU. `--device cuda` fails instead of silently falling back to CPU.
- Command-line flags override the TOML file, which overrides the built-in defaults.
- On a PitchSense error, the CLI prints `pitchsense: <message>` to stderr and exits with code 2. This includes a detector or tracker backend failing on a frame; the message names the backend and the frame index.

### Configuration

All fields, shown with their default values:

```toml
[detector]
backend = "rfdetr"
model = "rfdetr-nano"                 # or a local RF-DETR Nano checkpoint path
confidence_threshold = 0.3
device = "auto"                       # auto | cpu | cuda
classes = ["person", "sports ball"]

[tracker]
backend = "bytetrack"
classes = ["person"]                  # must be a subset of detector classes
track_activation_threshold = 0.25
lost_track_buffer = 30
minimum_matching_threshold = 0.8

[rendering]
enabled = true

[output]
codec = "MJPG"
```

Unknown sections and fields are rejected. After the model loads and before any frame is processed, the detector classes are checked against the model's own class list. A class the model does not provide, such as a typo, fails with the list of available classes. The pretrained model uses COCO class names. Tracker settings are passed directly to Supervision 0.30.5 `sv.ByteTrack`. Its frame rate is not configurable: it always comes from the source video.

### Outputs

| File in `RUN_DIR` | Content |
| --- | --- |
| `annotated.avi` | Source frames with tracked boxes and `class #id` labels (MJPG) |
| `results.jsonl` | One `FrameResult` JSON object per frame (schema `1.0.0`): frame index, timestamp, detections, tracks |
| `manifest.json` | Run provenance (schema `2.0.0`), written last |

Artifacts are first written as partial files. They are finalized only after the annotated video is read back and its frame count matches the number of processed frames. A failed run leaves no manifest.

### Provenance recorded in `manifest.json`

- Input filename, size, and SHA-256.
- Video metadata as read: width, height, source FPS, and reported frame count. Also the number of frames actually processed.
- Model identity:
  - backend and identifier (`rfdetr-nano`, or the local checkpoint path);
  - SHA-256 of the original local checkpoint file. None is recorded for the official weights, which RF-DETR downloads and caches; PitchSense does not hash them;
  - the model's full class list;
  - the resolved device.
- The complete effective configuration: detector, tracker, rendering, and output.
- Application and Python versions, plus installed versions of numpy, opencv-python-headless, pydantic, and (when installed) rfdetr, supervision, torch, torchvision, and transformers.
- Start and end UTC timestamps, and elapsed seconds for the processing run (model loading is not included).

### Real-backend smoke test (manual; not run by CI)

```powershell
uv sync --extra inference --group dev
uv run pitchsense run path\to\short_video.mp4 `
    --output runs\rfdetr-smoke `
    --model rfdetr-nano `
    --device cpu
```

- **Network and model download.** The first run downloads the official pretrained weights (`rf-detr-nano.pth`, about 350 MB) into RF-DETR's cache. The cache is `RF_HOME`, defaulting to `~/.roboflow/models`. RF-DETR checks the file's MD5. This is an explicit network/model operation; the automated tests never perform it.
- **Device.** CPU is chosen so the procedure is portable. Use `--device cuda` only if the installed PyTorch actually supports CUDA.
- **Success criteria.** The command exits with code 0, and `runs\rfdetr-smoke` contains `annotated.avi`, `results.jsonl`, and `manifest.json`.
- **Local checkpoint (optional).** Repeat with `--model path\to\checkpoint.pth`. The manifest's `model_identity.sha256` must equal `(Get-FileHash path\to\checkpoint.pth -Algorithm SHA256).Hash`, compared case-insensitively, and the file's hash must be unchanged after the run.
- **Not a benchmark.** This smoke test does not measure accuracy or performance.

### Development checks

```powershell
uv sync --frozen --group dev
uv run ruff format --check src/pitchsense tests/pitchsense
uv run ruff check src/pitchsense tests/pitchsense
uv run pytest
```

The default tests need no network, model weights, GPU, external video, or inherited code, and they perform no unsafe deserialization. The RF-DETR and ByteTrack adapters are tested with injected fakes. One extra test runs against the real Supervision ByteTrack, and only when the inference extra is installed. CI runs the checks above without the inference extra. `pytest` collects only `tests/pitchsense`.

### Why Supervision is pinned to 0.30.5

The inherited code already owns the top-level Python package name `trackers`. The modern standalone `trackers` package would collide with that frozen code. For M1, PitchSense therefore uses Supervision's deprecated `sv.ByteTrack` (removed in Supervision 0.31) behind its own `Tracker` interface. This is a temporary adapter, not a long-term commitment. A migration is expected once a later milestone isolates the historical namespace.

### M1 scope and limitations

- M1 implements an installable, configurable detection/tracking video pipeline; RF-DETR/ByteTrack adapters; and versioned results/provenance artifacts. The deterministic integration behavior is verified by the test suite.
- M1 does **not** establish detector accuracy, tracker accuracy, throughput or latency, physical measurement accuracy, or football analytics quality.
- The pretrained detector is a general COCO model. `person` covers players, referees, staff, and spectators alike. `sports ball` is detected, but ball tracking is not claimed: only `person` is tracked by default.
- The package has no team assignment, possession, camera-motion compensation, calibration/homography, metric coordinates, speed/distance, or ball interpolation. Those exist only in the historical baseline below and are not features of this package.

---

## Part 2: Historical inherited baseline (frozen)

The root-level code is the upstream football_analysis pipeline: YOLO detections, ByteTrack player/referee tracking, shirt-color team assignment, ball-possession heuristics, optical-flow camera movement, perspective mapping, and speed/distance overlays. The reconstruction added small maintenance changes. The code is frozen and kept for reference. It uses its own pinned environment (`requirements.txt`, including Ultralytics and Supervision 0.26.1) and is not part of the `pitchsense` package. Everything described in this part is historical upstream behavior, not PitchSense package functionality.

### Setup (Windows, CPU)

Run commands from the repository root. Python 3.11 is the reconstruction target, not a claim about the original author's environment. Install [uv](https://docs.astral.sh/uv/) and use the fully pinned dependency list:

```powershell
uv venv --python 3.11
uv pip sync requirements.txt --python .venv/Scripts/python.exe --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
$env:YOLO_CONFIG_DIR = "$PWD/.runtime/ultralytics"
$env:MPLCONFIGDIR = "$PWD/.runtime/matplotlib"
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

These instructions predate the package. They use the same `.venv` directory as `uv sync`, and each setup replaces the other's packages. Keep the two environments separate, for example by creating the baseline in another directory with `uv venv .venv-baseline --python 3.11` and adjusting the `--python` paths to match.

`requirements.in` records direct runtime choices; `requirements.txt` pins transitive dependencies for the tested Windows/Python 3.11 CPU environment. PyTorch and torchvision use CPU wheels from the official PyTorch index. Supervision 0.26.1 and Ultralytics share the single `opencv-python` distribution; do not also install a headless/contrib OpenCV wheel into this environment. Other operating systems, GPU wheels, and historical notebook environments have not been verified. To intentionally refresh the lock:

```powershell
uv pip compile requirements.in --python .venv/Scripts/python.exe --output-file requirements.txt --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

### Input assets and running the original pipeline

The Git repository does **not** contain the trained checkpoint or input video. The following are the original author's links, retained as provenance; their bytes, availability, and model identity have not been verified in this reconstruction:

- [Upstream trained checkpoint](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view): save as `models/best.pt`.
- [Upstream sample video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view): save as `input_videos/08fd33_4.mp4`.

Use only trusted model/checkpoint and pickle files. For an actual run, first obtain and verify these assets, then recompute the cached results: change **both** `read_from_stub=True` arguments in `main.py` to `False`. The tracked upstream stubs are clip-specific, are not bound to an input hash, and were not deserialized during this audit. Leaving the original defaults unchanged reuses them and does not demonstrate model inference.

```powershell
.venv/Scripts/python.exe main.py
```

Output: `output_videos/output_video.avi`. The original entry point assumes its specific clip, 24 frames/second, fixed camera-feature strips, and a manually calibrated pitch quadrilateral. A different video requires real calibration and code changes. Missing assets, unsuitable clips, or the documented edge cases can prevent a complete run. Dependency installation and synthetic checks are reproducible; end-to-end football inference and exact output reproduction are **NOT VERIFIED**. K-means is unseeded, so output may vary even for identical inputs.

### Repository map and processing order

| Location | Purpose |
| --- | --- |
| `main.py` | Frame loading, detection/caches, positions, camera adjustment, perspective mapping, ball interpolation, player speed, teams, possession, rendering and video writing |
| `trackers/` | Ultralytics inference in batches of 20, Supervision ByteTrack, bounding-box positions, ball interpolation and overlays |
| `team_assigner/` | Two-cluster shirt/background segmentation followed by two-cluster team assignment |
| `player_ball_assigner/` | Nearest player's foot within a 70-pixel threshold |
| `camera_movement_estimator/` | Shi-Tomasi features and Lucas-Kanade optical flow |
| `view_transformer/` | Four-point homography into a hard-coded 23.32 m by 68 m plane |
| `speed_and_distance_estimator/` | Endpoint displacement over five-frame windows at assumed 24 fps |
| `utils/` | Video I/O and bounding-box geometry |
| `training/`, `development_and_analysis/` | Historical YOLO training and color-clustering notebooks |
| `tests/test_smoke.py` | Synthetic algorithm smoke checks without trained weights/video |

### Historical notebooks

The training notebook specifies `yolov5x.pt`, 100 epochs and image size 640, but its training cell has no saved output. This is configuration evidence, not proof of a completed training run or the identity of `best.pt`. It references Roboflow project `football-players-detection-3zvbc`, version 1. Its data download and directory moves depend on an external account/dataset and are not idempotent.

Notebook-only packages (`roboflow`, Jupyter) are intentionally outside the runtime lock. Use a separate environment if revisiting training and set `ROBOFLOW_API_KEY` to your own credential. The credential labeled revoked in an upstream comment was replaced with an environment lookup; preserved Git history remains unchanged. The development notebook references its own local image path; it is exploratory evidence, not a portable executable workflow.

### Known limits and evidence

The original behavior is intentionally retained. In particular, first-frame unassigned possession can index an empty list; a terminal zero-duration speed window can divide by zero; optical-flow handling has stale-feature/status/cumulative-motion limitations; team assignment contains a clip-specific player-ID override; and video I/O assumes nonempty input. Full-frame lists are retained in memory. These are documented follow-up work, not silently claimed fixes.

See [RESUME_PROJECT_SUMMARY.md](RESUME_PROJECT_SUMMARY.md) for the full code audit, quantitative evidence, defensible resume bullets, and original-development opportunities. Detection accuracy, tracking accuracy, FPS/throughput, physical-measurement error, and performance improvements are **NOT VERIFIED**. The 24 fps constant is a timing assumption, not a throughput benchmark.

![Original upstream screenshot; not a newly reproduced result](output_videos/screenshot.png)
