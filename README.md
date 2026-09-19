# PitchSense

A reconstructed football-video analysis project: YOLO detections, ByteTrack player/referee tracking, shirt-color team assignment, ball-possession heuristics, optical-flow camera movement, perspective mapping, and speed/distance overlays.

## Attribution and project status

PitchSense was originally inspired by and reconstructed from [Abdullah Tarek's football_analysis](https://github.com/abdullahtarek/football_analysis). It is now being independently extended and engineered by Sathvick Sudarsan. The complete upstream Git history is preserved; the upstream implementation is Abdullah Tarek's work, not an original implementation by this repository's maintainer. This first reconstruction checkpoint adds environment documentation, small maintenance changes, and synthetic tests; it does not claim a new computer-vision method or improved model performance.

Upstream baseline: `e4799632cdf271cf57f4eb0c4d872cf1b7ab0f17`. No LICENSE or COPYING file exists in the fetched upstream history. No replacement license has been invented; existing files and attribution remain intact.

## Setup (Windows, CPU)

Run commands from the repository root. Python 3.11 is the reconstruction target, not a claim about the original author's environment. Install [uv](https://docs.astral.sh/uv/) and use the fully pinned dependency list:

```powershell
uv venv --python 3.11
uv pip sync requirements.txt --python .venv/Scripts/python.exe --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
$env:YOLO_CONFIG_DIR = "$PWD/.runtime/ultralytics"
$env:MPLCONFIGDIR = "$PWD/.runtime/matplotlib"
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

`requirements.in` records direct runtime choices; `requirements.txt` pins transitive dependencies for the tested Windows/Python 3.11 CPU environment. PyTorch and torchvision use CPU wheels from the official PyTorch index. Supervision 0.26.1 and Ultralytics share the single `opencv-python` distribution; do not also install a headless/contrib OpenCV wheel into this environment. Other operating systems, GPU wheels, and historical notebook environments have not been verified. To intentionally refresh the lock:

```powershell
uv pip compile requirements.in --python .venv/Scripts/python.exe --output-file requirements.txt --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

## Input assets and running the original pipeline

The Git repository does **not** contain the trained checkpoint or input video. The following are the original author's links, retained as provenance; their bytes, availability, and model identity have not been verified in this reconstruction:

- [Upstream trained checkpoint](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view): save as `models/best.pt`.
- [Upstream sample video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view): save as `input_videos/08fd33_4.mp4`.

Use only trusted model/checkpoint and pickle files. For an actual run, first obtain and verify these assets, then recompute the cached results: change **both** `read_from_stub=True` arguments in `main.py` to `False`. The tracked upstream stubs are clip-specific, are not bound to an input hash, and were not deserialized during this audit. Leaving the original defaults unchanged reuses them and does not demonstrate model inference.

```powershell
.venv/Scripts/python.exe main.py
```

Output: `output_videos/output_video.avi`. The original entry point assumes its specific clip, 24 frames/second, fixed camera-feature strips, and a manually calibrated pitch quadrilateral. A different video requires real calibration and code changes. Missing assets, unsuitable clips, or the documented edge cases can prevent a complete run. Dependency installation and synthetic checks are reproducible; end-to-end football inference and exact output reproduction are **NOT VERIFIED**. K-means is unseeded, so output may vary even for identical inputs.

## Repository map and processing order

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
| `tests/` | Synthetic algorithm smoke checks without trained weights/video |

## Historical notebooks

The training notebook specifies `yolov5x.pt`, 100 epochs and image size 640, but its training cell has no saved output. This is configuration evidence, not proof of a completed training run or the identity of `best.pt`. It references Roboflow project `football-players-detection-3zvbc`, version 1. Its data download and directory moves depend on an external account/dataset and are not idempotent.

Notebook-only packages (`roboflow`, Jupyter) are intentionally outside the runtime lock. Use a separate environment if revisiting training and set `ROBOFLOW_API_KEY` to your own credential. The credential labeled revoked in an upstream comment was replaced with an environment lookup; preserved Git history remains unchanged. The development notebook references its own local image path; it is exploratory evidence, not a portable executable workflow.

## Known limits and evidence

The original behavior is intentionally retained. In particular, first-frame unassigned possession can index an empty list; a terminal zero-duration speed window can divide by zero; optical-flow handling has stale-feature/status/cumulative-motion limitations; team assignment contains a clip-specific player-ID override; and video I/O assumes nonempty input. Full-frame lists are retained in memory. These are documented follow-up work, not silently claimed fixes.

See [RESUME_PROJECT_SUMMARY.md](RESUME_PROJECT_SUMMARY.md) for the full code audit, quantitative evidence, defensible resume bullets, and original-development opportunities. Detection accuracy, tracking accuracy, FPS/throughput, physical-measurement error, and performance improvements are **NOT VERIFIED**. The 24 fps constant is a timing assumption, not a throughput benchmark.

![Original upstream screenshot; not a newly reproduced result](output_videos/screenshot.png)
