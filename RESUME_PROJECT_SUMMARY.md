# PitchSense — resume evidence and complete code audit

## Provenance and evidence boundary

This is a reconstruction/adaptation of [Abdullah Tarek's football_analysis](https://github.com/abdullahtarek/football_analysis), inspected at upstream commit `e4799632cdf271cf57f4eb0c4d872cf1b7ab0f17`. Local upstream history contains three commits, all attributed to Abdullah Tarek (`edee084`, `da873f1`, `e479963`). The user reports implementing this project previously and losing the code; this checkout cannot independently establish that history or authorship of the upstream implementation. Cite the upstream project and distinguish personal reconstruction, cleanup, testing, and future original contributions from its existing functionality.

No LICENSE or COPYING file is present in the inspected upstream tree or those available commits. A public repository is not evidence of an open-source license; permission and reuse terms remain unresolved. Do not describe the upstream code as MIT-licensed or invent license terms. The original README's model/video links are references, not locally verified artifacts.

Evidence labels below: **SOURCE VERIFIED** means inspected implementation/configuration, **SAVED OUTPUT ONLY** means historical notebook or image content, and **NOT VERIFIED** means not established by execution or evaluation here. Source locations refer to the upstream commit above, so subsequent cleanup may shift current line numbers. Notebook cell numbers count all cells, including markdown and empty cells, starting at 1.

## Project overview (4 sentences)

PitchSense reconstructs and adapts Abdullah Tarek's Python football-video analysis pipeline, combining Ultralytics YOLO detections with Supervision ByteTrack identities for players and referees. It estimates team membership through scikit-learn KMeans jersey-color clustering, fills gaps in ball bounding boxes with pandas, and uses a proximity heuristic to estimate possession. OpenCV sparse optical flow and a fixed four-point perspective transform feed approximate player speed and distance calculations, with annotated AVI output. The repository supports explaining these implemented techniques, but model quality, tracking accuracy, physical measurement accuracy, training completion, and runtime performance have not been verified in this reconstruction.

## Architecture and complete module inventory

`video → YOLO detections → ByteTrack player/referee IDs → image positions → camera adjustment → fixed homography → player speed/distance → team colors → ball proximity/possession → annotation → AVI`

| Source | Implementation and role | Evidence |
|---|---|---|
| `main.py:12-84` | Orchestrates the entire offline pipeline; reads the clip, initializes model, loads caches by default, computes positions and overlays, writes AVI. | SOURCE VERIFIED; end-to-end NOT VERIFIED |
| `utils/video_utils.py:3-18` | OpenCV video decode into a complete list of frames; XVID AVI encoding. | SOURCE VERIFIED |
| `trackers/tracker.py:12-104` | Ultralytics YOLO wrapper; batched detection; `sv.Detections.from_ultralytics`; `sv.ByteTrack`; goalkeeper class remapped to player; dictionaries per frame for players/referees/ball. | SOURCE VERIFIED |
| `trackers/tracker.py:17-38` | Bottom-center foot positions for people, box center for ball; pandas linear interpolation then backfill of ball box coordinates. | SOURCE VERIFIED |
| `trackers/tracker.py:106-217` | Ellipse/ID labels for players, referee ellipses, triangle markers for ball/carrier, cumulative team-control percentages. | SOURCE VERIFIED |
| `team_assigner/team_assigner.py:8-73` | Two-stage KMeans: top-half crop pixels into jersey/background, then player color centers into two teams; corner-majority background heuristic; cached team per track ID. | SOURCE VERIFIED |
| `player_ball_assigner/player_ball_assigner.py:9-27` | Selects nearest player using ball-center distance to either lower corner of player bounding box, within a pixel threshold. | SOURCE VERIFIED |
| `camera_movement_estimator/camera_movement_estimator.py:9-99` | Grayscale Shi–Tomasi features (`goodFeaturesToTrack`) in fixed image strips; pyramidal Lucas–Kanade flow; largest feature displacement as camera movement; position subtraction, optional pickle cache, overlay. | SOURCE VERIFIED |
| `view_transformer/view_transformer.py:4-45` | Fixed four source points mapped to a rectangular field region with `getPerspectiveTransform`; polygon inclusion gate; `perspectiveTransform` maps retained positions. | SOURCE VERIFIED |
| `speed_and_distance_estimator/speed_and_distance_estimator.py:6-73` | Endpoint displacement over frame windows, meters/second converted to km/h, distance accumulation by player ID; skips ball/referees; draws labels. | SOURCE VERIFIED; physical units are assumptions |
| `utils/bbox_utils.py:1-16` | Bounding-box center/width/foot-point and Euclidean/signed coordinate-distance helpers. | SOURCE VERIFIED |
| `yolo_inference.py:1-9` | Standalone model prediction with saved detection output and first-frame box printing; separate from analysis pipeline. | SOURCE VERIFIED |
| All package `__init__.py` files | Re-export their corresponding class/helper; no additional pipeline behavior. | SOURCE VERIFIED |
| `training/football_training_yolo_v5.ipynb` | Eleven cells: dependency install, Roboflow dataset download, local folder moves, YOLO training command; empty cells and section headings add no behavior. | SOURCE VERIFIED; download output is SAVED OUTPUT ONLY |
| `development_and_analysis/color_assignement.ipynb` | Fourteen cells: RGB player crop, top-half crop, two-cluster segmentation, corner-background selection, extracted jersey center. | SOURCE VERIFIED; figures and printed centers are SAVED OUTPUT ONLY |
| `stubs/*.pkl` | Two supplied caches are present; neither was deserialized during this audit. Their content, video correspondence, provenance, and validity are NOT VERIFIED. | Presence only |
| `output_videos/screenshot.png`, `output_videos/cropped_image.jpg` | Upstream example assets, not generated evidence of a successful reconstruction run. | SAVED OUTPUT ONLY |
| Model/video placeholder text files, upstream README and `.gitignore` | Documentation/placeholders and ignore rules; no trained weights or input video bundled. Original ignore pattern misspells `__pycache__`. | SOURCE VERIFIED |

The actual `main.py` order interpolates the ball **after** camera adjustment and homography. Interpolation reconstructs ball entries containing only `bbox`, dropping their derived position fields. Player speed estimation excludes ball data, so this does not provide ball speed. The ball has a constant dictionary key `1`; it is taken from untracked detections and is not a persistent ByteTrack ball identity. Multiple ball detections overwrite that key in detection iteration order.

## Exact technology and model evidence

Runtime imports verify Python, Ultralytics (`YOLO`), Supervision (`ByteTrack`, `Detections`), OpenCV (`cv2`), NumPy, pandas, and scikit-learn (`KMeans`). Standard-library `pickle`, `os`, and `sys` support caches and imports. Matplotlib is used by the color-development notebook; Roboflow is used by the training notebook. No reproducible dependency versions are established by the original source; a historical Python 3.8 path in a saved warning is not a supported-version declaration.

- **Inference model:** `models/best.pt` (`main.py:17`, `yolo_inference.py:3`). This is a local filename, not an independently verified architecture/version. The weights are unavailable locally.
- **Training command model:** exactly `yolov5x.pt` (training notebook cell 9), with `task=detect mode=train`. The upstream README calls its linked artifact “Trained Yolo v5.” Neither proves that the absent `best.pt` came from this command or verifies checkpoint compatibility with a newly installed Ultralytics version.
- **Dataset reference:** Roboflow workspace `roboflow-jvuqo`, project `football-players-detection-3zvbc`, version `1`, export request `yolov5` (training notebook cell 4). No local dataset, split manifest, annotations, licensing review, or dataset quality validation was available.
- The notebook originally contains a credential-like value accompanied by an upstream comment saying it is revoked. This audit did not use or repeat it. A comment is not independent verification of revocation; credentials should be supplied through the environment.

## Quantitative facts safe to discuss

These are configuration/code facts, **not measured achievements**.

| Fact | Source | Evidence status |
|---|---|---|
| 20-frame inference batches and detection confidence argument `0.1` | `trackers/tracker.py:40-45` | SOURCE VERIFIED; not throughput/accuracy |
| Three output groups: players, referees, ball; four expected class names including goalkeeper | `trackers/tracker.py:57-98` | SOURCE VERIFIED; model class mapping unverified |
| Two pixel clusters and two team clusters; `n_init=1` and `10` respectively | `team_assigner/team_assigner.py:13,50` | SOURCE VERIFIED |
| Ball ownership threshold is strictly less than 70 pixels | `player_ball_assigner/player_ball_assigner.py:7,22` | SOURCE VERIFIED; no possession accuracy |
| Up to 100 features; 15×15 flow window; pyramid max level 2; displacement threshold greater than 5 pixels | `camera_movement_estimator/camera_movement_estimator.py:11-28,70` | SOURCE VERIFIED |
| Four fixed homography correspondences; target dimensions 23.32 by 68, treated as meters | `view_transformer/view_transformer.py:6-24` | SOURCE VERIFIED; calibration accuracy NOT VERIFIED |
| Five-frame speed window, assumed 24 fps, conversion factor 3.6 | `speed_and_distance_estimator/speed_and_distance_estimator.py:8-9,31-34` | SOURCE VERIFIED; 24 fps is not measured processing speed |
| AVI writer also hardcodes 24 fps | `utils/video_utils.py:14-15` | SOURCE VERIFIED; not source FPS validation |
| Training command requests 100 epochs and 640 image size | Training notebook cell 9 | SOURCE VERIFIED; no saved training output or completion proof |
| Notebook download/extraction progress reports completion | Training notebook cell 4 outputs | SAVED OUTPUT ONLY; archive progress counts are not dataset image counts |
| Color notebook's sample player cluster index is 0, background index 1; saved RGB center approximately `[171.38,235.65,142.85]` | Color notebook cells 10–12 | SAVED OUTPUT ONLY; illustrative example, not validation |

No mAP, precision/recall, IDF1, HOTA, MOTA, ID-switch count, possession F1, speed error, latency, processing FPS, training duration, GPU configuration, dataset image total, or measured improvement is established. Do not add those to a resume. Do not claim real-time operation, production readiness, tactical event detection, player re-identification, a dashboard, or deployed service.

## Source audit: limitations and failure modes

These findings describe the **upstream baseline**, not a claim that each survives the reconstruction's separate cleanup. Check the final diff and test results before describing a fix as completed.

1. **Missing inputs prevent a complete run.** Hardcoded input/model paths and immediate `video_frames[0]` access require actual assets. Initializing `Tracker` loads weights even if cached tracks are requested. Empty video reads and writer failures are not handled (`main.py:14-29`; `utils/video_utils.py:3-18`).
2. **Unsafe and unvalidated caches.** `pickle.load` executes Python deserialization and is inappropriate for untrusted files. Cache reads do not verify frame count, video identity, dimensions, model revision, or schema (`trackers/tracker.py:48-53`; camera module lines 43–47). Audit did not load them.
3. **Camera estimation is fragile.** It uses one maximum-displacement feature rather than a robust consensus; ignores optical-flow status/error and missing-feature cases; samples only columns 0–19 and 900–1049; and does not update feature coordinates when motion stays below threshold. Its `old-new` displacement sign and subtraction require careful validation. Subtracting isolated frame movement is not a cumulative stabilization transform; rotations, zoom, cuts, and parallax are unmodeled (camera module lines 19–74; `utils/bbox_utils.py:11-12`).
4. **Calibration is clip-specific.** A single fixed homography assumes a planar field and selected correspondence geometry. Outside-polygon points become `None`; an integer-rounded inclusion test precedes the floating-point transform. Camera-adjusted points are tested against fixed image vertices. Camera changes and bad calibration propagate directly into speed/distance (`view_transformer/view_transformer.py:6-44`).
5. **Speed estimates have numerical and sampling gaps.** A terminal one-frame window can divide by zero when frame count is 1 modulo 5. Missing endpoint IDs skip a segment; ID switches reset or misattribute accumulated distance; straight endpoint displacement undercounts curved paths; window totals are stamped across preceding frames; final frame can lack annotation. Fixed FPS and calibration are unverified (speed module lines 18–48).
6. **Team assignment is heuristic.** It needs sufficient valid crops and players in frame zero; no bounds checks protect empty/truncated crops. Corner background, top-half clothing, two-team clustering, and stable ID assumptions can fail for goalkeepers, referees, occlusion, lighting, or similar kits. Runtime has no fixed random seed and includes a clip-specific `player_id == 91` override. The notebook uses RGB whereas runtime OpenCV colors remain BGR; do not interchange their reported values without conversion (team module lines 18–71; color notebook cells 2,9).
7. **Ball interpolation invents unobserved positions.** It fills missing coordinates without a maximum gap, confidence, motion model, or observed/interpolated flag. Leading gaps are backfilled; a completely absent ball is not recoverable. Ball boxes may be overwritten by later detections. No ball trajectory accuracy is measured (`trackers/tracker.py:28-38,93-98`).
8. **Possession is proximity, not a validated event label.** Pixel threshold varies in physical meaning with perspective; feet are approximated by lower box corners. Unknown frames inherit previous team ownership; the first unknown frame raises an index error. Thus displayed percentages include imputed ownership and are not match ground truth (`main.py:58-70`; player-ball module; tracker lines 174–179).
9. **Offline memory and drawing assumptions.** All decoded and annotated frames are retained; inference batching does not make the pipeline streaming. Possession overlay coordinates assume a large frame and can fall off smaller images. No FPS/latency or memory benchmark is recorded (`utils/video_utils.py`; tracker lines 167–217; camera drawing loop).
10. **Reproducibility gaps.** Training notebook installs unpinned packages, uses environment-specific folder moves, and has no training metrics or evaluation cell. No executable test suite exists in the upstream commit. Model/video download links, source licensing, dataset rights, and package/checkpoint compatibility need independent checking before redistribution or claiming a reproduced result.

## Four conservative resume bullets

Use these as reconstruction/adaptation bullets, after reviewing and being able to explain the final code. They intentionally attribute the inherited design and contain no unverified performance claims.

- Reconstructed and documented PitchSense from Abdullah Tarek's football-analysis project, preserving upstream history and tracing its YOLO, ByteTrack, team-clustering, and annotated-video pipeline.
- Added a pinned Python runtime environment and six passing synthetic smoke tests for bounding-box geometry, ball assignment, homography, speed/distance, ball interpolation, and synthetic detection-to-ByteTrack conversion.
- Audited the upstream OpenCV optical-flow and four-point homography pipeline, documenting camera, calibration, and frame-rate assumptions behind approximate physical measurements.
- Reviewed inherited inference and training workflows, distinguishing the notebook's `yolov5x.pt` training configuration from unavailable inference weights and documenting reproducibility and measurement limitations.

If historical personal implementation can later be substantiated, adjust wording to match that evidence. Do not turn inherited features into claims of original algorithm design, personal model training, or measured impact. The cleanup and tests below were prepared with coding-assistant help; use personal contribution claims only after reviewing and understanding them.

## Reconstruction changes and verification (September 19, 2026)

The reconstruction preserved all upstream commits and existing assets, named the source remote `upstream`, and created local branch `pitchsense/reconstruction`. It added attribution/setup documentation, Python 3.11 runtime pins (45 resolved packages), six synthetic unittest cases, and corrected generated-file ignore patterns. Dead `sys.path` mutations and an unused import were removed, the `sekf` parameter typo was renamed to `self`, comments were clarified, and the notebook's credential literal was replaced with `ROBOFLOW_API_KEY`. Model algorithms, thresholds, hard-coded calibration, and the limitations listed above remain unchanged.

Verified on Windows with CPython 3.11.15: `python -m unittest discover -s tests -v` passed all **6** tests; `uv pip check` found all **45** installed packages metadata-compatible; importing `main` and its runtime dependencies succeeded; both notebook files parsed as JSON; `git diff --check` passed. OpenCV imported as 4.10.0 and PyTorch as 2.4.1+cpu. These checks do not load a detector, consume the supplied pickle caches, train a model, or validate full-video output. End-to-end inference, exact output repeatability, and performance remain **NOT VERIFIED** pending trusted weights/input video and correction or validation of clip-specific assumptions.

## Interview preparation

1. What did you personally reconstruct or change, and what came directly from Abdullah Tarek's repository? Show the diff and attribution.
2. How do YOLO detection and ByteTrack association differ? Where does this code assign identities, and why is its ball key not a tracked identity?
3. Why use upper-body pixels, two KMeans stages, and corner labels? What breaks for similar jerseys or goalkeeper kits?
4. What information is lost when interpolating ball boxes? How would you represent long gaps and uncertainty instead of fabricated continuity?
5. Why is a 70-pixel possession rule perspective-dependent, and how would you evaluate it against annotated possession?
6. What does Lucas–Kanade estimate here? Explain motion sign, feature lifecycle, outliers, and why a cumulative robust transform would be different.
7. What assumptions make a homography appropriate? How do four correspondences and calibration errors affect metric distances?
8. Derive speed from displacement and elapsed time. Why are 24 fps and five-frame windows assumptions rather than achieved runtime performance?
9. How do missed detections, ID switches, camera cuts, and curved paths affect accumulated distance?
10. Why is a saved training command not proof of a trained/validated model? What artifacts establish checkpoint provenance and reproducibility?
11. How would you evaluate detection, association, possession, and physical estimates separately? Avoid answering with metrics this project has not measured.
12. What tests can run without model weights, and what must wait for a trusted clip/checkpoint? Why avoid deserializing untrusted pickle caches?

## High-value future original extensions (not implemented claims)

1. **Reproducible evaluation first:** create a small, permission-cleared annotated clip set with explicit train/validation/test provenance, then measure each stage independently. Publish raw counts and evaluation scripts before quoting improvement percentages.
2. **Camera/calibration correctness:** implement robust feature filtering and cumulative transforms with synthetic-motion tests; supply per-clip field correspondences and actual source FPS; evaluate metric error against known reference distances.
3. **Explicit uncertainty:** distinguish observed, interpolated, and unknown ball positions; cap interpolation gaps; report unknown possession separately; evaluate the resulting precision/coverage tradeoff.
4. **Original temporal team assignment:** replace track-specific overrides with confidence-weighted color observations over time and evaluate failures on goalkeepers, occlusion, and illumination changes.
5. **Streaming and trustworthy artifacts:** process bounded frame batches, preserve video metadata, use non-executable cache formats with video/model fingerprints, and measure peak memory and throughput on a declared machine.

Keep proposed work separate from delivered work. A narrow, tested extension with a reproducible result is stronger evidence of original contribution than new labels for inherited functionality.
