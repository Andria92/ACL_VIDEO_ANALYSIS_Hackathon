# ACL Movement Explorer

Fresh hackathon prototype for exploring markerless 2D pose feasibility around documented ACL injury events in professional women's football.

This project is exploratory. It does not diagnose ACL injuries, predict ACL risk, estimate forces, infer true 3D joint angles, or classify safe/dangerous movement. Milestone 1 focuses only on whether stable, auditable whole-body 2D coordinates can be recovered from short clips.

## Milestone 1 Scope

Implemented foundation:

- typed case, video, annotation, pose, and quality models;
- replaceable `PoseBackend` contract;
- YOLO pose backend behind that contract;
- MediaPipe Pose Landmarker backend retained, but currently blocked in this desktop runtime by a native MediaPipe Metal service abort;
- manual static ROI support for target-athlete crops;
- raw frame-by-landmark coordinate export;
- skeleton overlay rendering;
- raw coordinate diagnostic plots;
- pytest coverage for the introduced core code.

Not implemented yet:

- ACL similarity, clustering, prediction, risk scores, automatic injury detection, 3D biomechanics, forces, torques, joint moments, or conditional biomechanical labels.

## Local Data

Do not commit copyrighted footage. Put local clips here:

```text
data/videos/
```

Generated outputs are written under:

```text
data/pose/
data/overlays/
data/diagnostics/
```

These directories are ignored except for their `.gitkeep` placeholders.

## Environment

Python 3.12 is the intended runtime.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

MediaPipe was attempted first, as preferred, but the installed MediaPipe Tasks runtime aborts natively during Pose Landmarker creation on this machine:

```text
DrishtiMetalHelper initWithCalculatorContext
Check failed: service_ Service is unavailable.
```

Because that abort happens in native code, Python cannot catch it safely. Milestone 1 therefore uses a YOLO pose fallback for runnable feasibility work. YOLO pose provides shoulders, elbows, wrists, hips, knees, and ankles, but not MediaPipe's heel or foot-index landmarks.

The MediaPipe backend remains in the package with an explicit model path for environments where the Tasks runtime is usable.

Download the default lite model:

```bash
python scripts/download_mediapipe_pose_model.py
```

By default, scripts look for:

```text
data/models/pose_landmarker_lite.task
```

Download the default YOLO pose fallback model:

```bash
python scripts/download_yolo_pose_model.py
```

By default, scripts look for:

```text
data/models/yolov8n-pose.pt
```

## Raw Pose Extraction

Example with a manually supplied target ROI:

```bash
python scripts/extract_pose.py \
  --video data/videos/test.mp4 \
  --backend yolo \
  --case-id TEST_CASE \
  --source-id TEST_VIEW_01 \
  --start-frame 0 \
  --end-frame 180 \
  --roi 320,120,260,520 \
  --model-path data/models/yolov8n-pose.pt \
  --output data/pose/test_raw.parquet
```

ROI format is:

```text
x,y,width,height
```

Coordinates are stored in long tabular format, one row per frame and landmark.

## Skeleton Overlay

```bash
python scripts/render_overlay.py \
  --video data/videos/test.mp4 \
  --pose data/pose/test_raw.parquet \
  --roi 320,120,260,520 \
  --output data/overlays/test_overlay.mp4
```

The overlay is for visual quality control: target bounding box, pose skeleton, landmark points, frame index, and timestamp.

## Coordinate Diagnostics

```bash
python scripts/plot_pose_diagnostics.py \
  --pose data/pose/test_raw.parquet \
  --output data/diagnostics/test_joint_trajectories.png
```

These plots are raw coordinate diagnostics only. They are not biomechanical feature results.

## Tests

```bash
pytest
```

## Human Annotation UI

Milestone 5.5 adds a local browser UI for the researcher to create human target-athlete ROI
keyframes and a Movement Window without editing CSV files by hand.

Launch:

```bash
python scripts/run_annotation_ui.py
```

Then open:

```text
http://127.0.0.1:8765
```

## Video Review Cutter

Launch the local review-and-cut app:

```bash
python scripts/run_video_cutter_ui.py
```

Then open:

```text
http://127.0.0.1:8770
```

The app scans `data/videos/` and the local `injury_videos` folder by default. You can also paste
a local video path in the browser, mark In and Out while reviewing playback, then export a new
clip under:

```text
data/videos/cuts/
```

Use `--video-root` to scan a different folder:

```bash
python scripts/run_video_cutter_ui.py --video-root /path/to/videos
```

The UI supports Christen Press and Ellie Carpenter as the first validation cases. Draw a box
around the documented injured athlete, add correction keyframes when the propagated ROI drifts
or clips visible limbs, mark Movement End, review the propagated boxes, and save.

Movement Start is inferred from the first manual ROI keyframe. Movement End is the frame where
the visible movement sequence has effectively finished. The operator does not need to identify an
ACL rupture frame, critical plant, initial contact, or injury instant.

Human annotations are written separately under:

```text
data/annotations/human/
```

The direct pipeline inputs are:

```text
*_target_roi_human.csv
*_movement_window_human.json
*_event_annotation_human.json
```

The event annotation JSON is retained as an internal compatibility file for timing code. For human
Movement Window runs, 0 ms means Movement End, not an ACL rupture or biomechanical event.

The full resumable UI session is stored as:

```text
*_annotation_session_human.json
```

Development annotations are not loaded into the annotation canvas by default. Comparison with
development annotations is available only after a human annotation has been saved, and measures
annotation agreement rather than biomechanical validity.

## First Human Results View

Once a human-run Movement Profile exists, the same local website exposes a simple Results view:

```text
http://127.0.0.1:8765/results?case=christen_press
```

The Christen Press Results view consumes only HUMAN namespace outputs. It shows the analysed
Movement Window, synchronized video and movement graphs, feature evidence cards, exact
source-frame traceability, and quality limitations. Cross-case similarity, UMAP, clustering, and
association rules remain unavailable until additional human-validated cases exist.

Milestone 5.7 adds a deterministic semantic movement layer on top of the same human outputs.
It produces user-facing MovementObservation records for movement path, hip-knee-ankle chain,
knee-to-ankle relationship, trunk/pelvis, upper body, bilateral limb relationship, timing, and
evidence. These observations remain projected, non-clinical, and traceable to source frames.

Generated semantic artifacts include:

```text
data/semantic/human/christen_press_movement_observations.json
data/path/human/christen_press_projected_movement_path.parquet
data/quality/human/christen_press_path_quality_summary.json
```

The semantic layer does not create opponent features, contact/support-foot cues, ankle angles
from COCO-17, true speed in m/s, knee-flexion labels without view validation, movement
archetypes, clustering, UMAP, association rules, or clinical/predictive conclusions.

Milestone 5.8 adds a phase-based Movement Story layer for the same HUMAN Christen Press
Movement Window. This is within-case temporal segmentation only: phases are contiguous
intervals in one observable sequence, not cross-case movement archetypes or injury-mechanism
classes. The phase engine uses quality-filtered projected geometry, supported robust dynamics,
and camera-compensated path descriptors, then builds a missing-aware multivariate movement
change score. Unsupported values are ignored rather than treated as zero movement.

Generated phase artifacts include:

```text
data/phases/human/christen_press_movement_phases.json
data/phases/human/christen_press_phase_frame_map.parquet
data/phases/human/christen_press_movement_change_score.parquet
data/phases/human/christen_press_phase_transitions.csv
data/diagnostics/human/christen_press_human_multivariate_change_score.png
data/diagnostics/human/christen_press_human_phase_timeline.png
```

The Results view now leads with one active scope, the movement video, the phase timeline,
and a selected-scope Movement Story. A compact metadata row and compact evidence row replace
the earlier dashboard-style header. Frame inspection, trajectory graphs, statistics, and
technical evidence remain available as progressive drill-down views.

Milestone 5.9 extends that story with conservative hierarchical phase refinement and a
multiscale metric explorer. Long phases are reviewed locally with within-phase robust
standardization; duration alone does not create a boundary. For Christen Press, the original
long final phase is split only where local supported evidence identifies sustained internal
movement changes.

The Results view now supports:

```text
WHOLE_MOVEMENT
PHASE
FIVE_FRAME_WINDOW
SINGLE_FRAME
```

Selecting a phase creates a bounded phase-subclip experience with Play Phase, Replay Phase,
Pause, Play Full Movement, and phase-clamped +/-1 and +/-5 frame navigation. The default
phase story shows start/mid/end video snapshots with pose overlay and only the most salient
supported movement families for that phase. Explore this movement exposes supporting
plain-language observations by movement concept. Research measurements keeps every useful
numeric HUMAN Press metric visualisable, chooses an appropriate visualization family, and
reports statistics using supported values only. Unsupported samples remain gaps and do not
enter mean, median, standard deviation, min/max, range, start/end, or change summaries.

Angular research measurements now have a dedicated second layer rather than living in the
default Movement Story. Supported degree-based metrics can be inspected over the whole
analysed subclip with phase boundaries overlaid, switched into individual phase scope, and
viewed as absolute angle or one of three explicit change modes: change from movement start,
change from phase start, or frame-to-frame change. Descriptive statistics include mean,
median, SD, min/max, Q1, Q3, IQR, start/end, signed change, absolute change, total absolute
change, and key frames. Phase bar summaries and a compact angular feature x phase heatmap
support deeper inspection without turning the default page into a metric wall.

Frame inspection also includes an optional target segmentation mask overlay. The current
implementation uses the human ROI as a target seed for an OpenCV GrabCut mask and saves
human positive-target / negative-opponent point prompts for refinement. This is a local,
human-correctable target-mask path; it does not claim SAM 2 quality or treat a rectangle ROI
as proof that every pixel or joint inside the crop belongs to the target.

Future cohort comparison hooks are architectural only. The UI and payload use the phrase
"ACL case-library reference" for future aggregate comparisons and do not run similarity, UMAP,
clustering, association rules, or archetype assignment.

## Scientific Boundary

Outputs describe observable image-plane geometry from markerless pose estimates. Unavailable measurements should remain unavailable rather than being filled or inferred. Raw pose rows are never overwritten by smoothing or later processing.
