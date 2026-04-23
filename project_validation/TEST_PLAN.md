# Project Validation Plan

This folder is the repeatable validation layer for the MOT16 project.

## Goal

Confirm that the current practical pipeline is ready to demo and ready to reference in the report:

- detector: `yolo26n.pt` by default, or another selected YOLO checkpoint
- tracker: `BoT-SORT` with `code/botsort_mot16_person.yaml`
- evaluation: `tracking_eval/run_mot16_tracker_report.py`
- demo app: `streamlit_app/app.py`

## Validation Cases

### 1. Environment and asset readiness

Checks:
- required Python packages import correctly
- model file exists
- tracker config exists
- MOT16 validation sequence exists

### 2. Tracker configuration sanity

Checks:
- tracker config contains `tracker_type: botsort`
- tracker config keeps `with_reid: True`

### 3. Single-frame model integration smoke test

Checks:
- load `yolo26n.pt`
- run tracking on one real MOT16 frame
- verify annotated output shape matches input
- verify at least one pedestrian track is returned

### 4. Sequence-level tracker smoke test

Checks:
- run tracking for a short real sequence slice on `MOT16-11`
- write preview MP4 and MOT-format TXT
- compute MOT metrics on the evaluated frames

### 5. Report-generation smoke test

Checks:
- invoke `tracking_eval/run_mot16_tracker_report.py` as a CLI command on a short frame budget
- verify `report.md`, `per_sequence_metrics.csv`, and `overall_metrics.json`

### 6. Existing full-report readiness check

Checks:
- locate the best full held-out report already produced on `MOT16-11` and `MOT16-13`
- verify key metrics clear a practical readiness floor

Current acceptance floor:
- `mota >= 0.35`
- `idf1 >= 0.45`
- `precision >= 0.85`

### 7. Streamlit live/offline helper smoke test

Checks:
- run `track_single_frame()` from the app code
- create a `LiveSessionRecorder`
- write one annotated live frame to MP4 and MOT TXT

### 8. Streamlit app boot check

Checks:
- launch the app headlessly
- verify Streamlit announces a local URL

## Output

Running the validation suite writes:

- `project_validation/runs/<timestamp>_validation/validation_report.json`
- `project_validation/runs/<timestamp>_validation/validation_report.md`

These files should be rerun after major app, tracker, or model-selection changes.
