# Project Validation Plan

This folder is the repeatable validation layer for the MOT16 class project.

## Goal

Confirm that the current best practical pipeline is ready to demo and ready to reference in the report:

- detector: `yolo26n.pt`
- tracker: `BoT-SORT` with `code/botsort_mot16_person.yaml`
- evaluation: `tracking_eval/run_mot16_tracker_report.py`
- demo app: `streamlit_app/app.py`

## Validation Cases

### 1. Environment and asset readiness

Checks:
- required Python packages import correctly
- best model file exists
- tracker config exists
- MOT16 validation sequence exists

Why it matters:
- avoids spending time debugging missing files or broken environments during demo week

Pass criteria:
- all required imports succeed
- all required files and folders exist

### 2. Tracker configuration sanity

Checks:
- tracker config contains `tracker_type: botsort`
- tracker config keeps `with_reid: True`

Why it matters:
- this project depends on ID stability, so the app and report should use the same tracker assumptions

Pass criteria:
- both settings are present in the active config file

### 3. Single-frame model integration smoke test

Checks:
- load `yolo26n.pt`
- run tracking on one real MOT16 frame
- verify annotated output shape matches input
- verify at least one pedestrian track is returned

Why it matters:
- proves the selected model and tracker can run locally in the current environment

Pass criteria:
- no runtime error
- at least one valid track ID or detection is produced on the test frame

### 4. Sequence-level tracker smoke test

Checks:
- run tracking for a short real sequence slice on `MOT16-11`
- write preview MP4 and MOT-format TXT
- compute MOT metrics on the evaluated frames

Why it matters:
- proves the tracker, annotation, MOT export, and evaluator work together end to end

Pass criteria:
- output artifacts exist
- processed frame count matches requested limit
- metrics are numerically valid

### 5. Report-generation smoke test

Checks:
- invoke `tracking_eval/run_mot16_tracker_report.py` as a CLI command on a short frame budget
- verify `report.md`, `per_sequence_metrics.csv`, and `overall_metrics.json`

Why it matters:
- this is the exact path used to create report-ready evidence

Pass criteria:
- CLI exits successfully
- all expected files are created and parse correctly

### 6. Existing full-report readiness check

Checks:
- locate the best full held-out report already produced on `MOT16-11` and `MOT16-13`
- verify key metrics clear a practical readiness floor

Why it matters:
- smoke tests prove plumbing; the full report proves we already have showcase-worthy results

Current acceptance floor:
- `mota >= 0.35`
- `idf1 >= 0.45`
- `precision >= 0.85`

### 7. Streamlit live/offline helper smoke test

Checks:
- run `track_single_frame()` from the app code
- create a `LiveSessionRecorder`
- write one annotated live frame to MP4 and MOT TXT

Why it matters:
- proves the app can reuse the tracker pipeline correctly and record outputs

Pass criteria:
- annotated frame is generated
- live MP4 and live TXT are created

### 8. Streamlit app boot check

Checks:
- launch the app headlessly
- verify Streamlit announces a local URL

Why it matters:
- confirms the app is at least bootable before browser-level testing

Pass criteria:
- process starts successfully and exposes a local URL

## Output

Running the validation suite writes:

- `project_validation/runs/<timestamp>_validation/validation_report.json`
- `project_validation/runs/<timestamp>_validation/validation_report.md`

These files are meant to be kept as project evidence and rerun after major changes.
