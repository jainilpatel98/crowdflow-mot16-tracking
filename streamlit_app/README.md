# MOT16 Streamlit Tracking App

This app runs people tracking with your local YOLO model and tracker settings, then exports:

- tracked video (`.mp4`)
- MOT-format track file (`.txt`)

It is isolated in one folder so the training and notebook code stays unchanged.

When the app opens, it also surfaces:

- the best full held-out MOT16 report found under `tracking_eval/runs/`
- the latest validation summary found under `project_validation/runs/`
- legacy laptop artifacts under `code_from_laptop/` if they are present

## Folder

- `streamlit_app/app.py` - main Streamlit UI and tracking pipeline
- `streamlit_app/requirements.txt` - app dependencies
- `streamlit_app/runs/` - generated outputs

## Supported inputs

1. `Live webcam` mode
- browser camera feed with live person tracking
- optional annotated MP4 recording
- MOT-format track ID export for the live session
- works on `localhost` or over `HTTPS`, because the browser camera API requires a secure context

2. `Upload video` mode
- upload `.mp4`, `.avi`, `.mov`, or `.mkv`
- app auto-detects source FPS

3. `MOT16 sequence` mode
- choose split (`train` or `test`)
- choose sequence (`MOT16-01`, `MOT16-12`, etc.)
- uses `seqinfo.ini` FPS for output timing

## Uses the active project assets

- Default tracker config: `code/botsort_mot16_person.yaml`
- Model list is auto-detected from:
  - project root `*.pt`
  - `code/*.pt`

This keeps the UI attached to the server repo instead of the archived laptop folder.

## Run

From project root:

```bash
.venv/bin/python -m pip install -r streamlit_app/requirements.txt
.venv/bin/python -m streamlit run streamlit_app/app.py
```

## Notes

- `Person class only` keeps class `0` (person) while tracking.
- `Limit number of frames` helps for quick tests and faster debugging.
- `streamlit-webrtc` is used for the browser webcam stream and live annotation path.
- Track file format is MOTChallenge-compatible with 10 columns:
  `frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z`
  where `x,y,z = -1`.
