# MOT16 Streamlit Tracking App

This app runs people tracking with your local YOLO model and tracker settings, then exports:

- tracked video (`.mp4`)
- MOT-format track file (`.txt`)

It is isolated in one folder so your current notebooks/code stay unchanged.

When the app opens, it now also surfaces:

- the best full held-out MOT16 report already present in the repo
- the latest validation summary
- quick access to the current showcase preview videos and report artifacts

## Folder

- `streamlit_app/app.py` - main Streamlit UI and tracking pipeline
- `streamlit_app/requirements.txt` - app dependencies
- `streamlit_app/runs/` - generated outputs (created at runtime)

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

## Uses your current project assets

- Default tracker config: `code/botsort_mot16_person.yaml`
- Model list is auto-detected from:
  - project root `*.pt`
  - `code/*.pt`

So your local files like `yolo26n.pt`, `yolo26m.pt`, and `code/yolo26n.pt` are available in the UI.

## Live mode notes

- This app is built for pedestrian tracking, so keeping `Person class only` enabled is the right default for this project.
- After changing live settings such as model, tracker thresholds, or recording FPS, click `Prepare New Live Session` before starting the webcam stream.
- Stop the webcam stream to finalize the recorded `.mp4` and make it downloadable.
- The live session also writes a MOT-format `.txt` file so the same app can support demo videos and report artifacts.

## Run

From project root:

```bash
python -m pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Or with your existing venv:

```bash
source .venv/bin/activate
python -m pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Recommended: use a dedicated virtual environment for the Streamlit app if you want to keep notebook dependency versions unchanged.

## Notes

- `Person class only` keeps class `0` (person) while tracking.
- `Limit number of frames` helps for quick tests and faster debugging.
- progress updates now reflect the expected frame budget when the source length is known
- `streamlit-webrtc` is used for the browser webcam stream and live annotation path.
- Track file format is MOTChallenge-compatible with 10 columns:
  `frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z`
  where `x,y,z = -1`.
