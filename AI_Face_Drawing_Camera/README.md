# AI Face Drawing Studio

A local webcam project using Python, Flask, OpenCV and MediaPipe.

## Features
- Live webcam feed
- MediaPipe face landmark detection
- Pencil sketch effect
- Black-and-white line-art effect
- Cartoon effect
- Digital painting effect
- Face mesh, glasses, crown and sparkle overlays
- Browser controls to change style and overlay

## Recommended setup
Use Python 3.11 (64-bit). This requirements file pins compatible package versions.

### Windows / VS Code PowerShell

Open this folder in VS Code and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Then open `http://127.0.0.1:5000` in Chrome or Edge.

## Troubleshooting
- Close other applications using the webcam.
- Windows: Settings > Privacy & security > Camera; enable camera access and desktop-app access.
- If the default camera is not found, change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `app.py`.
- Confirm VS Code is using the `.venv` interpreter.
- Stop the server with Ctrl+C.

## How it works
MediaPipe Face Mesh estimates facial landmarks. OpenCV applies image filters to the camera frame and draws optional overlays using the landmark coordinates. Flask streams the processed frames to the local browser.

## Important distinction
The pencil, line-art, cartoon and digital-painting styles in this starter project are computer-vision image-processing effects, not a cloud generative-AI model. Webcam frames are processed locally by the Python app; do not expose this development server publicly.
