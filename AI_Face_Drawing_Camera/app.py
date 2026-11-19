import atexit
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)

# -------------------- Camera and face tracker --------------------
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

camera_lock = threading.Lock()
settings_lock = threading.Lock()

selected_style = "original"
selected_overlay = "none"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55
)

STYLES = {"original", "pencil", "line_art", "cartoon", "digital_painting"}
OVERLAYS = {"none", "face_mesh", "glasses", "crown", "sparkles", "all"}


# -------------------- Artistic image styles --------------------
def pencil_sketch(frame):
    """Create a graphite-style sketch from the camera frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def line_art(frame):
    """Create a clean black-and-white contour drawing."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9, 9
    )
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def cartoon_style(frame):
    """Smooth color regions and retain strong outlines."""
    color = frame.copy()
    for _ in range(2):
        color = cv2.bilateralFilter(color, 9, 90, 90)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9, 7
    )
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges)


def digital_painting(frame):
    """A painterly effect made with OpenCV filters."""
    smooth = cv2.stylization(frame, sigma_s=60, sigma_r=0.45)
    # Add a subtle edge layer so facial details remain readable.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges = cv2.bitwise_not(edges)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(smooth, edges)


def apply_style(frame, style):
    if style == "pencil":
        return pencil_sketch(frame)
    if style == "line_art":
        return line_art(frame)
    if style == "cartoon":
        return cartoon_style(frame)
    if style == "digital_painting":
        return digital_painting(frame)
    return frame


# -------------------- Face landmark drawing helpers --------------------
def pixel_point(landmarks, index, width, height):
    lm = landmarks[index]
    return int(lm.x * width), int(lm.y * height)


def draw_mesh(frame, landmarks, width, height):
    for start, end in mp_face_mesh.FACEMESH_CONTOURS:
        p1 = pixel_point(landmarks, start, width, height)
        p2 = pixel_point(landmarks, end, width, height)
        cv2.line(frame, p1, p2, (80, 255, 210), 1, cv2.LINE_AA)


def draw_glasses(frame, landmarks, width, height):
    left_outer = pixel_point(landmarks, 33, width, height)
    left_inner = pixel_point(landmarks, 133, width, height)
    right_inner = pixel_point(landmarks, 362, width, height)
    right_outer = pixel_point(landmarks, 263, width, height)

    for a, b in ((left_outer, left_inner), (right_inner, right_outer)):
        cx = (a[0] + b[0]) // 2
        cy = (a[1] + b[1]) // 2
        rx = max(15, abs(b[0] - a[0]) // 2 + 9)
        ry = max(11, int(rx * 0.62))
        cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360,
                    (255, 190, 40), 3, cv2.LINE_AA)

    cv2.line(frame, left_inner, right_inner, (255, 190, 40), 3, cv2.LINE_AA)


def draw_crown(frame, landmarks, width, height):
    forehead = pixel_point(landmarks, 10, width, height)
    left = pixel_point(landmarks, 109, width, height)
    right = pixel_point(landmarks, 338, width, height)

    face_width = max(80, right[0] - left[0])
    crown_h = int(face_width * 0.34)
    base_y = forehead[1] + int(face_width * 0.06)
    half_w = int(face_width * 0.48)

    polygon = np.array([
        [forehead[0] - half_w, base_y],
        [forehead[0] - int(half_w * .85), base_y - crown_h],
        [forehead[0] - int(half_w * .35), base_y - int(crown_h * .48)],
        [forehead[0], base_y - int(crown_h * 1.12)],
        [forehead[0] + int(half_w * .35), base_y - int(crown_h * .48)],
        [forehead[0] + int(half_w * .85), base_y - crown_h],
        [forehead[0] + half_w, base_y]
    ], dtype=np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], (30, 190, 255))
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
    cv2.polylines(frame, [polygon], True, (30, 220, 255), 3, cv2.LINE_AA)


def draw_sparkles(frame, landmarks, width, height):
    for index in (50, 280, 205, 425):
        x, y = pixel_point(landmarks, index, width, height)
        size = 8
        color = (255, 255, 100)
        cv2.line(frame, (x - size, y), (x + size, y), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x, y - size), (x, y + size), color, 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)


def apply_overlay(frame, landmarks, overlay_name):
    if not landmarks or overlay_name == "none":
        return frame

    h, w = frame.shape[:2]
    if overlay_name in ("face_mesh", "all"):
        draw_mesh(frame, landmarks, w, h)
    if overlay_name in ("glasses", "all"):
        draw_glasses(frame, landmarks, w, h)
    if overlay_name in ("crown", "all"):
        draw_crown(frame, landmarks, w, h)
    if overlay_name in ("sparkles", "all"):
        draw_sparkles(frame, landmarks, w, h)
    return frame


# -------------------- Frame pipeline --------------------
def process_frame(frame):
    global selected_style, selected_overlay

    frame = cv2.flip(frame, 1)

    with settings_lock:
        style = selected_style
        overlay = selected_overlay

    # Find facial landmarks on the original camera frame.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    landmarks = None

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

    # Apply artistic effect, then add face-following drawings.
    output = apply_style(frame, style)
    if landmarks:
        output = apply_overlay(output, landmarks, overlay)

    h, w = output.shape[:2]
    label = f"STYLE: {style.replace('_', ' ').upper()}"
    cv2.putText(output, label, (16, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 255, 210), 2,
                cv2.LINE_AA)

    if not landmarks:
        cv2.putText(output, "Face not detected - face the camera",
                    (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (40, 210, 255), 2, cv2.LINE_AA)

    return output


def generate_frames():
    while True:
        with camera_lock:
            ok, frame = camera.read()

        if not ok:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera unavailable - check permissions",
                        (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)
        else:
            try:
                frame = process_frame(frame)
            except Exception as exc:
                print("Frame processing error:", exc)
                cv2.putText(frame, "Processing error - check terminal",
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")
        time.sleep(0.02)


# -------------------- Flask routes --------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/settings", methods=["POST"])
def update_settings():
    global selected_style, selected_overlay
    data = request.get_json(silent=True) or {}
    style = data.get("style", "original")
    overlay = data.get("overlay", "none")

    if style not in STYLES or overlay not in OVERLAYS:
        return jsonify({"error": "Invalid style or overlay"}), 400

    with settings_lock:
        selected_style = style
        selected_overlay = overlay

    return jsonify({"style": style, "overlay": overlay})


@app.route("/health")
def health():
    return jsonify({"camera_open": bool(camera.isOpened())})


def cleanup():
    with camera_lock:
        if camera.isOpened():
            camera.release()
    face_mesh.close()


atexit.register(cleanup)

if __name__ == "__main__":
    if not camera.isOpened():
        print("WARNING: Camera not detected. Check Windows camera permissions.")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
