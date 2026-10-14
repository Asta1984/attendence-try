import time

import cv2
import streamlit as st

from detector import YOLOVideoDetector

st.set_page_config(page_title="YOLO Live Detection", layout="wide")
st.title("YOLO Live Object Detection")

# ---------- sidebar controls ----------
st.sidebar.header("Settings")
source_type = st.sidebar.radio("Video source", ["Webcam", "Video file / URL / RTSP"])
if source_type == "Webcam":
    source = st.sidebar.number_input("Camera index", min_value=0, value=0, step=1)
else:
    source = st.sidebar.text_input("Path or stream URL", value="")

model_name = st.sidebar.selectbox(
    "YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt"]
)
conf = st.sidebar.slider("Confidence threshold", 0.05, 1.0, 0.25, 0.05)
iou = st.sidebar.slider("IoU threshold", 0.05, 1.0, 0.45, 0.05)
img_size = st.sidebar.select_slider("Inference size", options=[320, 416, 512, 640], value=640)

# ---------- session state ----------
if "detector" not in st.session_state:
    st.session_state.detector = None
if "running" not in st.session_state:
    st.session_state.running = False

col_start, col_stop = st.columns(2)
start_clicked = col_start.button("▶ Start", type="primary", use_container_width=True)
stop_clicked = col_stop.button("■ Stop", use_container_width=True)

if start_clicked and not st.session_state.running:
    try:
        with st.spinner(f"Loading {model_name}..."):
            det = YOLOVideoDetector(model_name=model_name, source=source,
                                    conf=conf, iou=iou, img_size=img_size)
            det.start()
        st.session_state.detector = det
        st.session_state.running = True
    except Exception as e:
        st.error(f"Failed to start: {e}")

if stop_clicked and st.session_state.detector:
    st.session_state.detector.stop()
    st.session_state.detector = None
    st.session_state.running = False

# ---------- live display ----------
frame_slot = st.empty()
info_col, table_col = st.columns([1, 2])
fps_slot = info_col.empty()
counts_slot = info_col.empty()
table_slot = table_col.empty()

det = st.session_state.detector
if det is None:
    frame_slot.info("Press **Start** to begin detection.")
else:
    # Threshold changes apply on the next Start (the model is loaded once per run).
    while det.is_running:
        frame = det.get_frame()
        if frame is not None:
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                             channels="RGB", use_container_width=True)
        fps_slot.metric("FPS", f"{det.get_fps():.1f}")

        detections = det.get_detections()
        counts = {}
        for d in detections:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        counts_slot.write(counts if counts else "No objects detected")
        table_slot.dataframe(
            [{"label": d["label"], "confidence": round(d["confidence"], 3), "bbox": d["bbox"]}
             for d in detections],
            use_container_width=True,
        )
        time.sleep(0.03)

    # Loop ended (stream closed or stopped)
    st.session_state.running = False
    st.session_state.detector = None
    st.info("Stream ended.")