import os
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load Model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path, task='detect')
    return model

# Streamlit UI
st.title("Surgical Tools Detection")
st.sidebar.header("Settings")

model_path = "./my_model.pt"
image_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp"])
video_file = st.sidebar.file_uploader("Upload Video", type=["avi", "mp4", "mov", "mkv"])
use_webcam = st.sidebar.checkbox("Use Webcam")
reference_size = st.sidebar.number_input("Reference Object Size (in cm)", min_value=1.0, value=5.0)

if model_path:
    model = load_model(model_path)
    labels = model.names
    min_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    resolution = st.sidebar.text_input("Resolution (WxH)", "640x480")

# Initialize Webcam or Video
cap = None  # Default to None

if use_webcam:
    cap = cv2.VideoCapture(0)  # Open webcam
elif video_file is not None:
    temp_video_path = f"./temp_video.{video_file.name.split('.')[-1]}"  # Create temp file
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())  # Save uploaded video
    cap = cv2.VideoCapture(temp_video_path)  # Load video

# Function to Run Inference
def run_inference(frame, reference_size):
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0
    bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
                   (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]

    pixel_per_cm = 10  

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'

            width_pixels = xmax - xmin
            height_pixels = ymax - ymin
            object_width_cm = width_pixels / pixel_per_cm
            object_height_cm = height_pixels / pixel_per_cm
            measurement_text = f"W: {object_width_cm:.2f} cm, H: {object_height_cm:.2f} cm"

            cv2.putText(frame, measurement_text, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            object_count += 1

    st.write(f"Detected {object_count} objects.")
    return frame

# Image Processing
if image_file:
    image = Image.open(image_file)
    frame = np.array(image)
    result_frame = run_inference(frame, reference_size)
    st.image(result_frame, channels="BGR", use_column_width=True)

# Video Processing
elif video_file:
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result_frame = run_inference(frame, reference_size)
        frame_placeholder.image(result_frame, channels="BGR", use_column_width=True)

    cap.release()
    os.remove(temp_video_path)  # Delete temp file after processing

# Webcam Streaming
elif use_webcam:
    frame_placeholder = st.empty()  # Placeholder for dynamic updates

    while use_webcam:
        ret, frame = cap.read()
        if not ret:
            st.write("Unable to fetch frame from webcam.")
            break

        result_frame = run_inference(frame, reference_size)
        frame_placeholder.image(result_frame, channels="BGR", use_column_width=True)

if cap:
    cap.release()
