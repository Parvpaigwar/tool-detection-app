import os
import time
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image


@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path, task='detect')
    return model


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

if use_webcam:
    cap = cv2.VideoCapture(0)
else:
    if image_file is not None:
        image = Image.open(image_file)
        frame = np.array(image)
    elif video_file is not None:
        cap = cv2.VideoCapture(video_file)

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

if image_file:
    result_frame = run_inference(frame, reference_size)
    st.image(result_frame, channels="BGR", use_column_width=True)
elif video_file:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result_frame = run_inference(frame, reference_size)
        st.image(result_frame, channels="BGR", use_column_width=True)
        time.sleep(0.1)  
elif use_webcam:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Unable to fetch frame from webcam.")
            break

        result_frame = run_inference(frame, reference_size)

        st.image(result_frame, channels="BGR", use_column_width=True)
        time.sleep(0.1)  

if use_webcam:
    cap.release()
