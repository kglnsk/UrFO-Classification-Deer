import streamlit as st
import zipfile
import os
import shutil
import tempfile
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load your YOLOv8 model
model = YOLO('det.pt',task = 'detect')

# Function to perform detection and save images by class
def detect_and_save_by_class(image, output_dir):
    results = model(image,augment = True, conf = 0.3)
    for result in results:
        img = result.orig_img
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Draw the bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 10)
            cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

            # Save the image in the corresponding class folder
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            file_path = os.path.join(class_dir, f'{len(os.listdir(class_dir)) + 1}.jpg')
            cv2.imwrite(file_path, img)
    return img

# Streamlit UI
st.title("Классификация парнокопытных ULTRA")

uploaded_file = st.file_uploader("Загрузите архив или изображение", type=["zip", "jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmp_dir:
        if uploaded_file.name.endswith('.zip'):
            # Handle ZIP archive
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
            file_list = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        else:
            # Handle single image
            file_list = [os.path.join(tmp_dir, uploaded_file.name)]
            with open(file_list[0], 'wb') as out_file:
                out_file.write(uploaded_file.getbuffer())

        output_dir = os.path.join(tmp_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        option = st.selectbox("Можете исправить класс",
    ("Кабарга", "Косуля", "Олень"))


        # Process each image
        for file_path in file_list:
            image = cv2.imread(file_path)
            detected_img = detect_and_save_by_class(image, output_dir)

            # Display the image with detection marks
            st.image(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB), caption=f'Detected: {os.path.basename(file_path)}', use_column_width=True)

        # Zip the output directory for download
        output_zip_path = os.path.join(tmp_dir, 'detected_images.zip')
        shutil.make_archive(output_zip_path.replace('.zip', ''), 'zip', output_dir)

        with open(output_zip_path, 'rb') as f:
            st.download_button('Download Detected Images', f, file_name='detected_images.zip')
