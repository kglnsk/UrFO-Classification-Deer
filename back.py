import os
import zipfile
import shutil
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import pandas as pd
import tempfile

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your YOLOv8 model
model = YOLO('yolov8_model.pt')

# Function to perform detection and get the class with the highest confidence
def detect_highest_conf_class(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    highest_confidence = 0
    highest_class = None

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            if confidence > highest_confidence:
                highest_confidence = confidence
                highest_class = class_id

    return highest_class

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if file.content_type not in ["application/zip", "image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    with tempfile.TemporaryDirectory() as tmp_dir:
        if file.content_type == "application/zip":
            # Handle ZIP archive
            file_path = os.path.join(tmp_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
            file_list = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        else:
            # Handle single image
            file_path = os.path.join(tmp_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            file_list = [file_path]

        # Store results
        data = []

        # Process each image
        for image_file in file_list:
            highest_class = detect_highest_conf_class(image_file)
            if highest_class is not None:
                data.append({'img_name': os.path.basename(image_file), 'class': highest_class})

        return JSONResponse(content=data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
