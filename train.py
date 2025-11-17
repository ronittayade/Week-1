from ultralytics import YOLO

# Path to the dataset yaml file
DATASET_PATH = "data.yaml"

# Load pretrained YOLO model (auto-downloads yolov8n.pt if not available)
model = YOLO("yolov8n.pt")

# Start Training
model.train(
    data=DATASET_PATH,
    epochs=1,      # Reduce from 30 → 10
    imgsz=416,      # Reduce resolution from 640 → 416
    batch=4,        # Smaller batch uses less processing
    project="PATH WHERE YOU WANT TO SAVE THE WEIGHTS",
    name="marine_plastic_fast",
)

print("\n======================================")
print(" 🎉 Training Completed Successfully!")
print(" Model saved under: runs/marine_plastic_model/weights/best.pt")
print("======================================")
