from ultralytics import YOLO
import os

# ---- Trained model path (update if needed) ----
MODEL_PATH = "best.pt"

# ---- Test images folder ----
TEST_FOLDER = "PATH OF TEST FOLDER"

# ---- Output folder for prediction results ----
OUTPUT_FOLDER = "PATH WHERE OUTPUT WILL BE SAVED"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Run prediction on all images in the folder
results = model.predict(
    source=TEST_FOLDER,
    save=True,
    conf=0.25,
    project=OUTPUT_FOLDER,
    name="inference_results"
)

print("\n🔥 Prediction completed for ALL test images!")
print(f"📁 Results saved here: {OUTPUT_FOLDER}\\inference_results")
