#conversion coco to yolo

from pycocotools.coco import COCO
import os
from tqdm import tqdm
import shutil

# Where your COCO dataset exists (BePLi dataset)
TRAIN_JSON = r"C:\Users\CW\Desktop\pbl\waste\98753\plastic_coco\annotation\train.json"
VAL_JSON   = r"C:\Users\CW\Desktop\pbl\waste\98753\plastic_coco\annotation\val.json"

TRAIN_IMAGES = r"C:\Users\CW\Desktop\pbl\waste\98753\plastic_coco\images\train"
VAL_IMAGES   = r"C:\Users\CW\Desktop\pbl\waste\98753\plastic_coco\images\val"

# Output folder for YOLO dataset
OUTPUT_TRAIN = r"C:\Users\CW\Desktop\pbl\waste\yolomodel\dataset\train"
OUTPUT_VAL   = r"C:\Users\CW\Desktop\pbl\waste\yolomodel\dataset\val"

os.makedirs(OUTPUT_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_VAL, exist_ok=True)


def convert_annotations(json_path, image_folder, output_folder):
    coco = COCO(json_path)

    # Loop images
    for img_id in tqdm(coco.imgs, desc=f"Processing {os.path.basename(json_path)}"):
        img_info = coco.imgs[img_id]
        img_name = img_info['file_name']
        img_path_src = os.path.join(image_folder, img_name)
        img_path_dst = os.path.join(output_folder, img_name)

        # Copy image to YOLO folder
        if os.path.exists(img_path_src):
            shutil.copy(img_path_src, img_path_dst)
        else:
            print(f"⚠ WARNING: Missing image file {img_path_src}, skipping.")
            continue

        # Create label file
        label_path = os.path.join(output_folder, img_name.replace(".png", ".txt").replace(".jpg", ".txt"))

        with open(label_path, "w") as f:
            annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

            for ann in annotations:
                bbox = ann["bbox"]

                # Convert COCO bbox → YOLO format
                x, y, w, h = bbox
                cx = (x + w / 2) / img_info['width']
                cy = (y + h / 2) / img_info['height']
                ww = w / img_info['width']
                hh = h / img_info['height']

                # Since you have only ONE class → class ID = 0
                f.write(f"0 {cx} {cy} {ww} {hh}\n")

    print(f"\n✔ Conversion completed: {output_folder}")


# -----------------------------
# Run Conversions
# -----------------------------

convert_annotations(TRAIN_JSON, TRAIN_IMAGES, OUTPUT_TRAIN)
convert_annotations(VAL_JSON, VAL_IMAGES, OUTPUT_VAL)

print("\n🎉 All COCO annotations successfully converted to YOLO format!")

#training model



# Path to the dataset yaml file
DATASET_PATH = r"C:\Users\CW\Desktop\pbl\waste\yolomodel\dataset\data.yaml"

# Load pretrained YOLO model (auto-downloads yolov8n.pt if not available)
model = YOLO("yolov8n.pt")

# Start Training
model.train(
    data=DATASET_PATH,
    epochs=1,      # Reduce from 30 → 10
    imgsz=416,      # Reduce resolution from 640 → 416
    batch=4,        # Smaller batch uses less processing
    project=r"C:\Users\CW\Desktop\pbl\waste\yolomodel\runs",
    name="marine_plastic_fast",
)

print("\n======================================")
print(" 🎉 Training Completed Successfully!")
print(" Model saved under: runs/marine_plastic_model/weights/best.pt")
print("======================================")

#testing 

# ---- Trained model path (update if needed) ----
MODEL_PATH = r"C:\Users\CW\Desktop\pbl\waste\yolomodel\runs\test_training_run\weights\best.pt"

# ---- Test images folder ----
TEST_FOLDER = r"C:\Users\CW\Desktop\pbl\waste\yolomodel\images\test"

# ---- Output folder for prediction results ----
OUTPUT_FOLDER = r"C:\Users\CW\Desktop\pbl\waste\yolomodel\output"

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

#Evaluation of model

# ---- Trained model path ----
MODEL_PATH = r"C:\Users\CW\Desktop\pbl\waste\yolomodel\runs\marine_plastic_fast2\weights\best.pt"

# ---- Dataset YAML ----
DATA_YAML = r"C:\Users\CW\Desktop\pbl\waste\yolomodel\dataset\data.yaml"

# Load model
model = YOLO(MODEL_PATH)

# Evaluate model
metrics = model.val(data=DATA_YAML)

# Extract values
precision = metrics.box.mp
recall = metrics.box.mr
f1 = metrics.box.f1[0]
map50 = metrics.box.map50
map50_95 = metrics.box.map

print("\n================ Evaluation Report ================")
print(f"📌 Precision:   {precision:.3f}")
print(f"📌 Recall:      {recall:.3f}")
print(f"📌 F1 Score:    {f1:.3f}")
print(f"📌 mAP@50:      {map50:.3f}")
print(f"📌 mAP@50-95:   {map50_95:.3f}")
print("===================================================")
