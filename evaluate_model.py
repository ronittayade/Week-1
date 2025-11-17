from ultralytics import YOLO

# ---- Trained model path ----
MODEL_PATH = "best.pt"

# ---- Dataset YAML ----
DATA_YAML = "data.yaml"

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
