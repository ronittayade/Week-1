from ultralytics import YOLO

model = YOLO("best.pt")
img = "test1.png"

model.predict(img, conf=0.4, save=True)

print("Prediction saved.")
