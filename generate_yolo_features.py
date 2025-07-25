import os
import json
import numpy as np
from ultralytics import YOLO
import cv2

yolo_model = YOLO('./runs/train/yolov8n-custom6/weights/best.pt')  # 替換成你訓練好的模型路徑

dataset_dir = 'dataset'
latest_folder = os.path.join(dataset_dir, sorted(os.listdir(dataset_dir))[-1])
image_dir = os.path.join(latest_folder, "images")
label_path = os.path.join(latest_folder, "labels.json")

with open(label_path, 'r') as f:
    records = json.load(f)

MAX_OBJECTS = 5
VECTOR_LENGTH = MAX_OBJECTS * 6  # 每個物件用 6 維（class, x, y, w, h, conf）
img_w, img_h = 1366, 768

def vectorize_yolo_results(results):
    vector = np.zeros(VECTOR_LENGTH, dtype=np.float32)
    boxes = results[0].boxes.cpu().numpy()
    for i, box in enumerate(boxes[:MAX_OBJECTS]):
        base = i * 6
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        vector[base + 0] = cls_id
        vector[base + 1] = cx / img_w
        vector[base + 2] = cy / img_h
        vector[base + 3] = w / img_w
        vector[base + 4] = h / img_h
        vector[base + 5] = conf
    return vector

yolo_features = {}

for r in records:
    img_path = os.path.join(image_dir, r['frame'])
    img = cv2.imread(img_path)
    results = yolo_model.predict(img, verbose=False)
    vec = vectorize_yolo_results(results)
    yolo_features[r['frame']] = vec.tolist()

with open(os.path.join(latest_folder, "yolo_features.json"), 'w') as f:
    json.dump(yolo_features, f, indent=2)

print("✅ 已儲存 YOLO 向量至 yolo_features.json")
