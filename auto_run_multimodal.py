import os
import time
import mss
import cv2
import numpy as np
import json
import keyboard
from tensorflow.keras.models import load_model
from ultralytics import YOLO

MONITORED_KEYS = ['left', 'right', 'up', 'down', 'z', 'x', 'c', 's']
INSTANT_KEYS = ['z', 'x', 'c', 's']
GAME_REGION = {"top": 0, "left": 0, "width": 1366, "height": 768}
INFER_INTERVAL = 0.1
IMG_W, IMG_H = 1366, 768
YOLO_VEC_LEN = 30  # 5 objects * 6 features

# 🔍 尋找最新資料夾中的模型與 label
def find_latest_model():
    dataset_dir = 'dataset'
    folders = sorted(os.listdir(dataset_dir))
    for folder in reversed(folders):
        folder_path = os.path.join(dataset_dir, folder)
        model_path = os.path.join(folder_path, 'trained_multimodal_model.h5')
        label_path = os.path.join(folder_path, 'labels.json')
        if os.path.exists(model_path) and os.path.exists(label_path):
            return folder_path, model_path, label_path
    raise Exception("❌ 找不到模型")

# 🧠 YOLO 向量化邏輯
def vectorize_yolo(results):
    vec = np.zeros(YOLO_VEC_LEN, dtype=np.float32)
    boxes = results[0].boxes.cpu().numpy()
    for i, box in enumerate(boxes[:5]):
        base = i * 6
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        vec[base + 0] = cls_id
        vec[base + 1] = cx / IMG_W
        vec[base + 2] = cy / IMG_H
        vec[base + 3] = w / IMG_W
        vec[base + 4] = h / IMG_H
        vec[base + 5] = conf
    return vec

# 📦 載入模型與 combo map
latest_folder, model_path, label_path = find_latest_model()
model = load_model(model_path)
with open(label_path, 'r') as f:
    records = json.load(f)
combo_list = sorted(set(tuple(r['keys']) for r in records))

# 🧿 載入 YOLO 模型（改成你的 best.pt）
yolo_model = YOLO(os.path.join(latest_folder, 'yolov8_best.pt'))

print(f"✅ 使用模型：{model_path}")
print("🎮 自動操作啟動，按 ESC 結束")

with mss.mss() as sct:
    try:
        while True:
            if keyboard.is_pressed('esc'):
                break

            # 擷取畫面
            screenshot = np.array(sct.grab(GAME_REGION))[:, :, :3]
            resized = cv2.resize(screenshot, (768, 1366))  # Tensorflow model input = (768, 1366)
            input_image = resized.astype('float32') / 255.0
            input_image = input_image[np.newaxis, ...]

            # 執行 YOLO 推論
            yolo_results = yolo_model.predict(screenshot, verbose=False)
            yolo_vector = vectorize_yolo(yolo_results)[np.newaxis, ...]

            # Multi-modal 推論
            pred = model.predict({'image_input': input_image, 'yolo_input': yolo_vector}, verbose=0)[0]
            index = np.argmax(pred)
            confidence = pred[index]
            keys = combo_list[index]

            # 放開舊鍵
            for k in MONITORED_KEYS:
                keyboard.release(k)

            # 執行按鍵
            if confidence > 0.2:
                for k in keys:
                    if k in INSTANT_KEYS:
                        keyboard.press_and_release(k)
                    else:
                        keyboard.press(k)

            print(f"🎯 {', '.join(keys) if keys else 'none'} | 🔍 conf: {confidence:.2f}")
            time.sleep(INFER_INTERVAL)

    finally:
        for k in MONITORED_KEYS:
            keyboard.release(k)
        print("🛑 自動控制已停止")
