import os
import time
import mss
import cv2
import numpy as np
import json
import keyboard
from tensorflow.keras.models import load_model

MONITORED_KEYS = ['left', 'right', 'up', 'down', 'z', 'x', 'c', 's']
INSTANT_KEYS = ['z', 'x', 'c', 's']
GAME_REGION = { "top": 0, "left": 0, "width": 1366, "height": 768 }
INFER_INTERVAL = 0.1

def find_latest_model():
    dataset_dir = 'dataset'
    folders = sorted(os.listdir(dataset_dir))
    for folder in reversed(folders):
        model_path = os.path.join(dataset_dir, folder, 'trained_model.h5')
        label_path = os.path.join(dataset_dir, folder, 'labels.json')
        if os.path.exists(model_path) and os.path.exists(label_path):
            return model_path, label_path
    raise Exception("❌ 找不到模型")

model_path, label_path = find_latest_model()
model = load_model(model_path)
with open(label_path, 'r') as f:
    records = json.load(f)
combo_list = sorted(set(tuple(r['keys']) for r in records))

print(f"✅ 使用模型：{model_path}")
print("🎮 自動操作啟動，按 ESC 結束")

with mss.mss() as sct:
    try:
        while True:
            if keyboard.is_pressed('esc'):
                break

            screenshot = np.array(sct.grab(GAME_REGION))[:, :, :3]
            resized = cv2.resize(screenshot, (224, 224)).astype('float32') / 255.0
            input_data = resized[np.newaxis, ...]

            pred = model.predict(input_data, verbose=0)[0]
            index = np.argmax(pred)
            confidence = pred[index]
            keys = combo_list[index]

            for k in MONITORED_KEYS:
                keyboard.release(k)

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
