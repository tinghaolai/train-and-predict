import os
import time
import json
import mss
import cv2
import numpy as np
import keyboard
from datetime import datetime

SAVE_DIR = 'dataset'
SCREEN_REGION = { "top": 0, "left": 0, "width": 1366, "height": 768 }
RECORD_DURATION = 2 * 60  # 錄製秒數
FRAME_INTERVAL = 0.1
MONITORED_KEYS = ['left', 'right', 'up', 'down', 'z', 'x', 'c', 's']

folder_name = datetime.now().strftime('%Y%m%d_%H%M%S')
folder_path = os.path.join(SAVE_DIR, folder_name)
image_dir = os.path.join(folder_path, "images")
os.makedirs(image_dir, exist_ok=True)

print(f"🎥 開始錄製 {RECORD_DURATION} 秒...")
records = []

with mss.mss() as sct:
    start_time = time.time()
    frame_id = 0

    while time.time() - start_time < RECORD_DURATION:
        screenshot = np.array(sct.grab(SCREEN_REGION))[:, :, :3]
        resized = cv2.resize(screenshot, (224, 224))
        filename = f"frame_{frame_id:05}.jpg"
        cv2.imwrite(os.path.join(image_dir, filename), resized)

        pressed = [k for k in MONITORED_KEYS if keyboard.is_pressed(k)]
        records.append({ "frame": filename, "keys": pressed })

        frame_id += 1
        time.sleep(FRAME_INTERVAL)

with open(os.path.join(folder_path, 'labels.json'), 'w') as f:
    json.dump(records, f, indent=2)

print(f"✅ 錄製完成！共 {frame_id} 張，儲存於 {folder_path}")
