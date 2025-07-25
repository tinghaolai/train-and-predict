import mss
import cv2
import numpy as np
import time
import keyboard
import os
import json
from datetime import datetime

GAME_REGION = {
    "top": 0,
    "left": 0,
    "width": 1366,
    "height": 768
}

MONITORED_KEYS = ['left', 'right', 'up', 'down', 'z', 'x', 'c', 's']

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("dataset", timestamp)
image_dir = os.path.join(output_dir, "images")
os.makedirs(image_dir, exist_ok=True)

records = []
frame_count = 0
total_duration = 60  # 5 分鐘
interval = 0.1
end_time = time.time() + total_duration

print(f"📸 開始錄製 5 分鐘整畫面操作...")

with mss.mss() as sct:
    while time.time() < end_time:
        screenshot = np.array(sct.grab(GAME_REGION))[:, :, :3]  # 原始畫面
        pressed_keys = [k for k in MONITORED_KEYS if keyboard.is_pressed(k)]

        filename = f"frame_{frame_count:05d}.png"
        filepath = os.path.join(image_dir, filename)
        cv2.imwrite(filepath, screenshot)  # 不做任何 resize

        records.append({
            'frame': filename,
            'keys': pressed_keys
        })

        frame_count += 1
        time.sleep(interval)

with open(os.path.join(output_dir, "labels.json"), 'w') as f:
    json.dump(records, f, indent=2)

print(f"✅ 錄製完成，共 {frame_count} 幀，儲存至：{output_dir}")
