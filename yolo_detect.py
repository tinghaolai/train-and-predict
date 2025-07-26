import os
import time
import mss
import cv2
import numpy as np
from ultralytics import YOLO

# 設定 YOLO 模型（替換為你的 best.pt 路徑）
yolo_model = YOLO('runs/train/yolov8n-custom8/weights/best.pt')

# 擷取畫面區域
GAME_REGION = {"top": 0, "left": 0, "width": 1366, "height": 768}

# 建立結果資料夾
output_dir = 'yolo_detect_result'
os.makedirs(output_dir, exist_ok=True)

with mss.mss() as sct:
    frame_index = 0
    while True:
        screenshot = np.array(sct.grab(GAME_REGION))[:, :, :3]

        # YOLO 偵測
        results = yolo_model.predict(screenshot, verbose=False)
        boxes = results[0].boxes.cpu().numpy()
        names = yolo_model.names

        # 複製畫面並準備畫 bounding box
        annotated_img = screenshot.copy()
        object_lines = []

        print("📦 偵測到的物件:")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 畫 bounding box 與 label
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 印出並記錄
            print(f"- {label} ({conf:.2f})")
            object_lines.append(f"{label} {conf:.2f}")

        print('-' * 40)

        # 儲存圖與對應標籤
        img_filename = f"frame_{frame_index:04d}.jpg"
        txt_filename = f"frame_{frame_index:04d}.txt"
        cv2.imwrite(os.path.join(output_dir, img_filename), annotated_img)

        with open(os.path.join(output_dir, txt_filename), 'w') as f:
            f.write('\n'.join(object_lines))

        print(f"💾 已儲存：{img_filename}, {txt_filename}\n")

        frame_index += 1
        time.sleep(0.5)
