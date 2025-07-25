import os
import time
import mss
import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 模型路徑（請替換成你的模型）
yolo_model = YOLO('runs/train/yolov8n-custom7/weights/best.pt')

# 截圖區域
GAME_REGION = {"top": 0, "left": 0, "width": 1366, "height": 768}

# 建立結果資料夾
os.makedirs('yolo_detect_result', exist_ok=True)

with mss.mss() as sct:
    frame_index = 0
    while True:
        screenshot = np.array(sct.grab(GAME_REGION))[:, :, :3]
        results = yolo_model.predict(screenshot, verbose=False)
        boxes = results[0].boxes.cpu().numpy()
        names = yolo_model.names

        # 複製畫面來畫圖
        annotated_img = screenshot.copy()

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 畫 bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 畫 label
            cv2.putText(
                annotated_img,
                f"{label} ({conf:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # 儲存圖片
        out_path = os.path.join('yolo_detect_result', f"frame_{frame_index:04d}.jpg")
        cv2.imwrite(out_path, annotated_img)
        print(f"💾 已儲存：{out_path}")

        frame_index += 1
        time.sleep(0.5)
