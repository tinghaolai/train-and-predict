from ultralytics import YOLO

# 使用 YOLOv8 nano 模型（或 yolov8s, yolov8m, yolov8l）
model = YOLO('yolov8n.yaml')  # 使用空模型開始訓練，或改用 yolov8n.pt 微調

# 訓練
model.train(
    data='./yolo/dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    workers=4,
    device='cpu',  # 如果有 GPU 就用 GPU，沒有可設為 'cpu'
    project='runs/train',
    name='yolov8n-custom',
    rect=True  # 若你是用絕對座標
)