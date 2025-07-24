import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 資料集路徑
dataset_dir = 'dataset'
folders = sorted(os.listdir(dataset_dir))
latest_folder = os.path.join(dataset_dir, folders[-1])
image_dir = os.path.join(latest_folder, "images")
label_path = os.path.join(latest_folder, "labels.json")

# 載入標籤
with open(label_path, 'r') as f:
    records = json.load(f)

# 建立 combo 編碼映射
combo_set = sorted(set(tuple(r['keys']) for r in records))
combo_to_index = {combo: i for i, combo in enumerate(combo_set)}
y = np.array([combo_to_index[tuple(r['keys'])] for r in records])
y_cat = to_categorical(y, num_classes=len(combo_set))

# 載入原始尺寸圖片 (不 resize)
X = []
for r in records:
    img_path = os.path.join(image_dir, r['frame'])
    img = load_img(img_path)  # ← 不指定 target_size，保留原圖 1366x768
    img_array = img_to_array(img) / 255.0
    X.append(img_array)
X = np.array(X)

# 檢查 shape 是否一致（有些圖可能壞掉）
if not all(img.shape == (768, 1366, 3) for img in X):
    raise ValueError("❌ 有圖片尺寸不是 1366x768x3，請檢查圖片是否完整")

# 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.1, random_state=42)

# 建立 CNN 模型（支援 1366x768）
model = Sequential([
    Conv2D(16, (5,5), activation='relu', input_shape=(768,1366,3)),
    MaxPooling2D((2,2)),
    Conv2D(32, (5,5), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(len(combo_set), activation='softmax')
])

# 編譯與訓練
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))  # batch_size 可視 GPU 調整

# 儲存模型與 combo map
model.save(os.path.join(latest_folder, "trained_model.h5"))
with open(os.path.join(latest_folder, "combo_map.json"), 'w') as f:
    json.dump({str(list(k)): v for k, v in combo_to_index.items()}, f, indent=2)

print("✅ 模型訓練完成並儲存於：", latest_folder)
