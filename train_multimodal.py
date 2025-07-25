from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import os, json
import numpy as np

# --- 載入資料 ---
dataset_dir = 'dataset'
latest_folder = os.path.join(dataset_dir, sorted(os.listdir(dataset_dir))[-1])
image_dir = os.path.join(latest_folder, "images")
label_path = os.path.join(latest_folder, "labels.json")
yolo_feat_path = os.path.join(latest_folder, "yolo_features.json")

with open(label_path, 'r') as f:
    records = json.load(f)

with open(yolo_feat_path, 'r') as f:
    yolo_features = json.load(f)

combo_set = sorted(set(tuple(r['keys']) for r in records))
combo_to_index = {combo: i for i, combo in enumerate(combo_set)}
y = np.array([combo_to_index[tuple(r['keys'])] for r in records])
y_cat = to_categorical(y, num_classes=len(combo_set))

X_img, X_yolo = [], []

for r in records:
    img_path = os.path.join(image_dir, r['frame'])
    img = load_img(img_path)
    img_array = img_to_array(img) / 255.0
    X_img.append(img_array)
    X_yolo.append(yolo_features[r['frame']])

X_img = np.array(X_img)
X_yolo = np.array(X_yolo, dtype=np.float32)

if not all(img.shape == (768, 1366, 3) for img in X_img):
    raise ValueError("❌ 有圖片尺寸不對")

X_train_img, X_test_img, X_train_yolo, X_test_yolo, y_train, y_test = train_test_split(
    X_img, X_yolo, y_cat, test_size=0.1, random_state=42
)

# --- 建立模型 ---
img_input = Input(shape=(768, 1366, 3), name='image_input')
yolo_input = Input(shape=(30,), name='yolo_input')

x = Conv2D(16, (5,5), activation='relu')(img_input)
x = MaxPooling2D((2,2))(x)
x = Conv2D(32, (5,5), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)

merged = Concatenate()([x, yolo_input])
x = Dropout(0.5)(merged)
x = Dense(128, activation='relu')(x)
output = Dense(len(combo_set), activation='softmax')(x)

model = Model(inputs=[img_input, yolo_input], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 訓練 ---
model.fit(
    {'image_input': X_train_img, 'yolo_input': X_train_yolo},
    y_train,
    epochs=10,
    batch_size=8,
    validation_data=(
        {'image_input': X_test_img, 'yolo_input': X_test_yolo},
        y_test
    )
)

model.save(os.path.join(latest_folder, "trained_multimodal_model.h5"))
with open(os.path.join(latest_folder, "combo_map.json"), 'w') as f:
    json.dump({str(list(k)): v for k, v in combo_to_index.items()}, f, indent=2)

print("✅ Multi-modal 模型訓練完成並儲存於：", latest_folder)
