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
num_classes = len(combo_set)

train_records, test_records = train_test_split(records, test_size=0.1, random_state=42)

# --- 建立 generator ---
def data_generator(records, yolo_features, image_dir, combo_to_index, batch_size=8, num_classes=10):
    i = 0
    while True:
        X_img, X_yolo, y_batch = [], [], []
        for _ in range(batch_size):
            if i >= len(records):
                i = 0
            r = records[i]
            img_path = os.path.join(image_dir, r['frame'])
            img = load_img(img_path)  # 原圖尺寸
            img_array = img_to_array(img).astype('float32') / 255.0
            X_img.append(img_array)
            X_yolo.append(yolo_features[r['frame']])
            label_index = combo_to_index[tuple(r['keys'])]
            y_batch.append(to_categorical(label_index, num_classes=num_classes))
            i += 1

        yield (
            {'image_input': np.array(X_img), 'yolo_input': np.array(X_yolo, dtype=np.float32)},
            np.array(y_batch)
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
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=[img_input, yolo_input], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 訓練 ---
batch_size = 8
steps_per_epoch = len(train_records) // batch_size
validation_steps = len(test_records) // batch_size

model.fit(
    data_generator(train_records, yolo_features, image_dir, combo_to_index, batch_size, num_classes),
    steps_per_epoch=steps_per_epoch,
    validation_data=data_generator(test_records, yolo_features, image_dir, combo_to_index, batch_size, num_classes),
    validation_steps=validation_steps,
    epochs=50
)

# --- 儲存 ---
model.save(os.path.join(latest_folder, "trained_multimodal_model.h5"))
with open(os.path.join(latest_folder, "combo_map.json"), 'w') as f:
    json.dump({str(list(k)): v for k, v in combo_to_index.items()}, f, indent=2)

print("✅ Multi-modal 模型訓練完成並儲存於：", latest_folder)
