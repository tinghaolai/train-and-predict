import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

SEQ_LEN = 5
dataset_dir = 'dataset'
folders = sorted(os.listdir(dataset_dir))
latest_folder = os.path.join(dataset_dir, folders[-1])
image_dir = os.path.join(latest_folder, "images")
label_path = os.path.join(latest_folder, "labels.json")

with open(label_path, 'r') as f:
    records = json.load(f)

combo_set = sorted(set(tuple(r['keys']) for r in records))
combo_to_index = {combo: i for i, combo in enumerate(combo_set)}
y_full = [combo_to_index[tuple(r['keys'])] for r in records]

X, y = [], []
for i in range(len(records) - SEQ_LEN):
    sequence = []
    valid = True
    for j in range(SEQ_LEN):
        r = records[i + j]
        img_path = os.path.join(image_dir, r['frame'])
        if not os.path.exists(img_path):
            valid = False
            break
        img = img_to_array(load_img(img_path, target_size=(224, 224))) / 255.0
        sequence.append(img)
    if valid:
        X.append(sequence)
        y.append(y_full[i + SEQ_LEN - 1])

X = np.array(X)
y = to_categorical(np.array(y), num_classes=len(combo_set))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(SEQ_LEN, 224, 224, 3)),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Flatten()),
    LSTM(128),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(len(combo_set), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
model.save(os.path.join(latest_folder, "trained_model_seq.h5"))
print("✅ LSTM 模型訓練完成！")
