import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train.csv')

flower_names = []
flower_images = []

for i in range (len(df)):
    row = df.iloc[i]
    img = Image.open('data/' + row['image:FILE'])
    
    img = img.resize((128, 128))

    img_array = np.array(img)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    img_array = img_array / 255.0

    flower_images.append(img_array)

    flower_names.append(row['category'])

val = pd.read_csv('data/val.csv')

val_names = []
val_images = []

for i in range (len(val)):
    row = val.iloc[i]
    img = Image.open('data/' + row['image:FILE'])
    
    img = img.resize((128, 128))

    img_array = np.array(img)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    img_array = img_array / 255.0

    val_images.append(img_array)

    val_names.append(row['category'])

flower_images = np.array(flower_images)
flower_names = np.array(flower_names)
val_images = np.array(val_images)
val_names = np.array(val_names)

x_train, x_test, y_train, y_test = train_test_split(flower_images, flower_names, train_size=0.8, random_state=42)

model = keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(14, activation=tf.nn.softmax)
])

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, validation_data=(val_images, val_names))

test_loss = model.evaluate(x_test, y_test)