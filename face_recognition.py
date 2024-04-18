import keras.layers
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.metrics import Precision, Recall, AUC
from keras.src.callbacks import EarlyStopping
from keras.preprocessing import image
import numpy as np

image_size = (128, 128)
batch_size = 8
epochs = 10

datagen = ImageDataGenerator(rescale=1./255)


def dataset(path, shuffle):
    return datagen.flow_from_directory(
        path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=shuffle
    )


train_generator = dataset("Dataset/Train", True)

val_generator = dataset("Dataset/Validate", False)

test_generator = dataset("Dataset/Test", False)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.99),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(8, activation='sigmoid')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall(), AUC(name='auc')])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

evaluation = model.evaluate(test_generator)
print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])
print("Test Precision:", evaluation[2])
print("Test Recall:", evaluation[3])
print("Test AUC:", evaluation[4])

model.save("cnn7.h5")

test_data, test_labels = test_generator.next()

print(test_generator.class_indices)

print(f'Test Data Shape: {test_data.shape}')
print(f'Test Labels Shape: {test_labels.shape}')
