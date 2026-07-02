import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 8

DATASET_PATH = r"C:\Users\filip\Desktop\archive\data"

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH + r"\train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["Awake", "Sleepy"]
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH + r"\val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["Awake", "Sleepy"]
)

test_data = test_datagen.flow_from_directory(
    DATASET_PATH + r"\test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["Awake", "Sleepy"],
    shuffle=False
)

print("\nDataset loaded successfully.")
print("\nStarting CNN training...\n")

model = models.Sequential([

    layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(64,64,3)
    ),

    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

test_loss, test_acc = model.evaluate(test_data)

train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

model.save("eye_model.h5")

print("\nModel saved as eye_model.h5")

plt.figure()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title(
    f'Train: {train_acc:.4f} | '
    f'Validation: {val_acc:.4f} | '
    f'Test: {test_acc:.4f}'
)

plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])

plt.savefig("accuracy_graph.png")

plt.show()

plt.figure()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])

plt.savefig("loss_graph.png")

plt.show()