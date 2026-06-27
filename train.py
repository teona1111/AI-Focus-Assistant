import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


DATASET_PATH = "spectrograms"

GRAPH_DIR = "model_graphs"

CLASSES = [
    "focus",
    "sleepy",
    "distracted"
]

IMAGE_SIZE = (
    64,
    64
)

MODEL_NAME = "rf_combined.pkl"

os.makedirs(
    GRAPH_DIR,
    exist_ok=True
)


def load_image(
    path
):

    img = Image.open(
        path
    )

    img = img.convert(
        "RGBA"
    )

    img = img.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    img = np.array(
        img
    )

    return img.flatten()


def load_dataset():

    X = []
    y = []

    for label_idx, class_name in enumerate(CLASSES):

        folder = os.path.join(
            DATASET_PATH,
            class_name
        )

        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        all_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(".png")
        ]

        gyro_files = [
            f for f in all_files
            if "gyro" in f.lower()
            or "gyroscope" in f.lower()
        ]

        print(f"{class_name}: {len(gyro_files)} gyro files")

        valid_pairs = 0

        for gyro_file in gyro_files:

            lower_name = gyro_file.lower()

            acc_file = lower_name.replace(
                "gyroscope",
                "accelerometer"
            )

            acc_file = acc_file.replace(
                "gyro",
                "acc"
            )

            gyro_path = os.path.join(
                folder,
                gyro_file
            )

            acc_path = os.path.join(
                folder,
                acc_file
            )

            if not os.path.exists(acc_path):

                # try original capitalization version too
                acc_file_alt = gyro_file.replace(
                    "Gyroscope",
                    "Accelerometer"
                )

                acc_file_alt = acc_file_alt.replace(
                    "gyro",
                    "acc"
                )

                acc_path = os.path.join(
                    folder,
                    acc_file_alt
                )

            if not os.path.exists(acc_path):
                print(f"Missing pair for: {gyro_file}")
                continue

            try:

                gyro_img = load_image(
                    gyro_path
                )

                acc_img = load_image(
                    acc_path
                )

                combined = np.concatenate(
                    [
                        gyro_img,
                        acc_img
                    ]
                )

                X.append(combined)
                y.append(label_idx)

                valid_pairs += 1

            except Exception as e:
                print(f"Error loading pair {gyro_file}: {e}")

        print(f"{class_name}: {valid_pairs} valid pairs")

    return np.array(X), np.array(y)

def plot_confusion_matrix(
    cm
):

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        cm
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.xticks(
        range(len(CLASSES)),
        CLASSES,
        rotation=45
    )

    plt.yticks(
        range(len(CLASSES)),
        CLASSES
    )

    plt.colorbar()

    for i in range(len(CLASSES)):

        for j in range(len(CLASSES)):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            GRAPH_DIR,
            "confusion_matrix.png"
        )
    )

    plt.show()


def main():

    print(
        "Loading dataset..."
    )

    X, y = load_dataset()

    if len(X) == 0:

        print(
            "Dataset is empty!"
        )

        return

    print()
    print(
        "Dataset loaded."
    )

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "Feature count:",
        X.shape[1]
    )

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    print()
    print(
        "Class counts:"
    )

    for u, c in zip(unique, counts):

        print(
            CLASSES[u],
            c
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print()
    print(
        "Training Random Forest..."
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training complete."
    )

    train_pred = model.predict(
        X_train
    )

    test_pred = model.predict(
        X_test
    )

    train_acc = accuracy_score(
        y_train,
        train_pred
    )

    test_acc = accuracy_score(
        y_test,
        test_pred
    )

    print()
    print(
        f"Train Accuracy: {train_acc:.4f}"
    )

    print(
        f"Test Accuracy: {test_acc:.4f}"
    )

    print()

    print(
        classification_report(
            y_test,
            test_pred,
            target_names=CLASSES
        )
    )

    cm = confusion_matrix(
        y_test,
        test_pred
    )

    print()
    print(
        "Confusion Matrix:"
    )

    print(
        cm
    )

    plot_confusion_matrix(
        cm
    )

    joblib.dump(
        model,
        MODEL_NAME
    )

    print()
    print(
        "Saved model:"
    )

    print(
        MODEL_NAME
    )


if __name__ == "__main__":
    main()