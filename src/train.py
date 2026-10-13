
"""
Training pipeline for brain tumor MRI classification.

This module:
1. Loads training and validation datasets.
2. Builds the model.
3. Trains the model.
4. Saves the best checkpoint.
5. Records training metrics.
6. Generates training graphs.
"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import tensorflow as tf

from src.config import (
    MODEL_DIR,
    OUTPUT_DIR,
    MODEL_PATH
)

from src.data import load_datasets

from src.model import build_model


def train_model(epochs=5):

    """
    Train the MobileNetV2 classifier.

    Args:
        epochs: Maximum number of training epochs.
    """

    # --------------------------------------------------
    # 1. CREATE OUTPUT DIRECTORIES
    # --------------------------------------------------

    MODEL_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    print("\nLoading datasets...")

    train_ds, val_ds, _ = load_datasets()

    # --------------------------------------------------
    # 2. BUILD MODEL
    # --------------------------------------------------

    print("\nBuilding MobileNetV2 model...")

    model = build_model()

    model.summary()

    # --------------------------------------------------
    # 3. CALLBACKS
    # --------------------------------------------------

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            filepath=str(MODEL_PATH),

            monitor="val_accuracy",

            mode="max",

            save_best_only=True,

            verbose=1

        ),

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=2,

            restore_best_weights=True,

            verbose=1

        ),

        tf.keras.callbacks.CSVLogger(

            str(

                OUTPUT_DIR / "training_history.csv"

            )

        )

    ]

    # --------------------------------------------------
    # 4. TRAIN MODEL
    # --------------------------------------------------

    print("\nStarting model training...\n")

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=epochs,

        callbacks=callbacks

    )

    # --------------------------------------------------
    # 5. SAVE TRAINING HISTORY
    # --------------------------------------------------

    history_data = {

        key: [

            float(value)

            for value in values

        ]

        for key, values in history.history.items()

    }

    history_path = (

        OUTPUT_DIR / "training_history.json"

    )

    with open(

        history_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            history_data,

            file,

            indent=4

        )

    # --------------------------------------------------
    # 6. GENERATE TRAINING GRAPHS
    # --------------------------------------------------

    accuracy = history.history["accuracy"]

    val_accuracy = history.history["val_accuracy"]

    loss = history.history["loss"]

    val_loss = history.history["val_loss"]

    fig, axes = plt.subplots(

        1,

        2,

        figsize=(12, 5)

    )

    # Accuracy graph.

    axes[0].plot(

        accuracy,

        label="Training Accuracy"

    )

    axes[0].plot(

        val_accuracy,

        label="Validation Accuracy"

    )

    axes[0].set_title(

        "Training vs Validation Accuracy"

    )

    axes[0].set_xlabel("Epoch")

    axes[0].set_ylabel("Accuracy")

    axes[0].legend()

    # Loss graph.

    axes[1].plot(

        loss,

        label="Training Loss"

    )

    axes[1].plot(

        val_loss,

        label="Validation Loss"

    )

    axes[1].set_title(

        "Training vs Validation Loss"

    )

    axes[1].set_xlabel("Epoch")

    axes[1].set_ylabel("Loss")

    axes[1].legend()

    fig.tight_layout()

    graph_path = (

        OUTPUT_DIR / "training_curves.png"

    )

    fig.savefig(

        graph_path,

        dpi=160

    )

    plt.close(fig)

    # --------------------------------------------------
    # 7. TRAINING SUMMARY
    # --------------------------------------------------

    print("\nTraining completed!")

    print("\nSaved model:")

    print(MODEL_PATH)

    print("\nSaved training history:")

    print(history_path)

    print("\nSaved training graph:")

    print(graph_path)

    print("\nNext step: evaluate the model.")

    return model


# --------------------------------------------------
# 8. RUN TRAINING DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--epochs",

        type=int,

        default=5

    )

    args = parser.parse_args()

    train_model(

        epochs=args.epochs

    )

