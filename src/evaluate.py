
"""
Evaluate the trained brain tumor classification model.

Outputs:
- Classification report
- Confusion matrix
- Example predictions
- JSON evaluation metrics
"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.config import (
    MODEL_PATH,
    OUTPUT_DIR,
    CLASS_NAMES
)

from src.data import load_datasets


def evaluate_model():

    # --------------------------------------------------
    # 1. CHECK MODEL
    # --------------------------------------------------

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(

            f"Model not found: {MODEL_PATH}\n"

            "Train the model before evaluation."

        )

    print("\nLoading datasets...")

    _, _, test_ds = load_datasets()

    # --------------------------------------------------
    # 2. LOAD TRAINED MODEL
    # --------------------------------------------------

    print("\nLoading trained model...")

    model = tf.keras.models.load_model(

        MODEL_PATH

    )

    # --------------------------------------------------
    # 3. GENERATE PREDICTIONS
    # --------------------------------------------------

    print("\nPredicting test images...")

    probabilities = model.predict(

        test_ds,

        verbose=1

    )

    y_true = np.concatenate(

        [

            labels.numpy().ravel()

            for _, labels in test_ds

        ]

    ).astype(int)

    y_pred = np.argmax(

        probabilities,

        axis=1

    )

    # --------------------------------------------------
    # 4. CLASSIFICATION REPORT
    # --------------------------------------------------

    report_text = classification_report(

        y_true,

        y_pred,

        labels=list(

            range(len(CLASS_NAMES))

        ),

        target_names=CLASS_NAMES,

        zero_division=0

    )

    report_dict = classification_report(

        y_true,

        y_pred,

        labels=list(

            range(len(CLASS_NAMES))

        ),

        target_names=CLASS_NAMES,

        zero_division=0,

        output_dict=True

    )

    print("\nCLASSIFICATION REPORT\n")

    print(report_text)

    # --------------------------------------------------
    # 5. CONFUSION MATRIX
    # --------------------------------------------------

    matrix = confusion_matrix(

        y_true,

        y_pred,

        labels=list(

            range(len(CLASS_NAMES))

        )

    )

    print("\nCONFUSION MATRIX\n")

    print(matrix)

    # --------------------------------------------------
    # 6. SAVE EVALUATION METRICS
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    metrics_path = (

        OUTPUT_DIR / "test_metrics.json"

    )

    with open(

        metrics_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            {

                "class_names": CLASS_NAMES,

                "classification_report": report_dict,

                "confusion_matrix": matrix.tolist()

            },

            file,

            indent=4

        )

    # --------------------------------------------------
    # 7. PLOT CONFUSION MATRIX
    # --------------------------------------------------

    display = ConfusionMatrixDisplay(

        confusion_matrix=matrix,

        display_labels=CLASS_NAMES

    )

    display.plot(

        cmap="Blues",

        xticks_rotation=30,

        values_format="d"

    )

    plt.title(

        "Brain MRI Classification - Test Confusion Matrix"

    )

    plt.tight_layout()

    confusion_path = (

        OUTPUT_DIR / "confusion_matrix.png"

    )

    plt.savefig(

        confusion_path,

        dpi=160

    )

    plt.close()

    # --------------------------------------------------
    # 8. VISUALIZE EXAMPLE PREDICTIONS
    # --------------------------------------------------

    images, labels = next(

        iter(test_ds)

    )

    batch_probabilities = model.predict(

        images,

        verbose=0

    )

    count = min(

        8,

        len(images)

    )

    fig, axes = plt.subplots(

        2,

        4,

        figsize=(13, 7)

    )

    for index, ax in enumerate(

        axes.flat

    ):

        if index >= count:

            ax.axis("off")

            continue

        actual_index = int(

            labels[index].numpy()

        )

        predicted_index = int(

            np.argmax(

                batch_probabilities[index]

            )

        )

        actual_label = CLASS_NAMES[

            actual_index

        ]

        predicted_label = CLASS_NAMES[

            predicted_index

        ]

        ax.imshow(

            images[index].numpy().astype(

                "uint8"

            )

        )

        ax.set_title(

            f"Actual: {actual_label}\n"

            f"Predicted: {predicted_label}",

            fontsize=8

        )

        ax.axis("off")

    fig.suptitle(

        "Example Test Predictions"

    )

    fig.tight_layout()

    examples_path = (

        OUTPUT_DIR / "prediction_examples.png"

    )

    fig.savefig(

        examples_path,

        dpi=160

    )

    plt.close(fig)

    # --------------------------------------------------
    # 9. PRINT OUTPUT PATHS
    # --------------------------------------------------

    print("\nEvaluation completed!")

    print("\nMetrics saved to:")

    print(metrics_path)

    print("\nConfusion matrix saved to:")

    print(confusion_path)

    print("\nPrediction examples saved to:")

    print(examples_path)

    print(

        "\nEducational research prototype only. "

        "Not for medical diagnosis."

    )


if __name__ == "__main__":

    evaluate_model()

