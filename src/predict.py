
"""
Single-image prediction for the brain MRI classifier.
"""

from pathlib import Path

import numpy as np

import tensorflow as tf

from PIL import Image

from src.config import (
    MODEL_PATH,
    IMAGE_SIZE,
    CLASS_NAMES
)


def predict_pil_image(image):

    """
    Predict the category of a PIL image.

    Args:
        image: PIL image object.

    Returns:
        Dictionary containing the predicted label and class scores.
    """

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(

            f"Model not found: {MODEL_PATH}\n"

            "Train the model before prediction."

        )

    # Convert to RGB and resize.

    image = image.convert("RGB").resize(

        IMAGE_SIZE

    )

    # Convert image to a numerical array.

    image_array = np.asarray(

        image,

        dtype=np.float32

    )

    # Add a batch dimension.

    image_array = np.expand_dims(

        image_array,

        axis=0

    )

    # Load trained model.

    model = tf.keras.models.load_model(

        MODEL_PATH

    )

    # Predict class scores.

    scores = model.predict(

        image_array,

        verbose=0

    )[0]

    # Find the highest-scoring class.

    predicted_index = int(

        np.argmax(scores)

    )

    predicted_label = CLASS_NAMES[

        predicted_index

    ]

    # Return result.

    return {

        "label": predicted_label,

        "scores": {

            class_name: float(score)

            for class_name, score in zip(

                CLASS_NAMES,

                scores

            )

        }

    }


def predict_image(image_path):

    """
    Predict the category of an image stored on disk.
    """

    image_path = Path(

        image_path

    )

    if not image_path.is_file():

        raise FileNotFoundError(

            f"Image not found: {image_path}"

        )

    with Image.open(

        image_path

    ) as image:

        result = predict_pil_image(

            image

        )

    print("\nBRAIN MRI CLASSIFICATION")

    print("=" * 40)

    print("Image:", image_path)

    print(

        "Predicted dataset category:",

        result["label"]

    )

    print("\nModel output scores:")

    for class_name, score in result["scores"].items():

        print(

            f"{class_name}: {score:.4f}"

        )

    print("=" * 40)

    print(

        "Educational prototype only. "

        "Not for medical diagnosis."

    )

    return result


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--image",

        required=True

    )

    args = parser.parse_args()

    predict_image(

        args.image

    )

