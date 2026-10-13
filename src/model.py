
"""
MobileNetV2 model for brain MRI image classification.

Dataset categories:
1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary

Educational and research use only.
"""

import tensorflow as tf
from tensorflow.keras import layers

from src.config import IMAGE_SIZE, NUM_CLASSES


def build_model():
    """
    Build a transfer-learning model using MobileNetV2.

    Returns:
        Compiled TensorFlow Keras model.
    """

    # 1. Data augmentation
    augmentation = tf.keras.Sequential(
        [
            layers.RandomRotation(0.03),
            layers.RandomZoom(0.08),
            layers.RandomContrast(0.08),
        ],
        name="data_augmentation",
    )

    # 2. Load pretrained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze pretrained layers during initial training
    base_model.trainable = False

    # 3. Define model input
    inputs = layers.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        name="mri_image",
    )

    # 4. Apply data augmentation
    x = augmentation(inputs)

    # 5. Normalize pixel values from [0, 255] to [-1, 1]
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1,
        name="mobilenet_preprocessing",
    )(x)

    # 6. Extract image features
    x = base_model(x, training=False)

    # 7. Reduce feature map dimensions
    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    # 8. Apply dropout regularization
    x = layers.Dropout(
        0.30,
        name="dropout",
    )(x)

    # 9. Classify into four categories
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="class_probabilities",
    )(x)

    # 10. Create the model
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Brain_Tumor_MobileNetV2",
    )

    # 11. Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )

    return model


if __name__ == "__main__":
    model = build_model()

    print("\nModel created successfully!\n")
    model.summary()

