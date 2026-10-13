"""
Dataset validation and image loading.

This module:
1. Checks the dataset folders.
2. Counts images in each class.
3. Loads training images.
4. Creates a validation split.
5. Loads the separate testing dataset.
"""

from pathlib import Path

import tensorflow as tf

from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
    CLASS_NAMES
)


# --------------------------------------------------
# 1. SUPPORTED IMAGE FORMATS
# --------------------------------------------------

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# --------------------------------------------------
# 2. COUNT IMAGES
# --------------------------------------------------

def count_images(folder):

    folder = Path(folder)

    if not folder.exists():
        return 0

    count = 0

    for file in folder.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        ):
            count += 1

    return count


# --------------------------------------------------
# 3. VALIDATE DATASET STRUCTURE
# --------------------------------------------------

def validate_dataset():

    print("\nChecking dataset structure...")

    # Check training and testing directories.

    for split_dir in [TRAIN_DIR, TEST_DIR]:

        if not split_dir.is_dir():

            raise FileNotFoundError(
                f"\nDataset directory not found: {split_dir}\n"
                "Check your dataset location."
            )

        # Check each expected class folder.

        for class_name in CLASS_NAMES:

            class_dir = split_dir / class_name

            if not class_dir.is_dir():

                raise FileNotFoundError(
                    f"\nMissing class directory: {class_dir}"
                )

            image_count = count_images(class_dir)

            if image_count == 0:

                raise ValueError(
                    f"\nNo images found in: {class_dir}"
                )

    print("\nDataset structure is valid.")

    # Print training image counts.

    print("\nTRAINING DATASET")

    for class_name in CLASS_NAMES:

        folder = TRAIN_DIR / class_name

        print(
            f"{class_name}: {count_images(folder)} images"
        )

    # Print testing image counts.

    print("\nTESTING DATASET")

    for class_name in CLASS_NAMES:

        folder = TEST_DIR / class_name

        print(
            f"{class_name}: {count_images(folder)} images"
        )


# --------------------------------------------------
# 4. LOAD DATASETS
# --------------------------------------------------

def load_datasets():

    validate_dataset()

    print("\nLoading training dataset...")

    train_ds = tf.keras.utils.image_dataset_from_directory(

        TRAIN_DIR,

        validation_split=0.15,

        subset="training",

        seed=SEED,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="int",

        class_names=CLASS_NAMES

    )

    print("\nLoading validation dataset...")

    val_ds = tf.keras.utils.image_dataset_from_directory(

        TRAIN_DIR,

        validation_split=0.15,

        subset="validation",

        seed=SEED,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="int",

        class_names=CLASS_NAMES

    )

    print("\nLoading testing dataset...")

    test_ds = tf.keras.utils.image_dataset_from_directory(

        TEST_DIR,

        shuffle=False,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="int",

        class_names=CLASS_NAMES

    )

    # Improve input pipeline efficiency.

    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(
        buffer_size=autotune
    )

    val_ds = val_ds.prefetch(
        buffer_size=autotune
    )

    test_ds = test_ds.prefetch(
        buffer_size=autotune
    )

    print("\nAll datasets loaded successfully.")

    return train_ds, val_ds, test_ds


# --------------------------------------------------
# 5. TEST THE DATA LOADER
# --------------------------------------------------

if __name__ == "__main__":

    train_ds, val_ds, test_ds = load_datasets()

    print("\nDataset Summary")

    print("Training batches:", len(train_ds))

    print("Validation batches:", len(val_ds))

    print("Testing batches:", len(test_ds))

    # Display the shape of one batch.

    for images, labels in train_ds.take(1):

        print("\nImage batch shape:", images.shape)

        print("Label batch shape:", labels.shape)

        print("Image data type:", images.dtype)

        print("Sample labels:", labels[:10].numpy())

    print("\nDataset verification completed.")