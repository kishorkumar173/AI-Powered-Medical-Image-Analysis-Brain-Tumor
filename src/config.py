"""
Central configuration for the Brain Tumor Classification project.
"""

from pathlib import Path

# --------------------------------------------------
# 1. PROJECT ROOT
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]


# --------------------------------------------------
# 2. DATASET PATHS
# --------------------------------------------------

DATA_DIR = ROOT_DIR / "data" / "brain_tumor_dataset"

TRAIN_DIR = DATA_DIR / "Training"

TEST_DIR = DATA_DIR / "Testing"


# --------------------------------------------------
# 3. MODEL AND OUTPUT PATHS
# --------------------------------------------------

MODEL_DIR = ROOT_DIR / "models"

OUTPUT_DIR = ROOT_DIR / "outputs"

IMAGE_DIR = ROOT_DIR / "images"

MODEL_PATH = MODEL_DIR / "brain_tumor_mobilenetv2.keras"


# --------------------------------------------------
# 4. IMAGE SETTINGS
# --------------------------------------------------

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16

SEED = 42


# --------------------------------------------------
# 5. CLASS CONFIGURATION
# --------------------------------------------------

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

NUM_CLASSES = len(CLASS_NAMES)


# --------------------------------------------------
# 6. DISPLAY CONFIGURATION
# --------------------------------------------------

PROJECT_NAME = "AI-Powered Medical Image Analysis"

TASK_NAME = "Brain Tumor Classification"


# --------------------------------------------------
# 7. PRINT CONFIGURATION
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 50)

    print(PROJECT_NAME)

    print(TASK_NAME)

    print("=" * 50)

    print("Project Root:", ROOT_DIR)

    print("Dataset Directory:", DATA_DIR)

    print("Training Directory:", TRAIN_DIR)

    print("Testing Directory:", TEST_DIR)

    print("Model Directory:", MODEL_DIR)

    print("Output Directory:", OUTPUT_DIR)

    print("Image Size:", IMAGE_SIZE)

    print("Batch Size:", BATCH_SIZE)

    print("Number of Classes:", NUM_CLASSES)

    print("Classes:", CLASS_NAMES)

    print("=" * 50)