
"""
AI-Powered Medical Image Analysis
Brain MRI Classification Web Application

Educational and research prototype only.
Not intended for medical diagnosis.
"""

import json
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image

from src.config import (
    MODEL_PATH,
    OUTPUT_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
)

from src.predict import predict_pil_image


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brain MRI Classification",
    page_icon="🧠",
    layout="wide",
)


# --------------------------------------------------
# 2. CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .disclaimer {
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #d9a441;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '🧠 Brain MRI Classification'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered educational image classification using MobileNetV2'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("About the Project")

    st.write(
        """
        This application uses a deep learning model based on
        MobileNetV2 to classify brain MRI images into one of
        four categories from the training dataset.
        """
    )

    st.subheader("Dataset Categories")

    st.write("1. Glioma")
    st.write("2. Meningioma")
    st.write("3. No Tumor")
    st.write("4. Pituitary")

    st.divider()

    st.caption(
        "Model: MobileNetV2 with transfer learning"
    )

    st.caption(
        "Input size: 224 × 224 pixels"
    )


# --------------------------------------------------
# 5. CHECK TRAINED MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():

    st.error(
        "Trained model not found. Please train the model first."
    )

    st.code(
        "python main.py train --epochs 5"
    )

    st.stop()


# --------------------------------------------------
# 6. IMAGE UPLOAD
# --------------------------------------------------

st.subheader("Upload a Brain MRI Image")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload an image file supported by this prototype.",
)


# --------------------------------------------------
# 7. PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        # Display uploaded image and information
        image_column, result_column = st.columns(
            [1, 1]
        )

        with image_column:

            st.subheader("Uploaded Image")

            st.image(
                image,
                caption="Uploaded MRI image",
                use_container_width=True,
            )

            st.write(
                f"Original image dimensions: "
                f"{image.width} × {image.height}"
            )

            st.write(
                f"Model input dimensions: "
                f"{IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}"
            )

        with result_column:

            st.subheader("Classification Result")

            with st.spinner(
                "Analyzing the image using the trained model..."
            ):

                result = predict_pil_image(image)

            predicted_label = result["label"]
            scores = result["scores"]

            st.markdown("### Predicted Dataset Category")

            st.success(
                predicted_label.replace("_", " ").title()
            )

            st.caption(
                "This is the model's predicted dataset category, "
                "not a medical diagnosis."
            )

            st.divider()

            st.markdown("### Model Output Scores")

            score_data = []

            for class_name in CLASS_NAMES:

                score = float(scores[class_name])

                score_data.append(
                    {
                        "Category": class_name.title(),
                        "Output score": score,
                    }
                )

                st.write(
                    f"**{class_name.title()}**: "
                    f"{score:.4f}"
                )

                st.progress(
                    max(0.0, min(1.0, score))
                )

        # ------------------------------------------
        # 8. SCORE COMPARISON
        # ------------------------------------------

        st.divider()

        st.subheader("Compare Model Output Scores")

        score_df = pd.DataFrame(score_data)

        st.bar_chart(
            score_df.set_index("Category")
        )

        st.caption(
            "Output scores are model outputs and are not "
            "necessarily calibrated clinical probabilities."
        )

    except Exception as error:

        st.error(
            "The image could not be processed."
        )

        st.exception(error)


# --------------------------------------------------
# 9. MODEL EVALUATION RESULTS
# --------------------------------------------------

st.divider()

st.subheader("Model Evaluation")

metrics_path = OUTPUT_DIR / "test_metrics.json"

if metrics_path.exists():

    try:

        with open(
            metrics_path,
            "r",
            encoding="utf-8",
        ) as file:

            metrics = json.load(file)

        st.write(
            "These results are loaded from the saved "
            "evaluation output."
        )

        # Display numeric test metrics if available.
        numeric_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
        }

        if numeric_metrics:

            metric_columns = st.columns(
                min(len(numeric_metrics), 4)
            )

            for index, (name, value) in enumerate(
                numeric_metrics.items()
            ):

                with metric_columns[
                    index % len(metric_columns)
                ]:

                    st.metric(
                        label=name.replace("_", " ").title(),
                        value=(
                            f"{value:.4f}"
                            if isinstance(value, float)
                            else value
                        ),
                    )

        else:

            st.json(metrics)

    except (OSError, json.JSONDecodeError) as error:

        st.warning(
            f"Could not read the metrics file: {error}"
        )

else:

    st.info(
        "Evaluation metrics are not available yet. "
        "Run the evaluation command first."
    )


# --------------------------------------------------
# 10. CONFUSION MATRIX
# --------------------------------------------------

confusion_matrix_path = (
    OUTPUT_DIR / "confusion_matrix.png"
)

if confusion_matrix_path.exists():

    st.subheader("Confusion Matrix")

    st.image(
        str(confusion_matrix_path),
        caption="Model classification results on the test dataset",
        use_container_width=True,
    )


# --------------------------------------------------
# 11. TRAINING CURVES
# --------------------------------------------------

training_curves_path = (
    OUTPUT_DIR / "training_curves.png"
)

if training_curves_path.exists():

    st.subheader("Training Performance")

    st.image(
        str(training_curves_path),
        caption="Training and validation performance",
        use_container_width=True,
    )


# --------------------------------------------------
# 12. EDUCATIONAL DISCLAIMER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="disclaimer">
        <strong>Important Educational Disclaimer</strong>
        <p>
        This application is an educational and research prototype.
        It is not a clinically validated medical device and must not
        be used to diagnose, exclude, or make treatment decisions
        about brain tumors.
        </p>
        <p>
        Predictions may be incorrect. If you have concerns about
        an MRI scan, consult a qualified medical professional.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "AI-Powered Medical Image Analysis | Brain MRI Classification"
)

