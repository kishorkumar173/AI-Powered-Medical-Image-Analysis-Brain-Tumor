# 🧠 AI-Powered Medical Image Analysis: Brain Tumor Classification

An AI-powered brain MRI image classification project using **Deep Learning and MobileNetV2 Transfer Learning** to classify MRI images into four dataset categories: Glioma, Meningioma, No Tumor, and Pituitary.

The project includes a complete machine learning pipeline for dataset preparation, model training, evaluation, image prediction, and an interactive web application built with Streamlit.

> **Disclaimer:** This project is an educational and research prototype. It is not a clinically validated medical device and must not be used for medical diagnosis or treatment decisions.

---

## 📌 Table of Contents

* [Project Overview](#-project-overview)
* [Problem Statement](#-problem-statement)
* [Objectives](#-objectives)
* [Features](#-features)
* [Technologies Used](#-technologies-used)
* [Dataset Description](#-dataset-description)
* [Model Architecture](#-model-architecture)
* [System Workflow](#-system-workflow)
* [Project Structure](#-project-structure)
* [Installation and Setup](#-installation-and-setup)
* [Model Training](#-model-training)
* [Model Evaluation](#-model-evaluation)
* [MRI Image Prediction](#-mri-image-prediction)
* [Run the Web Application](#-run-the-web-application)
* [Evaluation Metrics](#-evaluation-metrics)
* [Limitations](#-limitations)
* [Future Enhancements](#-future-enhancements)
* [Author](#-author)

---

## 📖 Project Overview

Brain MRI scans are important imaging tools used by healthcare professionals when investigating neurological conditions.

This project demonstrates how deep learning can be applied to MRI image classification using a pretrained convolutional neural network.

The system uses **MobileNetV2**, a convolutional neural network pretrained on ImageNet, as the feature extraction backbone. Transfer learning allows the project to reuse pretrained image features and train a classification layer for four categories in the selected brain MRI dataset.

The application allows users to upload an MRI image through a Streamlit interface and view the model's predicted dataset category and output scores.

The project is designed to demonstrate practical applications of computer vision, transfer learning, model evaluation, and web application development.

---

## 🎯 Problem Statement

Medical image datasets can contain large numbers of MRI images that need to be organized and analyzed.

This project explores the use of deep learning to classify brain MRI images into predefined dataset categories.

The goal is to develop an educational prototype that demonstrates an end-to-end image classification workflow, from dataset preparation and model training to prediction and visualization.

The model's output is intended for educational experimentation and is not a substitute for professional medical interpretation.

---

## 🚀 Objectives

* Develop a deep learning model for brain MRI image classification.
* Implement transfer learning using MobileNetV2.
* Apply image augmentation to support model generalization.
* Prepare training, validation, and testing datasets.
* Evaluate the trained model using classification metrics.
* Generate a confusion matrix and training performance plots.
* Develop an interactive Streamlit web application.
* Allow users to upload MRI images and view classification results.
* Document the implementation and make the source code available on GitHub.

---

## ✨ Features

### 1. Deep Learning Classification

Uses MobileNetV2 with pretrained ImageNet weights for feature extraction.

### 2. Transfer Learning

Freezes the pretrained backbone during initial training and learns a task-specific classification layer.

### 3. Data Augmentation

Applies random rotation, zoom, and contrast adjustments during training.

### 4. Model Evaluation

Supports evaluation using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

### 5. MRI Image Prediction

Accepts an MRI image and returns the model's predicted dataset category and class output scores.

### 6. Interactive Web Interface

Uses Streamlit to provide:

* MRI image upload
* Image preview
* Predicted category
* Class output scores
* Score comparison chart
* Evaluation visualizations

### 7. Training Visualization

Saves training history and performance plots for further analysis.

---

## 🛠 Technologies Used

| Technology   | Purpose                                     |
| ------------ | ------------------------------------------- |
| Python       | Primary programming language                |
| TensorFlow   | Deep learning framework                     |
| Keras        | Model development and training              |
| MobileNetV2  | Pretrained CNN backbone                     |
| NumPy        | Numerical operations                        |
| Pandas       | Tabular data handling                       |
| Matplotlib   | Visualization of evaluation results         |
| Scikit-learn | Classification metrics and confusion matrix |
| Pillow       | MRI image loading and processing            |
| Streamlit    | Interactive web application                 |
| Git          | Version control                             |
| GitHub       | Source code hosting                         |
| VS Code      | Development environment                     |

---

## 📂 Dataset Description

This project uses the Brain Tumor MRI Dataset available on Kaggle.

**Dataset source:** [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Download the dataset and place its contents in the expected project directory.

### Dataset Categories

| Category   | Description                                            |
| ---------- | ------------------------------------------------------ |
| Glioma     | MRI images labelled as glioma in the dataset           |
| Meningioma | MRI images labelled as meningioma in the dataset       |
| No Tumor   | MRI images labelled as having no tumor                 |
| Pituitary  | MRI images labelled as pituitary tumors in the dataset |

### Expected Directory Structure

```text
data/
└── brain_tumor_dataset/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    │
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

**Important:** The directory names must match those expected by the project's configuration and data-loading code. Adjust the folder structure if your downloaded dataset uses different names.

The dataset is excluded from the Git repository to avoid committing a large number of image files. Users must download it separately and comply with its applicable terms.

---

## 🧠 Model Architecture

The project uses **MobileNetV2 with transfer learning**.

MobileNetV2 is a convolutional neural network designed to extract image features efficiently.

### Architecture Components

1. **Input Layer**
   Accepts RGB images resized to 224 × 224 pixels.

2. **Data Augmentation**
   Applies random rotation, zoom, and contrast transformations during training.

3. **Normalization**
   Rescales pixel values from the range `[0, 255]` to `[-1, 1]`.

4. **MobileNetV2 Backbone**
   Uses pretrained ImageNet weights to extract image features. Its weights are frozen during initial training.

5. **Global Average Pooling**
   Converts spatial feature maps into a compact feature vector.

6. **Dropout Layer**
   Uses a dropout rate of 0.30 as a regularization measure.

7. **Dense Classification Layer**
   Uses four output units and softmax activation to generate scores for the four dataset categories.

### Model Configuration

| Parameter                         | Value                            |
| --------------------------------- | -------------------------------- |
| Model                             | MobileNetV2                      |
| Learning approach                 | Transfer learning                |
| Input dimensions                  | 224 × 224 × 3                    |
| Number of classes                 | 4                                |
| Optimizer                         | Adam                             |
| Initial learning rate             | 0.001                            |
| Loss function                     | Sparse categorical cross-entropy |
| Evaluation metric during training | Sparse categorical accuracy      |
| Dropout rate                      | 0.30                             |
| Batch size                        | 16                               |
| Initial training epochs           | 5, configurable                  |

The number of epochs can be changed through the training command.

---

## 🔄 System Workflow

```text
             Brain MRI Dataset
                     |
                     v
             Dataset Preparation
                     |
                     v
           Training / Validation
                     |
                     v
            MobileNetV2 Model
                     |
                     v
               Model Training
                     |
                     v
              Saved Model
                     |
           +---------+----------+
           |                    |
           v                    v
     Model Evaluation      Image Prediction
           |                    |
           v                    v
    Evaluation Metrics     Predicted Category
    Confusion Matrix       and Output Scores
           |                    |
           +---------+----------+
                     |
                     v
              Streamlit UI
                     |
                     v
             Display Results
```

---

## 📁 Project Structure

```text
AI-Powered-Medical-Image-Analysis-Brain-Tumor/
│
├── data/
│   └── brain_tumor_dataset/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── brain_tumor_mobilenetv2.keras
│
├── outputs/
│   ├── test_metrics.json
│   ├── confusion_matrix.png
│   ├── prediction_examples.png
│   ├── training_curves.png
│   └── training_history.csv
│
├── notebooks/
├── images/
├── docs/
│
├── main.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

The dataset, trained model, and generated output files may not all be included in the GitHub repository. The structure above represents the intended local project organization.

---

## ⚙️ Installation and Setup

### Prerequisites

Install the following:

* Python 3.11
* Visual Studio Code or another Python IDE
* Git
* Internet connection for downloading dependencies and pretrained weights

Python 3.11 is recommended for the specified TensorFlow environment.

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

Navigate into the project directory:

```bash
cd AI-Powered-Medical-Image-Analysis-Brain-Tumor
```

### Step 2: Create a Virtual Environment

On Windows:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, follow the appropriate Python virtual-environment instructions for your system rather than changing security settings unnecessarily.

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset

Download the dataset from Kaggle and extract it into:

```text
data/brain_tumor_dataset/
```

Verify that the `Training` and `Testing` directories contain the expected class folders.

### Step 5: Verify the Model Architecture

```bash
python -m src.model
```

This command builds the model and prints its architecture summary. It does not train the model.

---

## 🏋️ Model Training

To train the model for five epochs, run:

```bash
python main.py train --epochs 5
```

The training process uses:

* Training data
* Validation data
* Model checkpointing
* Early stopping
* CSV training logs
* Training performance plots

The best model checkpoint is saved at the configured model path:

```text
models/brain_tumor_mobilenetv2.keras
```

Training duration depends on the dataset size and available hardware.

---

## 📊 Model Evaluation

After training, evaluate the saved model using the test dataset:

```bash
python main.py evaluate
```

The evaluation script calculates classification metrics and generates visualizations.

Expected output files include:

```text
outputs/
├── test_metrics.json
├── confusion_matrix.png
└── prediction_examples.png
```

Training-related files may include:

```text
outputs/
├── training_history.csv
├── training_history.json
└── training_curves.png
```

The exact outputs depend on the scripts and configuration used.

Evaluation results should be reported using the actual generated metrics. Results on a particular dataset do not establish clinical reliability or guarantee performance on images from other hospitals, scanners, or populations.

---

## 🔍 MRI Image Prediction

To classify a single MRI image, run:

```bash
python main.py predict --image "PATH_TO_YOUR_MRI_IMAGE"
```

Replace the placeholder with the path to an actual image file.

For example, if your image is stored inside the testing dataset:

```powershell
python main.py predict --image "data/brain_tumor_dataset/Testing/glioma/Te-gl_1.jpg"
```

The filename above is an example and must match a file that exists in your local dataset.

The script displays:

* The image path
* The predicted dataset category
* Output scores for all four categories

The predicted category is the model's classification result, not a medical diagnosis.

---

## 🌐 Run the Streamlit Web Application

Start the web interface with:

```bash
streamlit run app.py
```

Streamlit will display a local address, usually:

```text
http://localhost:8501
```

Open this address in your browser.

### How to Use the Application

1. Launch the application.
2. Upload a supported brain MRI image.
3. View the uploaded image.
4. Review the predicted dataset category.
5. Examine the output scores for all four categories.
6. Compare scores using the chart.
7. Review available evaluation metrics and visualizations.

The web application loads the existing trained model. It does not retrain the model every time an image is uploaded.

---

## 📈 Evaluation Metrics

The project uses the following metrics to evaluate classification performance.

| Metric           | Meaning                                                         |
| ---------------- | --------------------------------------------------------------- |
| Accuracy         | Proportion of test images classified correctly                  |
| Precision        | Proportion of positive predictions for a class that are correct |
| Recall           | Proportion of actual images in a class correctly identified     |
| F1-score         | Harmonic mean of precision and recall                           |
| Confusion matrix | Shows correct and incorrect predictions for each class          |

### Results

Add the actual values from your evaluation output.

| Metric          | Result                  |
| --------------- | ----------------------- |
| Test accuracy   | Add your measured value |
| Macro precision | Add your measured value |
| Macro recall    | Add your measured value |
| Macro F1-score  | Add your measured value |

Do not publish placeholder values as measured results. For class-specific analysis, include precision, recall, and F1-score for each of the four categories.

---

## ⚠️ Limitations

* The model is trained on a specific public dataset and may not generalize to other MRI datasets.
* Incorrect predictions can occur, including confusion between different categories.
* Output scores are not necessarily calibrated clinical probabilities.
* The model has not been clinically validated.
* Dataset labels may contain errors or inconsistencies.
* Performance may be affected by image quality, acquisition settings, preprocessing, and dataset bias.
* A successful prediction does not establish that a model is safe or accurate for medical use.
* The application is not intended for diagnosis, treatment recommendations, or other clinical decisions.

All medical concerns should be discussed with qualified healthcare professionals.

---

## 🔮 Future Enhancements

Potential future improvements include:

* Fine-tuning selected MobileNetV2 layers.
* Comparing MobileNetV2 with other CNN architectures.
* Adding Grad-CAM visualizations to investigate model attention.
* Improving data-quality checks and duplicate detection.
* Evaluating class imbalance and per-class performance.
* Testing on an independent dataset from a different source.
* Adding automated experiment tracking.
* Improving interface accessibility and error handling.
* Deploying the educational application to a suitable hosting platform.

Any future model improvements should be verified experimentally rather than assumed to improve performance.

---

## 👨‍💻 Author

**Kishor Kumar L**

Computer Science and Engineering — Artificial Intelligence and Machine Learning

AMC Engineering College



---

## 📜 Disclaimer

This project was developed for educational and research purposes to demonstrate deep learning-based medical image classification.

It is not a clinically validated medical device. Its output must not be used to diagnose or rule out brain tumors, select treatments, or replace professional medical judgment.

The project author makes no claim that the model is suitable for clinical use.

---
