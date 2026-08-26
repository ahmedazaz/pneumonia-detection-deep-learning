# 🫁 Pneumonia Detection from Chest X-Ray Images Using Deep Learning

A deep learning project for detecting **pneumonia from chest X-ray images** and comparing multiple neural network architectures.

This project was developed as part of my **Master's Deep Learning coursework** and explores both models trained from scratch and transfer-learning approaches.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It is not intended for clinical diagnosis or medical decision-making.

---

## 📌 Project Overview

The goal of this project is to classify chest X-ray images into two categories:

* **NORMAL**
* **PNEUMONIA**

Four deep learning architectures were implemented and compared:

1. Baseline CNN
2. ResNet50 Fine-tuned
3. EfficientNetB0 Fine-tuned
4. Vision Transformer (ViT)

The project includes data preprocessing, augmentation, class imbalance handling, model training, fine-tuning, evaluation, confusion matrices, and model comparison.

---

## 📊 Dataset

The project uses the **Kaggle Chest X-Ray Pneumonia Dataset**.

Images are classified into:

* `NORMAL`
* `PNEUMONIA`

### Preprocessing

* Image size: **224 × 224**
* Batch size: **32**
* 20% of the training data used for validation
* Class weights used to reduce class imbalance

### Data Augmentation

The training pipeline includes:

* Random Rotation
* Random Zoom
* Random Translation

---

## 🧠 Models

### 1. Baseline CNN

A custom convolutional neural network built from scratch using:

* Conv2D
* MaxPooling2D
* GlobalAveragePooling2D
* Dense layers
* Dropout

The Baseline CNN provides a reference point for comparing more advanced architectures.

---

### 2. ResNet50 Fine-tuned

ResNet50 was initialized with **ImageNet pretrained weights**.

Training was performed in two stages:

1. Train the classification layers while keeping the pretrained backbone frozen.
2. Unfreeze the final layers and fine-tune them using a smaller learning rate.

ResNet50 achieved the **best balanced performance** in this project.

---

### 3. EfficientNetB0 Fine-tuned

EfficientNetB0 was also trained using transfer learning and later fine-tuned.

It achieved strong pneumonia detection performance and a high AUC score.

---

### 4. Vision Transformer (ViT)

A Vision Transformer was implemented and trained **from scratch**.

The model achieved high recall for pneumonia cases but had difficulty distinguishing normal X-ray images.

This highlights the challenge of training Vision Transformers without large datasets or pretrained weights.

---

## 📈 Model Performance

| Model                     |   Accuracy |  Precision |     Recall | Specificity |   F1 Score |        AUC |
| ------------------------- | ---------: | ---------: | ---------: | ----------: | ---------: | ---------: |
| Baseline CNN              |     77.40% |     74.75% |     96.41% |      45.73% |     84.21% |     89.98% |
| **ResNet50 Fine-tuned**   | **87.98%** | **85.23%** | **97.69%** |  **71.79%** | **91.04%** | **94.74%** |
| EfficientNetB0 Fine-tuned |     82.21% |     79.00% |     97.44% |      56.84% |     87.26% | **94.84%** |
| ViT                       |     66.99% |     65.92% |     97.69% |      15.81% |     78.72% |     72.26% |

---

## 🏆 Best Model: ResNet50 Fine-tuned

**ResNet50 Fine-tuned** provided the best overall balance between detecting pneumonia cases and correctly identifying normal images.

### Results

* **Accuracy:** 87.98%
* **Precision:** 85.23%
* **Recall:** 97.69%
* **Specificity:** 71.79%
* **F1 Score:** 91.04%
* **AUC:** 94.74%

From 390 pneumonia test images, the model correctly classified **381** and missed only **9**.

---

## 📉 Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* Specificity
* F1 Score
* ROC-AUC
* Confusion Matrix
* Training / Validation Accuracy
* Training / Validation Loss

Because missing a pneumonia case is particularly important in this classification problem, **Recall** was one of the key evaluation metrics.

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook

---

## 📁 Repository Structure

```text
pneumonia-detection-deep-learning/
│
├── pneumonia_detection_deep_learning_github.ipynb
├── README.md
└── .gitignore
```

The dataset and trained model files are not included in the repository because of their size.

---

## ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/ahmedazaz/pneumonia-detection-deep-learning.git
cd pneumonia-detection-deep-learning
```

Install the main dependencies:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

Place the dataset using the following structure:

```text
dataset/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── test/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── val/
```

Then open:

```text
pneumonia_detection_deep_learning_github.ipynb
```

and run the notebook cells.

---

## 🔍 Key Learning Outcomes

Through this project, I gained hands-on experience with:

* Medical image classification
* CNN architecture design
* Transfer learning
* Fine-tuning pretrained models
* Vision Transformers
* Data augmentation
* Class imbalance handling
* Model evaluation
* Confusion matrix analysis
* Comparing deep learning architectures

---

## 🚀 Future Improvements

Possible future improvements include:

* More extensive data augmentation
* Additional fine-tuning of pretrained models
* Grad-CAM for explainable AI
* Ensemble learning
* Pretrained Vision Transformer models
* Testing on additional external datasets

---

## 👨‍💻 Author

**Ahmed Azaz**

Master's Student in Computer Engineering
AI & Machine Learning
