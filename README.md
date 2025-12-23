# Skin Disease Classification using Deep Learning (HAM10000)

This project is a **deep learning–based skin disease classification system**
trained on the **HAM10000 (Human Against Machine)** dataset.
It demonstrates the application of **convolutional neural networks (CNNs)**
and **transfer learning** for multi-class medical image classification.

> ⚠️ **Disclaimer:**  
> This project is intended for **educational and research purposes only** and
> must **not** be used for real-world medical diagnosis or clinical decision-making.

---

## 📌 Project Overview

- Image-based classification of skin lesions
- Multi-class prediction across 7 dermatological categories
- Built to showcase **model training, evaluation, and ML workflow**
- Focus on handling **class imbalance** and **model generalization**

---

## 🗂 Dataset

- **HAM10000** (Human Against Machine with 10,000 training images)
- Publicly available dermatology image dataset
- **7 Classes:**
  - `akiec` – Actinic keratoses
  - `bcc` – Basal cell carcinoma
  - `bkl` – Benign keratosis-like lesions
  - `df` – Dermatofibroma
  - `mel` – Melanoma
  - `nv` – Melanocytic nevi
  - `vasc` – Vascular lesions

📁 *Note:* The dataset is **not included** in this repository due to size constraints.

---

## 🧠 Model Architecture

- **Base model:** MobileNet
- **Approach:** Transfer learning
- Fine-tuning of top layers for domain adaptation
- Custom classification head for 7-class output
- Class imbalance addressed using **class weighting**

---

## 📊 Evaluation Metrics

The model was evaluated using multiple performance metrics:

- **Accuracy:** ~79%
- **Weighted F1-score:** ~0.79
- **Macro F1-score:** ~0.57

Evaluation includes:
- Confusion matrix
- Training and validation curves
- Class-wise performance analysis

These metrics highlight the challenges of **imbalanced medical datasets**.

---

## 🛠 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **OpenCV**
- **Matplotlib / Seaborn** (for evaluation & visualization)

---

## 📁 Repository Structure

```text
skin-disease-detection/
│
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── outputs.txt                # Model outputs / logs (optional)
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore rules
│
├── static/
│   │   └── style.css          # Main stylesheet
│   │   ├── script.js          # Core frontend logic
│   │   └── style.js           # Toggle / UI interactions
│   │   └── ananya_narang.jpg  # Author image
│
├── templates/
│   ├── index.html
│   ├── result.html
│   ├── reports.html
│   ├── chat.html
│   ├── appointments.html
│   ├── clinics.html
│   ├── tips.html
│   ├── contact.html
│   └── about.html
│
└── training/
    └── train_cnn_improved.py  # Model training script
```
---

## 📌 Notes

- Trained model weights and datasets are intentionally excluded
- Focus of this repository is **code clarity and ML methodology**
- Results are dataset-dependent and may vary with hyperparameter tuning

---

## 👤 Author

**Ananya Narang**  
Deep Learning & AI/ML Enthusiast  

---

## 📄 License

This project is released strictly for **educational and research use**.
