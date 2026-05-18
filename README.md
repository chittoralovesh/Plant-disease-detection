<div align="center">

# 🌿 Plant Disease Detection

### *Detect plant diseases from a single leaf image, powered by Deep Learning.*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-features">Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-roadmap">Roadmap</a> •
  <a href="#-contributing">Contributing</a>
</p>

</div>

---

## 🌱 Overview

**Plant Disease Detection** is a deep learning–based application that helps farmers, agronomists, and home gardeners identify diseases in plants from a single photograph of a leaf. By leveraging a Convolutional Neural Network (CNN) trained on a large dataset of healthy and diseased plant leaves, the system can classify diseases in seconds and provide actionable insights.

Crop diseases are responsible for an estimated **20–40% loss in global food production every year**. Early and accurate detection is the single most effective intervention a farmer can make. This project aims to put that capability into anyone's hands — using nothing more than a smartphone camera.

---

## ✨ Features

- 🍃 **Real-time disease classification** from leaf images
- 🧠 **Deep CNN model** trained on the PlantVillage dataset
- 🌾 **Multi-crop support** — detects diseases across multiple plant species (tomato, potato, apple, grape, corn, and more)
- 🖼️ **Drag-and-drop image upload** through an intuitive web interface
- ⚡ **Fast inference** — results in under a second on CPU
- 📊 **Confidence scores** for every prediction
- 💊 **Suggested remedies & preventive measures** for detected diseases
- 🌐 **Lightweight web app** — runs locally or deploys to the cloud with one command

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Deep Learning** | TensorFlow / Keras |
| **Image Processing** | OpenCV, Pillow, NumPy |
| **Web Framework** | Streamlit *(or Flask — update to match your `app/`)* |
| **Visualization** | Matplotlib, Seaborn |
| **Dataset** | [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) |

---

## 📸 Demo

> *Add a screenshot or GIF of your app in action here.*

<div align="center">
  <img src="https://via.placeholder.com/800x400.png?text=Add+a+screenshot+of+your+app+here" alt="App Demo" width="80%">
</div>

---

## 📂 Project Structure

```
Plant-disease-detection/
│
├── app/                       # Application source code
│   ├── main.py                # Entry point for the web app
│   ├── model/                 # Trained model files (.h5 / .keras)
│   ├── utils/                 # Helper functions (preprocessing, prediction)
│   └── static/                # Static assets (CSS, images)
│
├── notebooks/                 # Jupyter notebooks for training & experiments
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

> *Tweak this tree to mirror your actual `app/` folder once finalized.*

---

## ⚙️ Installation

### Prerequisites
- Python **3.9 or higher**
- `pip` package manager
- (Optional) A virtual environment tool — `venv` or `conda`

### Step 1 — Clone the repository
```bash
git clone https://github.com/chittoralovesh/Plant-disease-detection.git
cd Plant-disease-detection
```

### Step 2 — Create and activate a virtual environment
```bash
# On macOS / Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **Note:** If `requirements.txt` isn't present, install the core packages manually:
> ```bash
> pip install tensorflow streamlit numpy pillow opencv-python matplotlib
> ```

---

## 🚀 Usage

### Run the web application
```bash
cd app
streamlit run main.py
```

Then open your browser at **http://localhost:8501** and:
1. Upload a clear image of a plant leaf 🍃
2. Click **Predict**
3. View the predicted disease, confidence score, and recommended remedies

### Run inference from Python
```python
from app.utils.predict import predict_disease

result = predict_disease("path/to/leaf.jpg")
print(result)
# → {'disease': 'Tomato___Late_blight', 'confidence': 0.974}
```

---

## 🧠 Model Architecture

The classifier is built on a Convolutional Neural Network optimized for leaf image classification.

| Layer | Details |
|---|---|
| Input | 224 × 224 × 3 RGB image |
| Conv Blocks | Multiple Conv2D + ReLU + MaxPooling layers |
| Regularization | Dropout, Batch Normalization |
| Classifier Head | Global Average Pooling → Dense → Softmax |
| Loss | Categorical Cross-Entropy |
| Optimizer | Adam (lr = 1e-3) |

> *Replace the table above with your actual architecture details once finalized.*

### 📈 Results

| Metric | Score |
|---|---|
| Training Accuracy | `XX.XX%` |
| Validation Accuracy | `XX.XX%` |
| Test Accuracy | `XX.XX%` |

> *Plug in your real numbers once training is complete.*

---

## 🗂️ Dataset

This project uses the **[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)** — a publicly available collection of over **50,000 expert-curated images** of healthy and diseased plant leaves across **38 classes** spanning **14 crop species**, including:

🍅 Tomato &nbsp;•&nbsp; 🥔 Potato &nbsp;•&nbsp; 🍎 Apple &nbsp;•&nbsp; 🍇 Grape &nbsp;•&nbsp; 🌽 Corn &nbsp;•&nbsp; 🍓 Strawberry &nbsp;•&nbsp; 🍑 Peach &nbsp;•&nbsp; 🫑 Bell Pepper

---

## 🗺️ Roadmap

- [x] Build and train the base CNN classifier
- [x] Develop the web interface
- [ ] Add disease-specific treatment recommendations
- [ ] Add multilingual support (Hindi, Marathi, Spanish, French)
- [ ] Deploy a public demo on Streamlit Cloud / Hugging Face Spaces
- [ ] Build a mobile companion app (Flutter / React Native)
- [ ] Integrate live camera input for real-time detection
- [ ] Expand dataset to include field-condition (non-lab) imagery

---

## 🤝 Contributing

Contributions, suggestions, and feedback are warmly welcomed! 🌟

1. Fork the project
2. Create your feature branch — `git checkout -b feature/AmazingFeature`
3. Commit your changes — `git commit -m 'Add some AmazingFeature'`
4. Push to the branch — `git push origin feature/AmazingFeature`
5. Open a Pull Request

For major changes, please open an issue first to discuss what you'd like to change.

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

## 🙏 Acknowledgements

- [PlantVillage Project](https://plantvillage.psu.edu/) at Penn State University
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) teams
- [Streamlit](https://streamlit.io/) for the simplest ML deployment framework around
- The open-source community that makes projects like this possible 💚

---

## 📬 Contact

**Lovesh Chittora** — [@chittoralovesh](https://github.com/chittoralovesh)

Project Link: [https://github.com/chittoralovesh/Plant-disease-detection](https://github.com/chittoralovesh/Plant-disease-detection)

---

<div align="center">

### 🌾 *Built with the mission of helping farmers grow healthier crops.* 🌾

⭐ **If you find this project useful, please consider giving it a star!** ⭐

</div>
