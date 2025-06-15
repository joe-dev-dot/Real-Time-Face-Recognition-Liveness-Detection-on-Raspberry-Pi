# 🧠 Real-Time Face Recognition & Liveness Detection on Raspberry Pi

This project implements a **real-time face recognition and liveness detection system** optimized for the Raspberry Pi. It uses TensorFlow Lite, MediaPipe, and PiCamera2 for efficient, fast performance using multiprocessing.

---

## 📁 Project Structure

```
.
├── models/
│   ├── face_recognition_model.tflite
│   └── liveness_detection_model.tflite
├── dataset/
│   └── person_name/
│       ├── image1.jpg
│       └── image2.jpg
├── face_recognition.py
├── liveness_detection.py
├── generate_known_embeddings.py
├── main.py
├── requirements.txt
└── known_faces_encodings.pkl  ← Generated after running embedding script
```

---

## ⚙️ Features

- ✅ **Face Recognition**
  - TFLite-based embedding model (160x160 input)
  - MediaPipe face detection
  - Cosine similarity for identity matching
- ✅ **Liveness Detection**
  - Binary classifier: live vs spoof
  - ImageNet normalization (mean/std)
- ✅ **Multiprocessing**
  - Separates recognition and liveness tasks
  - Smooth frame rate on Raspberry Pi
- ✅ **Camera Integration**
  - PiCamera2 for live video feed

---

## 🔧 Installation

1. Clone this repository:

```bash
git clone https://github.com/joe-dev-dot/Real-Time-Face-Recognition-Liveness-Detection-on-Raspberry-Pi
cd project
```

2. (Optional) Create a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

3. Install the requirements:

```bash
pip install -r requirements.txt
```

---

## 🧬 Prepare Dataset

1. Add person folders with images to `dataset/`:

```
dataset/
├── alice/
│   ├── 1.jpg
│   └── 2.jpg
├── bob/
│   ├── 1.jpg
│   └── 2.jpg
```

2. Generate the `.pkl` embeddings:

```bash
python generate_known_faces_encodings.py
```

This will create `known_faces_encodings.pkl`.

---

## 🚀 Run the App

Launch the real-time recognition system:

```bash
python main.py
```

**Controls:**

- Press `q` to quit the app.

---

## 💡 Description of Files

| File                           | Description                                                    |
| ------------------------------ | -------------------------------------------------------------- |
| `main.py`                      | Captures video, starts multiprocessing, handles result display |
| `face_recognition.py`          | Face detection (MediaPipe), embedding extraction (TFLite)      |
| `liveness_detection.py`        | Liveness model preprocessing, inference, and output            |
| `generate_known_embeddings.py` | Extracts embeddings from `dataset/` and saves as `.pkl`        |

---

## 📊 Model Details

- **Face Recognition Model**

  - Input: `160x160`, normalized to `[-1, 1]`
  - Output: 128-d embedding vector

- **Liveness Detection Model**

  - Input: `224x224`, normalized using ImageNet mean/std
  - Output: Sigmoid probability (1D)
  - Threshold: `> 0.5` means spoof

---

## 🚨 Example Output

```
Alice (0.81)  
Liveness: live (0.27)

Unknown (0.23)  
Liveness: spoof (0.92)
```

---

## 🛈 Contact

- Maintainer: [Youssef Toumi]
- Email: [youssef.toumi@polytechnicien.tn]

Feel free to open issues or contribute with improvements!

