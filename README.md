# 🚀 Face Recognition using Data Augmentation and CNN

A state-of-the-art attendance tracking system leveraging **Synthetic Data Augmentation**, **FaceNet (Inception-ResNet-v1 CNN)**, and **MTCNN (Detection CNN)**.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-Firestore-FFCA28?style=for-the-badge&logo=firebase&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

---

## 🌟 Key Features

- 🧠 **Advanced AI Recognition**: Uses MTCNN for robust face detection and Keras-FaceNet for high-accuracy embeddings.
- 🔄 **Synthetic Data Augmentation**: Automatically generates image variations (flip, brightness, contrast) during training to increase model accuracy and robustness.
- ☁️ **Cloud Integration**: Real-time sync with Firebase Firestore (metadata/logs) and Firebase Storage (trained models).
- 📊 **Dynamic Dashboard**: View daily attendance trends, student statistics, and real-time logs.
- 📹 **Multi-Mode Recognition**:
  - **Webcam**: Real-time browser-based recognition.
  - **Video Files**: Process pre-recorded `.mp4` videos for batch attendance.
- ✅ **Incremental Learning**: Add new students on-the-fly without retraining the entire database.
- 📱 **Network Ready**: Accessible across your local Wi-Fi network.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras-FaceNet, MTCNN
- **Computer Vision**: OpenCV
- **Database**: Firebase Firestore (Logs & Student Info)
- **Cloud Storage**: Firebase Storage (Face Embeddings)
- **Frontend**: Vanilla HTML/JS with Tailwind-inspired styling (Premium Dark/Glassmorphism design)

---

## ⚙️ Setup & Installation

### 1. Clone & Environment
```powershell
# Clone the repository
git clone https://github.com/ragav3883-blip/smart-attendence.git
cd smart-attendence

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
> [!IMPORTANT]
> If you are on a machine without a GUI (like a server) and only need the Web UI, `opencv-python-headless` is fine. For **Webcam preview** or **Video playback** (local GUI), you MUST install the standard version: `pip install opencv-python`.

```powershell
pip install -r requirements.txt
```

### 3. 🔥 Firebase Configuration (CRITICAL)

The system requires a Firebase Service Account key to communicate with the database.

1.  Go to the [Firebase Console](https://console.firebase.google.com/).
2.  Create a new project (e.g., `Smart Attendance`).
3.  **Firestore Database**: Enable it in "Production" or "Test" mode.
4.  **Storage**: Enable it (set up default bucket).
5.  **Service Account Key**:
    - Go to **Project Settings** (⚙️ icon) > **Service Accounts**.
    - Click **Generate New Private Key**.
    - A `.json` file will download to your computer.
6.  **Setup the File**:
    - **Rename** the downloaded file (e.g., `my-project-firebase-adm.json`) exactly to: **`serviceAccountKey.json`**.
    - **Place** this file in the **root directory** of this project (`smart-attendence/`).

---

## 🚀 How to Run

### Option 1: Web Management (Recommended)
This starts the Flask server with the modern control panel.
```powershell
python app.py
```
*Access via browser:* `http://localhost:5000`

### Option 2: Production Server (Waitress)
More stable for multiple connections.
```powershell
python server.py
```

### Option 3: Process Video File
Run recognition on a pre-recorded video.
```powershell
python video_attendance.py
```

### Option 4: Command Line
```powershell
# Train embeddings from local images
python main.py --train

# Run webcam recognition directly
python main.py --recognize
```

---

## 📂 Project Structure

```text
smart-attendence/
├── data/
│   ├── students/
│   │   ├── images/       # Raw student photos (organized by Roll No)
│   │   └── videos/       # .mp4 files for batch processing
├── src/
│   ├── pipeline.py       # Core ML logic (MTCNN + FaceNet)
│   ├── firebase_db.py    # Firebase Firestore/Storage handlers
│   └── logger.py         # Local CSV logging fallback
├── templates/            # Flask Web UI components
├── serviceAccountKey.json # [REQUIRED] Your Firebase credentials
├── app.py                # Flask Server
└── requirements.txt      # Python dependencies
```

---

## 📝 How it Works

1.  **Enrollment**: Add student details and capture images via the Web UI.
2.  **Training**: The system extracts 128-dimensional face embeddings and stores them in `embeddings.json`.
3.  **Sync**: The `embeddings.json` is synced with **Firebase Storage** so multiple devices can share the same database.
4.  **Recognition**: live frames are compared against embeddings using **Cosine Similarity**.
5.  **Logging**: Attendance is marked in **Firestore** with timestamps and confidence scores.

---

## 🤝 Contributing
Feel free to fork this project and submit PRs.

## 📄 License
Educational use only.
