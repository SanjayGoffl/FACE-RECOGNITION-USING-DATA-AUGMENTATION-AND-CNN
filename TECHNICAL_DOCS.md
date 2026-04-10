# 🛠️ Technical Documentation: Face Recognition Attendance System

This document provides a comprehensive overview of the technical architecture, models, dependencies, and user flows of the **Face Recognition using Data Augmentation and CNN** project.

---

## 1. 🏗️ High-Level Architecture
The system follows a **Cloud-Hybrid Architecture**:
- **Edge Inference**: Face detection and recognition happen on the server (Flask) receiving frames via a Web API.
- **Cloud Persistence**: Student metadata, daily logs, and trained embeddings are synchronized with **Firebase**.
- **Ephemeral Training**: Student images are processed into mathematical vectors (embeddings) and then discarded, ensuring privacy and storage efficiency.

---

## 2. 🧠 Machine Learning Stack (CNN)
The core of the system is built on two primary Convolutional Neural Network (CNN) architectures:

### A. Face Detection: MTCNN
- **Architecture**: Multi-task Cascaded Convolutional Networks.
- **Purpose**: Locates faces in a frame, identifies 5 facial landmarks (eyes, nose, mouth), and produces a bounding box.
- **Optimization**: Input frames are downscaled to **480p** before feeding into MTCNN to significantly reduce CPU load on cloud environments (Railway/Render).

### B. Face Embeddings: FaceNet
- **Architecture**: Inception-ResNet-v1 (Pre-trained on VGGFace2).
- **Purpose**: Converts a cropped face image (160x160) into a **512-dimensional vector** (embedding).
- **Matching Logic**: Uses **Cosine Similarity** to compare the incoming face vector against the database of stored vectors.
- **Threshold**: Defaults to `0.6`. A similarity score $>0.6$ is considered a "Match."

### C. Data Augmentation
During the training phase, the system applies **synthetic augmentation** to every captured frame to increase robustness:
- **Horizontal Flipping**: Mimics mirror-image camera angles.
- **Brightness Jittering**: Simulates low-light and high-light environments.
- **Scaling**: Adjusts for different distances from the camera.

---

## 3. 📦 Dependency Breakdown

| Dependency | Purpose |
| :--- | :--- |
| `tensorflow-cpu` | The engine that runs the FaceNet and MTCNN models. |
| `keras-facenet` | Provides the Pre-trained FaceNet model wrapper. |
| `mtcnn` | The detection framework for finding faces in video frames. |
| `opencv-python-headless` | Image manipulation (resizing, cropping, BGR-to-RGB conversion). |
| `firebase-admin` | Manages Firestore (logs) and Storage (syncing `embeddings.json`). |
| `numpy` | Used for mathematical operations, such as calculating the **Mean Embedding** for a student. |
| `gunicorn` | A production-grade WSGI server used for cloud deployment. |
| `flask-cors` | Handles Cross-Origin Resource Sharing for mobile/network access. |

---

## 🖥️ 4. Application Pages & UI

- **🏠 Dashboard (`/`)**:
    - Displays real-time stats (Total Students vs. Present Today).
    - Features a **Chart.js** line graph showing attendance trends over the last 7 days.
- **📸 Live Scan (`/attendance`)**:
    - Real-time video stream with an overlay canvas.
    - Features a "Switch Camera" toggle for mobile front/back camera support.
- **✍️ Enrollment (`/enroll`)**:
    - Step-based UI: 1. Student Info -> 2. AI Face Capture -> 3. Training/Sync.
- **👥 Student Management (`/students`)**:
    - Central database view with search/filter functionality.
    - Features **"Add Photos"** (Append Mode) to improve existing profiles.
- **📅 History (`/history`)**:
    - Allows admins to pick any date and view all attendance logs for that day.

---

## 🔄 5. Core User Flows

### A. The "Smart Enrollment" Flow
1. User enters Name and Registration Number.
2. Web browser requests Camera Permission (Front camera prioritized).
3. **Capture Mode**: System takes 10 burst photos, instructing the user to move slightly.
4. **Backend Training**:
    - MTCNN detects face in each photo.
    - Augmentation creates 4 variants per photo (Total: 40 face samples).
    - FaceNet generates embeddings for all 40 samples.
    - System calculates a **Mathematical Mean** of all 40 vectors to form a uniquely robust ID.
5. Result is pushed to **Firestore** and the `.json` model is uploaded to **Cloud Storage**.

### B. The "Real-Time Recognition" Flow
1. Camera feed samples a frame every 500ms.
2. Frame is downscaled and sent via POST to `/api/recognize`.
3. System checks **In-Memory Cache** for existing students (No disk I/O for speed).
4. MTCNN finds the face; FaceNet calculates current embedding.
5. If match is found, system logs attendance to **Firebase** and returns Name/Confidence.
6. UI draws a **Green Box** and triggers a "Success" toast.

### C. The "Data Append" Flow
1. Admin selects an existing student in `/students`.
2. System auto-fills credentials and switches to **Append Mode**.
3. New photos are taken and processed.
4. The existing face vector is recalculated using BOTH old and new data, making the recognition "smarter" over time.

---
