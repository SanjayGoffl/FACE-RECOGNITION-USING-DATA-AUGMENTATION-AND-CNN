"""
Smart Attendance Pipeline
Complete end-to-end face detection, embedding, training, and recognition
All logic contained in single file - no external dependencies
"""

import json
import os
import glob
import csv
import shutil
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from keras_facenet import FaceNet


import sys
# Ensure src is in path if running directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import firebase_db
except ImportError:
    try:
        from src import firebase_db
    except ImportError:
        print("⚠️ Could not import firebase_db. Ensure it is in the same directory or src package.")
        firebase_db = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "reference", "embeddings.json")
STUDENTS_DIR = os.path.join(BASE_DIR, "data", "students", "images")
LOGS_DIR = os.path.join(BASE_DIR, "logs", "attendance.csv")


# ========== FACE DETECTION ==========

class FaceDetector:
    """MTCNN Face Detection"""
    
    def __init__(self):
        self.detector = MTCNN()
    
    def detect(self, frame_bgr):
        """Detect faces in BGR frame.
        Returns: [{'box': [x, y, w, h], 'confidence': float}, ...]
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            results = self.detector.detect_faces(rgb)
        except Exception as e:
            # Some frames can fail inside MTCNN when shapes are invalid; skip quietly
            print(f"⚠️ Face detection skipped: {e}")
            return []
        detections = []
        for r in results:
            box = r.get('box', [0, 0, 0, 0])
            conf = r.get('confidence', 0.0)
            detections.append({'box': box, 'confidence': conf})
        return detections
    
    @staticmethod
    def clamp_box(box, width, height, margin=0):
        """Clamp box coordinates within frame bounds."""
        x, y, w, h = box
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = w + 2 * margin
        h = h + 2 * margin
        x2 = min(width, x + w)
        y2 = min(height, y + h)
        x = max(0, x)
        y = max(0, y)
        w = max(1, x2 - x)
        h = max(1, y2 - y)
        return [x, y, w, h]


# ========== FACE EMBEDDING ==========

class FaceEmbedder:
    """FaceNet Embedding Generation"""
    
    def __init__(self):
        cache_dir = os.path.join(BASE_DIR, "models")
        os.makedirs(cache_dir, exist_ok=True)
        self.model = FaceNet(cache_folder=cache_dir)
    
    @staticmethod
    def crop_and_resize(frame_bgr, box, size=160):
        """Crop and resize face from frame."""
        x, y, w, h = box
        h_img, w_img = frame_bgr.shape[:2]
        x = max(0, x)
        y = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        face_bgr = frame_bgr[y:y2, x:x2]
        if face_bgr.size == 0:
            return None
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        pil_img = pil_img.resize((size, size))
        return np.asarray(pil_img)
    
    def embed(self, face_rgb_160):
        """Generate 512-d embedding for face image."""
        if face_rgb_160 is None:
            return None
        embeddings = self.model.embeddings([face_rgb_160])
        return embeddings[0].astype(np.float32)


# ========== UI DRAWING ==========

def draw_box(frame_bgr, box, color=(0, 255, 0), thickness=2):
    """Draw bounding box on frame."""
    x, y, w, h = box
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, thickness)


def draw_label(frame_bgr, text, box, color=(0, 255, 0)):
    """Draw label text above box."""
    x, y, w, h = box
    label_bg_color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame_bgr, (x, max(0, y - th - 10)), (x + tw + 10, y), label_bg_color, -1)
    cv2.putText(frame_bgr, text, (x + 5, y - 5), font, font_scale, color, thickness, cv2.LINE_AA)


# ========== ATTENDANCE LOGGING ==========

class AttendanceLogger:
    """Log attendance to CSV file"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._marked = False
    
    def mark_once(self, name: str, roll_no: str, confidence: float = 1.0):
        """Mark attendance once per session."""
        if self._marked:
            return False
        
        # Log to Firebase
        if firebase_db:
            try:
                firebase_db.log_attendance(roll_no, name, confidence, method="web-ui")
            except Exception as e:
                print(f"⚠️ Firebase logging failed: {e}")

        # Keep local CSV backup
        now = datetime.now().isoformat(timespec='seconds')
        exists = os.path.exists(self.csv_path)
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(['name', 'roll_no', 'timestamp', 'confidence'])
                writer.writerow([name, roll_no, now, confidence])
        except Exception as e:
            print(f"⚠️ CSV logging failed: {e}")
            
        self._marked = True
        return True


# ========== PIPELINE ==========

class AttendancePipeline:
    """Complete attendance pipeline - all ML logic in one class."""
    
    def __init__(self):
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        self.logger = AttendanceLogger(LOGS_DIR)
        
        # Sync from cloud on init
        self.sync_from_cloud()

    def sync_from_cloud(self):
        """Download embeddings.json from Firebase Storage."""
        if not firebase_db: return
        
        try:
            bucket = firebase_db.storage.bucket()
            blob = bucket.blob("embeddings.json")
            if blob.exists():
                print("☁️ Syncing embeddings from Cloud Storage...")
                os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
                blob.download_to_filename(EMBEDDINGS_FILE)
                print("✅ Embeddings synced.")
            else:
                print("☁️ No embeddings found in cloud (Fresh Start).")
        except Exception as e:
            print(f"⚠️ Cloud sync failed: {e}")

    def sync_to_cloud(self):
        """Upload embeddings.json to Firebase Storage."""
        if not firebase_db: return
        
        try:
            if not os.path.exists(EMBEDDINGS_FILE):
                return
                
            bucket = firebase_db.storage.bucket()
            blob = bucket.blob("embeddings.json")
            blob.upload_from_filename(EMBEDDINGS_FILE)
            print("☁️ ✅ Embeddings uploaded to Cloud Storage.")
        except Exception as e:
            print(f"⚠️ Cloud upload failed: {e}")
    
    def detect_and_embed(self, image):
        """Detect face in image and generate embedding.
        
        Returns:
            (embedding, box) or (None, None)
        """
        detections = self.detector.detect(image)
        if not detections:
            return None, None
        
        det = max(detections, key=lambda d: d['confidence'])
        box = self.detector.clamp_box(det['box'], image.shape[1], image.shape[0], margin=10)
        face = self.embedder.crop_and_resize(image, box)
        emb = self.embedder.embed(face)
        
        return emb, box
    
    def compare_embeddings(self, emb1, emb2):
        """Cosine similarity between embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        
        emb1 = np.array(emb1, dtype=np.float32)
        emb2 = np.array(emb2, dtype=np.float32)
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def normalize_embedding(self, emb):
        """Normalize embedding vector."""
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return (emb / (norm + 1e-8)).tolist()
    
    def _get_all_students_from_csv(self):
        """Get all students from Firebase (fallback to CSV if fail)."""
        if firebase_db:
            try:
                print("🔄 Fetching students from Firebase...")
                return firebase_db.get_all_students_dict()
            except Exception as e:
                print(f"⚠️ Firebase fetch failed: {e}")
        
        # Fallback to local CSV? Or just empty. 
        # For now, let's assuming we rely on Firebase. 
        # But if students.csv exists, we can use it.
        csv_path = os.path.join(BASE_DIR, "data", "students", "info", "students.csv")
        if not os.path.exists(csv_path):
            return {}
        
        students = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                roll = row['roll_number'].strip()
                name = row['name'].strip()
                students[roll] = name
        return students
    
    def _process_student(self, roll_no, name):
        """Process a single student and return their embedding data.
        
        Returns:
            dict: {"name": str, "embedding": list} or None if failed
        """
        student_dir = os.path.join(STUDENTS_DIR, roll_no)
        if not os.path.isdir(student_dir):
            print(f"⚠️  Skipping {roll_no} - folder not found")
            return None
        
        # Get all photos (case-insensitive extensions)
        photos = sorted(
            glob.glob(os.path.join(student_dir, "*.jpg")) +
            glob.glob(os.path.join(student_dir, "*.JPG")) +
            glob.glob(os.path.join(student_dir, "*.jpeg")) +
            glob.glob(os.path.join(student_dir, "*.JPEG")) +
            glob.glob(os.path.join(student_dir, "*.png")) +
            glob.glob(os.path.join(student_dir, "*.PNG"))
        )
        
        if len(photos) < 2:
            print(f"⚠️  Skipping {roll_no} - need at least 2 photos, found {len(photos)}")
            return None
        
        # Generate embeddings from all photos
        embs = []
        for photo_path in photos:
            img = cv2.imread(photo_path)
            if img is None:
                continue
            emb, _ = self.detect_and_embed(img)
            if emb is not None:
                embs.append(emb)
        
        if not embs:
            print(f"❌ {roll_no} - no valid faces detected")
            return None
        
        # Compute mean embedding
        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
        mean_emb = self.normalize_embedding(mean_emb)
        
        print(f"✓ {roll_no}: {name} ({len(embs)} faces)")
        return {
            "name": name,
            "embedding": mean_emb
        }

    def delete_student(self, roll_no, delete_data=True):
        """Delete student from embeddings model and optionally delete data.

        Args:
            roll_no: Student roll number
            delete_data: If True, delete the image folder in data/students/images

        Returns:
            bool: True if deleted (or not found but processed), False on error
        """
        # 1. Update Embeddings
        embeddings = self.load_embeddings()
        if roll_no in embeddings:
            del embeddings[roll_no]
            # Save back
            try:
                os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
                with open(EMBEDDINGS_FILE, "w") as f:
                    json.dump(embeddings, f)
                print(f"🗑️ Removed {roll_no} from embeddings model.")
            except Exception as e:
                print(f"⚠️ Failed to save embeddings during deletion: {e}")
                return False
        else:
            print(f"⚠️ {roll_no} not found in embeddings (safe to ignore if not trained yet).")

        # 2. Delete Data Folder
        if delete_data:
            student_dir = os.path.join(STUDENTS_DIR, roll_no)
            if os.path.exists(student_dir):
                try:
                    shutil.rmtree(student_dir)
                    print(f"🗑️ Deleted data folder: {student_dir}")
                except Exception as e:
                    print(f"⚠️ Failed to delete data folder: {e}")
                    return False
            else:
                print(f"⚠️ specific data folder {student_dir} not found.")

        return True

    def add_students_incremental(self, specific_roll_nos=None):
        """Add new students incrementally WITHOUT retraining existing ones.
        
        Args:
            specific_roll_nos: Optional list of roll numbers to add. If None, adds all new students.
        
        Returns:
            int: Number of students added
        """
        # Load existing embeddings
        existing_embeddings = self.load_embeddings()
        
        # Get all students from CSV
        all_students = self._get_all_students_from_csv()
        
        if not all_students:
            print("❌ No students in CSV")
            return 0
        
        # Determine which students to process
        if specific_roll_nos:
            # Process only specified roll numbers
            students_to_add = {roll: all_students[roll] for roll in specific_roll_nos if roll in all_students}
        else:
            # Process only NEW students (not in existing embeddings)
            # For cloud functionality, we assume training happens LOCALLY usually, or we need to download images.
            # But "Free Tier Deployment" usually means INFERENCE only.
            # If we want to TRAIN on cloud, we need to download images from storage?
            # Complexity: High. 
            # Simplified approach: We assume images are present locally (or uploaded via API). 
            # If this is Cloud Run, images from API are saved to "STUDENTS_DIR" which is ephemeral.
            # So we process them, update embeddings, push embeddings to cloud. The images are lost on restart, but that's fine as long as we have the embedding.
            # We strictly don't need to keep images for inference.
            
            students_to_add = {roll: name for roll, name in all_students.items() 
                             if roll not in existing_embeddings}
        
        if not students_to_add:
            print("✅ No new students to add. All students already trained.")
            return 0
        
        print(f"\n🔄 Adding {len(students_to_add)} new student(s) incrementally...")
        print(f"   (Keeping {len(existing_embeddings)} existing student(s))\n")
        
        added_count = 0
        for roll_no, name in students_to_add.items():
            result = self._process_student(roll_no, name)
            if result:
                existing_embeddings[roll_no] = result
                added_count += 1
        
        # Save updated embeddings (existing + new)
        os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
        with open(EMBEDDINGS_FILE, "w") as f:
            json.dump(existing_embeddings, f)
            
        # Push to cloud
        self.sync_to_cloud()
        
        print(f"\n✅ Added {added_count} new student(s)")
        print(f"   Total students: {len(existing_embeddings)}\n")
        return added_count
    
    def train_batch(self, incremental=True):
        """Train embeddings for students from CSV and folders.
        
        Args:
            incremental: If True (default), only processes new students.
                        If False, retrains ALL students from scratch.
        """
        if incremental:
            # INCREMENTAL MODE: Only add new students
            self.add_students_incremental()
        else:
            # FULL RETRAIN MODE: Retrain everything from scratch
            print("\n⚠️  FULL RETRAIN MODE - Processing ALL students from scratch\n")
            
            all_students = self._get_all_students_from_csv()
            
            if not all_students:
                print("❌ No students in CSV")
                return
            
            print(f"Training {len(all_students)} students...")
            embeddings = {}
            
            for roll_no, name in all_students.items():
                result = self._process_student(roll_no, name)
                if result:
                    embeddings[roll_no] = result
            
            # Save embeddings
            os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
            with open(EMBEDDINGS_FILE, "w") as f:
                json.dump(embeddings, f)
            
            print(f"\n✓ Trained {len(embeddings)} students\n")
    
    def load_embeddings(self):
        """Load all student embeddings."""
        if not os.path.exists(EMBEDDINGS_FILE):
            return {}
        with open(EMBEDDINGS_FILE, "r") as f:
            return json.load(f)
    
    def recognize_live(self, threshold=0.6):
        """Run real-time attendance recognition."""
        embeddings = self.load_embeddings()
        if not embeddings:
            print("❌ No embeddings found. Run --train first.")
            return
        
        print(f"Recognizing {len(embeddings)} students. Press 'q' to quit.\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            emb, box = self.detect_and_embed(frame)
            
            if emb is not None and box is not None:
                # Find best match across all students
                best_sim = 0
                best_roll = None
                best_name = None
                
                for roll_no, data in embeddings.items():
                    ref_emb = data['embedding']
                    sim = self.compare_embeddings(emb, ref_emb)
                    
                    if sim > best_sim:
                        best_sim = sim
                        best_roll = roll_no
                        best_name = data['name']
                
                draw_box(frame, box)
                
                if best_sim >= threshold:
                    label = f"{best_name} ({best_roll}) | {best_sim:.2f}"
                    self.logger.mark_once(best_name, best_roll)  # idempotent; no repeated logs
                else:
                    label = f"Unknown | {best_sim:.2f}"
                
                draw_label(frame, label, box)
            
            cv2.imshow('Smart Attendance', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


    def recognize_frame(self, frame, threshold=0.6):
        """Recognize face in a single frame (for Web UI)."""
        embeddings = self.load_embeddings()
        if not embeddings:
            return {"status": "error", "message": "No embeddings found"}
            
        emb, box = self.detect_and_embed(frame)
        if emb is None:
            return {"status": "no_face"}
            
        # Find best match
        best_sim = 0
        best_roll = None
        best_name = None
        
        for roll_no, data in embeddings.items():
            ref_emb = data['embedding']
            sim = self.compare_embeddings(emb, ref_emb)
            if sim > best_sim:
                best_sim = sim
                best_roll = roll_no
                best_name = data['name']
        
        result = {
            "status": "success",
            "box": box,
            "confidence": float(best_sim)
        }
        
        if best_sim >= threshold:
            result.update({
                "match": True,
                "name": best_name,
                "roll_no": best_roll
            })
            # Log it (async or sync)
            # Create a new logger instance for stateless requests? 
            # Or use self.logger, but self.logger has state `_marked`.
            # For web API, we want logging.
            # But we might want to debounce logic.
            # For now, log every hit.
            if firebase_db:
                firebase_db.log_attendance(best_roll, best_name, best_sim, method="web-ui")
                
        else:
            result.update({
                "match": False,
                "name": "Unknown"
            })
            
        return result

# Convenience functions for main.py
pipeline = AttendancePipeline()

def train(incremental=True):
    """Train embeddings.
    
    Args:
        incremental: If True, only adds new students. If False, retrains all.
    """
    pipeline.train_batch(incremental=incremental)

def add_student(roll_no=None):
    """Add a specific student or all new students incrementally.
    
    Args:
        roll_no: Optional roll number. If None, adds all new students.
    """
    if roll_no:
        pipeline.add_students_incremental([roll_no])
    else:
        pipeline.add_students_incremental()

def recognize(threshold=0.6):
    """Run recognition."""
    pipeline.recognize_live(threshold)

def delete_student(roll_no, delete_data=True):
    """Delete a student by roll no."""
    pipeline.delete_student(roll_no, delete_data)

