from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import os
import cv2
import numpy as np
import base64
import json
from src.pipeline import pipeline, add_student as pipeline_add_student, train as pipeline_train
from src import firebase_db
from datetime import datetime

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENTS_DIR = os.path.join(BASE_DIR, "data", "students", "images")

def decode_base64_image(base64_str):
    """Convert base64 string to numpy image."""
    try:
        # Check if header exists
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Image decode error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enroll', methods=['GET'])
def enroll_page():
    return render_template('enroll.html')

@app.route('/attendance', methods=['GET'])
def attendance_page():
    return render_template('attendance.html')

@app.route('/students', methods=['GET'])
def students_page():
    """Page to manage/view students."""
    return render_template('students.html')

@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    """Receive student data and burst images."""
    data = request.json
    name = data.get('name')
    reg_no = data.get('reg_no')
    images = data.get('images') # List of base64 strings
    
    if not name or not reg_no or not images:
        return jsonify({"error": "Missing data"}), 400
        
    print(f"Received enrollment for {name} ({reg_no}) with {len(images)} images.")
    
    # Create directory
    student_dir = os.path.join(STUDENTS_DIR, reg_no)
    os.makedirs(student_dir, exist_ok=True)
    
    # Save images
    count = 0
    for idx, img_str in enumerate(images):
        img = decode_base64_image(img_str)
        if img is not None:
            filename = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.jpg"
            cv2.imwrite(os.path.join(student_dir, filename), img)
            count += 1
            
    # Add to Firebase
    try:
        firebase_db.add_student_to_db(reg_no, name)
    except Exception as e:
        return jsonify({"display_error": f"Firebase Error: {str(e)}"}), 500
    
    # Trigger incremental training
    try:
        pipeline_add_student(reg_no)
    except Exception as e:
        return jsonify({"display_error": f"Training Error: {str(e)}"}), 500
        
    return jsonify({"success": True, "message": f"Saved {count} images and trained model."})

@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        pipeline_train(incremental=False)
        return jsonify({"success": True, "message": "Full training complete."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    data = request.json
    image_str = data.get('image')
    threshold = data.get('threshold', 0.6)
    
    if not image_str:
        return jsonify({"error": "No image provided"}), 400
        
    img = decode_base64_image(image_str)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400
        
    result = pipeline.recognize_frame(img, threshold=threshold)
    
    # Convert numpy types to native python types for JSON serialization
    if 'box' in result and result['box']:
        result['box'] = [int(x) for x in result['box']]
        
    return jsonify(result)

@app.route('/api/students', methods=['GET'])
def api_get_students():
    """Fetch all enrolled students."""
    try:
        students = firebase_db.get_all_students_details()
        return jsonify(students)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs', methods=['GET'])
def get_logs():
    # Simple log fetcher from Firebase (limit 50 recent)
    try:
        logs_ref = firebase_db.db.collection('attendance_logs').order_by('timestamp', direction=firebase_db.firestore.Query.DESCENDING).limit(50)
        docs = logs_ref.stream()
        logs = []
        for doc in docs:
            log = doc.to_dict()
            # Convert timestamp to string
            if 'timestamp' in log and log['timestamp']:
                log['timestamp'] = log['timestamp'].isoformat() if hasattr(log['timestamp'], 'isoformat') else str(log['timestamp'])
            logs.append(log)
        return jsonify(logs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def api_analytics():
    """Get analytics data for dashboard."""
    try:
        stats = firebase_db.get_analytics_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/dashboard', methods=['GET'])
def dashboard_page():
    return render_template('dashboard.html')

@app.route('/history', methods=['GET'])
def history_page():
    return render_template('history.html')

@app.route('/api/history', methods=['GET'])
def api_history():
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({"error": "Date required"}), 400
    try:
        logs = firebase_db.get_logs_by_date(date_str)
        return jsonify(logs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/students/<reg_no>', methods=['DELETE'])
def api_delete_student(reg_no):
    """Delete a student by register number (roll number)."""
    try:
        # 1. Delete from Firebase
        firebase_db.delete_student_from_db(reg_no)
        
        # 2. Delete from Local Data & Embeddings
        pipeline.delete_student(reg_no, delete_data=True)
        
        return jsonify({"success": True, "message": f"Student {reg_no} deleted successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure data dirs exist
    os.makedirs(STUDENTS_DIR, exist_ok=True)
    # Host 0.0.0.0 for local network access
    app.run(host='0.0.0.0', port=5000, debug=True)
