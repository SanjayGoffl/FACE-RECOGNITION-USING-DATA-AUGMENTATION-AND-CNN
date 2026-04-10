import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime, timedelta
import os
import json

# Initialize Firebase
# Note: For local development, we expect GOOGLE_APPLICATION_CREDENTIALS to be set
# or we use the default application credentials if running in an environment with them.
# If you have a serviceAccountKey.json, place it in the root and uncomment the lines below.

if not firebase_admin._apps:
    # Option A: Use Service Account File (Local)
    if os.path.exists("serviceAccountKey.json"):
        cred = credentials.Certificate("serviceAccountKey.json")
        try:
            with open("serviceAccountKey.json") as f:
                sa_key = json.load(f)
                project_id = sa_key.get("project_id")
        except:
            project_id = "smart-attend-sys-2025" # Fallback
            
    # Option B: Use Environment Variable (Railway/Cloud)
    elif os.environ.get("FIREBASE_SERVICE_ACCOUNT"):
        print("[INFO] Using FIREBASE_SERVICE_ACCOUNT env var...")
        service_account_info = json.loads(os.environ.get("FIREBASE_SERVICE_ACCOUNT"))
        cred = credentials.Certificate(service_account_info)
        project_id = service_account_info.get("project_id")
    else:
        raise ValueError("[ERROR] No serviceAccountKey.json found AND no FIREBASE_SERVICE_ACCOUNT env var set.")

    # Initialize App
    firebase_admin.initialize_app(cred, {
        'storageBucket': f"{project_id}.firebasestorage.app"
    })

db = firestore.client()

def add_student_to_db(reg_no, name):
    """Add or update student in Firestore."""
    doc_ref = db.collection('students').document(reg_no)
    doc_ref.set({
        'reg_no': reg_no,
        'name': name,
        'created_at': firestore.SERVER_TIMESTAMP,
        'last_trained': None
    }, merge=True)
    print(f"[SUCCESS] Student {name} ({reg_no}) added to Firestore.")

# Simple in-memory cache for debouncing: {reg_no: timestamp}
_log_cache = {}

def log_attendance(reg_no, name, confidence, method="web-cam"):
    """Log attendance to Firestore (with 60s cooldown)."""
    global _log_cache
    
    # Prune old cache entries to prevent memory leak (optional, but good practice)
    now = datetime.now()
    if len(_log_cache) > 1000:
        _log_cache = {k: v for k, v in _log_cache.items() if (now - v).total_seconds() < 60}

    # Check cooldown
    if reg_no in _log_cache:
        last_log_time = _log_cache[reg_no]
        if (now - last_log_time).total_seconds() < 60:
            print(f"[WAIT] Skipped duplicate log for {name} (cooldown active)")
            return

    # Update cache
    _log_cache[reg_no] = now
    
    log_data = {
        'reg_no': reg_no,
        'name': name,
        'timestamp': firestore.SERVER_TIMESTAMP,
        'confidence': float(confidence),
        'method': method,
        'date_str': now.strftime("%Y-%m-%d") # Helper for querying
    }
    
    # We can use a subcollection or a root collection. Root is easier for querying all logs.
    db.collection('attendance_logs').add(log_data)
    print(f"[SUCCESS] Attendance logged for {name} ({reg_no})")

def get_all_students_dict():
    """Get all students as a dictionary {reg_no: name}."""
    docs = db.collection('students').stream()
    students = {}
    for doc in docs:
        data = doc.to_dict()
        students[data['reg_no']] = data['name']
    return students

def get_all_students_details():
    """Get all students as a list of dicts with full details."""
    docs = db.collection('students').stream()
    students_list = []
    for doc in docs:
        students_list.append(doc.to_dict())
    return students_list

def delete_student_from_db(reg_no):
    """Delete student from Firestore."""
    try:
        db.collection('students').document(reg_no).delete()
        print(f"[INFO] Student {reg_no} deleted from Firestore.")
        return True
    except Exception as e:
        print(f"[WARNING] Failed to delete from Firestore: {e}")
        return False
        return False

def get_analytics_stats():
    """Get statistics for the analytics dashboard."""
    stats = {
        "total_students": 0,
        "total_attendance_today": 0,
        "daily_trends": {} 
    }
    
    # 1. Total Students
    # Note: count() query is cheaper but sticking to stream for simplicity if small scale
    students_coll = db.collection('students')
    stats["total_students"] = len(list(students_coll.stream()))
    
    # 2. Daily Trends (Last 7 Days)
    today = datetime.now()
    date_strs = []
    for i in range(7):
        d = today - timedelta(days=i)
        date_strs.append(d.strftime("%Y-%m-%d"))
    
    # Query logs for these dates
    # We'll do a simple query for logs in range. 
    # Ideally should use composite index on date_str, but "in" query works for up to 10 items.
    
    logs_ref = db.collection('attendance_logs')
    # Filter for dates in our list
    # Use batches or just stream all recent logs? Stream recent 500 logs is safer without indexes.
    # Actually, let's just query by date_str for today to get "Present Today" accurate
    
    today_str = today.strftime("%Y-%m-%d")
    today_logs = logs_ref.where('date_str', '==', today_str).stream()
    
    today_present = set()
    for log in today_logs:
        to_dict = log.to_dict()
        today_present.add(to_dict.get('reg_no'))
    stats["total_attendance_today"] = len(today_present)
    
    # 3. Last 7 Days Trend
    # Building a map of {date: count}
    # Since 'in' query requires an index we might not have, let's just do it manually for now
    # or iterate. For a "hackathon" scale, let's just fetch logs from last 7 days.
    
    end_date = today
    start_date = today - timedelta(days=6)
    
    # This might fail if you don't have composite indexes created.
    # Safer approach: iterate last 7 days and make 7 queries? (Inefficient but robust without indexes)
    trend_data = {}
    
    for d_str in reversed(date_strs):
        daily_logs = logs_ref.where('date_str', '==', d_str).stream()
        unique_students = set()
        for log in daily_logs:
            unique_students.add(log.to_dict().get('reg_no'))
        trend_data[d_str] = len(unique_students)
        
    stats["daily_trends"] = trend_data
    
    return stats

def get_logs_by_date(date_str):
    """Get all logs for a specific date (YYYY-MM-DD)."""
    try:
        logs_ref = db.collection('attendance_logs')
        
        # Parse date string "YYYY-MM-DD"
        start_dt = datetime.strptime(date_str, "%Y-%m-%d")
        
        # Create end of day (23:59:59.999999)
        end_dt = start_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Filter by timestamp range
        # Note: timestamps in Firestore are stored in UTC (usually)
        # This query assumes the server local time aligns with intended query date or relies on naive datetime handling
        query = logs_ref.where('timestamp', '>=', start_dt)\
                        .where('timestamp', '<=', end_dt)\
                        .order_by('timestamp', direction=firestore.Query.DESCENDING)
        
        docs = query.stream()
        
        logs = []
        for doc in docs:
            log = doc.to_dict()
            if 'timestamp' in log and log['timestamp']:
                # Handle Firestore datetime object
                ts = log['timestamp']
                if hasattr(ts, 'isoformat'):
                    log['timestamp'] = ts.isoformat()
                else:
                    log['timestamp'] = str(ts)
            logs.append(log)
        return logs
    except Exception as e:
        print(f"Error querying logs by date: {e}")
        return []
