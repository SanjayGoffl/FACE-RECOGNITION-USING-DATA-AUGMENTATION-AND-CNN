from waitress import serve
from app import app
import os
import socket

def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == "__main__":
    # Ensure data dirs exist (same as app.py)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    STUDENTS_DIR = os.path.join(BASE_DIR, "data", "students", "images")
    os.makedirs(STUDENTS_DIR, exist_ok=True)
    
    port = 5000
    host = "0.0.0.0"
    
    local_ip = get_ip_address()
    
    print("="*60)
    print(f"🚀 PRODUCTION SERVER STARTED")
    print(f"   Using Waitress WSGI Server")
    print("-" * 60)
    print(f"📡 Local Access:     http://localhost:{port}")
    print(f"🌐 Network Access:   http://{local_ip}:{port}")
    print("-" * 60)
    print("logs:")
    
    serve(app, host=host, port=port, threads=6)
