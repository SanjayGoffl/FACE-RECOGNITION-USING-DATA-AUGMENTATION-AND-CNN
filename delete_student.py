import argparse
import sys
import os

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import delete_student, pipeline

def main():
    parser = argparse.ArgumentParser(description="Delete a student from the system.")
    parser.add_argument("roll_no", help="The roll number/register number of the student to delete")
    parser.add_argument("--no-data", action="store_true", help="Do NOT delete the image folder (keep images)")
    
    args = parser.parse_args()
    
    print(f"🗑️ Attempting to delete student: {args.roll_no}")
    
    # Check if student exists in embeddings
    embeddings = pipeline.load_embeddings()
    if args.roll_no not in embeddings:
        print(f"⚠️ Warning: Student {args.roll_no} not found in trained model.")
        choice = input("Proceed anyway? (y/N): ")
        if choice.lower() != 'y':
            print("Aborted.")
            return

    delete_data = not args.no_data
    
    success = delete_student(args.roll_no, delete_data=delete_data)
    
    if success:
        print(f"✅ Successfully deleted {args.roll_no}")
    else:
        print(f"❌ Failed to delete {args.roll_no}")

if __name__ == "__main__":
    main()
