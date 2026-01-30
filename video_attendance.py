"""
Video Attendance Recognition
Process pre-recorded videos for attendance using trained embeddings
Works exactly like webcam test but with video files
"""

import cv2
import os
import glob
import argparse
from src.pipeline import pipeline, draw_box, draw_label


# Default video folder
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_VIDEO_DIR = os.path.join(BASE_DIR, "data", "students", "videos")


def list_available_videos():
    """List all .mp4 videos in the default video folder.
    
    Returns:
        list: List of video filenames
    """
    if not os.path.exists(DEFAULT_VIDEO_DIR):
        os.makedirs(DEFAULT_VIDEO_DIR, exist_ok=True)
        print(f"📁 Created video folder: {DEFAULT_VIDEO_DIR}")
        return []
    
    videos = glob.glob(os.path.join(DEFAULT_VIDEO_DIR, "*.mp4"))
    videos.extend(glob.glob(os.path.join(DEFAULT_VIDEO_DIR, "*.MP4")))
    return sorted(videos)


def select_video_interactive():
    """Show menu to select a video from the default folder.
    
    Returns:
        str: Path to selected video or None
    """
    videos = list_available_videos()
    
    if not videos:
        print(f"\n❌ No videos found in: {DEFAULT_VIDEO_DIR}")
        print(f"📌 Please place .mp4 files in the videos folder and try again.")
        return None
    
    print(f"\n🎬 Available Videos ({len(videos)}):")
    print("=" * 60)
    
    for idx, video_path in enumerate(videos, 1):
        filename = os.path.basename(video_path)
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        print(f"   [{idx}] {filename} ({file_size:.2f} MB)")
    
    print("=" * 60)
    
    while True:
        try:
            choice = input(f"\nSelect video (1-{len(videos)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("👋 Exiting...")
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(videos):
                selected_video = videos[choice_num - 1]
                print(f"✅ Selected: {os.path.basename(selected_video)}\n")
                return selected_video
            else:
                print(f"⚠️  Please enter a number between 1 and {len(videos)}")
        except ValueError:
            print("⚠️  Invalid input. Please enter a number or 'q' to quit.")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            return None


def recognize_from_video(video_path, threshold=0.6):
    """Run attendance recognition on a pre-recorded video file.
    
    Args:
        video_path: Path to the video file (.mp4 format)
        threshold: Recognition threshold (default 0.6)
    """
    # Load student database
    embeddings = pipeline.load_embeddings()
    if not embeddings:
        print("❌ No embeddings found. Run 'python main.py --train' first.")
        return
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    print(f"\n🎥 Processing video: {os.path.basename(video_path)}")
    print(f"📊 Recognizing {len(embeddings)} students")
    print(f"🎯 Threshold: {threshold}")
    print(f"Press 'q' to quit early\n")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Cannot open video file")
        return
    
    # Get video properties for display
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📹 Video Info:")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {duration:.2f}s\n")
    
    # Create resizable window (same as webcam test)
    cv2.namedWindow('Video Attendance', cv2.WINDOW_NORMAL)
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\n✅ Finished processing video")
            break
        
        frame_count += 1
        
        # Detect and embed faces in the current frame
        emb, box = pipeline.detect_and_embed(frame)
        
        if emb is not None and box is not None:
            best_sim = 0
            best_roll, best_name = None, None
            
            # Search for the best match in trained database
            for roll_no, data in embeddings.items():
                sim = pipeline.compare_embeddings(emb, data['embedding'])
                if sim > best_sim:
                    best_sim, best_roll, best_name = sim, roll_no, data['name']
            
            draw_box(frame, box)
            
            if best_sim >= threshold:
                label = f"{best_name} ({best_roll}) | {best_sim:.2f}"
                # Logs to logs/attendance.csv once per session (same as webcam)
                if pipeline.logger.mark_once(best_name, best_roll):
                    print(f"✓ Attendance marked: {best_name} ({best_roll}) - Similarity: {best_sim:.2f}")
            else:
                label = f"Unknown | {best_sim:.2f}"
            
            draw_label(frame, label, box)
        
        # Add frame counter overlay
        counter_text = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(frame, counter_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Video Attendance', frame)
        
        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️  Video processing stopped by user")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n🎬 Video processing complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video-based Attendance Recognition')
    parser.add_argument('--video', type=str, 
                        help='Path to video file (.mp4). If not provided, shows interactive menu.')
    parser.add_argument('--threshold', type=float, default=0.6, 
                        help='Recognition threshold (default 0.6)')
    
    args = parser.parse_args()
    
    # If video path is provided, use it directly
    if args.video:
        recognize_from_video(args.video, args.threshold)
    else:
        # Interactive mode: show menu to select from default folder
        selected_video = select_video_interactive()
        if selected_video:
            recognize_from_video(selected_video, args.threshold)
