import os
import cv2
from tqdm import tqdm

def preprocess_video(input_path, output_path):
    """
    Preprocess video by extracting 2 frames per minute and creating a video.
    The frames are inverted to correct orientation.
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_minutes = total_frames / (fps * 60)
    
    print(f"Processing video: {input_path}")
    print(f"Resolution: {width}x{height} at {fps}fps")
    print(f"Duration: {duration_minutes:.1f} minutes")
    
    # Calculate frame indices to extract (2 frames per minute)
    frames_per_minute = fps * 60
    frame_indices = []
    for minute in range(int(duration_minutes) + 1):
        # Extract frames at 15s and 45s of each minute
        frame_indices.extend([
            int(minute * frames_per_minute + 15 * fps),
            int(minute * frames_per_minute + 45 * fps)
        ])
    # Remove indices that exceed total frames
    frame_indices = [i for i in frame_indices if i < total_frames]
    
    # Create output video writer (2 FPS for the extracted frames)
    # Using H.264 codec for better compatibility
    if os.path.splitext(output_path)[1].lower() == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    else:
        # Fallback to XVID for other formats
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter(output_path, fourcc, 2, (width, height))
    if not out.isOpened():
        print("Failed to create video writer. Trying alternative codec...")
        # Try alternative codec if first one fails
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = os.path.splitext(output_path)[0] + '.avi'
        out = cv2.VideoWriter(output_path, fourcc, 2, (width, height))
        if not out.isOpened():
            print("Error: Could not create video writer")
            return False
    
    # Process frames
    current_frame = 0
    frames_processed = 0
    pbar = tqdm(total=len(frame_indices), desc="Processing frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame in frame_indices:
            # Rotate frame 180 degrees
            frame_inverted = cv2.rotate(frame, cv2.ROTATE_180)
            
            # Write frame to video
            out.write(frame_inverted)
            frames_processed += 1
            pbar.update(1)
            
            if frames_processed == len(frame_indices):
                break
                
        current_frame += 1
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"Preprocessed video saved to: {output_path}")
    print(f"Total frames in output: {frames_processed} (2 frames per minute)")
    return True

def main():
    # Create directories if they don't exist
    os.makedirs("raw_videos", exist_ok=True)
    os.makedirs("preprocessed_videos", exist_ok=True)
    
    # Process all videos in raw_videos directory
    videos = [f for f in os.listdir("raw_videos") if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not videos:
        print("No videos found in raw_videos directory")
        return
    
    for video in videos:
        input_path = os.path.join("raw_videos", video)
        output_path = os.path.join("preprocessed_videos", f"preprocessed_{video}")
        preprocess_video(input_path, output_path)

if __name__ == "__main__":
    main() 