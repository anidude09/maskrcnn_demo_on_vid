import os
import cv2
import numpy as np
import json
from glob import glob
import matplotlib.pyplot as plt

# Add constant for video inversion
INVERT_VIDEO = True  # Set to True for upside-down videos

def load_mask_data(mask_data_dir, detection_file):
    """Load mask data and its corresponding JSON metadata."""
    # Load detection JSON
    with open(detection_file, 'r') as f:
        detection_data = json.load(f)
    
    masks = []
    for detection in detection_data['detections']:
        mask_file = detection['mask_file']
        mask_path = os.path.join(mask_data_dir, mask_file)
        mask = np.load(mask_path)
        masks.append({
            'mask': mask,
            'confidence': detection['confidence'],
            'bbox': detection['bbox'],
            'cow_id': detection['cow_id']
        })
    
    return detection_data['timestamp'], masks

def visualize_masks(video_path, timestamp, masks, output_dir='mask_visualizations'):
    """Visualize masks on the video frame."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video and get to the correct frame
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Parse timestamp and convert to frame number
    minutes, seconds = map(int, timestamp.split(':'))
    target_frame = (minutes * 60 + seconds) * fps
    
    # Get to the correct frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Could not read frame at timestamp {timestamp}")
        return
    
    # Invert frame if video is upside down
    if INVERT_VIDEO:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    # Convert frame to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]
    
    # Create figure with subplots
    n_masks = len(masks)
    fig = plt.figure(figsize=(20, 5*((n_masks+1)//2)))
    
    # Plot original frame
    plt.subplot(((n_masks+1)//2), 2, 1)
    plt.imshow(frame_rgb)
    plt.title(f"Original Frame (Timestamp: {timestamp})")
    plt.axis('off')
    
    # Plot each mask
    for idx, mask_data in enumerate(masks, start=2):
        mask = mask_data['mask']
        bbox = mask_data['bbox']
        confidence = mask_data['confidence']
        cow_id = mask_data['cow_id']
        
        # Create a colored overlay for the mask
        colored_mask = np.zeros_like(frame_rgb)
        color = np.random.randint(0, 255, 3)
        
        # Resize mask to bbox size and place it in the frame
        y1 = int(bbox['y1'] * height)
        x1 = int(bbox['x1'] * width)
        y2 = int(bbox['y2'] * height)
        x2 = int(bbox['x2'] * width)
        
        # If video is inverted, we need to adjust the y-coordinates
        if INVERT_VIDEO:
            y1, y2 = height - y2, height - y1
        
        mask_resized = cv2.resize(mask, (x2-x1, y2-y1))
        mask_binary = mask_resized > 0.5
        
        # Create full-size mask
        full_mask = np.zeros((height, width), dtype=bool)
        full_mask[y1:y2, x1:x2] = mask_binary
        
        # Apply the mask
        for c in range(3):
            colored_mask[:, :, c] = np.where(full_mask, color[c], 0)
        
        # Blend with original frame
        alpha = 0.5
        blended = cv2.addWeighted(frame_rgb, 1, colored_mask, alpha, 0)
        
        # Draw bounding box
        cv2.rectangle(blended, (x1, y1), (x2, y2), color.tolist(), 2)
        
        # Plot the result
        plt.subplot(((n_masks+1)//2), 2, idx)
        plt.imshow(blended)
        plt.title(f"Cow {idx-1} (Confidence: {confidence:.2f})")
        plt.axis('off')
    
    # Save the visualization
    timestamp_str = timestamp.replace(':', '_')
    plt.savefig(os.path.join(output_dir, f'masks_{timestamp_str}.png'))
    plt.close()
    
    print(f"Visualization saved to {output_dir}/masks_{timestamp_str}.png")

def main():
    # Find all detection files
    mask_data_dir = 'output/mask_data'
    detection_files = glob(os.path.join(mask_data_dir, 'detections_*.json'))
    
    if not detection_files:
        print("No detection files found in output/mask_data")
        return
    
    # Process each detection file
    for detection_file in detection_files:
        print(f"\nProcessing {detection_file}")
        timestamp, masks = load_mask_data(mask_data_dir, detection_file)
        
        # Extract the full video name from the detection filename
        # Example: from 'detections_08_20241025095958_04_00.json' get '08_20241025095958.mp4'
        filename = os.path.basename(detection_file)  # get just the filename
        video_name = '_'.join(filename.split('_')[1:3])  # get '08_20241025095958'
        video_path = f'videos/{video_name}.mp4'
        
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue
        
        print(f"Found {len(masks)} masks at timestamp {timestamp}")
        visualize_masks(video_path, timestamp, masks)

if __name__ == "__main__":
    main() 