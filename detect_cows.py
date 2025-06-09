import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

MODEL_PATH = 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/saved_model'

class CowDetector:
    def __init__(self, model_path, confidence_threshold=0.7):
        print("Loading model...")
        self.model = tf.saved_model.load(model_path)
        print("Model loaded successfully!")
        
        # Enable TensorFlow optimizations
        tf.config.optimizer.set_jit(True)
        
        # Set confidence threshold
        self.confidence_threshold = confidence_threshold
    
    def refine_mask(self, mask, kernel_size=3):
        """Refine mask using morphological operations."""
        # Convert probability mask to binary
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Create kernels for morphological operations
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Remove noise and fill small holes
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Smooth edges
        binary_mask = cv2.GaussianBlur(binary_mask, (3, 3), 0)
        binary_mask = (binary_mask > 0.5).astype(np.uint8)
        
        return binary_mask
    
    @tf.function(experimental_relax_shapes=True)
    def detect_cows(self, image_tensor):
        """Run detection with TensorFlow optimization."""
        return self.model(image_tensor)
    
    def process_frame(self, frame):
        """Process a single frame and return detections."""
        # Prepare the frame for detection
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (1024, 1024))
        
        # Convert to tensor
        image_tensor = tf.convert_to_tensor(image_resized)
        image_tensor = tf.expand_dims(image_tensor, 0)
        
        # Run detection
        detections = self.detect_cows(image_tensor)
        
        # Process detections
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        masks = detections['detection_masks'][0].numpy()
        
        # Filter for cows (class 21 in COCO) and higher confidence threshold
        indices = np.where((classes == 21) & (scores > self.confidence_threshold))[0]
        
        # Refine masks for selected detections
        refined_masks = []
        for idx in indices:
            refined_mask = self.refine_mask(masks[idx])
            refined_masks.append(refined_mask)
        
        return {
            'boxes': boxes[indices],
            'scores': scores[indices],
            'masks': np.array(refined_masks) if refined_masks else masks[indices]
        }
    
    def visualize_detections(self, frame, detections):
        """Draw detections and masks on the frame."""
        height, width = frame.shape[:2]
        output_frame = frame.copy()
        
        # Generate random colors for visualization
        colors = np.random.randint(0, 255, (len(detections['boxes']), 3))
        
        for i, (box, score, mask) in enumerate(zip(detections['boxes'], 
                                                 detections['scores'],
                                                 detections['masks'])):
            # Get pixel coordinates
            y1, x1, y2, x2 = map(int, [
                box[0] * height,
                box[1] * width,
                box[2] * height,
                box[3] * width
            ])
            
            # Resize mask to match the region size
            mask_resized = cv2.resize(mask.astype(float), (x2-x1, y2-y1))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            
            # Create colored mask with edge highlighting
            color = colors[i]
            colored_mask = np.zeros_like(frame)
            
            # Fill the mask
            for c in range(3):
                colored_mask[y1:y2, x1:x2, c] = mask_binary * color[c]
            
            # Find and draw mask edges
            mask_edges = cv2.Canny(mask_binary, 100, 200)
            edge_thickness = 2
            dilated_edges = cv2.dilate(mask_edges, np.ones((edge_thickness, edge_thickness), np.uint8))
            
            for c in range(3):
                colored_mask[y1:y2, x1:x2, c][dilated_edges > 0] = 255
            
            # Blend mask with frame
            alpha = 0.6  # Increased opacity for better visibility
            output_frame = cv2.addWeighted(output_frame, 1, colored_mask, alpha, 0)
            
            # Draw bounding box and score
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color.tolist(), 2)
            cv2.putText(output_frame, f'Cow: {score:.2f}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
        
        return output_frame

    def save_isolated_mask(self, frame, mask, box, output_path):
        """Save an isolated mask image with just the cow pixels."""
        height, width = frame.shape[:2]
        
        # Get pixel coordinates
        y1, x1, y2, x2 = map(int, [
            box[0] * height,
            box[1] * width,
            box[2] * height,
            box[3] * width
        ])
        
        # Resize mask to match the region size
        mask_resized = cv2.resize(mask.astype(float), (x2-x1, y2-y1))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # Create full-size mask
        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask_binary
        
        # Apply edge refinement
        kernel = np.ones((3,3), np.uint8)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
        
        # Create isolated cow image with edge highlighting
        isolated_cow = np.zeros_like(frame)
        
        # Copy original pixels for cow
        for c in range(3):
            isolated_cow[:, :, c] = frame[:, :, c] * full_mask
        
        # Add edge highlighting
        edges = cv2.Canny(full_mask, 100, 200)
        edge_thickness = 2
        dilated_edges = cv2.dilate(edges, np.ones((edge_thickness, edge_thickness), np.uint8))
        
        # Add white edges
        for c in range(3):
            isolated_cow[:, :, c][dilated_edges > 0] = 255
        
        # Save the isolated mask
        cv2.imwrite(output_path, isolated_cow)
        
        return full_mask

def process_video(video_path, detector, output_dir='output'):
    """Process video for cow detection with mask extraction."""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'detections'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'isolated_masks'), exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_minute = fps * 60
    
    # Setup output video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(output_dir, f'{video_name}_detected.mp4')
    
    # Using H.264 codec for better compatibility
    if os.path.splitext(output_video_path)[1].lower() == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    else:
        # Fallback to XVID for other formats
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Failed to create video writer. Trying alternative codec...")
        # Try alternative codec if first one fails
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video_path = os.path.splitext(output_video_path)[0] + '.avi'
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print("Error: Could not create video writer")
            return
    
    # Process frames
    frame_count = 0
    detections_data = []
    pbar = tqdm(total=total_frames, desc="Processing video")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        pbar.update(1)
        
        # Get timestamp (for preprocessed video, each frame represents 30 seconds)
        minutes = (frame_count - 1) // 2  # 2 frames per minute
        seconds = 15 if frame_count % 2 == 1 else 45  # 15s for odd frames, 45s for even frames
        timestamp = f"{minutes:02d}:{seconds:02d}"
        
        # Detect cows
        detections = detector.process_frame(frame)
        
        if len(detections['boxes']) > 0:
            # Save detection data
            detection_data = {
                'timestamp': timestamp,
                'frame_number': frame_count,
                'detections': []
            }
            
            for i, (box, score, mask) in enumerate(zip(detections['boxes'],
                                                     detections['scores'],
                                                     detections['masks'])):
                # Generate filenames
                mask_filename = f'mask_{video_name}_{timestamp.replace(":", "_")}_{i}.npy'
                isolated_mask_filename = f'isolated_mask_{video_name}_{timestamp.replace(":", "_")}_{i}.png'
                
                # Save numpy mask
                mask_path = os.path.join(output_dir, 'detections', mask_filename)
                np.save(mask_path, mask)
                
                # Save isolated mask image
                isolated_mask_path = os.path.join(output_dir, 'isolated_masks', isolated_mask_filename)
                detector.save_isolated_mask(frame, mask, box, isolated_mask_path)
                
                detection_data['detections'].append({
                    'cow_id': f'cow_{timestamp}_{i}',
                    'confidence': float(score),
                    'bbox': {
                        'y1': float(box[0]),
                        'x1': float(box[1]),
                        'y2': float(box[2]),
                        'x2': float(box[3])
                    },
                    'mask_file': mask_filename,
                    'isolated_mask_file': isolated_mask_filename
                })
            
            detections_data.append(detection_data)
        
        # Visualize detections
        frame = detector.visualize_detections(frame, detections)
        
        # Add timestamp to frame
        cv2.putText(frame, f"Time: {timestamp}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(frame)
    
    pbar.close()
    cap.release()
    out.release()
    
    # Save all detections to a summary file
    summary_file = os.path.join(output_dir, 'detections', f'{video_name}_detections.json')
    summary_data = {
        'video_name': video_name,
        'total_frames': total_frames,
        'fps': fps,
        'duration_seconds': total_frames / fps,
        'frames_analyzed': len(detections_data),
        'detections': detections_data,
        'processing_date': datetime.now().isoformat()
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nProcessed video saved to: {output_video_path}")
    print(f"Detection data saved to: {summary_file}")
    print(f"Isolated mask images saved to: {os.path.join(output_dir, 'isolated_masks')}")

def main():
    # Create directories
    os.makedirs("preprocessed_videos", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    # Initialize detector
    detector = CowDetector(MODEL_PATH)
    
    # Process all preprocessed videos
    videos = [f for f in os.listdir("preprocessed_videos") if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not videos:
        print("No preprocessed videos found. Please run preprocess_video.py first.")
        return
    
    for video in videos:
        video_path = os.path.join("preprocessed_videos", video)
        print(f"\nProcessing {video_path}")
        process_video(video_path, detector)

if __name__ == "__main__":
    main()  