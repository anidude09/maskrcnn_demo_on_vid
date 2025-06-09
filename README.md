# Cow Detection and Segmentation System

A computer vision system that detects and segments cows in pen videos using Mask R-CNN. The system processes upside-down surveillance videos, corrects their orientation, and generates detailed segmentation masks for each detected cow.

## Features

- **Video Preprocessing**
  - Extracts 2 frames per minute (at 15s and 45s marks)
  - Automatically corrects video orientation (180° rotation)
  - Supports multiple video formats (MP4, AVI, MOV)

- **Cow Detection & Segmentation**
  - Uses Mask R-CNN with Inception ResNet v2 backbone
  - Generates instance segmentation masks
  - Provides confidence scores for each detection
  - Creates visualization overlays

- **Output Generation**
  - Processed video with detection visualizations
  - Individual mask files for each detected cow
  - Detection metadata in JSON format
  - Frame-by-frame segmentation masks

## Getting Started


## Demo video Link
https://drive.google.com/file/d/1Sf0X_3J3qF3y0aXJ9mQwSHqBm0SfRJ2O/view?usp=sharing


### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Sufficient storage for video processing

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/cow-detection-system.git
cd cow-detection-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the Mask R-CNN model:
```bash
# Create model directory
mkdir -p mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/saved_model

# Download model from TensorFlow Model Garden
# https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1
```

### Usage

1. Place your original (upside-down) videos in the `raw_videos` directory.

2. Run the preprocessing script:
```bash
python preprocess_video.py
```

3. Run the detection and segmentation:
```bash
python detect_cows.py
```

### Output Structure

```
project_directory/
├── raw_videos/                 # Original upside-down videos
├── preprocessed_videos/        # Corrected orientation videos
└── output/
    ├── video_name_detected.mp4 # Visualization video
    ├── detections/            # Mask data and metadata
    │   ├── mask_*.npy        # Raw mask arrays
    │   └── metadata.json     # Detection information
    └── isolated_masks/       # Individual cow masks
```

## Output Details

### Detection Video
- Shows bounding boxes around detected cows
- Displays segmentation masks with different colors
- Includes confidence scores and timestamps

### Mask Files
- `.npy` files containing raw mask data
- PNG files showing isolated cow masks
- Each mask is named with timestamp and cow ID

### Metadata JSON
```json
{
    "timestamp": "00:15",
    "frame_number": 1800,
    "detections": [
        {
            "cow_id": "cow_00_15_0",
            "confidence": 0.98,
            "bbox": {...},
            "mask_file": "mask_00_15_0.npy"
        }
    ]
}
```

## Performance Notes

- Processing time depends on:
  * Video length and resolution
  * GPU capabilities
  * Number of cows per frame
- Typical processing rates:
  * Preprocessing: ~100 frames/second
  * Detection: ~2-5 frames/second with GPU


## Contributing

Feel free to open issues or submit pull requests. For major changes, please open an issue first to discuss what you'd like to change.

