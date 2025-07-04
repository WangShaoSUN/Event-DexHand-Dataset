# Event-based Dexterous Hand Motion Transfer

## 📌 Overview
This project implements an event-based motion transfer system for dexterous robotic hands, using synchronized event camera data and RGB images to capture and transfer natural hand movements.

## 🗂 Project Structure
## Project Structure
```
.
├── dataset/
│   ├── event_images/         # Event camera data (28 sequences)
│   ├── rgb_images/           # RGB image data (27 sequences)
│   └── meta_data/            # Metadata (21 sequences)
├── hand_embodiment/          # Hand embodiment implementations
├── models/
│   ├── mano/                 # MANO hand model components
│   └── smplx/                # SMPLX model components
├── load_dataset.py           # Dataset loading utilities
├── render_from_pickle.py     # Data visualization from pickle files
├── requirements.txt          # Python dependencies
├── shadow_pose_generate.py   # Pose generation from event data
├── train.py                  # Main training script
├── vis_dexhand.py            # DexHand visualization tools
└── README.md                 # This documentation file
```


## Dataset Description
```
dataset/
├── event_images/ # Event camera data
│ ├── event_sequence_0.npy # Event sequence 0
│ ├── event_sequence_1.npy # Event sequence 1
│ └── ... 
│
├── rgb_images/ # Corresponding RGB images
│ ├── rgb_frame_0.npy # RGB sequence 0
│ ├── rgb_frame_1.npy # RGB sequence 1
│ └── ... 
│
├── meta_data/ # Metadata files
│ ├── metadata_0.pickle # Sequence 0 metadata
│ ├── metadata_1.pickle # Sequence 1 metadata
│ └── ... 
```
## Dataset Download
The dataset can be downloaded from Baidu Netdisk:

**Download Link**: [https://pan.baidu.com/s/1ygX9PAg0pdOZdiR7sKP45g?pwd=sfd9 ]( https://pan.baidu.com/s/1ygX9PAg0pdOZdiR7sKP45g?pwd=sfd9 )  
**Extraction Code**: sfd9

### Organization

| Directory | File Type | Count | Description |
|-----------|-----------|-------|-------------|
| event_images/ | .npy | 150000 | Event camera sequences |
| rgb_images/ | .npy | 150000 | RGB image sequences |
| meta_data/ | .pickle | 150000 | Sequence metadata |

### Key Features
- Temporally aligned event and RGB data
- Compact NumPy array storage format
- Comprehensive metadata including:
  - Timestamps
  - Hand pose parameters
  - Motion annotations

### Loading Example

Here is a simple utility function to load a synchronized data sequence:

```python
import numpy as np
import pickle

def load_sequence(seq_id, base_path='dataset/'):
    """Load synchronized data for a given sequence ID."""
    try:
        event_path = f'{base_path}event_images/event_image_{seq_id}.npy'
        event_data = np.load(event_path)
    except FileNotFoundError:
        print(f"Event data for sequence {seq_id} not found.")
        event_data = None

    try:
        rgb_path = f'{base_path}rgb_images/rgb_image_{seq_id}.npy'
        rgb_data = np.load(rgb_path)
    except FileNotFoundError:
        print(f"RGB data for sequence {seq_id} not found.")
        rgb_data = None

    try:
        meta_path = f'{base_path}meta_data/meta_data_{seq_id}.pickle'
        with open(meta_path, 'rb') as f:
            meta_data = pickle.load(f)
    except FileNotFoundError:
        print(f"Metadata for sequence {seq_id} not found.")
        meta_data = None

    return event_data, rgb_data, meta_data

# Example: Load sequence 5
event, rgb, meta = load_sequence(5)
```

## 🚀 Getting Started

Follow these steps to set up the project environment.

### Prerequisites

Ensure you have the following installed:
-   Python 3.7+
-   PyTorch
-   NumPy
-   OpenCV
-   MANO / SMPLX models (follow their respective installation guides)

A complete list of dependencies can be found in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/event-dexterous-hand.git
    cd event-dexterous-hand
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 💡 Usage

The repository includes scripts for training, data processing, and visualization.

### Training

To start training the model, run the `train.py` script. You can specify the dataset path and other hyperparameters as command-line arguments.

```bash
python train.py 
```

### Data Processing

To generate hand poses from raw event data, use the `shadow_pose_generate.py` script.

```bash
# Example: Generate poses from an event data file
python shadow_pose_generate.py input_events.npy output_poses.pickle
```

### Visualization

To visualize the results for a specific sequence, use the `vis_dexhand.py` script.

```bash
# View DexHand results for a processed sequence
python vis_dexhand.py 
```

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:
1.  Open an issue to discuss your proposed changes or new features.
2.  Fork the repository.
3.  Create a new branch for your feature (`git checkout -b feature/YourFeature`).
4.  Commit your changes (`git commit -m 'Add YourFeature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
