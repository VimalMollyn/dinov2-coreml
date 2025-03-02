# DINOv2 CoreML
DINOv2 in CoreML. Conversion scripts and live webcam demo.



https://github.com/user-attachments/assets/578808c7-aad3-40ce-ab28-6b446b1e3d4d



## Installation

```bash
# Create virtual environment
conda create -n dino-coreml python=3.11
conda activate dino-coreml

# Install dependencies
python -m pip install -r requirements.txt
```

## Convert DINOv2 to CoreML

```bash
python convert_coreml.py
```

## Run DINOv2 live on webcam

```bash
python dino_live.py
```

Roughly 10 FPS on my M2 MacBook Air.

