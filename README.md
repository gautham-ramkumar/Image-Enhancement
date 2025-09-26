# Image-Enhancement
## A Hybrid U-Net for Low-Light Image Enhancement
This project implements a novel, two-stage deep learning pipeline for robust low-light image enhancement. The model is a hybrid inspired by the principles of Zero-DCE and LIVENet. It uses a U-Net architecture to perform simultaneous denoising and curve-based enhancement, trained in a supervised, multi-task manner to produce high-quality, artifact-free results.

## Key Features:
- Simultaneous Denoising & Enhancement: A single, unified U-Net model learns to both clean noise and improve lighting at the same time.
- Hybrid Architecture: Fuses Zero-DCE's curve estimation enhancement with LIVENet's supervised, multi-task training strategy.
- Two-Stage Training: Employs a robust training methodology where a denoiser is pre-trained and then frozen, allowing the enhancement network to be trained for optimal quality.

## Results
<img width="1701" height="469" alt="image" src="https://github.com/user-attachments/assets/2ebcb1a6-17ef-4e51-9790-c933e7d48a96" />
<img width="1701" height="469" alt="image" src="https://github.com/user-attachments/assets/e715d0ce-8282-4954-b787-bfe213d5feb3" />
<img width="1701" height="469" alt="image" src="https://github.com/user-attachments/assets/5aca0e35-df75-4715-a725-fe8a731a2cc9" />

## Setup and Installation:
```bash
# Clone the repository
git clone https://github.com/gautham-ramkumar/Image-Enhancement.git
cd Image-Enhancement

# Create and activate a virtual environment
python3 -m venv venv
# Source the environment
source venv/bin/activate

# Install required packages
pip3 install -r requirements.txt
```

## 
