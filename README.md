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

## Training and testing
To train the model, Run following command from ```/src``` directory.
```bash
python3 train.py
```
This will train for 100 epochs and save the best-performing model weights. 

To test the model, Run following command from ```/src``` directory.
```bash
python3 test.py
```
Enhanced images will be saved to the ```final_test_outputs/``` directory.


├── Papers
│   ├── LIVENet.pdf
│   └── Zero-DCE.pdf
├── src
│   ├── dataloader.py
│   ├── losses.py
│   ├── model.py
│   ├── test.py
│   └── train.py
└── test_outputs
    ├── 111.png
    ├── 146.png
    ├── 179.png
    ├── 1.png
    ├── 22.png
    ├── 23.png
    ├── 493.png
    ├── 547.png
    ├── 55.png
    ├── 665.png
    ├── 669.png
    ├── 748.png
    ├── 778.png
    ├── 780.png
    ├── 79.png
    ├── Training_results1.png
    ├── Training_results2.png
    └── Training_results3.png

