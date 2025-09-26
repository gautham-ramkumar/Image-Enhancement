# Image-Enhancement
## A Hybrid U-Net for Low-Light Image Enhancement
This project implements a novel, two-stage deep learning pipeline for robust low-light image enhancement. The model is a hybrid inspired by the principles of Zero-DCE and LIVENet. It uses a U-Net architecture to perform simultaneous denoising and curve-based enhancement, trained in a supervised, multi-task manner to produce high-quality, artifact-free results.

## Key Features:
- Simultaneous Denoising & Enhancement: A single, unified U-Net model learns to both clean noise and improve lighting at the same time.
- Hybrid Architecture: Fuses Zero-DCE's curve estimation enhancement with LIVENet's supervised, multi-task training strategy.
- Two-Stage Training: Employs a robust training methodology where a denoiser is pre-trained and then frozen, allowing the enhancement network to be trained for optimal quality.
