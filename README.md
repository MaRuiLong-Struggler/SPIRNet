# SPIRNet
For polarization image demosaicing and super-resolution tasks


# Requirements
- python=3.8.18
- pytorch=2.0.1
- cv2
- numpy
- tqdm
- scikit-image

- 
# Dataset
The datasets employed in this study were collected by Zhou et al. and Yu et al.
[dataset1](https://drive.google.com/drive/folders/1VWcpbmJsKV9PCCu9xXQCPqZav-FYyon4?dmr=1&ec=wgc-drive-hero-goto), which contains a total of 138 images.
[dataset2](https://drive.google.com/file/d/1IucTM_KQfdmvEVTCPO1QjEyndnFSc2fc/view?usp=sharing.)

data/
├── train/
│   ├── 001/
│   │   ├── RGB_0.png
│   │   ├── RGB_45.png
│   │   ├── RGB_90.png
│   │   └── RGB_135.png
│   ├── 002/
│   │   ├── RGB_0.png
│   │   ├── RGB_45.png
│   │   ├── RGB_90.png
│   │   └── RGB_135.png
│   └── .../
└── test/
    ├── 001/
    │   ├── RGB_0.png
    │   ├── RGB_45.png
    │   ├── RGB_90.png
    │   └── RGB_135.png
    ├── 002/
    │   ├── RGB_0.png
    │   ├── RGB_45.png
    │   ├── RGB_90.png
    │   └── RGB_135.png
    └── .../

# Train  
```bash
python train.py
```
# Acknowledgements
This code is built on [Restormer]([https://github.com/swz30/Restormer](https://github.com/PRIS-CV/PIDSR). We thank the authors for sharing their codes.

# Citation
[1] Zhou S, Zhou C, Lyu Y, et al. PIDSR: Complementary Polarized Image Demosaicing and Super-Resolution[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 16081-16090.
[2] Yu D, Li Q, Zhang Z, et al. Color polarization image super-resolution reconstruction via a cross-branch supervised learning strategy[J]. Optics and Lasers in Engineering, 2023, 165: 107469.
