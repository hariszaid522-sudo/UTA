Show Image
Show Image
Show Image
Official implementation of "Uncertainty Trajectory Analysis for Reliable Chest X-ray Classification"

Authors: Zahid Ullah, Chonnam National University, South Korea

📋 Overview
This repository contains the code for Uncertainty Trajectory Analysis (UTA), a novel approach for uncertainty quantification in medical image classification. UTA leverages deep supervision with auxiliary classifiers to track prediction uncertainty across network layers, enabling more reliable predictions without the computational overhead of Monte Carlo Dropout.
Show Image
Key Contributions

Uncertainty Trajectory Analysis (UTA): A novel uncertainty estimation method that tracks how prediction confidence evolves through network layers
Efficient Uncertainty Quantification: Single forward pass (20x faster than MC Dropout)
Improved Classification: Deep supervision improves both classification accuracy and uncertainty estimation
Optimized UTA Formula: UTA = 0.4×(1-U₃) + 0.2×Confidence + 0.4×Agreement

🏆 Results Summary
NIH ChestX-ray14 Dataset (25,596 test images)
MethodMean AUCCorrelationSP@50%Baseline + Confidence0.72030.330094.37%Baseline + MC Dropout0.72030.279394.02%Ours + UTA0.78230.446694.93%

AUC Improvement: +8.6% over baseline
Wins: 14/14 diseases

CheXpert Dataset (202 validation images, 30% training)
MethodMean AUCCorrelationSP@50%Baseline + Confidence0.85130.470386.93%Baseline + MC Dropout0.85130.287084.55%Ours + UTA0.88000.622690.89%

AUC Improvement: +3.4% over baseline
Correlation Improvement: +32.4% over baseline
Selective Prediction Improvement: +3.96% absolute
Wins: 4/5 diseases in AUC

📦 Pretrained Models
All pretrained models are available for download. After downloading, place them in the pretrained_models/ directory.
Download Links
ModelDatasetClassesAUCSizeDownloadbaseline_nih.pthNIH ChestX-ray14140.7203~30 MBGoogle Drivetrajectory_nih.pthNIH ChestX-ray14140.7823~35 MBGoogle Drivebaseline_chexpert.pthCheXpert50.8513~30 MBGoogle Drivetrajectory_chexpert.pthCheXpert50.8800~35 MBGoogle Drive
Original Model Paths (for reference)
If you have access to the training server, models are located at:
bash# NIH ChestX-ray14 Models
/home/dsplab/4tb/zahid/nih_trajectory_experiment_v2/baseline_best.pth
/home/dsplab/4tb/zahid/trajectory_option_a_deep_supervision/option_a_deep_supervision_best.pth

# CheXpert Models  
/home/dsplab/4tb/zahid/chexpert_trajectory_30pct/baseline_best.pth
/home/dsplab/4tb/zahid/chexpert_trajectory_30pct/trajectory_best.pth
Directory Structure After Download
pretrained_models/
├── nih/
│   ├── baseline_best.pth           # Baseline DenseNet121 (14 classes)
│   └── trajectory_best.pth         # Trajectory DenseNet121 (14 classes)
└── chexpert/
    ├── baseline_best.pth           # Baseline DenseNet121 (5 classes)
    └── trajectory_best.pth         # Trajectory DenseNet121 (5 classes)
Prepare Models for Upload (Run on Server)
bash# Create directory
mkdir -p ~/models_to_upload

# Copy and rename models
cp /home/dsplab/4tb/zahid/nih_trajectory_experiment_v2/baseline_best.pth ~/models_to_upload/baseline_nih.pth
cp /home/dsplab/4tb/zahid/trajectory_option_a_deep_supervision/option_a_deep_supervision_best.pth ~/models_to_upload/trajectory_nih.pth
cp /home/dsplab/4tb/zahid/chexpert_trajectory_30pct/baseline_best.pth ~/models_to_upload/baseline_chexpert.pth
cp /home/dsplab/4tb/zahid/chexpert_trajectory_30pct/trajectory_best.pth ~/models_to_upload/trajectory_chexpert.pth

# Verify
ls -lh ~/models_to_upload/
🛠️ Installation
Requirements
bash# Clone repository
git clone https://github.com/YOUR_USERNAME/uta-chest-xray.git
cd uta-chest-xray

# Create conda environment
conda create -n uta python=3.8
conda activate uta

# Install dependencies
pip install -r requirements.txt
Dependencies
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
pandas>=1.3.0
scikit-learn>=0.24.0
Pillow>=8.0.0
tqdm>=4.60.0
matplotlib>=3.4.0
📁 Project Structure
uta-chest-xray/
├── README.md
├── requirements.txt
├── LICENSE
│
├── models/
│   ├── baseline.py              # Baseline DenseNet121
│   └── trajectory.py            # Trajectory DenseNet121 with aux classifiers
│
├── datasets/
│   ├── nih_dataset.py           # NIH ChestX-ray14 dataset loader
│   └── chexpert_dataset.py      # CheXpert dataset loader
│
├── training/
│   ├── train_baseline.py        # Train baseline model
│   ├── train_trajectory.py      # Train trajectory model (Option A)
│   └── train_trajectory_v2.py   # Train with strong trajectory losses (Option B)
│
├── evaluation/
│   ├── evaluate_nih.py          # Evaluate on NIH dataset
│   ├── evaluate_chexpert.py     # Evaluate on CheXpert dataset
│   └── uta_optimization.py      # UTA weight optimization
│
├── utils/
│   ├── metrics.py               # Evaluation metrics
│   ├── uncertainty.py           # Uncertainty computation functions
│   └── visualization.py         # Plotting functions
│
├── pretrained_models/           # Download pretrained models here
│   ├── nih/
│   └── chexpert/
│
└── figures/                     # Generated figures
🚀 Quick Start
1. Download Pretrained Models
bash# Create directories
mkdir -p pretrained_models/nih pretrained_models/chexpert

# Download models (replace with actual links)
wget -O pretrained_models/nih/trajectory_best.pth "YOUR_DOWNLOAD_LINK"
wget -O pretrained_models/chexpert/trajectory_best.pth "YOUR_DOWNLOAD_LINK"
2. Inference Example
pythonimport torch
from models.trajectory import TrajectoryDenseNet
from utils.uncertainty import compute_uta_score

# Load model
model = TrajectoryDenseNet(num_classes=14)
model.load_state_dict(torch.load('pretrained_models/nih/trajectory_best.pth'))
model.eval()

# Inference
with torch.no_grad():
    main_out, aux_outs = model(image)
    probs = torch.sigmoid(main_out)
    
    # Compute UTA score
    uta_score = compute_uta_score(aux_outs, main_out, 
                                   weights={'u3': 0.4, 'conf': 0.2, 'agree': 0.4})
3. Evaluate on NIH Dataset
bashpython evaluation/evaluate_nih.py \
    --data_root /path/to/nih-chest-xray \
    --model_path pretrained_models/nih/trajectory_best.pth \
    --output_dir results/
4. Evaluate on CheXpert Dataset
bashpython evaluation/evaluate_chexpert.py \
    --data_root /path/to/chexpert \
    --model_path pretrained_models/chexpert/trajectory_best.pth \
    --output_dir results/
📊 UTA Formula
The Uncertainty Trajectory Analysis score is computed as:
UTA = α × (1 - U₃) + β × Confidence + γ × Agreement
Where:

U₃: Entropy from auxiliary classifier 3 (after DenseBlock 3)
Confidence: 1 - entropy of main classifier output
Agreement: Fraction of auxiliary classifiers agreeing with main prediction

Optimized weights: α=0.4, β=0.2, γ=0.4
🏋️ Training
Train Baseline Model
bashpython training/train_baseline.py \
    --data_root /path/to/dataset \
    --output_dir checkpoints/baseline \
    --epochs 15 \
    --batch_size 32 \
    --lr 1e-4
Train Trajectory Model (Option A - Recommended)
bashpython training/train_trajectory.py \
    --data_root /path/to/dataset \
    --output_dir checkpoints/trajectory \
    --epochs 15 \
    --batch_size 32 \
    --lr 1e-4 \
    --lambda_aux 0.3
📈 Detailed Results
NIH ChestX-ray14 Per-Disease AUC
DiseaseBaselineOursΔAtelectasis0.69710.7548+0.0578Cardiomegaly0.74450.8530+0.1085Effusion0.74300.7996+0.0566Infiltration0.63640.6748+0.0384Mass0.71630.7987+0.0825Nodule0.67980.7471+0.0673Pneumonia0.65820.6894+0.0311Pneumothorax0.74190.8217+0.0798Consolidation0.67100.7274+0.0565Edema0.75240.8124+0.0599Emphysema0.81200.8936+0.0816Fibrosis0.71210.7809+0.0688Pleural Thickening0.69870.7384+0.0397Hernia0.82050.8607+0.0402Mean0.72030.7823+0.0621
CheXpert Per-Disease AUC
DiseaseBaselineOursΔWinnerAtelectasis0.80280.8336+0.0308OursCardiomegaly0.78010.8603+0.0802OursConsolidation0.81600.8518+0.0358OursEdema0.92350.9384+0.0149OursPleural Effusion0.93410.9161-0.0180BaseMean0.85130.8800+0.0287
Selective Prediction Accuracy (NIH ChestX-ray14)
CoverageBase+ConfBase+MCOurs+UTA100%92.16%92.16%92.18%90%92.61%92.57%92.74%80%93.03%92.92%93.24%70%93.46%93.27%93.76%60%93.88%93.68%94.29%50%94.37%94.02%94.93%
Selective Prediction Accuracy (CheXpert)
CoverageBase+ConfBase+MCOurs+UTA100%79.41%79.41%78.71%90%81.10%80.22%80.66%80%82.48%80.99%83.35%70%83.97%81.99%86.38%60%85.62%83.47%88.93%50%86.93%84.55%90.89%
🔬 Model Architecture
Trajectory DenseNet121
Input Image (224×224×3)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Conv + BN + ReLU + MaxPool (64 channels)                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ DenseBlock 1 (256 channels, 56×56)                          │
│     └──► Auxiliary Classifier 1 → U₁ Entropy                │
└─────────────────────────────────────────────────────────────┘
    │ Transition 1
    ▼
┌─────────────────────────────────────────────────────────────┐
│ DenseBlock 2 (512 channels, 28×28)                          │
│     └──► Auxiliary Classifier 2 → U₂ Entropy                │
└─────────────────────────────────────────────────────────────┘
    │ Transition 2
    ▼
┌─────────────────────────────────────────────────────────────┐
│ DenseBlock 3 (1024 channels, 14×14)                         │
│     └──► Auxiliary Classifier 3 → U₃ Entropy                │
└─────────────────────────────────────────────────────────────┘
    │ Transition 3
    ▼
┌─────────────────────────────────────────────────────────────┐
│ DenseBlock 4 (1024 channels, 7×7)                           │
│     └──► Auxiliary Classifier 4 → U₄ Entropy                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Global Average Pooling + Main Classifier                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Output Probabilities + UTA Score
📝 Citation
If you find this work useful, please cite:
bibtex@article{ullah2025uta,
  title={Uncertainty Trajectory Analysis for Reliable Chest X-ray Classification},
  author={Ullah, Zahid},
  journal={},
  year={2025}
}
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgements

NIH ChestX-ray14 dataset: Wang et al., 2017
CheXpert dataset: Irvin et al., 2019
DenseNet architecture: Huang et al., 2017

📧 Contact
For questions or issues, please open an issue or contact:

Zahid Ullah - zahid@jnu.ac.kr
Chonnam National University, South Korea


⭐ If you find this repository helpful, please consider giving it a star!