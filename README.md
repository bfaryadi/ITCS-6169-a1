# Scene Recognition CNN

This repository contains a PyTorch image-classification project for a 16-class natural scene recognition task. The model is trained on the `data/train` split and evaluated on the `data/test` split using a compact CNN architecture with BatchNorm, dropout, and augmentation strategies such as MixUp and CutMix.

## Project overview

The code in this repo trains and evaluates a convolutional neural network for classifying scene categories like `Bedroom`, `Coast`, `Forest`, `Mountain`, `Street`, and `TallBuilding`.

The implementation includes:
- A custom CNN model in `model.py`
- Training logic with augmentation and checkpointing in `train.py`
- Evaluation and confusion-matrix reporting in `evaluate.py`
- Experiment notebook: `ITCS_6169_8169_Assignment1_2026.ipynb`
- Saved final artifacts in `checkpoints/`

## Dataset

The project expects a folder structure like this:

```text
data/
├── train/
│   ├── Bedroom/
│   ├── Coast/
│   ├── Flower/
│   ├── Forest/
│   ├── Highway/
│   ├── Industrial/
│   ├── InsideCity/
│   ├── Kitchen/
│   ├── LivingRoom/
│   ├── Mountain/
│   ├── Office/
│   ├── OpenCountry/
│   ├── Store/
│   ├── Street/
│   ├── Suburb/
│   └── TallBuilding/
└── test/
    └── same 16 classes
```

The dataset contains 16 scene classes and is loaded via `torchvision.datasets.ImageFolder`.

## Model architecture

The CNN is implemented in `model.py` as `TNet`:
- 4 convolutional stages
- Conv-BN-ReLU blocks
- Max pooling after the first three stages
- Global average pooling
- Dropout + final linear classifier

## Training pipeline

Training is handled by `train.py` and supports:
- Reproducible random seeds
- CUDA training when available
- DirectML fallback for AMD GPUs on Windows
- CPU fallback
- Validation split creation from the training set
- MixUp and CutMix augmentation
- Checkpoint saving for best and latest models
- Loss and accuracy plotting

Typical usage:

```bash
python train.py --data-root data --out-dir checkpoints --epochs 120
```

Optional visualization:

```bash
python train.py --data-root data --out-dir checkpoints --visualize
```

## Evaluation

Evaluation is handled by `evaluate.py`.

To evaluate both saved checkpoints on validation and test data:

```bash
python evaluate.py --data-root data --ckpt-dir checkpoints
```

To evaluate only the best checkpoint:

```bash
python evaluate.py --data-root data --ckpt-dir checkpoints --checkpoint best
```

The script prints accuracy and generates confusion-matrix plots for each split.

## Repository structure

```text
.
├── AI_USAGE.md                 # Notes on AI-assisted experimentation
├── ITCS_6169_8169_Assignment1_2026.ipynb  # Notebook with exploration and experiments
├── data/                       # Training and test image folders
├── checkpoints/                # Final saved training artifacts
├── model.py                    # CNN architecture
├── train.py                    # Training logic and checkpoint export
├── evaluate.py                 # Model evaluation and class metrics
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore
```

## Results

The project includes final saved artifacts in `checkpoints/`.

The best validation checkpoint achieved approximately:
- Validation accuracy: 88.96%

Artifacts in `checkpoints/` include:
- `best.pt`
- `last.pt`
- `history.json`
- Loss and accuracy plots
- Confusion matrices for validation and test splits

## Environment setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If a GPU is available, the training script will automatically use CUDA. On AMD Windows systems, it will use DirectML when available.

## Notes

This project is designed as a machine-learning assignment focused on scene recognition and practical model tuning. It combines a compact CNN architecture with modern training augmentation techniques to achieve strong performance on a 16-class scene dataset.

## Related file

- `AI_USAGE.md` documents how AI tools were used during development and experimentation.