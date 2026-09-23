# ITCS 6169/8169 Assignment 1

This repository contains a simple CNN training pipeline for the 16-class scene recognition task described in the assignment.

## Project structure

- `src/model.py` — CNN model and data transforms
- `train.py` — training and validation workflow
- `evaluate.py` — final evaluation on the test set
- `AI_USAGE.md` — brief description of AI-assisted development work
- `requirements.txt` — Python dependencies

## Environment setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

On AMD systems, if you want to use DirectML, install the optional package:

```bash
python -m pip install torch-directml
```

## Run training

```bash
python train.py --data-root data --epochs 20 --batch-size 64 --lr 0.002 --img-size 64
```

## Run evaluation

```bash
python evaluate.py --data-root data --checkpoint checkpoints/best_model.pth
```

## Notes

- The default setup mirrors the notebook baseline, but you should adjust architecture, augmentation, and regularization to improve validation accuracy.
- Use validation data for model selection and reserve the test set for the final evaluation.
- Record your experiments and model choices in the repository README or experiment notes.
