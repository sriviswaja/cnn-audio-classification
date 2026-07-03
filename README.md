# CNN Audio Classification 🎧

A CNN-based audio classifier that identifies urban sounds from Mel-spectrogram
representations, built with TensorFlow/Keras on the UrbanSound8K dataset.

## Overview

This project converts raw audio waveforms into Mel-spectrograms and trains a
convolutional neural network to classify them into 10 categories of urban sound.
Built end-to-end in Google Colab, from data download through EDA, preprocessing,
model training, and evaluation.

## Dataset

[UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) — 8,732
labeled sound excerpts (≤4s) across 10 classes:

`air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`,
`engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music`

## Pipeline

1. **EDA** — waveform and Mel-spectrogram visualization across classes, class
   distribution check, sample rate/duration variability analysis.
2. **Preprocessing** — audio resampled to 22050 Hz, padded/truncated to a fixed
   4-second length, converted to 128-bin log-Mel-spectrograms.
3. **Model** — a 3-block CNN (Conv2D + MaxPooling) with GlobalAveragePooling2D
   and Dropout(0.5), trained with EarlyStopping.
4. **Evaluation** — confusion matrix and per-class classification report.

## Results

- **Test accuracy: 90%**
- Macro avg F1-score: 0.90 across all 10 classes
- Strongest performance: `gun_shot` (0.99 recall) — acoustically distinct
- Most common confusions: `drilling` ↔ `jackhammer` (both sustained mechanical
  noise), `children_playing` ↔ `street_music` (both broadband ambient noise)

## Key learning: fixing overfitting

An initial architecture (large Dense layer after Flatten) reached ~99% train
accuracy but only ~88% validation accuracy, with validation loss increasing —
classic overfitting from a parameter-heavy classifier head (4.3M of 4.45M total
params sat in one Dense layer).

**Fix:** replaced `Flatten() + Dense(128)` with `GlobalAveragePooling2D() +
Dense(64)`, and increased Dropout to 0.5. This cut the trainable parameters
drastically and produced training curves where train/val accuracy and loss
tracked closely together — a sign of genuine generalization rather than
memorization.

## Tech stack

- TensorFlow / Keras
- librosa (audio processing)
- NumPy, pandas
- scikit-learn (train/test split, metrics)
- matplotlib, seaborn (visualization)
- Google Colab (GPU runtime)

## Project structure
cnn-audio-classification/
├── notebooks/
│   └── audio_classification.ipynb   # full pipeline: EDA → training → evaluation
├── src/                              # (reserved for future modularized scripts)
├── data/                             # dataset (gitignored, downloaded via Kaggle API)
└── .gitignore

## Running it yourself

1. Get a [Kaggle API key](https://www.kaggle.com/settings) and a GitHub
   personal access token.
2. Open `notebooks/audio_classification.ipynb` in Colab.
3. Run all cells top to bottom — the notebook downloads the dataset via Kaggle,
   preprocesses it, and trains the model in one pass (GPU runtime recommended).

## Future improvements

- Switch to UrbanSound8K's official 10-fold cross-validation split (used a
  random stratified split here) to fully rule out data leakage between
  train/test sets.
- Try data augmentation (time-shifting, pitch-shifting, noise injection) to
  improve performance on the weakest classes (`children_playing`,
  `street_music`).
- Experiment with MFCCs as an alternative/complementary feature set.
