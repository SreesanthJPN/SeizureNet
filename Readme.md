# Epileptic Seizure Detection System

A deep learning-based system for detecting epileptic seizures using EEG signals. This project implements a hybrid CNN-LSTM architecture to analyze multi-channel EEG data and classify seizure events.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to develop an automated system for detecting epileptic seizures using EEG signals. The system processes multi-channel EEG data through a sophisticated deep learning pipeline to identify seizure events in real-time. The implementation includes data preprocessing, feature engineering, model training, and evaluation components.

## Features

- Multi-channel EEG signal processing
- Hybrid CNN-LSTM architecture
- Real-time seizure detection
- Class imbalance handling
- Comprehensive data preprocessing pipeline
- Model checkpointing and early stopping
- Performance evaluation metrics

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- NumPy
- scikit-learn
- imbalanced-learn
- Matplotlib

### Setup
```bash
# Clone the repository
git clone https://github.com/SreesanthJPN/SeizureNet.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Pipeline

1. **Data Collection**
   - Raw EEG data from multiple patients
   - Format: (samples, channels, timepoints)
   - Initial shape: (710, 35, 61440)

2. **Preprocessing**
   - Data cleaning and normalization
   - Channel selection and reduction
   - Time window segmentation
   - Feature extraction

3. **Data Balancing**
   - Original distribution:
     - Seizure (class 1): 557 samples
     - Non-seizure (class 0): 262 samples
   - Balanced distribution:
     - 400 seizure samples
     - 262 non-seizure samples

## Model Architecture

The system uses a hybrid CNN-LSTM architecture:

```python
Input: (None, 13, 61440)  # 13 channels, 61440 timepoints

# Convolutional Layers
Conv1D(32) → Conv1D(32) → MaxPool1D
Conv1D(64) → Conv1D(32) → MaxPool1D

# LSTM Layers
LSTM(64) → LSTM(64)

# Dense Layers
Dense(64) → Dense(128) → Dense(2)
```

## Usage

### Data Preparation
```python
# Load and preprocess data
data = np.load('data/processed/balanced_data.npy')
labels = np.load('data/processed/one_hot_encoded_labels.npy')

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)
```

### Model Training
```python
# Load and compile model
model = load_model('models/80model.keras')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, 
                   validation_split=0.1,
                   epochs=100,
                   batch_size=32,
                   callbacks=[EarlyStopping(patience=5)])
```

### Evaluation
```python
# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')
```

## Results

The model achieves:
- Accuracy: 80% on test set
- Precision: 0.82
- Recall: 0.78
- F1-score: 0.80

## License

This project is licensed under the MIT License

## Acknowledgments

- Dataset: [https://physionet.org/content/siena-scalp-eeg/1.0.0/]
