# Node.js Multiclass Sigmoid ONNX Model Test

This directory contains the Node.js implementation for testing multiclass sigmoid ONNX models, specifically designed for emotion classification.

## Requirements

- Node.js 14.0+
- Dependencies listed in `package.json`

## Setup

1. Install dependencies:
```bash
npm install
```

2. Place your model files in this directory:
- `model.onnx` - Your trained ONNX model
- `vocab.json` - TF-IDF vectorizer vocabulary and parameters
- `scaler.json` - Class labels for emotion categories

## Usage

Run the test with default text:
```bash
node test_onnx_model.js
```

Run with custom text:
```bash
node test_onnx_model.js "I feel so happy today!"
```

## Expected Output

The test will output:
- System information
- TF-IDF preprocessing details
- Emotion analysis results with probabilities for each emotion
- Performance metrics including timing and memory usage
- Overall performance rating

## Model Format

This implementation expects:
- **Input**: TF-IDF vectorized text features
- **Output**: Sigmoid probabilities for multiple emotion classes
- **Preprocessing**: TF-IDF vectorization matching sklearn behavior
- **Classes**: Emotion categories (anger, disgust, fear, happiness, sadness, surprise, etc.)

## Performance

Node.js implementation provides:
- Fast TF-IDF preprocessing with sklearn compatibility
- Efficient ONNX Runtime inference
- Detailed performance metrics and memory tracking
- Cross-platform compatibility 