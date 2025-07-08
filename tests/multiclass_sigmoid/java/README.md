# Java Multiclass Sigmoid ONNX Model Test

This directory contains the Java implementation for testing multiclass sigmoid ONNX models, specifically designed for emotion classification.

## Requirements

- Java 11+
- Maven 3.6+

## Setup

1. Compile the project:
```bash
mvn compile
```

2. Place your model files in this directory:
- `model.onnx` - Your trained ONNX model
- `vocab.json` - TF-IDF vectorizer vocabulary and parameters
- `scaler.json` - Class labels for emotion categories

## Usage

Run the test with default text:
```bash
mvn exec:java
```

Run with custom text:
```bash
mvn exec:java -Dexec.args="I feel so happy today!"
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

## Dependencies

- ONNX Runtime Java: Model inference
- Gson: JSON parsing for model artifacts
- Java Management API: System monitoring 