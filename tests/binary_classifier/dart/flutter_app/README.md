# Binary Classifier Flutter App

A simple Flutter application that demonstrates binary text classification using ONNX Runtime.

## Features

- Clean, simple UI for text input and classification
- Real-time binary classification (Positive/Negative)
- Probability scores with percentage display
- ONNX model integration with preprocessing pipeline

## Architecture

The app consists of:

1. **Text Preprocessing**: TF-IDF vectorization with vocabulary mapping and scaling
2. **ONNX Model Inference**: Binary classification using ONNX Runtime
3. **Result Display**: Clean UI showing classification results and confidence scores

## Model Files

The app requires three model files in `assets/models/`:
- `model.onnx` - The trained binary classification model
- `vocab.json` - Vocabulary mapping and IDF weights
- `scaler.json` - Feature scaling parameters (mean and scale)

## Usage

1. Enter text in the input field
2. Tap "Classify" to run the model
3. View the classification result (Positive/Negative) with probability score

## Dependencies

- `flutter` - UI framework
- `onnxruntime` - Model inference engine
- `cupertino_icons` - iOS-style icons

## Building

```bash
flutter pub get
flutter build web --release  # For web deployment
flutter build apk --release  # For Android
```

## Testing

The app can be tested through the GitHub Actions workflow by selecting "flutter" as the language option. 