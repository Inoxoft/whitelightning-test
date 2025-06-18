# 🎯 Dart Multiclass Classification ONNX Model

This directory contains a **Dart/Flutter implementation** for multiclass text classification using ONNX Runtime. The model performs **topic classification** on news articles and other text content, categorizing them into 10 predefined categories.

## 📁 Directory Structure

```
dart/
├── lib/
│   └── main.dart              # Main Dart implementation
├── test/
│   └── onnx_inference_test.dart # Unit tests
├── model.onnx                 # Multiclass classification ONNX model
├── vocab.json                 # Vocabulary mapping for tokenization
├── scaler.json                # Label mapping for categories
├── pubspec.yaml               # Dart dependencies
├── pubspec.lock               # Lock file
└── README.md                  # This file
```

## 🛠️ Prerequisites

### Required Software
- **Flutter SDK 3.0+**: [Install Flutter](https://flutter.dev/docs/get-started/install)
- **Dart SDK 2.17+**: (included with Flutter)

### Installation

```bash
# Navigate to the Dart directory
cd tests/multiclass_classifier/dart

# Get dependencies
flutter pub get

# For console app (if available)
dart pub get
```

## 🚀 Usage

### Flutter App Usage
```bash
# Navigate to the Dart directory
cd tests/multiclass_classifier/dart

# Run Flutter app (mobile/desktop)
flutter run

# Run tests
flutter test
```

### Console Usage (if implemented)
```bash
# Run with default text
dart run lib/main.dart

# Run with custom text
dart run lib/main.dart "NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"
```

### Expected Output
```
🤖 ONNX MULTICLASS CLASSIFIER - DART IMPLEMENTATION
==================================================
🔄 Processing: "NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"

💻 SYSTEM INFORMATION:
   Platform: linux
   Version: Linux 5.4.0
   CPU Cores: 4 logical
   Total Memory: Not available in Dart Flutter
   Implementation: Dart with ONNX Runtime

📊 TOPIC CLASSIFICATION RESULTS:
⏱️  Processing Time: 114ms
   🏆 Predicted Category: SPORTS 📝
   📈 Confidence: 100.0%
   📝 Input Text: "NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"

📊 DETAILED PROBABILITIES:
   📝 Business: 0.0% 
   📝 Education: 0.0% 
   📝 Entertainment: 0.0% 
   📝 Environment: 0.0% 
   📝 Health: 0.0% 
   📝 Politics: 0.0% 
   📝 Science: 0.0% 
   📝 Sports: 100.0% ████████████████████ ⭐
   📝 Technology: 0.0% 
   📝 World: 0.0% 

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 114ms
   ┣━ Preprocessing: 34ms (30%)
   ┣━ Model Inference: 68ms (60%)
   ┗━ Post-processing: 11ms (10%)
   🧠 CPU Usage: Not available in Dart Flutter
   💾 Memory: Not available in Dart Flutter
   🚀 Throughput: 8.8 texts/sec
   Performance Rating: ✅ GOOD
```

## 🎯 Performance Characteristics

- **Total Time**: ~114ms (good performance)
- **Preprocessing**: Sequence tokenization (30 tokens)
- **Model Input**: Int32 array [1, 30]
- **Inference Engine**: ONNX Runtime Flutter
- **Memory Usage**: Estimated ~4MB
- **CPU Usage**: Estimated ~20%
- **Throughput**: 8.8 texts per second

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 