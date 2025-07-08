# 🎭 Multiclass Sigmoid ONNX Model Tests

This directory contains cross-language implementations for testing **multiclass sigmoid ONNX models**, specifically designed for **emotion classification** tasks.

## 🔬 What is Multiclass Sigmoid?

Unlike traditional multiclass classification (which uses softmax and outputs a single class), multiclass sigmoid classification:

- **Outputs probabilities for each class independently** using sigmoid activation
- **Allows multiple classes to be positive** (multi-label classification)
- **Ideal for emotion detection** where text can express multiple emotions simultaneously
- **Uses TF-IDF vectorization** for text preprocessing (similar to binary classification)

## 🎯 Emotion Classification Task

The implementations test emotion classification with categories such as:
- **Anger** - Expressions of frustration, irritation, or rage
- **Disgust** - Reactions of revulsion or strong dislike  
- **Fear** - Expressions of anxiety, worry, or terror
- **Happiness** - Joy, contentment, or positive emotions
- **Sadness** - Expressions of sorrow, grief, or melancholy
- **Surprise** - Reactions to unexpected events or information

## 🛠️ Supported Languages

| Language | Status | Key Features |
|----------|--------|--------------|
| **Python** | ✅ Complete | sklearn-compatible TF-IDF, comprehensive metrics |
| **Node.js** | ✅ Complete | Fast preprocessing, memory efficient |
| **Java** | ✅ Complete | Enterprise-ready, robust error handling |
| **C++** | ✅ Complete | High performance, native implementation |
| **C** | ✅ Complete | Lightweight, embedded systems ready |
| **Rust** | ✅ Complete | Memory safe, ultra-fast processing |
| **Dart** | ✅ Complete | Cross-platform, Flutter compatible |
| **Swift** | ✅ Complete | iOS/macOS optimized, Core ML integration |

## 📊 Expected Model Format

Each implementation expects these files:

### `model.onnx`
- **Input**: TF-IDF vectorized text features (Float32, shape: [1, N])
- **Output**: Sigmoid probabilities for each emotion class (Float32, shape: [1, num_classes])
- **Architecture**: Dense layers with sigmoid activation (NOT softmax)

### `vocab.json`
```json
{
  "vocabulary": {"word1": 0, "word2": 1, ...},
  "idf": [idf_val1, idf_val2, ...],
  "max_features": 5000
}
```

### `scaler.json`
```json
{
  "0": "anger",
  "1": "disgust", 
  "2": "fear",
  "3": "happiness",
  "4": "sadness",
  "5": "surprise"
}
```

## 🚀 Usage Examples

### Python
```bash
cd tests/multiclass_sigmoid/python
python test_onnx_model.py "I'm so excited but also nervous about tomorrow!"
```

### Node.js
```bash
cd tests/multiclass_sigmoid/nodejs
node test_onnx_model.js "I'm so excited but also nervous about tomorrow!"
```

### Java
```bash
cd tests/multiclass_sigmoid/java
mvn exec:java -Dexec.args="I'm so excited but also nervous about tomorrow!"
```

### C++
```bash
cd tests/multiclass_sigmoid/cpp
make && ./test_onnx_model "I'm so excited but also nervous about tomorrow!"
```

### Rust
```bash
cd tests/multiclass_sigmoid/rust
cargo run --release -- "I'm so excited but also nervous about tomorrow!"
```

### Dart
```bash
cd tests/multiclass_sigmoid/dart
dart run lib/main.dart "I'm so excited but also nervous about tomorrow!"
```

### Swift
```bash
cd tests/multiclass_sigmoid/SwiftONNXRunner
swift run SwiftONNXRunner "I'm so excited but also nervous about tomorrow!"
```

## 📈 Performance Characteristics

Multiclass sigmoid models typically show:

- **Preprocessing**: Similar to binary classification (TF-IDF vectorization)
- **Inference**: Slightly slower than binary due to multiple outputs
- **Memory**: Comparable to binary classification
- **Accuracy**: Depends on emotion complexity in text

## 🔧 Key Differences from Binary/Multiclass

| Aspect | Binary | Multiclass | Multiclass Sigmoid |
|--------|--------|------------|-------------------|
| **Output** | Single probability | Single class (softmax) | Multiple probabilities |
| **Activation** | Sigmoid | Softmax | Sigmoid per class |
| **Use Case** | Spam detection | Topic classification | Emotion detection |
| **Multi-label** | No | No | Yes |
| **Preprocessing** | TF-IDF | Tokenization | TF-IDF |

## 🎯 Integration with Testing Framework

This complements the existing test suite:
- **Binary Classifier**: Spam detection, sentiment analysis
- **Multiclass Classifier**: News topic classification  
- **Multiclass Sigmoid**: Emotion classification ⭐ NEW

All implementations follow the same performance reporting format and can be used in GitHub Actions workflows for comprehensive testing across languages and model types. 