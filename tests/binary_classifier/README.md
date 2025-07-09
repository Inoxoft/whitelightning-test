# ⚖️ Binary Classifier ONNX Model Tests

This directory contains cross-language implementations for testing **binary classification ONNX models**, specifically designed for **sentiment analysis** tasks.

## 🔬 What is Binary Classification?

Binary classification predicts one of two possible outcomes using **sigmoid activation**:

- **Outputs a single probability score** (0.0 to 1.0)
- **Threshold-based decision making** (typically 0.5)
- **Ideal for sentiment analysis** (positive vs negative sentiment)
- **Uses TF-IDF vectorization** for text feature extraction

## 🎯 Sentiment Analysis Task

The implementations test sentiment analysis on text inputs:

- **Positive sentiment**: Probability ≥ 0.5 (celebratory, congratulatory, happy content)
- **Negative sentiment**: Probability < 0.5 (complaints, concerns, sad content)
- **Input format**: Raw text strings
- **Output format**: Single probability score with sentiment label

## 🌍 Language Implementations

### Performance Comparison

| Language | Avg. Processing Time | Performance Rating | Status |
|----------|---------------------|-------------------|---------|
| **C++** | ~2-5ms | 🚀 EXCELLENT | ✅ |
| **Rust** | ~5-10ms | 🚀 EXCELLENT | ✅ |
| **Python** | ~15-25ms | ✅ GOOD | ✅ |
| **Java** | ~20-30ms | ✅ GOOD | ✅ |
| **Node.js** | ~25-35ms | ✅ GOOD | ✅ |
| **Dart** | ~30-40ms | ✅ GOOD | ✅ |
| **C** | ~40-60ms | ⚠️ ACCEPTABLE | ✅ |
| **Swift** | ~45-70ms | ⚠️ ACCEPTABLE | ✅ |

### 📁 Directory Structure

```
tests/binary_classifier/
├── README.md
├── python/
│   ├── test_onnx_model.py      # Main test implementation
│   ├── vocab.json              # TF-IDF vocabulary (5000+ terms)
│   ├── scaler.json             # Feature scaling parameters
│   └── requirements.txt        # Python dependencies
├── cpp/
│   ├── test_onnx_model.cpp     # C++ implementation
│   ├── Makefile               # Build configuration
│   └── CMakeLists.txt         # CMake build file
├── rust/
│   ├── src/main.rs            # Rust implementation
│   ├── Cargo.toml             # Rust dependencies
│   └── Cargo.lock             # Dependency lock file
├── java/
│   ├── TestONNXModel.java     # Java implementation
│   ├── build.gradle           # Gradle build file
│   └── pom.xml               # Maven configuration
├── nodejs/
│   ├── test_onnx_model.js     # Node.js implementation
│   ├── package.json           # NPM dependencies
│   └── package-lock.json      # Dependency lock file
├── dart/
│   ├── lib/main.dart          # Dart implementation
│   ├── pubspec.yaml           # Dart dependencies
│   └── test/                  # Dart test files
├── c/
│   ├── test_onnx_model.c      # C implementation
│   ├── Makefile               # Build configuration
│   └── onnx_runtime_libs/     # ONNX Runtime libraries
└── SwiftONNXRunner/
    ├── Package.swift          # Swift package configuration
    └── Sources/SwiftONNXRunner/
        └── main.swift         # Swift implementation
```

## 🔧 Technical Specifications

### Model Architecture
- **Input Shape**: `[1, 5000]` (TF-IDF feature vector)
- **Output Shape**: `[1, 1]` (single probability score)
- **Activation**: Sigmoid
- **Model Format**: ONNX Runtime compatible

### Text Processing Pipeline
1. **Tokenization**: Split text into individual words
2. **TF-IDF Vectorization**: Convert to numerical features using vocabulary
3. **Scaling**: Apply standardization using mean/scale parameters
4. **Inference**: Pass through ONNX model
5. **Classification**: Apply 0.5 threshold for binary decision

### Configuration Files

#### `vocab.json` Structure
```json
{
  "vocab": {
    "word1": 0,
    "word2": 1,
    ...
  },
  "idf": [4.2, 3.8, 5.1, ...]
}
```

#### `scaler.json` Structure
```json
{
  "mean": [0.0001, 0.0064, ...],
  "scale": [0.009, 0.044, ...]
}
```

## 🚀 Quick Start

### Python
```bash
cd tests/binary_classifier/python/
pip install -r requirements.txt
python test_onnx_model.py
```

### C++
```bash
cd tests/binary_classifier/cpp/
make
./test_onnx_model
```

### Rust
```bash
cd tests/binary_classifier/rust/
cargo run
```

### Java
```bash
cd tests/binary_classifier/java/
./gradlew run
```

### Node.js
```bash
cd tests/binary_classifier/nodejs/
npm install
npm start
```

### Dart
```bash
cd tests/binary_classifier/dart/
dart pub get
dart run lib/main.dart
```

### C
```bash
cd tests/binary_classifier/c/
make
./test_onnx_model
```

### Swift
```bash
cd tests/binary_classifier/SwiftONNXRunner/
swift run
```

## 📊 Example Output

```
🤖 ONNX BINARY CLASSIFIER - [LANGUAGE] IMPLEMENTATION
====================================================
🔄 Processing: "Congratulations! You've won a free iPhone!"

💻 SYSTEM INFORMATION:
   Platform: macOS
   CPU Cores: 8 physical, 16 logical
   Total Memory: 32.0 GB
   Runtime: [Language] Implementation

📊 SENTIMENT ANALYSIS RESULTS:
⏱️  Processing Time: 23.45ms
   🏆 Predicted Sentiment: Positive
   📈 Confidence: 87.3% (0.8732)
   📝 Input Text: "Congratulations! You've won a free iPhone!"

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 23.45ms
   ┣━ Preprocessing: 12.1ms (51.6%)
   ┣━ Model Inference: 8.2ms (35.0%)
   ┗━ Postprocessing: 3.1ms (13.4%)

🚀 THROUGHPUT:
   Texts per second: 42.7

💾 RESOURCE USAGE:
   Memory Delta: +2.4 MB
   CPU Usage: 45.2% avg, 78.9% peak

🎯 PERFORMANCE RATING: ✅ GOOD
   (23.4ms total - Target: <100ms)
```

## 🧪 Test Cases

### Standard Test Inputs
- **Positive**: `"Congratulations! You've won a free iPhone — click here to claim your prize now!"`
- **Negative**: `"I'm very disappointed with this terrible service and poor quality product"`
- **Neutral**: `"The weather today is partly cloudy with a chance of rain"`

### Expected Outputs
- **Positive texts**: Probability > 0.5, Label: "Positive"
- **Negative texts**: Probability < 0.5, Label: "Negative"
- **Confidence scores**: Typically 0.6-0.9 for clear sentiment

## 🔧 Dependencies

### Core Requirements
- **ONNX Runtime**: v1.16.0+
- **Model files**: `model.onnx`, `vocab.json`, `scaler.json`
- **Language-specific**: See individual implementation directories

### Development Tools
- **Performance profiling**: Built-in timing and memory monitoring
- **CI/CD support**: GitHub Actions compatible
- **Cross-platform**: Windows, macOS, Linux support

## 🐛 Common Issues

### Model Loading Errors
- **Issue**: "Model file not found"
- **Solution**: Ensure `model.onnx` is in the correct directory
- **CI Environment**: Graceful handling when model files are missing

### Memory Issues
- **Issue**: High memory usage during processing
- **Solution**: Optimize batch processing and clear intermediate results
- **Monitoring**: Built-in memory tracking in all implementations

### Performance Bottlenecks
- **TF-IDF Processing**: Most time-consuming step (40-60% of total time)
- **Model Inference**: Typically 20-40% of processing time
- **Optimization**: Consider vocabulary pruning for production use

## 📚 Additional Resources

- **ONNX Runtime Documentation**: https://onnxruntime.ai/
- **Binary Classification Theory**: Standard machine learning textbooks
- **TF-IDF Explanation**: Text mining and information retrieval resources
- **Model Performance Tuning**: ONNX optimization guides

## 🤝 Contributing

1. Follow the existing code structure for new language implementations
2. Include comprehensive error handling and logging
3. Add performance benchmarking with standardized metrics
4. Ensure CI/CD compatibility for automated testing
5. Document any new dependencies or configuration requirements

---

*This implementation provides a robust foundation for binary sentiment classification across multiple programming languages with consistent performance monitoring and error handling.* 