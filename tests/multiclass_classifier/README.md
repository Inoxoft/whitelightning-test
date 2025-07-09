# 🏷️ Multiclass Classifier ONNX Model Tests

This directory contains cross-language implementations for testing **multiclass classification ONNX models**, specifically designed for **topic classification** tasks.

## 🔬 What is Multiclass Classification?

Multiclass classification predicts one of multiple possible categories using **softmax activation**:

- **Outputs probability distribution** across all classes
- **Argmax selection** for final prediction
- **Ideal for topic classification** (news categories, document types)
- **Uses token-based preprocessing** with vocabulary mapping

## 🎯 Topic Classification Task

The implementations test topic classification on text inputs:

- **Business** 💼: Financial news, market updates, corporate announcements
- **Health** 🏥: Medical news, health tips, wellness articles
- **Politics** 🏛️: Government news, policy updates, election coverage
- **Sports** ⚽: Game results, player news, tournament coverage

## 🌍 Language Implementations

### Performance Comparison

| Language | Avg. Processing Time | Performance Rating | Status |
|----------|---------------------|-------------------|---------|
| **C++** | ~3-8ms | 🚀 EXCELLENT | ✅ |
| **Rust** | ~8-15ms | 🚀 EXCELLENT | ✅ |
| **Python** | ~20-35ms | ✅ GOOD | ✅ |
| **Java** | ~25-40ms | ✅ GOOD | ✅ |
| **Node.js** | ~30-45ms | ✅ GOOD | ✅ |
| **Dart** | ~35-50ms | ✅ GOOD | ✅ |
| **C** | ~50-75ms | ⚠️ ACCEPTABLE | ✅ |
| **Swift** | ~55-85ms | ⚠️ ACCEPTABLE | ✅ |

### 📁 Directory Structure

```
tests/multiclass_classifier/
├── README.md
├── python/
│   ├── test_onnx_model.py      # Main test implementation
│   ├── vocab.json              # Token vocabulary mapping
│   ├── scaler.json             # Class label mapping
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
- **Input Shape**: `[1, 30]` (tokenized and padded sequence)
- **Output Shape**: `[1, 4]` (probability distribution over 4 classes)
- **Activation**: Softmax
- **Model Format**: ONNX Runtime compatible

### Text Processing Pipeline
1. **Tokenization**: Split text into individual words
2. **Token Mapping**: Convert words to vocabulary indices
3. **Sequence Padding**: Pad/truncate to fixed length (30 tokens)
4. **Inference**: Pass through ONNX model
5. **Classification**: Apply argmax to select highest probability class

### Configuration Files

#### `vocab.json` Structure
```json
{
  "word1": 1,
  "word2": 2,
  "word3": 3,
  "<OOV>": 1,
  ...
}
```

#### `scaler.json` Structure (Label Mapping)
```json
{
  "0": "Business",
  "1": "Health", 
  "2": "Politics",
  "3": "Sports"
}
```

## 🚀 Quick Start

### Python
```bash
cd tests/multiclass_classifier/python/
pip install -r requirements.txt
python test_onnx_model.py
```

### C++
```bash
cd tests/multiclass_classifier/cpp/
make
./test_onnx_model
```

### Rust
```bash
cd tests/multiclass_classifier/rust/
cargo run
```

### Java
```bash
cd tests/multiclass_classifier/java/
./gradlew run
```

### Node.js
```bash
cd tests/multiclass_classifier/nodejs/
npm install
npm start
```

### Dart
```bash
cd tests/multiclass_classifier/dart/
dart pub get
dart run lib/main.dart
```

### C
```bash
cd tests/multiclass_classifier/c/
make
./test_onnx_model
```

### Swift
```bash
cd tests/multiclass_classifier/SwiftONNXRunner/
swift run
```

## 📊 Example Output

```
🤖 ONNX MULTICLASS CLASSIFIER - [LANGUAGE] IMPLEMENTATION
========================================================
🔄 Processing: "NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"

💻 SYSTEM INFORMATION:
   Platform: macOS
   CPU Cores: 8 physical, 16 logical
   Total Memory: 32.0 GB
   Runtime: [Language] Implementation

📊 TOPIC CLASSIFICATION RESULTS:
⏱️  Processing Time: 28.7ms
   🏆 Predicted Category: SPORTS ⚽
   📈 Confidence: 94.2%
   📝 Input Text: "NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"

📊 DETAILED PROBABILITIES:
   💼 Business: 2.1% ████
   🏥 Health: 1.8% ███
   🏛️ Politics: 1.9% ███
   ⚽ Sports: 94.2% ████████████████ ⭐

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 28.7ms
   ┣━ Preprocessing: 15.2ms (53.0%)
   ┣━ Model Inference: 10.8ms (37.6%)
   ┗━ Postprocessing: 2.7ms (9.4%)

🚀 THROUGHPUT:
   Texts per second: 34.8

💾 RESOURCE USAGE:
   Memory Delta: +1.8 MB
   CPU Usage: 42.8% avg, 71.4% peak

🎯 PERFORMANCE RATING: ✅ GOOD
   (28.7ms total - Target: <100ms)
```

## 🧪 Test Cases

### Standard Test Inputs

#### Business 💼
- **Example**: `"Apple Inc. reports record quarterly earnings with revenue up 15% year-over-year"`
- **Expected**: High confidence for Business category

#### Health 🏥
- **Example**: `"New study shows Mediterranean diet reduces risk of heart disease by 30%"`
- **Expected**: High confidence for Health category

#### Politics 🏛️
- **Example**: `"Congress passes bipartisan infrastructure bill with $1.2 trillion funding"`
- **Expected**: High confidence for Politics category

#### Sports ⚽
- **Example**: `"NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"`
- **Expected**: High confidence for Sports category

### Expected Outputs
- **Clear categories**: Confidence > 80% for well-defined topics
- **Ambiguous text**: More distributed probabilities across categories
- **Confidence scores**: Typically 0.7-0.95 for clear topic classification

## 🔧 Technical Details

### Vocabulary Processing
- **Vocabulary Size**: ~10,000 unique tokens
- **OOV Handling**: Out-of-vocabulary words mapped to `<OOV>` token
- **Sequence Length**: Fixed at 30 tokens (pad with zeros, truncate if longer)
- **Token Indexing**: Integer indices starting from 1

### Model Characteristics
- **Architecture**: Typically embedding + LSTM/GRU + dense layers
- **Training Data**: News articles from business, health, politics, sports domains
- **Accuracy**: Expected >90% on clean, well-formatted text
- **Inference Speed**: Optimized for real-time classification

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

### Vocabulary Issues
- **Issue**: High OOV rate affecting accuracy
- **Solution**: Expand vocabulary or use subword tokenization
- **Monitoring**: Track OOV percentage in preprocessing

### Performance Bottlenecks
- **Token Processing**: Most time-consuming step (40-60% of total time)
- **Model Inference**: Typically 30-50% of processing time
- **Optimization**: Consider vocabulary pruning and batch processing

## 📈 Performance Optimization

### Preprocessing Optimizations
- **Vocabulary Caching**: Keep vocabulary in memory
- **Batch Processing**: Process multiple texts simultaneously
- **Parallel Tokenization**: Use multi-threading for large batches

### Model Optimizations
- **Quantization**: Use INT8 quantized models for faster inference
- **Graph Optimization**: Enable ONNX Runtime graph optimizations
- **Provider Selection**: Use GPU/CPU providers based on hardware

## 📚 Additional Resources

- **ONNX Runtime Documentation**: https://onnxruntime.ai/
- **Multiclass Classification Theory**: Pattern recognition textbooks
- **Text Classification Techniques**: NLP and machine learning resources
- **Topic Modeling**: Information retrieval and text mining guides

## 🤝 Contributing

1. Follow the existing code structure for new language implementations
2. Include comprehensive error handling and logging
3. Add performance benchmarking with standardized metrics
4. Ensure CI/CD compatibility for automated testing
5. Document any new dependencies or configuration requirements

### Adding New Categories
1. Update `scaler.json` with new label mappings
2. Retrain model with additional category data
3. Update test cases and documentation
4. Verify performance across all language implementations

---

*This implementation provides a robust foundation for multiclass topic classification across multiple programming languages with consistent performance monitoring and error handling.* 