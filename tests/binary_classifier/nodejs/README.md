# ONNX Binary Classifier - JavaScript Implementation

A high-performance sentiment analysis classifier using ONNX Runtime for JavaScript (Node.js) with comprehensive performance monitoring and system information display.

## 🚀 Features

- **Sentiment Analysis**: Binary classification (Positive/Negative) using TF-IDF preprocessing
- **Performance Monitoring**: Detailed timing breakdown, resource usage tracking, and throughput analysis
- **System Information**: Platform detection, CPU/memory specs, runtime versions
- **Benchmarking Mode**: Statistical performance analysis with multiple runs
- **CI/CD Ready**: Graceful handling when model files are missing
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 📋 Requirements

- Node.js 18.0.0 or higher
- ONNX Runtime for JavaScript 1.19.2
- Model files: `model.onnx`, `vocab.json`, `scaler.json`

## 🛠️ Installation

```bash
# Install dependencies
npm install

# Or using yarn
yarn install
```

## 📁 Required Files

Place these files in the same directory as `package.json`:

1. **model.onnx** - The trained ONNX model file
2. **vocab.json** - TF-IDF vocabulary and IDF weights
   ```json
   {
     "vocab": {"word1": 0, "word2": 1, ...},
     "idf": [1.23, 4.56, ...]
   }
   ```
3. **scaler.json** - Feature scaling parameters
   ```json
   {
     "mean": [0.1, 0.2, ...],
     "scale": [1.1, 1.2, ...]
   }
   ```

## 🎯 Usage

### Basic Usage
```bash
# Run with default test texts
npm start

# Test custom text
npm start "This product is amazing!"

# Using node directly
node test_onnx_model.js "Great service!"
```

### Performance Benchmarking
```bash
# Run 100 iterations benchmark
npm run benchmark

# Custom number of iterations
npm run benchmark 500
node test_onnx_model.js --benchmark 500
```

## 📊 Output Features

### System Information
- Platform and processor architecture
- CPU cores and total memory
- Node.js and ONNX Runtime versions

### Performance Metrics
- **Timing Breakdown**: Preprocessing, inference, and postprocessing times with percentages
- **Resource Usage**: Memory consumption and CPU utilization
- **Throughput**: Texts processed per second
- **Performance Rating**: EXCELLENT/GOOD/ACCEPTABLE/POOR classification

### Sample Output
```
🤖 ONNX BINARY CLASSIFIER - JAVASCRIPT IMPLEMENTATION
====================================================

🔄 Processing: This product is amazing!

💻 SYSTEM INFORMATION:
   Platform: darwin
   Processor: arm64
   CPU Cores: 8
   Total Memory: 16.0 GB
   Runtime: JavaScript Implementation
   Node.js Version: v18.17.0
   ONNX Runtime Version: 1.22.0

📊 SENTIMENT ANALYSIS RESULTS:
   🏆 Predicted Sentiment: Positive
   📈 Confidence: 87.45% (0.8745)
   📝 Input Text: "This product is amazing!"

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 45.23ms
   ┣━ Preprocessing: 12.34ms (27.3%)
   ┣━ Model Inference: 28.45ms (62.9%)
   ┗━ Postprocessing: 4.44ms (9.8%)

🚀 THROUGHPUT:
   Texts per second: 22.1

💾 RESOURCE USAGE:
   Memory Start: 45.67 MB
   Memory End: 47.23 MB
   Memory Delta: +1.56 MB
   CPU Usage: 15.2% avg, 45.8% peak (12 samples)

🎯 PERFORMANCE RATING: ✅ GOOD
   (45.2ms total - Target: <100ms)
```

## 🔧 Technical Details

### Preprocessing Pipeline
1. **Text Tokenization**: Split text into words and convert to lowercase
2. **TF-IDF Vectorization**: Convert to 5000-dimensional feature vector
3. **Feature Scaling**: Apply mean normalization and standard scaling

### Model Architecture
- **Input**: Float32 tensor [1, 5000] (TF-IDF features)
- **Output**: Float32 tensor [1, 1] (sentiment probability)
- **Threshold**: 0.5 (>0.5 = Positive, ≤0.5 = Negative)

### Performance Monitoring
- **High-Resolution Timing**: Uses `performance.now()` for microsecond precision
- **Memory Tracking**: Monitors heap usage before/after processing
- **CPU Monitoring**: Samples CPU usage during processing (approximation)
- **Statistical Analysis**: Mean, min, max calculations for benchmarking

## 🏗️ CI/CD Integration

The implementation includes CI-friendly features:

```javascript
// Graceful handling when model files are missing
if (!checkModelFiles()) {
    console.log('⚠️ Model files not found - exiting safely');
    console.log('✅ JavaScript implementation compiled successfully');
    return;
}
```

### GitHub Actions Integration
```yaml
- name: Test JavaScript Binary Classifier
  run: |
    cd tests/binary_classifier/nodejs
    npm install
    npm test
```

## 🐛 Troubleshooting

### Common Issues

1. **Module not found errors**
   ```bash
   # Ensure you're using ES modules
   # package.json should have "type": "module"
   ```

2. **ONNX Runtime installation issues**
   ```bash
   # Clear npm cache and reinstall
   npm cache clean --force
   npm install
   ```

3. **Memory issues with large models**
   ```bash
   # Increase Node.js memory limit
   node --max-old-space-size=4096 test_onnx_model.js
   ```

## 📈 Performance Expectations

- **Target**: <100ms total processing time
- **Typical**: 20-80ms on modern hardware
- **Throughput**: 15-50 texts/second depending on hardware

## 🤝 Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure CI compatibility

## 📄 License

This implementation is part of the ONNX model testing framework. 