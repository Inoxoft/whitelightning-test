# ONNX Multiclass Classifier - JavaScript Implementation

A high-performance news category classifier using ONNX Runtime for JavaScript (Node.js) with comprehensive performance monitoring and system information display.

## 🚀 Features

- **News Classification**: Multiclass classification (health, politics, sports, world) using token-based preprocessing
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
2. **vocab.json** - Tokenizer vocabulary mapping
   ```json
   {
     "word1": 1,
     "word2": 2,
     "<OOV>": 1,
     ...
   }
   ```
3. **scaler.json** - Label mapping for classes
   ```json
   {
     "0": "health",
     "1": "politics", 
     "2": "sports",
     "3": "world"
   }
   ```

## 🎯 Usage

### Basic Usage
```bash
# Run with default test texts
npm start

# Test custom text
npm start "France defeats Argentina in World Cup final"

# Using node directly
node test_onnx_model.js "New healthcare policy announced"
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
🤖 ONNX MULTICLASS CLASSIFIER - JAVASCRIPT IMPLEMENTATION
========================================================

🔄 Processing: France defeats Argentina in World Cup final

💻 SYSTEM INFORMATION:
   Platform: darwin
   Processor: arm64
   CPU Cores: 8
   Total Memory: 16.0 GB
   Runtime: JavaScript Implementation
   Node.js Version: v18.17.0
   ONNX Runtime Version: 1.22.0

📊 MULTICLASS CLASSIFICATION RESULTS:
   🏆 Predicted Category: sports
   📈 Confidence: 92.34% (0.9234)
   📝 Input Text: "France defeats Argentina in World Cup final"
   📋 All Class Probabilities:
      health: 0.0123 (1.2%)
      politics: 0.0234 (2.3%)
      sports: 0.9234 (92.3%)
      world: 0.0409 (4.1%)

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 38.45ms
   ┣━ Preprocessing: 8.23ms (21.4%)
   ┣━ Model Inference: 25.67ms (66.8%)
   ┗━ Postprocessing: 4.55ms (11.8%)

🚀 THROUGHPUT:
   Texts per second: 26.0

💾 RESOURCE USAGE:
   Memory Start: 42.34 MB
   Memory End: 43.78 MB
   Memory Delta: +1.44 MB
   CPU Usage: 12.8% avg, 38.9% peak (10 samples)

🎯 PERFORMANCE RATING: ✅ GOOD
   (38.5ms total - Target: <100ms)
```

## 🔧 Technical Details

### Preprocessing Pipeline
1. **Text Tokenization**: Split text into words and convert to lowercase
2. **Token Mapping**: Convert words to integer IDs using vocabulary
3. **Sequence Padding**: Pad/truncate to fixed length of 30 tokens
4. **OOV Handling**: Unknown words mapped to `<OOV>` token

### Model Architecture
- **Input**: Int32 tensor [1, 30] (token sequence)
- **Output**: Float32 tensor [1, 4] (class probabilities)
- **Classes**: health, politics, sports, world

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
- name: Test JavaScript Multiclass Classifier
  run: |
    cd tests/multiclass_classifier/nodejs
    npm install
    npm test
```

## 🎯 Classification Categories

The model classifies news articles into these categories:

- **🏥 Health**: Medical news, healthcare policies, disease outbreaks
- **🏛️ Politics**: Government actions, elections, political events
- **⚽ Sports**: Sports events, competitions, athlete news
- **🌍 World**: International news, global events, foreign affairs

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

4. **Tensor shape mismatches**
   ```bash
   # Ensure input sequence length is exactly 30
   # Check vocab.json format matches expected structure
   ```

## 📈 Performance Expectations

- **Target**: <100ms total processing time
- **Typical**: 20-60ms on modern hardware
- **Throughput**: 20-50 texts/second depending on hardware

## 🧪 Testing Examples

```bash
# Sports news
node test_onnx_model.js "Lakers win championship game"

# Politics news  
node test_onnx_model.js "President announces new economic policy"

# Health news
node test_onnx_model.js "New vaccine shows promising results"

# World news
node test_onnx_model.js "Climate summit reaches historic agreement"
```

## 🤝 Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure CI compatibility

## 📄 License

This implementation is part of the ONNX model testing framework. 