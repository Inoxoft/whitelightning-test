# Dart ONNX Multiclass Classifier

A comprehensive Dart implementation of the ONNX multiclass classifier with system monitoring, performance metrics, and support for multiple text categories.

## Features

### 🖥️ System Information
- Platform and architecture detection
- CPU information and core count
- Memory usage monitoring
- Dart version and ONNX runtime version

### 📊 Performance Metrics
- Detailed timing breakdown (preprocessing, inference, postprocessing)
- Throughput analysis (predictions per second)
- Memory usage tracking
- CPU usage monitoring (simulated)
- Performance rating system

### 🎯 Multiclass Classification
- Support for multiple text categories
- Confidence scores for predictions
- Flexible vocabulary handling
- Training bias detection and reporting

## Setup

### Console Application

1. **Install Dart SDK** (3.0.0 or higher)
2. **Install dependencies**:
   ```bash
   cd tests/multiclass_classifier/dart
   dart pub get
   ```

3. **Place model files** in the directory:
   - `model.onnx`
   - `vocab.json`
   - `scaler.json` (contains class labels)

## Usage

### Console Application

#### Basic Classification
```bash
dart run bin/main.dart
```

#### Custom Text Classification
```bash
dart run bin/main.dart "Scientists discover new species in the Amazon rainforest"
```

#### Benchmark Mode
```bash
dart run bin/main.dart --benchmark 50
```

## Sample Output

```
🖥️  SYSTEM INFORMATION:
   Platform: macos 14.5.0
   Architecture: x86_64
   CPU: Intel(R) Core(TM) i5-8257U CPU @ 1.40GHz
   CPU Cores: 4 physical, 8 logical
   CPU Frequency: 1400 MHz
   Total Memory: 16.00 GB
   Available Memory: 3.33 GB
   Dart Version: 3.2.0
   ONNX Runtime: 1.16.0

🔍 Testing custom text: 'Scientists discover new species in the Amazon rainforest'

📊 PREDICTION RESULTS:
   Text: 'Scientists discover new species in the Amazon rainforest'
   Predicted Class: science

📊 PERFORMANCE METRICS:
   Total Processing Time: 12.45ms
   ├─ Preprocessing: 1.23ms (9.9%)
   ├─ Model Inference: 10.12ms (81.3%)
   └─ Postprocessing: 1.10ms (8.8%)

🚀 THROUGHPUT:
   Predictions per second: 80.32
   Total predictions: 1
   Average time per prediction: 12.45ms

💾 MEMORY USAGE:
   Memory Start: 234.56 MB
   Memory End: 235.01 MB
   Memory Peak: 235.67 MB
   Memory Delta: +0.45 MB

🔥 CPU USAGE:
   Average CPU: 42.3%
   Peak CPU: 58.9%
   Samples: 12

🎯 PERFORMANCE RATING: ✅ VERY GOOD
   (12.5ms total - Target: <100ms)
```

## Architecture

### Console Application (`bin/main.dart`)
- **SystemInfo**: Platform and hardware detection
- **PerformanceMetrics**: Comprehensive performance tracking
- **ResourceMonitor**: Background CPU and memory monitoring
- **MulticlassClassifier**: ONNX model wrapper with timing and class handling

### Key Components

#### MulticlassClassifier
- Handles different vocabulary formats (binary vs multiclass)
- Token-based preprocessing (30 tokens max)
- Class label extraction from scaler.json
- Confidence scoring and prediction

#### Performance Monitoring
- Real-time CPU usage simulation
- Memory usage tracking
- Detailed timing breakdown
- Throughput calculations

## File Formats

### vocab.json
The classifier supports two formats:

**Binary Classifier Format:**
```json
{
  "vocab": {"word1": 0, "word2": 1, ...},
  "idf": [0.1, 0.2, ...]
}
```

**Multiclass Classifier Format:**
```json
{
  "word1": 0,
  "word2": 1,
  "<OOV>": 2,
  ...
}
```

### scaler.json
Contains class labels:
```json
{
  "0": "business",
  "1": "education", 
  "2": "entertainment",
  "3": "science",
  "4": "sports"
}
```

## Supported Categories

The model typically supports categories like:
- **Business**: Financial news, market updates
- **Education**: Academic content, learning materials
- **Entertainment**: Movies, music, celebrity news
- **Science**: Research, discoveries, technology
- **Sports**: Games, athletes, competitions

## Performance Characteristics

### Preprocessing
- Tokenization: ~1-2ms
- Vocabulary lookup with OOV handling
- Fixed sequence length (30 tokens)
- Padding and truncation

### Inference
- Model execution: ~8-12ms
- Input shape: [1, 30] (int32)
- Output: Class probabilities

### Postprocessing
- Argmax for class selection: ~1ms
- Class label mapping
- Confidence calculation

## Performance Ratings

- 🚀 **EXCELLENT**: < 10ms
- ✅ **VERY GOOD**: 10-50ms
- 👍 **GOOD**: 50-100ms
- ⚠️ **ACCEPTABLE**: 100-200ms
- ❌ **POOR**: > 200ms

## CI/CD Integration

The console application automatically detects missing model files:

```bash
⚠️ Model files not found in current directory
Expected files: model.onnx, vocab.json, scaler.json
✅ Dart implementation compiled successfully
🏗️ Build verification completed - would run with actual model files
```

## Known Issues

### Training Bias
The model may exhibit training bias, often predicting certain classes more frequently:

```
ℹ️ Note: This model may have training bias issues - most predictions tend toward certain classes
```

### Recommendations
1. **Balanced Dataset**: Retrain with balanced class distribution
2. **Evaluation**: Test with diverse text samples
3. **Monitoring**: Track prediction distribution over time

## Dependencies

- `onnxruntime: ^1.16.0` - ONNX model inference
- `system_info2: ^4.0.0` - System information
- `platform: ^3.1.3` - Platform detection
- `json_annotation: ^4.8.1` - JSON handling

## Troubleshooting

### Common Issues

1. **Vocabulary mismatch**: Ensure vocab.json matches model training
2. **Class labels**: Verify scaler.json contains correct class mappings
3. **Sequence length**: Model expects exactly 30 tokens
4. **OOV handling**: Unknown words use `<OOV>` token or are skipped

### Performance Tips

1. **Warmup**: First prediction includes model loading time
2. **Batch processing**: Use benchmark mode for multiple predictions
3. **Memory management**: Monitor for memory leaks in long-running applications
4. **Platform optimization**: Performance varies by platform

## License

This implementation is part of the ONNX model testing suite and follows the same license as the parent project. 