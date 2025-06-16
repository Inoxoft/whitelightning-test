# Dart Binary Classifier Implementation

A comprehensive Dart implementation of the ONNX binary classifier with advanced performance monitoring and system information tracking.

## Features

### 🚀 Core Functionality
- **ONNX Model Inference**: Binary text classification using ONNX Runtime
- **Text Preprocessing**: TF-IDF vectorization with vocabulary mapping
- **Feature Scaling**: StandardScaler normalization for optimal performance

### 📊 Performance Monitoring
- **Detailed Timing**: Preprocessing, inference, and postprocessing breakdown
- **Memory Tracking**: Real-time memory usage monitoring with peak detection
- **CPU Monitoring**: Background CPU usage tracking with average and peak metrics
- **Throughput Analysis**: Predictions per second calculation
- **Performance Rating**: Color-coded performance assessment (EXCELLENT/VERY GOOD/GOOD/ACCEPTABLE/POOR)

### 🖥️ System Information
- **Platform Details**: Operating system and architecture detection
- **Hardware Info**: CPU cores and memory information
- **Runtime Versions**: Dart SDK and ONNX Runtime versions
- **Resource Utilization**: Memory and CPU usage statistics

### 🎯 Advanced Features
- **Custom Text Input**: Test with your own text samples
- **Benchmark Mode**: Performance testing with configurable iterations
- **Model Warmup**: Initial runs to optimize performance
- **Progress Tracking**: Real-time progress updates for long benchmarks
- **Resource Cleanup**: Proper memory management and resource disposal

## CI/CD Compatibility

This implementation includes mock ONNX Runtime functionality for CI environments where the actual ONNX Runtime package may not be available. The mock implementation:

- Simulates realistic inference timing and results
- Provides build verification without requiring model files
- Maintains API compatibility with the real ONNX Runtime
- Enables comprehensive testing in GitHub Actions

## Usage

### Basic Usage
```bash
# Install dependencies
dart pub get

# Run with model files (if available)
dart run bin/main.dart "Your text to classify"

# Build verification (without model files)
dart run bin/main.dart
```

### Benchmark Mode
```bash
# Run performance benchmark
dart run bin/main.dart --benchmark 50

# Compile to executable
dart compile exe bin/main.dart -o test_onnx_model
./test_onnx_model --benchmark 100
```

## Dependencies

- `json_annotation`: JSON serialization support
- `platform`: Cross-platform system information
- `io`: File I/O operations
- `path`: File path utilities
- `async`: Asynchronous programming utilities
- `collection`: Enhanced collection operations

## Performance Metrics

The implementation provides comprehensive performance analysis:

- **Total Time**: End-to-end processing time
- **Preprocessing Time**: Text vectorization and scaling
- **Inference Time**: ONNX model execution
- **Postprocessing Time**: Result extraction and formatting
- **Memory Usage**: Start, end, peak, and delta measurements
- **CPU Usage**: Average and peak utilization during processing
- **Throughput**: Predictions per second calculation

## Architecture

```
Input Text → Preprocessing → ONNX Inference → Postprocessing → Classification Result
     ↓              ↓              ↓              ↓
Performance    Memory         CPU          Resource
Monitoring     Tracking    Monitoring      Cleanup
```

## GitHub Actions Integration

This implementation is fully integrated with GitHub Actions for automated testing:

- **Build Verification**: Ensures code compiles successfully
- **Dependency Resolution**: Automatic package installation
- **Mock Testing**: Simulated inference when model files are unavailable
- **Performance Benchmarking**: Automated performance testing
- **Artifact Collection**: Build outputs and logs collection

The workflow supports both binary and multiclass classifier testing with configurable parameters for model type, language selection, and custom text input.

## License

This implementation is part of the ONNX model testing suite and follows the same license as the parent project. 