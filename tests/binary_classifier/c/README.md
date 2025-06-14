# Binary Classifier C Implementation

A high-performance C implementation for ONNX binary sentiment classification with comprehensive performance monitoring.

## 🚀 Features

- **Advanced Performance Monitoring**: CPU usage tracking, memory monitoring, detailed timing analysis
- **System Information Collection**: Hardware specs, platform details, resource utilization
- **TF-IDF Text Processing**: Vocabulary mapping, IDF weighting, standardization
- **Comprehensive Benchmarking**: Statistical analysis with percentiles and throughput metrics
- **Real-time CPU Monitoring**: Multi-threaded continuous CPU usage tracking
- **Memory Tracking**: Before/after memory usage with delta calculations
- **Performance Classification**: Automatic rating (Excellent/Good/Acceptable/Poor)

## 📋 Requirements

### Dependencies
- **ONNX Runtime C API** (v1.22.0+)
- **cJSON library** for JSON parsing
- **pthread** for multi-threading
- **Standard C libraries** (math, time, sys)

### Model Files
- `model.onnx` - Binary classification ONNX model
- `vocab.json` - TF-IDF vocabulary with IDF weights
- `scaler.json` - Standardization parameters (mean/scale)

## 🛠️ Installation

### macOS
```bash
# Install cJSON
brew install cjson

# Download ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-universal2-1.22.0.tgz
tar -xzf onnxruntime-osx-universal2-1.22.0.tgz
```

### Linux
```bash
# Install cJSON
sudo apt-get install libcjson-dev  # Ubuntu/Debian
# or
sudo yum install cjson-devel       # CentOS/RHEL

# Download ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar -xzf onnxruntime-linux-x64-1.22.0.tgz
```

## 🔨 Compilation

```bash
# Compile the program
make

# Or manually
gcc -Wall -Wextra -O2 -std=c99 \
    -I./onnxruntime-osx-universal2-1.22.0/include \
    -o test_onnx_model test_onnx_model.c \
    -L./onnxruntime-osx-universal2-1.22.0/lib \
    -lonnxruntime -lcjson -lpthread -lm
```

## 🎯 Usage

### Basic Testing
```bash
# Run default tests
./test_onnx_model

# Test custom text
./test_onnx_model "This product is amazing!"
```

### Performance Benchmarking
```bash
# Run benchmark with 100 iterations
./test_onnx_model --benchmark 100

# Quick benchmark
make benchmark
```

### Makefile Commands
```bash
make            # Compile
make test       # Run default tests
make benchmark  # Run performance benchmark
make clean      # Clean build files
make install-deps  # Show dependency installation guide
```

## 📊 Output Features

### System Information
- Platform and CPU details
- Memory and core count
- Implementation type

### Performance Metrics
- **Timing Breakdown**: Preprocessing, inference, post-processing
- **CPU Monitoring**: Average and peak usage with continuous tracking
- **Memory Tracking**: Usage deltas and current consumption
- **Throughput Analysis**: Texts per second processing rates
- **Statistical Analysis**: Mean, min, max, percentiles, standard deviation

### Sentiment Analysis
- **Confidence Scoring**: Probability-based sentiment classification
- **Visual Indicators**: Confidence bars and emoji representations
- **Detailed Results**: Raw scores and classification thresholds

## 🔧 Model Requirements

### Input Format
- **Shape**: [1, 5000] float32 tensor
- **Data**: TF-IDF standardized feature vector
- **Input Name**: "float_input"

### Preprocessing Pipeline
1. **Text Normalization**: Lowercase conversion
2. **Tokenization**: Word splitting and vocabulary mapping
3. **TF Calculation**: Term frequency with normalization
4. **IDF Application**: Inverse document frequency weighting
5. **Standardization**: Mean centering and scaling

### Output Format
- **Shape**: [1, 1] float32 tensor
- **Range**: 0.0 to 1.0 (sigmoid probability)
- **Interpretation**: >0.5 = Positive, ≤0.5 = Negative

## 📈 Performance Benchmarking

### Benchmark Features
- **Warmup Runs**: Model optimization before measurement
- **Statistical Analysis**: Comprehensive timing statistics
- **Progress Tracking**: Real-time benchmark progress
- **Performance Classification**: Automatic quality rating
- **Throughput Calculation**: Overall and per-text processing rates

### Performance Targets
- **🚀 Excellent**: <10ms per text
- **✅ Good**: 10-50ms per text
- **⚠️ Acceptable**: 50-100ms per text
- **❌ Poor**: >100ms per text

## 🐛 Troubleshooting

### Common Issues

1. **Library Not Found**
   ```bash
   # Ensure ONNX Runtime path is correct
   export LD_LIBRARY_PATH=./onnxruntime-osx-universal2-1.22.0/lib:$LD_LIBRARY_PATH
   ```

2. **cJSON Missing**
   ```bash
   # Install cJSON library
   brew install cjson  # macOS
   sudo apt-get install libcjson-dev  # Linux
   ```

3. **Model Files Missing**
   ```bash
   # Ensure these files exist in the same directory:
   ls -la model.onnx vocab.json scaler.json
   ```

4. **Permission Denied**
   ```bash
   chmod +x test_onnx_model
   ```

### Debug Mode
```bash
# Compile with debug symbols
gcc -g -DDEBUG -Wall -Wextra -std=c99 \
    -I./onnxruntime-osx-universal2-1.22.0/include \
    -o test_onnx_model_debug test_onnx_model.c \
    -L./onnxruntime-osx-universal2-1.22.0/lib \
    -lonnxruntime -lcjson -lpthread -lm
```

## 🔍 Architecture

### Core Components
- **Performance Monitoring**: Multi-threaded CPU tracking, memory monitoring
- **Text Processing**: TF-IDF pipeline with vocabulary mapping
- **ONNX Integration**: Model loading, inference execution, result processing
- **Statistical Analysis**: Comprehensive benchmarking with detailed metrics
- **System Integration**: Platform-specific optimizations and resource detection

### Threading Model
- **Main Thread**: Model inference and data processing
- **Monitor Thread**: Continuous CPU usage collection
- **Synchronization**: Mutex-protected shared data structures

## 📝 Example Output

```
🤖 ONNX BINARY CLASSIFIER - C IMPLEMENTATION
==============================================

💻 SYSTEM INFORMATION:
   Platform: macOS/Linux
   CPU: Apple M1 Pro
   CPU Cores: 8 physical, 8 logical
   Total Memory: 16.0 GB
   Implementation: C with ONNX Runtime

📊 SENTIMENT ANALYSIS RESULTS:
   🏆 Predicted Sentiment: Positive
   📈 Confidence: 87.34% (0.8734)

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 12.45ms
   ┣━ Preprocessing: 8.23ms (66.1%)
   ┣━ Model Inference: 3.12ms (25.1%)
   ┗━ Post-processing: 1.10ms (8.8%)
   🧠 CPU Usage: 45.2% avg, 78.9% peak (124 readings)
   💾 Memory: 12.3MB → 12.8MB (Δ+0.5MB)
   🚀 Throughput: 80.3 texts/sec
   Performance Rating: 🚀 EXCELLENT
```

## 🤝 Integration

This C implementation provides the same comprehensive performance monitoring as the Python version, including:
- Real-time CPU usage tracking
- Memory consumption analysis
- Detailed timing breakdowns
- Statistical performance analysis
- System information collection
- Automated performance classification

Perfect for production environments requiring high-performance text classification with detailed monitoring capabilities. 