# Multiclass Classifier C Implementation

A high-performance C implementation for ONNX multiclass text classification with comprehensive performance monitoring and Cyrillic text support.

## 🚀 Features

- **Advanced Performance Monitoring**: CPU usage tracking, memory monitoring, detailed timing analysis
- **System Information Collection**: Hardware specs, platform details, resource utilization
- **Cyrillic Text Support**: Proper Unicode handling and case conversion
- **Token-based Processing**: Vocabulary mapping with padding to 30 tokens
- **Comprehensive Benchmarking**: Statistical analysis with percentiles and throughput metrics
- **Real-time CPU Monitoring**: Multi-threaded continuous CPU usage tracking
- **Memory Tracking**: Before/after memory usage with delta calculations
- **Performance Classification**: Automatic rating (Excellent/Good/Acceptable/Poor)
- **Multi-class Output**: Support for 4+ classification categories

## 📋 Requirements

### Dependencies
- **ONNX Runtime C API** (v1.22.0+)
- **cJSON library** for JSON parsing
- **pthread** for multi-threading
- **Standard C libraries** (math, time, sys)

### Model Files
- `model.onnx` - Multiclass classification ONNX model
- `vocab.json` - Token vocabulary mapping (word → token_id)
- `scaler.json` - Label mapping (index → class_name)

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
./test_onnx_model "шляк би тебе трафив"

# Test English text
./test_onnx_model "This is a health related topic about medicine"
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

### Classification Results
- **Multi-class Probabilities**: Confidence scores for all classes
- **Class Predictions**: Highest probability class selection
- **Detailed Analysis**: Per-class confidence breakdown
- **Visual Indicators**: Star marking for predicted class

## 🔧 Model Requirements

### Input Format
- **Shape**: [1, 30] int32 tensor
- **Data**: Tokenized and padded sequence
- **Input Name**: "input"
- **Padding**: Zero-padded to exactly 30 tokens

### Preprocessing Pipeline
1. **Text Normalization**: Cyrillic case conversion
2. **Tokenization**: Word splitting and vocabulary lookup
3. **Token Mapping**: Word → token_id conversion
4. **Sequence Padding**: Zero-padding to 30 tokens
5. **OOV Handling**: Unknown words mapped to `<OOV>` token

### Output Format
- **Shape**: [1, N] float32 tensor (N = number of classes)
- **Data**: Probability distribution over classes
- **Interpretation**: Argmax for predicted class

### Supported Classes
- **health**: Medical and health-related topics
- **politics**: Political news and discussions
- **sports**: Sports events and activities
- **world**: International news and events

## 🌐 Cyrillic Text Support

### Unicode Handling
- **UTF-8 Processing**: Proper multi-byte character handling
- **Case Conversion**: Cyrillic uppercase → lowercase
- **Character Ranges**: Support for А-Я, а-я, Ё, ё
- **Mixed Text**: Cyrillic + Latin character support

### Supported Cyrillic Ranges
- **А-Я**: 0xD090-0xD0AF → 0xD0B0-0xD0CF
- **Р-Я**: 0xD080-0xD08F → 0xD190-0xD19F
- **Special Characters**: Ё, ё and other variants

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

4. **Unicode Issues**
   ```bash
   # Set proper locale for Cyrillic support
   export LC_ALL=en_US.UTF-8
   export LANG=en_US.UTF-8
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
- **Text Processing**: Unicode-aware tokenization with vocabulary mapping
- **ONNX Integration**: Model loading, inference execution, result processing
- **Statistical Analysis**: Comprehensive benchmarking with detailed metrics
- **System Integration**: Platform-specific optimizations and resource detection

### Threading Model
- **Main Thread**: Model inference and data processing
- **Monitor Thread**: Continuous CPU usage collection
- **Synchronization**: Mutex-protected shared data structures

### Unicode Processing
- **Multi-byte Handling**: Proper UTF-8 character processing
- **Case Conversion**: Language-specific uppercase/lowercase mapping
- **Character Classification**: ASCII vs. Cyrillic character detection

## 📝 Example Output

```
🤖 ONNX MULTICLASS CLASSIFIER - C IMPLEMENTATION
==================================================

💻 SYSTEM INFORMATION:
   Platform: macOS/Linux
   CPU: Apple M1 Pro
   CPU Cores: 8 physical, 8 logical
   Total Memory: 16.0 GB
   Implementation: C with ONNX Runtime

🎯 CLASS PROBABILITIES:
   health: 0.0234
   politics: 0.1456
   sports: 0.0892
   world: 0.7418 ⭐

📊 CLASSIFICATION RESULT:
   🏆 Predicted Class: world
   📈 Confidence: 74.18% (0.7418)

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 8.73ms
   ┣━ Preprocessing: 2.45ms (28.1%)
   ┣━ Model Inference: 4.12ms (47.2%)
   ┗━ Post-processing: 2.16ms (24.7%)
   🧠 CPU Usage: 52.3% avg, 89.1% peak (87 readings)
   💾 Memory: 11.2MB → 11.7MB (Δ+0.5MB)
   🚀 Throughput: 114.5 texts/sec
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

### Key Advantages
- **High Performance**: Native C implementation for maximum speed
- **Unicode Support**: Proper Cyrillic text handling
- **Comprehensive Monitoring**: Detailed performance analytics
- **Production Ready**: Robust error handling and resource management
- **Cross-platform**: macOS and Linux support

Perfect for production environments requiring high-performance multilingual text classification with detailed monitoring capabilities. 