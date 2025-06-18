# 🚀 C++ Binary Classification ONNX Model

This directory contains a **C++ implementation** for binary text classification using ONNX Runtime. The model performs **sentiment analysis** on text input using TF-IDF vectorization and a trained neural network.

## 📁 Directory Structure

```
cpp/
├── test_onnx_model.cpp    # Main C++ implementation
├── model.onnx             # Binary classification ONNX model
├── vocab.json             # TF-IDF vocabulary and IDF weights
├── scaler.json            # Feature scaling parameters
├── Makefile               # Build configuration
├── CMakeLists.txt         # CMake build file
└── README.md              # This file
```

## 🛠️ Prerequisites

### Required Libraries
- **ONNX Runtime C++**: For model inference
- **nlohmann/json**: For JSON parsing
- **Standard C++17**: Compiler support

### Linux/Ubuntu Installation
```bash
# Install ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
sudo mv onnxruntime-linux-x64-1.16.0 /opt/onnxruntime

# Install nlohmann/json
sudo apt-get install nlohmann-json3-dev

# Install build tools
sudo apt-get install build-essential cmake
```

### macOS Installation
```bash
# Install ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-osx-x86_64-1.16.0.tgz
tar -xzf onnxruntime-osx-x86_64-1.16.0.tgz
sudo mv onnxruntime-osx-x86_64-1.16.0 /opt/onnxruntime

# Install nlohmann/json via Homebrew
brew install nlohmann-json

# Install build tools
xcode-select --install
```

## 🏗️ Building

### Option 1: Using Makefile
```bash
# Navigate to the C++ directory
cd tests/binary_classifier/cpp

# Compile the application
make

# Run with default text
./test_onnx_model

# Run with custom text
./test_onnx_model "Your custom text here"
```

### Option 2: Using CMake
```bash
# Navigate to the C++ directory
cd tests/binary_classifier/cpp

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Run the executable
./test_onnx_model "Your text here"
```

### Option 3: Manual Compilation
```bash
g++ -std=c++17 -O3 \
    -I/opt/onnxruntime/include \
    -L/opt/onnxruntime/lib \
    test_onnx_model.cpp \
    -lonnxruntime \
    -o test_onnx_model
```

## 🚀 Usage

### Basic Usage
```bash
# Default test text
./test_onnx_model

# Custom text analysis
./test_onnx_model "Congratulations! You've won a free iPhone — click here to claim your prize now!"
```

### Expected Output
```
🤖 ONNX BINARY CLASSIFIER - C++ IMPLEMENTATION
==============================================
🔄 Processing: Your text here

💻 SYSTEM INFORMATION:
   Platform: Linux
   Processor: AMD EPYC 7763 64-Core Processor
   CPU Cores: 4 physical, 4 logical
   Total Memory: 15.6 GB
   Runtime: C++ Implementation

📊 SENTIMENT ANALYSIS RESULTS:
   🏆 Predicted Sentiment: Positive
   📈 Confidence: 99.98% (0.9998)
   📝 Input Text: "Your text here"

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 43.54ms
   ┣━ Preprocessing: 9.19ms (21.1%)
   ┣━ Model Inference: 34.28ms (78.7%)
   ┗━ Postprocessing: 0.00ms (0.0%)

🚀 THROUGHPUT:
   Texts per second: 23.0

💾 RESOURCE USAGE:
   Memory Start: 6.10 MB
   Memory End: 43.82 MB
   Memory Delta: +37.72 MB
   CPU Usage: 0.0% avg, 0.0% peak (1 samples)
```

## 🎯 Performance Characteristics

- **Preprocessing**: TF-IDF vectorization (5000 features)
- **Model Input**: Float32 array [1, 5000]
- **Inference Engine**: ONNX Runtime CPU Provider
- **Memory Usage**: ~38MB peak
- **Speed**: ~44ms total processing time

## 🔧 Technical Details

### Model Architecture
- **Type**: Binary Classification
- **Input**: Text string
- **Features**: TF-IDF vectors (5000 dimensions)
- **Output**: Probability [0.0 - 1.0]
- **Threshold**: >0.5 = Positive, ≤0.5 = Negative

### Processing Pipeline
1. **Text Preprocessing**: Tokenization and lowercasing
2. **TF-IDF Vectorization**: Using vocabulary and IDF weights
3. **Feature Scaling**: Standardization using mean/scale parameters
4. **Model Inference**: ONNX Runtime execution
5. **Post-processing**: Probability interpretation

## 🐛 Troubleshooting

### Common Issues

**1. "onnxruntime not found"**
```bash
# Set library path
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH

# Or copy libraries
sudo cp /opt/onnxruntime/lib/* /usr/local/lib/
sudo ldconfig
```

**2. "nlohmann/json.hpp not found"**
```bash
# Install development headers
sudo apt-get install nlohmann-json3-dev
```

**3. "Permission denied"**
```bash
# Make executable
chmod +x test_onnx_model
```

## 📊 Model Files

- **`model.onnx`**: Binary classification neural network (~10MB)
- **`vocab.json`**: TF-IDF vocabulary mapping and IDF weights
- **`scaler.json`**: Feature standardization parameters (mean/scale)

## 🚀 Integration

To integrate this into your C++ application:

```cpp
#include "your_onnx_classifier.h"

// Initialize classifier
BinaryClassifier classifier("model.onnx", "vocab.json", "scaler.json");

// Classify text
auto result = classifier.predict("Your text here");
cout << "Sentiment: " << (result > 0.5 ? "Positive" : "Negative") << endl;
cout << "Confidence: " << (result * 100) << "%" << endl;
```

## 📈 Performance Tips

1. **Reuse Session**: Load ONNX model once, reuse for multiple predictions
2. **Batch Processing**: Process multiple texts together when possible
3. **Memory Management**: Monitor memory usage for long-running applications
4. **CPU Optimization**: Use optimized BLAS libraries if available

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 