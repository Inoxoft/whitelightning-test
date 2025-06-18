# 🚀 C++ Multiclass Classification ONNX Model

This directory contains a **C++ implementation** for multiclass text classification using ONNX Runtime. The model performs **topic classification** on news articles and other text content, categorizing them into multiple predefined categories.

## 📁 Directory Structure

```
cpp/
├── test_onnx_model.cpp    # Main C++ implementation
├── model.onnx             # Multiclass classification ONNX model
├── vocab.json             # Vocabulary mapping for tokenization
├── scaler.json            # Label mapping for categories
├── Makefile               # Build configuration
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
sudo apt-get install build-essential
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

### Using Makefile
```bash
# Navigate to the C++ directory
cd tests/multiclass_classifier/cpp

# Compile the application
make

# Run with default text
./test_onnx_model

# Run with custom text
./test_onnx_model "NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"
```

### Manual Compilation
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

# Custom news text classification
./test_onnx_model "NASA launches new space telescope to study distant galaxies"

# Business news classification
./test_onnx_model "Apple reports record quarterly earnings amid strong iPhone sales"
```

### Expected Output
```
🤖 ONNX MULTICLASS CLASSIFIER - C++ IMPLEMENTATION
==================================================
🔄 Processing: NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship

💻 SYSTEM INFORMATION:
   Platform: Linux
   Processor: AMD EPYC 7763 64-Core Processor
   CPU Cores: 4 physical, 4 logical
   Total Memory: 15.6 GB
   Implementation: C++ with ONNX Runtime

📊 TOPIC CLASSIFICATION RESULTS:
⏱️  Processing Time: 32.8ms
   🏆 Predicted Category: SPORTS 📝
   📈 Confidence: 100.0%
   📝 Input Text: "NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"

📊 DETAILED PROBABILITIES:
   📝 Business: 0.0% 
   📝 Education: 0.0% 
   📝 Entertainment: 0.0% 
   📝 Environment: 0.0% 
   📝 Health: 0.0% 
   📝 Politics: 0.0% 
   📝 Science: 0.0% 
   📝 Sports: 100.0% ████████████████████ ⭐
   📝 Technology: 0.0% 
   📝 World: 0.0% 

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 32.84ms
   ┣━ Preprocessing: 1.97ms (6.0%)
   ┣━ Model Inference: 30.76ms (93.7%)
   ┗━ Post-processing: 0.11ms (0.3%)
   🧠 CPU Usage: 0.0% avg, 0.0% peak (4 readings)
   💾 Memory: 6.0MB → 27.6MB (Δ+21.6MB)
   🚀 Throughput: 30.4 texts/sec
```

## 🎯 Performance Characteristics

- **Total Time**: ~33ms (excellent performance)
- **Preprocessing**: Sequence tokenization (30 tokens)
- **Model Input**: Int32 array [1, 30]
- **Inference Engine**: ONNX Runtime CPU Provider
- **Memory Usage**: ~22MB additional
- **Speed**: 30+ texts per second

## 🔧 Technical Details

### Model Architecture
- **Type**: Multiclass Classification (10 categories)
- **Input**: Text string → Token sequence
- **Features**: Sequential tokens (max 30 tokens)
- **Output**: Probability distribution over 10 classes
- **Categories**: Business, Education, Entertainment, Environment, Health, Politics, Science, Sports, Technology, World

### Processing Pipeline
1. **Text Preprocessing**: Tokenization and sequence padding
2. **Vocabulary Mapping**: Convert words to token IDs
3. **Sequence Padding**: Pad/truncate to 30 tokens
4. **Model Inference**: ONNX Runtime execution
5. **Post-processing**: Softmax probabilities and category mapping

### Input Format
- **Max Sequence Length**: 30 tokens
- **Vocabulary Size**: Variable (depends on training data)
- **Unknown Tokens**: Mapped to `<OOV>` token ID
- **Padding**: Zero-padding for shorter sequences

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

**3. "Segmentation fault"**
```bash
# Check model files exist
ls -la model.onnx vocab.json scaler.json

# Run with debug info
gdb ./test_onnx_model
```

**4. "Invalid token sequence"**
- Ensure vocab.json contains proper word mappings
- Check for `<OOV>` token in vocabulary
- Verify sequence length doesn't exceed 30 tokens

## 📊 Model Files

- **`model.onnx`**: Multiclass classification neural network (~2.8MB)
- **`vocab.json`**: Word-to-token ID mapping for text tokenization
- **`scaler.json`**: Category ID to label mapping (0="Business", 1="Education", etc.)

## 🎯 Categories Supported

1. **Business** - Financial news, corporate announcements, market reports
2. **Education** - Academic research, educational policies, school news
3. **Entertainment** - Movies, music, celebrity news, TV shows
4. **Environment** - Climate change, conservation, environmental policies
5. **Health** - Medical research, health policies, disease outbreaks
6. **Politics** - Government news, elections, political events
7. **Science** - Scientific discoveries, research breakthroughs
8. **Sports** - Athletic events, sports news, player updates
9. **Technology** - Tech innovations, gadget reviews, software updates
10. **World** - International news, global events, foreign affairs

## 🚀 Integration Example

```cpp
#include "multiclass_classifier.h"

// Initialize classifier
MulticlassClassifier classifier("model.onnx", "vocab.json", "scaler.json");

// Classify text
auto results = classifier.predict("Breaking: New AI breakthrough achieved");

// Get top prediction
auto top_category = results[0].first;
auto confidence = results[0].second;

cout << "Category: " << top_category << endl;
cout << "Confidence: " << (confidence * 100) << "%" << endl;

// Print all probabilities
for (auto& result : results) {
    cout << result.first << ": " << (result.second * 100) << "%" << endl;
}
```

## 📈 Performance Comparison

### vs Other Implementations
- **C++**: 32.84ms (this implementation)
- **C**: 32.54ms (slightly faster)
- **Rust**: 1.24ms (fastest)
- **Swift**: 7.47ms (mobile optimized)
- **Node.js**: 24.40ms (web friendly)

### Optimization Features
- **Efficient Tokenization**: Fast word-to-ID mapping
- **Memory Management**: Minimal allocations
- **Vectorized Operations**: Optimized sequence processing
- **Session Reuse**: Single model loading for multiple predictions

## 🔄 Testing Examples

```bash
# Technology news
./test_onnx_model "Apple announces new iPhone with revolutionary AI features"

# Health news  
./test_onnx_model "Researchers discover new treatment for rare genetic disease"

# Politics news
./test_onnx_model "President signs new climate legislation into law"

# Science news
./test_onnx_model "Scientists detect gravitational waves from black hole merger"
```

## 📝 Notes

- **Fast Performance**: Excellent for real-time classification
- **Low Memory**: Efficient resource usage
- **High Accuracy**: Reliable category predictions
- **Production Ready**: Suitable for high-throughput applications

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 