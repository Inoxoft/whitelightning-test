# 🚀 C++ Multiclass Classification ONNX Model

This directory contains a **C++ implementation** for multiclass text classification using ONNX Runtime. The model performs **topic classification** on news articles and other text content, categorizing them into multiple predefined categories with cross-platform support.

## 📋 System Requirements

### Minimum Requirements
- **CPU**: x86_64 or ARM64 architecture
- **RAM**: 4GB available memory
- **Storage**: 100MB free space
- **Compiler**: C++17 compatible (GCC 7+, Clang 5+, MSVC 2017+)
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+

### Supported Platforms
- ✅ **Windows**: 10, 11 (x64, ARM64)
- ✅ **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 9+, Fedora 30+
- ✅ **macOS**: 10.15+ (Intel & Apple Silicon)

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

## 🛠️ Step-by-Step Installation

### 🪟 Windows Installation

#### Step 1: Install Visual Studio
```powershell
# Install Visual Studio Community 2022 (Recommended)
# Download from: https://visualstudio.microsoft.com/vs/community/
# During installation, select:
# - "Desktop development with C++"
# - "CMake tools for C++"
# - "Git for Windows"
```

#### Step 2: Install vcpkg Package Manager
```powershell
# Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg

# Bootstrap vcpkg
.\bootstrap-vcpkg.bat

# Integrate with Visual Studio
.\vcpkg integrate install
```

#### Step 3: Install Dependencies
```powershell
# Install nlohmann-json
C:\vcpkg\vcpkg install nlohmann-json:x64-windows

# Install CMake (if not installed with Visual Studio)
winget install Kitware.CMake
```

#### Step 4: Download ONNX Runtime
```powershell
# Create project directory
mkdir C:\whitelightning-cpp-multiclass
cd C:\whitelightning-cpp-multiclass

# Download ONNX Runtime for Windows
curl -L -o onnxruntime-win-x64-1.22.0.zip https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip

# Extract
Expand-Archive -Path onnxruntime-win-x64-1.22.0.zip -DestinationPath .
```

#### Step 5: Build & Run
```powershell
# Copy source files to project directory
# test_onnx_model.cpp, model.onnx, vocab.json, scaler.json

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake ^
         -DONNXRUNTIME_ROOT_PATH=..\onnxruntime-win-x64-1.22.0

# Build
cmake --build . --config Release

# Copy ONNX Runtime DLL
copy "..\onnxruntime-win-x64-1.22.0\lib\onnxruntime.dll" Release\

# Run
.\Release\test_onnx_model.exe "Technology news about artificial intelligence"
```

---

### 🐧 Linux Installation

#### Step 1: Install Build Tools
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake git wget curl

# CentOS/RHEL/Fedora
sudo dnf install -y gcc-c++ cmake git wget curl make
```

#### Step 2: Install nlohmann/json
```bash
# Ubuntu 20.04+
sudo apt install -y nlohmann-json3-dev

# Build from source (if package not available)
cd /tmp
wget https://github.com/nlohmann/json/archive/v3.11.2.tar.gz
tar -xzf v3.11.2.tar.gz && cd json-3.11.2
mkdir build && cd build && cmake .. && make -j$(nproc) && sudo make install
```

#### Step 3: Download ONNX Runtime
```bash
# Create project directory
mkdir -p ~/whitelightning-cpp-multiclass
cd ~/whitelightning-cpp-multiclass

# Download ONNX Runtime for Linux
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar -xzf onnxruntime-linux-x64-1.22.0.tgz
```

#### Step 4: Build & Run
```bash
# Copy source files to project directory
# Manual compilation
g++ -std=c++17 -O3 -Wall -Wextra \
    -I./onnxruntime-linux-x64-1.22.0/include \
    -o test_onnx_model test_onnx_model.cpp \
    -L./onnxruntime-linux-x64-1.22.0/lib \
    -lonnxruntime -lpthread

# Set library path
export LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.22.0/lib:$LD_LIBRARY_PATH

# Run
./test_onnx_model "Technology news about artificial intelligence"
```

---

### 🍎 macOS Installation

#### Step 1: Install Development Tools
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Step 2: Install Dependencies
```bash
# Install nlohmann/json and CMake
brew install nlohmann-json cmake wget
```

#### Step 3: Download ONNX Runtime
```bash
# Create project directory
mkdir -p ~/whitelightning-cpp-multiclass
cd ~/whitelightning-cpp-multiclass

# Universal build (works on Intel and Apple Silicon)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-universal2-1.22.0.tgz
tar -xzf onnxruntime-osx-universal2-1.22.0.tgz
```

#### Step 4: Build & Run
```bash
# Copy source files to project directory
# Manual compilation
g++ -std=c++17 -O3 -Wall -Wextra \
    -I./onnxruntime-osx-universal2-1.22.0/include \
    -I$(brew --prefix nlohmann-json)/include \
    -o test_onnx_model test_onnx_model.cpp \
    -L./onnxruntime-osx-universal2-1.22.0/lib \
    -lonnxruntime

# Run
./test_onnx_model "Technology news about artificial intelligence"
```

## 📊 Expected Output

```
🤖 ONNX MULTICLASS CLASSIFIER - C++ IMPLEMENTATION
=================================================
🔄 Processing: "Technology news about artificial intelligence"

💻 SYSTEM INFORMATION:
   Platform: macOS 14.0 (Darwin)
   Processor: Apple M2 Pro
   CPU Cores: 12 physical, 12 logical
   Total Memory: 32.0 GB
   Runtime: C++ Implementation with ONNX Runtime v1.22.0

📊 MULTICLASS CLASSIFICATION RESULTS:
   🏆 Predicted Category: Technology ✅
   📈 Confidence: 94.2% (0.9420)
   📝 Input Text: "Technology news about artificial intelligence"

🎯 TOP 3 PREDICTIONS:
   1. Technology: ████████████████████ 94.2%
   2. Science: ██████ 3.8%
   3. Business: ██ 1.2%

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 35.67ms
   ┣━ Preprocessing: 8.45ms (23.7%)
   ┣━ Model Inference: 25.12ms (70.4%)
   ┗━ Postprocessing: 2.10ms (5.9%)

🚀 THROUGHPUT:
   Texts per second: 28.0

💾 RESOURCE USAGE:
   Memory Start: 6.10 MB
   Memory End: 42.35 MB
   Memory Delta: +36.25 MB
```

## 🎯 Performance Characteristics

- **Categories**: 10 topic classifications (Technology, Politics, Sports, etc.)
- **Input Processing**: Tokenization (30 tokens max)
- **Model Input**: Float32 array [1, 30]
- **Inference Engine**: ONNX Runtime CPU Provider
- **Memory Usage**: ~36MB peak
- **Speed**: ~36ms total processing time

## 🔧 Technical Details

### Model Architecture
- **Type**: Multiclass Classification (10 categories)
- **Input**: Text string
- **Features**: Tokenized word indices (30 dimensions)
- **Output**: Probability distribution [0.0 - 1.0] across 10 categories
- **Categories**: Technology, Politics, Sports, Entertainment, Business, Science, Health, World, Education, Environment

### Processing Pipeline
1. **Text Preprocessing**: Tokenization and vocabulary mapping
2. **Sequence Processing**: Convert to fixed-length token sequence
3. **Model Inference**: ONNX Runtime execution
4. **Post-processing**: Probability interpretation and ranking

## 🐛 Troubleshooting

### Common Issues Across Platforms

**1. ONNX Runtime Library Not Found**
```bash
# Linux/macOS: Set library path
export LD_LIBRARY_PATH=./onnxruntime-*/lib:$LD_LIBRARY_PATH

# Windows: Copy DLL to executable directory
copy "onnxruntime-*\lib\onnxruntime.dll" .
```

**2. nlohmann/json Header Not Found**
```bash
# Install development package or build from source
# See platform-specific instructions above
```

**3. Compilation Errors**
```bash
# Ensure C++17 support
g++ --version  # Should be 7.0+
clang++ --version  # Should be 5.0+
```

## 📈 Usage Examples

```bash
# Technology news
./test_onnx_model "Apple releases new iPhone with advanced AI features"

# Sports news
./test_onnx_model "Manchester United wins the championship final"

# Politics news
./test_onnx_model "President announces new economic policy changes"

# Business news
./test_onnx_model "Stock market reaches record high amid tech rally"
```

## 🚀 Integration Example

```cpp
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <vector>
#include <string>

class MulticlassClassifier {
private:
    Ort::Env env;
    Ort::Session session;
    nlohmann::json vocab;
    nlohmann::json scaler;
    std::vector<std::string> categories;

public:
    MulticlassClassifier(const std::string& model_path, 
                        const std::string& vocab_path,
                        const std::string& scaler_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "MulticlassClassifier"),
          session(env, model_path.c_str(), Ort::SessionOptions{nullptr}) {
        
        // Load vocabulary and categories
        std::ifstream vocab_file(vocab_path);
        vocab_file >> vocab;
        
        std::ifstream scaler_file(scaler_path);
        scaler_file >> scaler;
        
        // Initialize categories
        categories = {"Technology", "Politics", "Sports", "Entertainment", 
                     "Business", "Science", "Health", "World", "Education", "Environment"};
    }
    
    std::pair<std::string, float> predict(const std::string& text) {
        // Preprocess text to token sequence
        auto tokens = preprocessText(text);
        
        // Create input tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_shape = {1, 30};
        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, tokens.data(), tokens.size(), 
            input_shape.data(), input_shape.size());
        
        // Run inference
        std::vector<const char*> input_names = {"input_ids"};
        std::vector<const char*> output_names = {"output"};
        
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                        input_names.data(), &input_tensor, 1,
                                        output_names.data(), 1);
        
        // Extract results
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        
        // Find max probability category
        int max_idx = 0;
        float max_prob = output_data[0];
        for (int i = 1; i < 10; i++) {
            if (output_data[i] > max_prob) {
                max_prob = output_data[i];
                max_idx = i;
            }
        }
        
        return {categories[max_idx], max_prob};
    }
    
private:
    std::vector<int64_t> preprocessText(const std::string& text) {
        // Implement tokenization logic
        std::vector<int64_t> tokens(30, 0);  // Pad to 30 tokens
        // ... tokenization implementation ...
        return tokens;
    }
};

// Usage
int main() {
    try {
        MulticlassClassifier classifier("model.onnx", "vocab.json", "scaler.json");
        
        auto result = classifier.predict("Technology news about artificial intelligence");
        
        std::cout << "Category: " << result.first << std::endl;
        std::cout << "Confidence: " << (result.second * 100) << "%" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## 📊 Model Files

- **`model.onnx`**: Multiclass classification neural network (~15MB)
- **`vocab.json`**: Vocabulary mapping for tokenization
- **`scaler.json`**: Category label mappings

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 