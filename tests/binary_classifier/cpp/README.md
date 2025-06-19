# 🚀 C++ Binary Classification ONNX Model

This directory contains a **C++ implementation** for binary text classification using ONNX Runtime. The model performs **sentiment analysis** on text input using TF-IDF vectorization and a trained neural network with cross-platform support.

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
├── model.onnx             # Binary classification ONNX model
├── vocab.json             # TF-IDF vocabulary and IDF weights
├── scaler.json            # Feature scaling parameters
├── Makefile               # Build configuration
├── CMakeLists.txt         # CMake build file
└── README.md              # This file
```

## 🛠️ Step-by-Step Installation

### 🪟 Windows Installation

#### Step 1: Install Visual Studio
```powershell
# Option A: Install Visual Studio Community 2022 (Recommended)
# Download from: https://visualstudio.microsoft.com/vs/community/
# During installation, select:
# - "Desktop development with C++"
# - "CMake tools for C++"
# - "Git for Windows"

# Option B: Install Build Tools for Visual Studio 2022
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
```

#### Step 2: Install Package Manager (vcpkg)
```powershell
# Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg

# Bootstrap vcpkg
.\bootstrap-vcpkg.bat

# Integrate with Visual Studio
.\vcpkg integrate install

# Set environment variable
$env:VCPKG_ROOT = "C:\vcpkg"
```

#### Step 3: Install Dependencies
```powershell
# Install nlohmann-json
C:\vcpkg\vcpkg install nlohmann-json:x64-windows

# Install CMake (if not installed with Visual Studio)
# Download from: https://cmake.org/download/
# Or use winget:
winget install Kitware.CMake
```

#### Step 4: Download ONNX Runtime
```powershell
# Create project directory
mkdir C:\whitelightning-cpp
cd C:\whitelightning-cpp

# Download ONNX Runtime for Windows
curl -L -o onnxruntime-win-x64-1.22.0.zip https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip

# Extract
Expand-Archive -Path onnxruntime-win-x64-1.22.0.zip -DestinationPath .
```

#### Step 5: Copy Source Files
```powershell
# Copy your source files to the project directory
# test_onnx_model.cpp, model.onnx, vocab.json, scaler.json, CMakeLists.txt
```

#### Step 6: Build with CMake
```powershell
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake ^
         -DONNXRUNTIME_ROOT_PATH=..\onnxruntime-win-x64-1.22.0

# Build
cmake --build . --config Release
```

#### Step 7: Run the Program
```powershell
# Copy ONNX Runtime DLL
copy "..\onnxruntime-win-x64-1.22.0\lib\onnxruntime.dll" Release\

# Run with default text
.\Release\test_onnx_model.exe

# Run with custom text
.\Release\test_onnx_model.exe "This product is amazing!"
```

---

### 🐧 Linux Installation

#### Step 1: Update System & Install Build Tools
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake git wget curl

# CentOS/RHEL 8+
sudo dnf update -y
sudo dnf install -y gcc-c++ cmake git wget curl make

# CentOS/RHEL 7
sudo yum update -y
sudo yum install -y gcc-c++ cmake3 git wget curl make
# Use cmake3 instead of cmake on CentOS 7

# Fedora
sudo dnf update -y
sudo dnf install -y gcc-c++ cmake git wget curl make
```

#### Step 2: Install nlohmann/json
```bash
# Ubuntu 20.04+ / Debian 11+
sudo apt install -y nlohmann-json3-dev

# Ubuntu 18.04 / Debian 10 (older versions)
sudo apt install -y nlohmann-json-dev

# CentOS/RHEL/Fedora
sudo dnf install -y json-devel
# or
sudo yum install -y json-devel

# Build from source (if package not available)
cd /tmp
wget https://github.com/nlohmann/json/archive/v3.11.2.tar.gz
tar -xzf v3.11.2.tar.gz
cd json-3.11.2
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

#### Step 3: Download ONNX Runtime
```bash
# Create project directory
mkdir -p ~/whitelightning-cpp
cd ~/whitelightning-cpp

# Download ONNX Runtime for Linux
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz

# Extract
tar -xzf onnxruntime-linux-x64-1.22.0.tgz
```

#### Step 4: Copy Source Files
```bash
# Copy your source files to the project directory
# test_onnx_model.cpp, model.onnx, vocab.json, scaler.json, CMakeLists.txt, Makefile
```

#### Step 5: Build the Program
```bash
# Option A: Using CMake (Recommended)
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT_PATH=../onnxruntime-linux-x64-1.22.0
make -j$(nproc)

# Option B: Using Makefile
make

# Option C: Manual compilation
g++ -std=c++17 -O3 -Wall -Wextra \
    -I./onnxruntime-linux-x64-1.22.0/include \
    -o test_onnx_model test_onnx_model.cpp \
    -L./onnxruntime-linux-x64-1.22.0/lib \
    -lonnxruntime -lpthread
```

#### Step 6: Set Library Path & Run
```bash
# Set library path
export LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.22.0/lib:$LD_LIBRARY_PATH

# Make executable
chmod +x test_onnx_model

# Run with default text
./test_onnx_model

# Run with custom text
./test_onnx_model "This product is amazing!"
```

---

### 🍎 macOS Installation

#### Step 1: Install Xcode Command Line Tools
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Verify installation
xcode-select -p
gcc --version
```

#### Step 2: Install Homebrew
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Update Homebrew
brew update

# Add Homebrew to PATH (if needed)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc
```

#### Step 3: Install Dependencies
```bash
# Install nlohmann/json
brew install nlohmann-json

# Install CMake
brew install cmake

# Install wget (optional)
brew install wget
```

#### Step 4: Download ONNX Runtime
```bash
# Create project directory
mkdir -p ~/whitelightning-cpp
cd ~/whitelightning-cpp

# For Intel Macs
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-x86_64-1.22.0.tgz
tar -xzf onnxruntime-osx-x86_64-1.22.0.tgz

# For Apple Silicon Macs (M1/M2/M3)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-arm64-1.22.0.tgz
tar -xzf onnxruntime-osx-arm64-1.22.0.tgz

# Universal build (recommended - works on both)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-universal2-1.22.0.tgz
tar -xzf onnxruntime-osx-universal2-1.22.0.tgz
```

#### Step 5: Copy Source Files
```bash
# Copy your source files to the project directory
# test_onnx_model.cpp, model.onnx, vocab.json, scaler.json, CMakeLists.txt, Makefile
```

#### Step 6: Build the Program
```bash
# Option A: Using CMake (Recommended)
mkdir build && cd build

# For Intel Macs
cmake .. -DONNXRUNTIME_ROOT_PATH=../onnxruntime-osx-x86_64-1.22.0

# For Apple Silicon Macs
cmake .. -DONNXRUNTIME_ROOT_PATH=../onnxruntime-osx-arm64-1.22.0

# For Universal build
cmake .. -DONNXRUNTIME_ROOT_PATH=../onnxruntime-osx-universal2-1.22.0

# Build
make -j$(sysctl -n hw.ncpu)

# Option B: Using Makefile
make

# Option C: Manual compilation (Intel Mac)
g++ -std=c++17 -O3 -Wall -Wextra \
    -I./onnxruntime-osx-x86_64-1.22.0/include \
    -I$(brew --prefix nlohmann-json)/include \
    -o test_onnx_model test_onnx_model.cpp \
    -L./onnxruntime-osx-x86_64-1.22.0/lib \
    -lonnxruntime

# Option D: Manual compilation (Apple Silicon)
g++ -std=c++17 -O3 -Wall -Wextra \
    -I./onnxruntime-osx-arm64-1.22.0/include \
    -I$(brew --prefix nlohmann-json)/include \
    -o test_onnx_model test_onnx_model.cpp \
    -L./onnxruntime-osx-arm64-1.22.0/lib \
    -lonnxruntime

# Option E: Manual compilation (Universal)
g++ -std=c++17 -O3 -Wall -Wextra \
    -I./onnxruntime-osx-universal2-1.22.0/include \
    -I$(brew --prefix nlohmann-json)/include \
    -o test_onnx_model test_onnx_model.cpp \
    -L./onnxruntime-osx-universal2-1.22.0/lib \
    -lonnxruntime
```

#### Step 7: Run the Program
```bash
# Run with default text
./test_onnx_model

# Run with custom text
./test_onnx_model "This product is amazing!"
```

## 🔧 Advanced Configuration

### CMakeLists.txt Example
```cmake
cmake_minimum_required(VERSION 3.16)
project(ONNXBinaryClassifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(nlohmann_json REQUIRED)

# Set ONNX Runtime path
set(ONNXRUNTIME_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.22.0" CACHE PATH "ONNX Runtime root directory")

# Include directories
include_directories(${ONNXRUNTIME_ROOT_PATH}/include)

# Add executable
add_executable(test_onnx_model test_onnx_model.cpp)

# Link libraries
target_link_libraries(test_onnx_model 
    ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so
    nlohmann_json::nlohmann_json
    pthread
)

# Set RPATH for Linux/macOS
if(UNIX)
    set_target_properties(test_onnx_model PROPERTIES
        INSTALL_RPATH "${ONNXRUNTIME_ROOT_PATH}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
```

### Makefile Example
```makefile
# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    ONNX_DIR = onnxruntime-linux-x64-1.22.0
    LIBS = -lonnxruntime -lpthread
endif
ifeq ($(UNAME_S),Darwin)
    ONNX_DIR = onnxruntime-osx-universal2-1.22.0
    LIBS = -lonnxruntime
    CXXFLAGS += -I$(shell brew --prefix nlohmann-json)/include
endif

# Include and library paths
INCLUDES = -I./$(ONNX_DIR)/include
LIBPATH = -L./$(ONNX_DIR)/lib

# Target
TARGET = test_onnx_model
SOURCE = test_onnx_model.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCE) $(LIBPATH) $(LIBS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	./$(TARGET) "This is a test message"

.PHONY: all clean test
```

## 🎯 Usage Examples

### Basic Usage
```bash
# Default test
./test_onnx_model

# Positive sentiment
./test_onnx_model "I love this product! It's amazing!"

# Negative sentiment
./test_onnx_model "This is terrible and disappointing."

# Neutral/Mixed sentiment
./test_onnx_model "The product is okay, nothing special."
```

### Batch Testing
```bash
# Test multiple sentences
./test_onnx_model "Great service and fast delivery!"
./test_onnx_model "Poor quality, would not recommend."
./test_onnx_model "Average product, meets expectations."
```

## 🐛 Troubleshooting

### Windows Issues

**1. "VCRUNTIME140.dll is missing"**
```powershell
# Install Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

**2. "Cannot open include file: 'nlohmann/json.hpp'"**
```powershell
# Reinstall nlohmann-json via vcpkg
C:\vcpkg\vcpkg remove nlohmann-json:x64-windows
C:\vcpkg\vcpkg install nlohmann-json:x64-windows

# Or specify include path manually
# -I"C:\vcpkg\installed\x64-windows\include"
```

**3. "onnxruntime.dll not found"**
```powershell
# Copy DLL to output directory
copy "onnxruntime-win-x64-1.22.0\lib\onnxruntime.dll" Release\

# Or add to PATH
$env:PATH = "C:\path\to\onnxruntime\lib;$env:PATH"
```

### Linux Issues

**1. "libonnxruntime.so: cannot open shared object file"**
```bash
# Set library path permanently
echo 'export LD_LIBRARY_PATH=$HOME/whitelightning-cpp/onnxruntime-linux-x64-1.22.0/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Or copy to system library directory
sudo cp ./onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so* /usr/local/lib/
sudo ldconfig
```

**2. "nlohmann/json.hpp: No such file or directory"**
```bash
# Ubuntu/Debian
sudo apt install nlohmann-json3-dev

# CentOS/RHEL/Fedora
sudo dnf install json-devel

# Build from source
wget https://github.com/nlohmann/json/archive/v3.11.2.tar.gz
tar -xzf v3.11.2.tar.gz && cd json-3.11.2
mkdir build && cd build && cmake .. && make && sudo make install
```

**3. "CMake version too old"**
```bash
# Ubuntu 18.04 - install newer CMake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt update && sudo apt install cmake
```

### macOS Issues

**1. "dyld: Library not loaded: @rpath/libonnxruntime.dylib"**
```bash
# Fix library paths
install_name_tool -change @rpath/libonnxruntime.dylib \
    ./onnxruntime-osx-universal2-1.22.0/lib/libonnxruntime.dylib test_onnx_model

# Or set DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=./onnxruntime-osx-universal2-1.22.0/lib:$DYLD_LIBRARY_PATH
```

**2. "nlohmann/json.hpp not found"**
```bash
# Reinstall nlohmann-json
brew uninstall nlohmann-json
brew install nlohmann-json

# Check installation path
brew --prefix nlohmann-json
```

**3. "Apple Silicon compatibility issues"**
```bash
# Check binary architecture
file test_onnx_model

# Use universal ONNX Runtime build
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-universal2-1.22.0.tgz
```

## 📊 Expected Output

```
🤖 ONNX BINARY CLASSIFIER - C++ IMPLEMENTATION
==============================================
🔄 Processing: "This product is amazing!"

💻 SYSTEM INFORMATION:
   Platform: macOS 14.0 (Darwin)
   Processor: Apple M2 Pro
   CPU Cores: 12 physical, 12 logical
   Total Memory: 32.0 GB
   Runtime: C++ Implementation with ONNX Runtime v1.22.0

📊 SENTIMENT ANALYSIS RESULTS:
   🏆 Predicted Sentiment: Positive ✅
   📈 Confidence: 99.8% (0.9982)
   📝 Input Text: "This product is amazing!"

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 43.54ms
   ┣━ Preprocessing: 9.19ms (21.1%)
   ┣━ Model Inference: 34.28ms (78.7%)
   ┗━ Postprocessing: 0.07ms (0.2%)

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
- **Throughput**: ~23 texts per second

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

## 📊 Model Files

- **`model.onnx`**: Binary classification neural network (~10MB)
- **`vocab.json`**: TF-IDF vocabulary mapping and IDF weights
- **`scaler.json`**: Feature standardization parameters (mean/scale)

## 🚀 Integration Example

```cpp
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <vector>
#include <string>

class BinaryClassifier {
private:
    Ort::Env env;
    Ort::Session session;
    nlohmann::json vocab;
    nlohmann::json scaler;

public:
    BinaryClassifier(const std::string& model_path, 
                    const std::string& vocab_path,
                    const std::string& scaler_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "BinaryClassifier"),
          session(env, model_path.c_str(), Ort::SessionOptions{nullptr}) {
        
        // Load vocabulary and scaler
        std::ifstream vocab_file(vocab_path);
        vocab_file >> vocab;
        
        std::ifstream scaler_file(scaler_path);
        scaler_file >> scaler;
    }
    
    float predict(const std::string& text) {
        // Preprocess text to TF-IDF features
        auto features = preprocessText(text);
        
        // Create input tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_shape = {1, 5000};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, features.data(), features.size(), 
            input_shape.data(), input_shape.size());
        
        // Run inference
        std::vector<const char*> input_names = {"float_input"};
        std::vector<const char*> output_names = {"output"};
        
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                        input_names.data(), &input_tensor, 1,
                                        output_names.data(), 1);
        
        // Extract result
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        return output_data[0];
    }
    
private:
    std::vector<float> preprocessText(const std::string& text) {
        // Implement TF-IDF preprocessing
        // This is a simplified version - implement full TF-IDF logic
        std::vector<float> features(5000, 0.0f);
        // ... TF-IDF implementation ...
        return features;
    }
};

// Usage
int main() {
    try {
        BinaryClassifier classifier("model.onnx", "vocab.json", "scaler.json");
        
        float probability = classifier.predict("This product is amazing!");
        std::string sentiment = (probability > 0.5) ? "Positive" : "Negative";
        
        std::cout << "Sentiment: " << sentiment << std::endl;
        std::cout << "Confidence: " << (probability * 100) << "%" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## 📈 Performance Tips

1. **Compiler Optimizations**: Use `-O3 -march=native` for maximum performance
2. **Session Reuse**: Create session once, reuse for multiple predictions
3. **Memory Management**: Use move semantics and avoid unnecessary copies
4. **Batch Processing**: Process multiple texts together when possible
5. **Threading**: Use OpenMP or std::thread for parallel processing

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 