# 🚀 Binary Classifier C Implementation

A high-performance C implementation for ONNX binary sentiment classification with comprehensive performance monitoring and cross-platform support.

## 📋 System Requirements

### Minimum Requirements
- **CPU**: x86_64 or ARM64 architecture
- **RAM**: 2GB available memory
- **Storage**: 50MB free space
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+

### Supported Platforms
- ✅ **Windows**: 10, 11 (x64, ARM64)
- ✅ **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 9+, Fedora 30+
- ✅ **macOS**: 10.15+ (Intel & Apple Silicon)

## 🛠️ Step-by-Step Installation

### 🪟 Windows Installation

#### Step 1: Install Build Tools
```powershell
# Option A: Install Visual Studio Community (Recommended)
# Download from: https://visualstudio.microsoft.com/vs/community/
# During installation, select "Desktop development with C++"

# Option B: Install Build Tools for Visual Studio
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
```

#### Step 2: Install Dependencies
```powershell
# Install vcpkg (Package Manager)
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Install cJSON
.\vcpkg install cjson:x64-windows

# Alternative: Download pre-built cJSON
# From: https://github.com/DaveGamble/cJSON/releases
```

#### Step 3: Download ONNX Runtime
```powershell
# Create project directory
mkdir whitelightning-c
cd whitelightning-c

# Download ONNX Runtime for Windows
curl -L -o onnxruntime-win-x64-1.22.0.zip https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip

# Extract
Expand-Archive -Path onnxruntime-win-x64-1.22.0.zip -DestinationPath .
```

#### Step 4: Compile the Program
```powershell
# Using Visual Studio Developer Command Prompt
# Start -> Visual Studio 2022 -> Developer Command Prompt

# Navigate to your project directory
cd path\to\whitelightning-c

# Compile
cl /I"onnxruntime-win-x64-1.22.0\include" ^
   /I"vcpkg\installed\x64-windows\include" ^
   test_onnx_model.c ^
   /link "onnxruntime-win-x64-1.22.0\lib\onnxruntime.lib" ^
   "vcpkg\installed\x64-windows\lib\cjson.lib" ^
   /OUT:test_onnx_model.exe
```

#### Step 5: Run the Program
```powershell
# Copy ONNX Runtime DLL to your directory
copy "onnxruntime-win-x64-1.22.0\lib\onnxruntime.dll" .

# Run with default text
.\test_onnx_model.exe

# Run with custom text
.\test_onnx_model.exe "This product is amazing!"
```

---

### 🐧 Linux Installation

#### Step 1: Update System & Install Build Tools
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential wget curl unzip

# CentOS/RHEL/Fedora
sudo yum update -y  # or dnf update -y for Fedora
sudo yum install -y gcc make wget curl unzip  # or dnf install
```

#### Step 2: Install cJSON Library
```bash
# Ubuntu/Debian
sudo apt install -y libcjson-dev

# CentOS/RHEL
sudo yum install -y epel-release
sudo yum install -y cjson-devel

# Fedora
sudo dnf install -y cjson-devel

# Build from source (if package not available)
wget https://github.com/DaveGamble/cJSON/archive/v1.7.15.tar.gz
tar -xzf v1.7.15.tar.gz
cd cJSON-1.7.15
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

#### Step 3: Download ONNX Runtime
```bash
# Create project directory
mkdir -p ~/whitelightning-c
cd ~/whitelightning-c

# Download ONNX Runtime for Linux
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz

# Extract
tar -xzf onnxruntime-linux-x64-1.22.0.tgz
```

#### Step 4: Copy Source Files
```bash
# Copy your source files to the project directory
# test_onnx_model.c, model.onnx, vocab.json, scaler.json, Makefile
```

#### Step 5: Compile the Program
```bash
# Option A: Using Makefile (if available)
make

# Option B: Manual compilation
gcc -Wall -Wextra -O2 -std=c99 \
    -I./onnxruntime-linux-x64-1.22.0/include \
    -o test_onnx_model test_onnx_model.c \
    -L./onnxruntime-linux-x64-1.22.0/lib \
    -lonnxruntime -lcjson -lpthread -lm

# Set library path
export LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.22.0/lib:$LD_LIBRARY_PATH
```

#### Step 6: Run the Program
```bash
# Make executable
chmod +x test_onnx_model

# Run with default text
./test_onnx_model

# Run with custom text
./test_onnx_model "This product is amazing!"

# Run benchmark
./test_onnx_model --benchmark 100
```

---

### 🍎 macOS Installation

#### Step 1: Install Xcode Command Line Tools
```bash
# Install Xcode Command Line Tools
xcode-select --install

# If already installed, ensure it's up to date
sudo xcode-select --reset
```

#### Step 2: Install Homebrew (if not installed)
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Update Homebrew
brew update
```

#### Step 3: Install Dependencies
```bash
# Install cJSON
brew install cjson

# Install wget (optional, for downloading)
brew install wget
```

#### Step 4: Download ONNX Runtime
```bash
# Create project directory
mkdir -p ~/whitelightning-c
cd ~/whitelightning-c

# For Intel Macs
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-x86_64-1.22.0.tgz
tar -xzf onnxruntime-osx-x86_64-1.22.0.tgz

# For Apple Silicon Macs (M1/M2/M3)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-arm64-1.22.0.tgz
tar -xzf onnxruntime-osx-arm64-1.22.0.tgz

# Universal build (works on both Intel and Apple Silicon)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-universal2-1.22.0.tgz
tar -xzf onnxruntime-osx-universal2-1.22.0.tgz
```

#### Step 5: Copy Source Files
```bash
# Copy your source files to the project directory
# test_onnx_model.c, model.onnx, vocab.json, scaler.json, Makefile
```

#### Step 6: Compile the Program
```bash
# Option A: Using Makefile (if available)
make

# Option B: Manual compilation (Intel Mac)
gcc -Wall -Wextra -O2 -std=c99 \
    -I./onnxruntime-osx-x86_64-1.22.0/include \
    -I$(brew --prefix cjson)/include \
    -o test_onnx_model test_onnx_model.c \
    -L./onnxruntime-osx-x86_64-1.22.0/lib \
    -L$(brew --prefix cjson)/lib \
    -lonnxruntime -lcjson -lpthread -lm

# Option C: Manual compilation (Apple Silicon Mac)
gcc -Wall -Wextra -O2 -std=c99 \
    -I./onnxruntime-osx-arm64-1.22.0/include \
    -I$(brew --prefix cjson)/include \
    -o test_onnx_model test_onnx_model.c \
    -L./onnxruntime-osx-arm64-1.22.0/lib \
    -L$(brew --prefix cjson)/lib \
    -lonnxruntime -lcjson -lpthread -lm

# Option D: Universal build
gcc -Wall -Wextra -O2 -std=c99 \
    -I./onnxruntime-osx-universal2-1.22.0/include \
    -I$(brew --prefix cjson)/include \
    -o test_onnx_model test_onnx_model.c \
    -L./onnxruntime-osx-universal2-1.22.0/lib \
    -L$(brew --prefix cjson)/lib \
    -lonnxruntime -lcjson -lpthread -lm
```

#### Step 7: Run the Program
```bash
# Run with default text
./test_onnx_model

# Run with custom text
./test_onnx_model "This product is amazing!"

# Run benchmark
./test_onnx_model --benchmark 100
```

## 🔧 Advanced Configuration

### Environment Variables
```bash
# Linux/macOS
export ONNX_RUNTIME_PATH=/path/to/onnxruntime
export LD_LIBRARY_PATH=$ONNX_RUNTIME_PATH/lib:$LD_LIBRARY_PATH

# Windows (PowerShell)
$env:ONNX_RUNTIME_PATH = "C:\path\to\onnxruntime"
$env:PATH = "$env:ONNX_RUNTIME_PATH\lib;$env:PATH"
```

### Compiler Optimizations
```bash
# Debug build
gcc -g -DDEBUG -Wall -Wextra -std=c99 ...

# Release build with optimizations
gcc -O3 -DNDEBUG -march=native -Wall -Wextra -std=c99 ...

# Profile-guided optimization
gcc -fprofile-generate ...
# Run program with representative data
gcc -fprofile-use -O3 ...
```

## 🎯 Usage Examples

### Basic Usage
```bash
# Default test
./test_onnx_model

# Custom text
./test_onnx_model "I love this product!"

# Negative sentiment
./test_onnx_model "This is terrible and disappointing."
```

### Benchmarking
```bash
# Quick benchmark (10 iterations)
./test_onnx_model --benchmark 10

# Comprehensive benchmark (1000 iterations)
./test_onnx_model --benchmark 1000

# Save results to file
./test_onnx_model --benchmark 100 > benchmark_results.txt
```

### Performance Testing
```bash
# Test with different text lengths
./test_onnx_model "Short text"
./test_onnx_model "This is a much longer text that contains multiple sentences and should test the preprocessing performance with more complex vocabulary matching and TF-IDF calculations."
```

## 🐛 Troubleshooting

### Windows Issues

**1. "MSVCR120.dll is missing"**
```powershell
# Install Visual C++ Redistributable
# Download from: https://www.microsoft.com/en-us/download/details.aspx?id=40784
```

**2. "Cannot find onnxruntime.dll"**
```powershell
# Copy DLL to executable directory
copy "onnxruntime-win-x64-1.22.0\lib\onnxruntime.dll" .

# Or add to PATH
$env:PATH = "C:\path\to\onnxruntime\lib;$env:PATH"
```

**3. "cjson.h not found"**
```powershell
# Reinstall cJSON via vcpkg
.\vcpkg remove cjson:x64-windows
.\vcpkg install cjson:x64-windows
```

### Linux Issues

**1. "libonnxruntime.so not found"**
```bash
# Set library path
export LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.22.0/lib:$LD_LIBRARY_PATH

# Or copy to system library directory
sudo cp ./onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so* /usr/local/lib/
sudo ldconfig
```

**2. "cjson/cJSON.h not found"**
```bash
# Install development package
sudo apt install libcjson-dev  # Ubuntu/Debian
sudo yum install cjson-devel   # CentOS/RHEL

# Or build from source
wget https://github.com/DaveGamble/cJSON/archive/v1.7.15.tar.gz
tar -xzf v1.7.15.tar.gz && cd cJSON-1.7.15
mkdir build && cd build && cmake .. && make && sudo make install
```

**3. "Permission denied"**
```bash
chmod +x test_onnx_model
```

### macOS Issues

**1. "dyld: Library not loaded"**
```bash
# Check library paths
otool -L test_onnx_model

# Fix library paths
install_name_tool -change @rpath/libonnxruntime.dylib \
    ./onnxruntime-osx-universal2-1.22.0/lib/libonnxruntime.dylib test_onnx_model
```

**2. "Apple Silicon compatibility"**
```bash
# Check architecture
file test_onnx_model

# Use universal build
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-universal2-1.22.0.tgz
```

**3. "Command Line Tools not found"**
```bash
# Reinstall Xcode Command Line Tools
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install
```

## 📊 Expected Output

```
🤖 ONNX BINARY CLASSIFIER - C IMPLEMENTATION
==========================================
🔄 Processing: "This product is amazing!"

💻 SYSTEM INFORMATION:
   Platform: macOS 14.0 (Darwin)
   Processor: Apple M2 Pro
   CPU Cores: 12 physical, 12 logical
   Total Memory: 32.0 GB
   Implementation: C with ONNX Runtime v1.22.0

📊 SENTIMENT ANALYSIS RESULTS:
   🏆 Predicted Sentiment: Positive ✅
   📈 Confidence: 99.8% (0.9982)
   📊 Confidence Bar: ████████████████████ (20/20)
   📝 Input Text: "This product is amazing!"

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 87.21ms
   ┣━ Preprocessing: 50.93ms (58.4%)
   ┣━ Model Inference: 0.31ms (0.4%)
   ┗━ Post-processing: 0.02ms (0.0%)
   
🚀 THROUGHPUT:
   Texts per second: 11.5
   
💾 RESOURCE USAGE:
   Memory Start: 8.2 MB
   Memory End: 45.5 MB
   Memory Delta: +37.3 MB
   CPU Usage: 0.0% avg, 0.0% peak (1 samples)
   
⭐ Performance Rating: ✅ GOOD
```

## 🚀 Features

- **Cross-Platform**: Windows, Linux, macOS support
- **Advanced Performance Monitoring**: CPU usage, memory tracking, timing analysis
- **TF-IDF Text Processing**: Vocabulary mapping, IDF weighting, standardization
- **Comprehensive Benchmarking**: Statistical analysis with percentiles
- **Real-time Monitoring**: Multi-threaded CPU usage tracking
- **System Information**: Hardware specs, platform details
- **Performance Classification**: Automatic rating system

## 📈 Performance Targets

- **🚀 Excellent**: <10ms per text
- **✅ Good**: 10-50ms per text  
- **⚠️ Acceptable**: 50-100ms per text
- **❌ Poor**: >100ms per text

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 