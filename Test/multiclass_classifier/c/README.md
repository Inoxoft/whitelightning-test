# 🔧 C Multiclass Classification ONNX Model

A high-performance C implementation for ONNX multiclass text classification with comprehensive performance monitoring, Cyrillic text support, and cross-platform compatibility.

## 📋 System Requirements

### Minimum Requirements
- **CPU**: x86_64 or ARM64 architecture
- **RAM**: 2GB available memory
- **Storage**: 500MB free space
- **C Compiler**: GCC 7+, Clang 5+, MSVC 2017+
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+

### Supported Platforms
- ✅ **Windows**: 10, 11 (x64, ARM64)
- ✅ **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 9+, Fedora 30+
- ✅ **macOS**: 10.15+ (Intel & Apple Silicon)

## 📁 Directory Structure

```
c/
├── test_onnx_model.c          # Main C implementation
├── model.onnx                 # Multiclass classification ONNX model
├── vocab.json                 # Token vocabulary mapping
├── scaler.json                # Label mapping for categories
├── Makefile                   # Build configuration
└── README.md                  # This file
```

## 🛠️ Step-by-Step Installation

### 🪟 Windows Installation

#### Step 1: Install Build Tools
```powershell
# Option A: Install Visual Studio Community (Recommended)
# Download from: https://visualstudio.microsoft.com/vs/community/
# During installation, select "Desktop development with C++"

# Option B: Install Build Tools for Visual Studio
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Option C: Install MinGW-w64
# Download from: https://www.mingw-w64.org/downloads/
# Or via MSYS2:
winget install MSYS2.MSYS2
```

#### Step 2: Install vcpkg Package Manager
```powershell
# Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg

# Bootstrap vcpkg
.\bootstrap-vcpkg.bat

# Add to PATH
$env:PATH += ";$(Get-Location)"

# Set VCPKG_ROOT environment variable
[Environment]::SetEnvironmentVariable("VCPKG_ROOT", (Get-Location), "User")
```

#### Step 3: Install Dependencies via vcpkg
```powershell
# Install cJSON
.\vcpkg install cjson:x64-windows

# Install pthread (if using MinGW)
.\vcpkg install pthreads:x64-windows

# Integrate with Visual Studio
.\vcpkg integrate install
```

#### Step 4: Download ONNX Runtime
```powershell
# Download ONNX Runtime for Windows
$onnxUrl = "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip"
Invoke-WebRequest -Uri $onnxUrl -OutFile "onnxruntime-win-x64-1.22.0.zip"

# Extract
Expand-Archive -Path "onnxruntime-win-x64-1.22.0.zip" -DestinationPath "."

# Set environment variable
[Environment]::SetEnvironmentVariable("ONNXRUNTIME_ROOT", "$PWD\onnxruntime-win-x64-1.22.0", "User")
```

#### Step 5: Create Project Directory
```powershell
# Create project directory
mkdir C:\whitelightning-c-multiclass
cd C:\whitelightning-c-multiclass

# Copy source files
# test_onnx_model.c, model.onnx, vocab.json, scaler.json
```

#### Step 6: Compile and Run
```powershell
# Using Visual Studio Developer Command Prompt
cl /I"%ONNXRUNTIME_ROOT%\include" /I"%VCPKG_ROOT%\installed\x64-windows\include" test_onnx_model.c /link "%ONNXRUNTIME_ROOT%\lib\onnxruntime.lib" "%VCPKG_ROOT%\installed\x64-windows\lib\cjson.lib" /out:test_onnx_model.exe

# Using MinGW-w64
gcc -Wall -Wextra -O2 -std=c99 -I"%ONNXRUNTIME_ROOT%/include" -I"%VCPKG_ROOT%/installed/x64-windows/include" test_onnx_model.c -L"%ONNXRUNTIME_ROOT%/lib" -L"%VCPKG_ROOT%/installed/x64-windows/lib" -lonnxruntime -lcjson -lpthread -lm -o test_onnx_model.exe

# Run the program
.\test_onnx_model.exe "France defeats Argentina in World Cup final"
```

---

### 🐧 Linux Installation

#### Step 1: Install Build Tools
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential gcc make

# CentOS/RHEL 8+
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y gcc make

# CentOS/RHEL 7
sudo yum groupinstall -y "Development Tools"
sudo yum install -y gcc make

# Fedora
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y gcc make
```

#### Step 2: Install cJSON Library
```bash
# Ubuntu/Debian
sudo apt install -y libcjson-dev

# CentOS/RHEL 8+
sudo dnf install -y cjson-devel

# CentOS/RHEL 7
sudo yum install -y epel-release
sudo yum install -y cjson-devel

# Fedora
sudo dnf install -y cjson-devel

# Alternative: Build from source
git clone https://github.com/DaveGamble/cJSON.git
cd cJSON
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

#### Step 3: Install pthread (usually included)
```bash
# pthread is typically included with glibc
# Verify availability
echo '#include <pthread.h>' | gcc -E - > /dev/null && echo "pthread available" || echo "pthread missing"
```

#### Step 4: Download ONNX Runtime
```bash
# Download ONNX Runtime for Linux
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar -xzf onnxruntime-linux-x64-1.22.0.tgz

# Set environment variable
echo "export ONNXRUNTIME_ROOT=$PWD/onnxruntime-linux-x64-1.22.0" >> ~/.bashrc
source ~/.bashrc
```

#### Step 5: Create Project Directory
```bash
# Create project directory
mkdir -p ~/whitelightning-c-multiclass
cd ~/whitelightning-c-multiclass

# Copy source files
# test_onnx_model.c, model.onnx, vocab.json, scaler.json
```

#### Step 6: Compile and Run
```bash
# Create Makefile or compile directly
gcc -Wall -Wextra -O2 -std=c99 \
    -I${ONNXRUNTIME_ROOT}/include \
    test_onnx_model.c \
    -L${ONNXRUNTIME_ROOT}/lib \
    -lonnxruntime -lcjson -lpthread -lm \
    -o test_onnx_model

# Set library path
export LD_LIBRARY_PATH=${ONNXRUNTIME_ROOT}/lib:$LD_LIBRARY_PATH

# Run the program
./test_onnx_model "France defeats Argentina in World Cup final"
```

---

### 🍎 macOS Installation

#### Step 1: Install Xcode Command Line Tools
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Accept license
sudo xcodebuild -license accept

# Verify installation
gcc --version
make --version
```

#### Step 2: Install Homebrew (if not installed)
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH (Apple Silicon)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc

# Add to PATH (Intel)
echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc
```

#### Step 3: Install Dependencies
```bash
# Install cJSON
brew install cjson

# Install wget (for downloading ONNX Runtime)
brew install wget
```

#### Step 4: Download ONNX Runtime
```bash
# For Apple Silicon Macs
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-universal2-1.22.0.tgz
tar -xzf onnxruntime-osx-universal2-1.22.0.tgz

# For Intel Macs
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-x86_64-1.22.0.tgz
tar -xzf onnxruntime-osx-x86_64-1.22.0.tgz

# Set environment variable
echo "export ONNXRUNTIME_ROOT=$PWD/onnxruntime-osx-universal2-1.22.0" >> ~/.zshrc
source ~/.zshrc
```

#### Step 5: Create Project Directory
```bash
# Create project directory
mkdir -p ~/whitelightning-c-multiclass
cd ~/whitelightning-c-multiclass

# Copy source files
# test_onnx_model.c, model.onnx, vocab.json, scaler.json
```

#### Step 6: Compile and Run
```bash
# Compile with proper flags
gcc -Wall -Wextra -O2 -std=c99 \
    -I${ONNXRUNTIME_ROOT}/include \
    -I$(brew --prefix cjson)/include \
    test_onnx_model.c \
    -L${ONNXRUNTIME_ROOT}/lib \
    -L$(brew --prefix cjson)/lib \
    -lonnxruntime -lcjson -lpthread -lm \
    -o test_onnx_model

# Run the program
./test_onnx_model "France defeats Argentina in World Cup final"
```

## 🔧 Advanced Configuration

### Makefile Template
```makefile
# Makefile for C Multiclass Classifier

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
TARGET = test_onnx_model
SOURCE = test_onnx_model.c

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Default paths
ONNXRUNTIME_ROOT ?= ./onnxruntime-linux-x64-1.22.0

# Platform-specific settings
ifeq ($(UNAME_S),Linux)
    INCLUDES = -I$(ONNXRUNTIME_ROOT)/include
    LDFLAGS = -L$(ONNXRUNTIME_ROOT)/lib
    LIBS = -lonnxruntime -lcjson -lpthread -lm
    RPATH = -Wl,-rpath,$(ONNXRUNTIME_ROOT)/lib
endif

ifeq ($(UNAME_S),Darwin)
    BREW_PREFIX = $(shell brew --prefix)
    INCLUDES = -I$(ONNXRUNTIME_ROOT)/include -I$(BREW_PREFIX)/include
    LDFLAGS = -L$(ONNXRUNTIME_ROOT)/lib -L$(BREW_PREFIX)/lib
    LIBS = -lonnxruntime -lcjson -lpthread -lm
    RPATH = -Wl,-rpath,$(ONNXRUNTIME_ROOT)/lib
endif

# Windows (MinGW)
ifeq ($(OS),Windows_NT)
    INCLUDES = -I$(ONNXRUNTIME_ROOT)/include -I$(VCPKG_ROOT)/installed/x64-windows/include
    LDFLAGS = -L$(ONNXRUNTIME_ROOT)/lib -L$(VCPKG_ROOT)/installed/x64-windows/lib
    LIBS = -lonnxruntime -lcjson -lpthread -lm
    TARGET = test_onnx_model.exe
endif

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(INCLUDES) $(SOURCE) $(LDFLAGS) $(LIBS) $(RPATH) -o $(TARGET)

test: $(TARGET)
	./$(TARGET) "France defeats Argentina in World Cup final"

benchmark: $(TARGET)
	./$(TARGET) --benchmark 100

clean:
	rm -f $(TARGET)

install-deps:
	@echo "Install dependencies for your platform:"
	@echo "Linux: sudo apt install libcjson-dev"
	@echo "macOS: brew install cjson"
	@echo "Windows: vcpkg install cjson:x64-windows"

.PHONY: all test benchmark clean install-deps
```

### Environment Variables
```bash
# Linux/macOS
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH

# Windows (PowerShell)
$env:ONNXRUNTIME_ROOT = "C:\path\to\onnxruntime"
$env:PATH += ";$env:ONNXRUNTIME_ROOT\lib"
```

## 🎯 Usage Examples

### Basic Usage
```bash
# Default test
./test_onnx_model

# Sports classification
./test_onnx_model "France defeats Argentina in World Cup final"

# Health classification
./test_onnx_model "New study reveals breakthrough in cancer treatment"

# Politics classification
./test_onnx_model "President signs new legislation on healthcare reform"

# Technology classification
./test_onnx_model "Apple announces new iPhone with revolutionary AI features"
```

### Performance Benchmarking
```bash
# Quick benchmark (10 iterations)
./test_onnx_model --benchmark 10

# Comprehensive benchmark (1000 iterations)
./test_onnx_model --benchmark 1000

# Using Makefile
make benchmark
```

### Cyrillic Text Support
```bash
# Test Cyrillic text
./test_onnx_model "шляк би тебе трафив"

# Mixed language text
./test_onnx_model "This is здоровье related topic"
```

## 🐛 Troubleshooting

### Windows Issues

**1. "MSVCR120.dll not found"**
```powershell
# Install Visual C++ Redistributable
# Download from: https://www.microsoft.com/en-us/download/details.aspx?id=40784
```

**2. "'cl' is not recognized as an internal or external command"**
```powershell
# Use Visual Studio Developer Command Prompt
# Or add Visual Studio to PATH
$env:PATH += ";C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64"
```

**3. "Cannot open include file 'cjson/cJSON.h'"**
```powershell
# Ensure vcpkg is properly integrated
.\vcpkg integrate install

# Or specify include path manually
/I"%VCPKG_ROOT%\installed\x64-windows\include"
```

**4. "LNK2019: unresolved external symbol"**
```powershell
# Ensure all libraries are linked
"%VCPKG_ROOT%\installed\x64-windows\lib\cjson.lib" "%ONNXRUNTIME_ROOT%\lib\onnxruntime.lib"
```

### Linux Issues

**1. "fatal error: cjson/cJSON.h: No such file or directory"**
```bash
# Install cJSON development headers
sudo apt install libcjson-dev  # Ubuntu/Debian
sudo dnf install cjson-devel   # CentOS/RHEL/Fedora
```

**2. "error while loading shared libraries: libonnxruntime.so"**
```bash
# Set library path
export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH

# Or copy to system library directory
sudo cp $ONNXRUNTIME_ROOT/lib/libonnxruntime.so* /usr/local/lib/
sudo ldconfig
```

**3. "undefined reference to pthread functions"**
```bash
# Link pthread library explicitly
gcc ... -lpthread
```

**4. "Permission denied" when running"**
```bash
# Make executable
chmod +x test_onnx_model
```

### macOS Issues

**1. "fatal error: 'cjson/cJSON.h' file not found"**
```bash
# Install cJSON via Homebrew
brew install cjson

# Or specify include path
-I$(brew --prefix cjson)/include
```

**2. "dyld: Library not loaded: libonnxruntime.dylib"**
```bash
# Check library path
otool -L test_onnx_model

# Set DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$DYLD_LIBRARY_PATH
```

**3. "Apple Silicon compatibility issues"**
```bash
# Use universal binary for ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-universal2-1.22.0.tgz

# Check architecture
file test_onnx_model
```

**4. "Xcode license agreement"**
```bash
# Accept Xcode license
sudo xcodebuild -license accept
```

## 📊 Expected Output

```
🤖 ONNX MULTICLASS CLASSIFIER - C IMPLEMENTATION
==============================================
🔄 Processing: "France defeats Argentina in World Cup final"

💻 SYSTEM INFORMATION:
   Platform: macOS 14.0 (Darwin)
   Processor: arm64
   CPU Cores: 12 physical, 12 logical
   Total Memory: 32.0 GB
   Runtime: C Implementation
   Compiler: GCC 13.2.0
   ONNX Runtime Version: 1.22.0

📊 MULTICLASS CLASSIFICATION RESULTS:
   🏆 Predicted Category: sports ⭐
   📈 Confidence: 92.34% (0.9234)
   📝 Input Text: "France defeats Argentina in World Cup final"
   
   📋 All Class Probabilities:
      health: 0.0123 (1.2%)
      politics: 0.0234 (2.3%)
      sports: 0.9234 (92.3%) ⭐
      world: 0.0409 (4.1%)

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 0.8ms
   ┣━ Preprocessing: 0.3ms (37.5%)
   ┣━ Model Inference: 0.4ms (50.0%)
   ┗━ Postprocessing: 0.1ms (12.5%)

🚀 THROUGHPUT:
   Texts per second: 1250.0

💾 RESOURCE USAGE:
   Memory Start: 8.45 MB
   Memory End: 9.12 MB
   Memory Delta: +0.67 MB
   CPU Usage: 18.3% avg, 45.2% peak (12 samples)

🎯 PERFORMANCE RATING: 🚀 EXCELLENT
   (0.8ms total - Target: <10ms)
```

## 🚀 Features

- **Blazing Fast Performance**: Sub-millisecond inference times
- **Advanced Performance Monitoring**: CPU usage tracking, memory monitoring, detailed timing analysis
- **System Information Collection**: Hardware specs, platform details, resource utilization
- **Cyrillic Text Support**: Proper Unicode handling and case conversion
- **Token-based Processing**: Vocabulary mapping with padding to 30 tokens
- **Comprehensive Benchmarking**: Statistical analysis with percentiles and throughput metrics
- **Real-time CPU Monitoring**: Multi-threaded continuous CPU usage tracking
- **Memory Tracking**: Before/after memory usage with delta calculations
- **Performance Classification**: Automatic rating (Excellent/Good/Acceptable/Poor)
- **Multi-class Output**: Support for 4+ classification categories

## 🎯 Performance Characteristics

- **Total Time**: ~0.8ms (fastest implementation)
- **Memory Usage**: Minimal (~0.7MB additional)
- **CPU Efficiency**: Low CPU usage with high throughput
- **Platform**: Consistent performance across operating systems
- **Scalability**: Suitable for high-throughput applications

## 🔧 Technical Details

### Model Architecture
- **Type**: Multiclass Classification (4+ categories)
- **Input**: Text string
- **Features**: Token sequences (30 tokens)
- **Output**: Probability distribution for each class
- **Prediction**: Argmax of output probabilities

### Processing Pipeline
1. **Text Normalization**: Cyrillic case conversion and cleaning
2. **Tokenization**: Word splitting and vocabulary lookup
3. **Token Mapping**: Word → token_id conversion with OOV handling
4. **Sequence Padding**: Zero-padding to exactly 30 tokens
5. **Model Inference**: ONNX Runtime execution
6. **Post-processing**: Probability interpretation and class mapping

### Supported Classes
- **🏥 Health**: Medical and health-related topics
- **🏛️ Politics**: Political news and discussions
- **⚽ Sports**: Sports events and activities
- **🌍 World**: International news and events

### Cyrillic Text Support
- **UTF-8 Processing**: Proper multi-byte character handling
- **Case Conversion**: Cyrillic uppercase → lowercase
- **Character Ranges**: Support for А-Я, а-я, Ё, ё
- **Mixed Text**: Cyrillic + Latin character support

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

## 📝 Notes

- **Fastest Implementation**: Sub-millisecond performance with minimal overhead
- **Production Ready**: Suitable for high-throughput, real-time applications
- **Memory Efficient**: Minimal memory footprint and no memory leaks
- **Cross-Platform**: Consistent behavior across operating systems

### When to Use C Implementation
- ✅ **High Performance**: Real-time or high-throughput requirements
- ✅ **System Programming**: Low-level control and optimization
- ✅ **Embedded Systems**: Resource-constrained environments
- ✅ **Legacy Integration**: Integration with existing C/C++ codebases
- ✅ **Minimal Dependencies**: Lightweight deployment requirements
- ❌ **Rapid Development**: Longer development time vs. higher-level languages
- ❌ **Complex Logic**: Better alternatives for complex business logic

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 