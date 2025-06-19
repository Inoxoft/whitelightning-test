# 📊 Node.js Multiclass Classification ONNX Model

A high-performance news category classifier using ONNX Runtime for JavaScript (Node.js) with comprehensive performance monitoring, system information display, and cross-platform compatibility.

## 📋 System Requirements

### Minimum Requirements
- **Node.js**: 18.0.0 or higher (LTS recommended)
- **RAM**: 2GB available memory
- **Storage**: 150MB free space
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+

### Recommended Versions
- **Node.js**: 18.17.0 or 20.x LTS
- **npm**: 9.0.0+ (comes with Node.js)
- **ONNX Runtime**: 1.22.0

### Supported Platforms
- ✅ **Windows**: 10, 11 (x64, ARM64)
- ✅ **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 9+, Fedora 30+
- ✅ **macOS**: 10.15+ (Intel & Apple Silicon)

## 📁 Directory Structure

```
nodejs/
├── test_onnx_model.js         # Main Node.js implementation
├── package.json               # Project configuration & dependencies
├── package-lock.json          # Dependency lock file
├── model.onnx                 # Multiclass classification ONNX model
├── vocab.json                 # Token vocabulary mapping
├── scaler.json                # Label mapping for categories
└── README.md                  # This file
```

## 🛠️ Step-by-Step Installation

### 🪟 Windows Installation

#### Step 1: Install Node.js
```powershell
# Option A: Download from official website
# Visit: https://nodejs.org/en/download/
# Download Windows Installer (.msi) - LTS version recommended

# Option B: Using winget
winget install OpenJS.NodeJS

# Option C: Using Chocolatey
choco install nodejs

# Option D: Using Scoop
scoop install nodejs

# Verify installation
node --version
npm --version
```

#### Step 2: Create Project Directory
```powershell
# Create project directory
mkdir C:\whitelightning-nodejs-multiclass
cd C:\whitelightning-nodejs-multiclass

# Copy project files
# package.json, test_onnx_model.js, model.onnx, vocab.json, scaler.json
```

#### Step 3: Install Dependencies
```powershell
# Install dependencies
npm install

# Or install manually
npm install onnxruntime-node@1.22.0

# Verify installation
npm list
```

#### Step 4: Run the Application
```powershell
# Run with default test
npm start

# Run with custom text
npm start "France defeats Argentina in World Cup final"

# Run benchmark
npm run benchmark

# Using node directly
node test_onnx_model.js "New healthcare policy announced"
```

---

### 🐧 Linux Installation

#### Step 1: Install Node.js
```bash
# Option A: Using NodeSource repository (Recommended)
# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# CentOS/RHEL/Fedora
curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
sudo dnf install -y nodejs  # or yum for older versions

# Option B: Using package manager (may be older version)
# Ubuntu/Debian
sudo apt update
sudo apt install -y nodejs npm

# CentOS/RHEL 8+
sudo dnf install -y nodejs npm

# CentOS/RHEL 7
sudo yum install -y nodejs npm

# Fedora
sudo dnf install -y nodejs npm

# Option C: Using snap
sudo snap install node --classic

# Option D: Using nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 18
nvm use 18

# Verify installation
node --version
npm --version
```

#### Step 2: Create Project Directory
```bash
# Create project directory
mkdir -p ~/whitelightning-nodejs-multiclass
cd ~/whitelightning-nodejs-multiclass

# Copy project files
# package.json, test_onnx_model.js, model.onnx, vocab.json, scaler.json
```

#### Step 3: Install Dependencies
```bash
# Install dependencies
npm install

# Or install manually
npm install onnxruntime-node@1.22.0

# Verify installation
npm list
```

#### Step 4: Run the Application
```bash
# Run with default test
npm start

# Run with custom text
npm start "France defeats Argentina in World Cup final"

# Run benchmark
npm run benchmark

# Using node directly
node test_onnx_model.js "New healthcare policy announced"
```

---

### 🍎 macOS Installation

#### Step 1: Install Node.js
```bash
# Option A: Download from official website
# Visit: https://nodejs.org/en/download/
# Download macOS Installer (.pkg) - LTS version recommended

# Option B: Using Homebrew (Recommended)
# Install Homebrew first if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH (Apple Silicon)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc

# Add to PATH (Intel)
echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc

# Install Node.js
brew install node

# Option C: Using MacPorts
sudo port install nodejs18

# Option D: Using nvm (Node Version Manager)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.zshrc
nvm install 18
nvm use 18

# Verify installation
node --version
npm --version
```

#### Step 2: Create Project Directory
```bash
# Create project directory
mkdir -p ~/whitelightning-nodejs-multiclass
cd ~/whitelightning-nodejs-multiclass

# Copy project files
# package.json, test_onnx_model.js, model.onnx, vocab.json, scaler.json
```

#### Step 3: Install Dependencies
```bash
# Install dependencies
npm install

# Or install manually
npm install onnxruntime-node@1.22.0

# Verify installation
npm list
```

#### Step 4: Run the Application
```bash
# Run with default test
npm start

# Run with custom text
npm start "France defeats Argentina in World Cup final"

# Run benchmark
npm run benchmark

# Using node directly
node test_onnx_model.js "New healthcare policy announced"
```

## 🔧 Advanced Configuration

### package.json Template
```json
{
  "name": "onnx-multiclass-classifier",
  "version": "1.0.0",
  "description": "ONNX Multiclass Text Classification with Node.js",
  "main": "test_onnx_model.js",
  "type": "module",
  "scripts": {
    "start": "node test_onnx_model.js",
    "test": "node test_onnx_model.js",
    "benchmark": "node test_onnx_model.js --benchmark 100",
    "benchmark-quick": "node test_onnx_model.js --benchmark 10",
    "benchmark-comprehensive": "node test_onnx_model.js --benchmark 1000"
  },
  "keywords": [
    "onnx",
    "machine-learning",
    "text-classification",
    "multiclass",
    "news-classification",
    "nodejs"
  ],
  "author": "White Lightning Team",
  "license": "MIT",
  "dependencies": {
    "onnxruntime-node": "^1.22.0"
  },
  "engines": {
    "node": ">=18.0.0",
    "npm": ">=9.0.0"
  }
}
```

### Environment Variables
```bash
# Linux/macOS
export NODE_ENV=production
export NODE_OPTIONS="--max-old-space-size=4096"

# Windows (PowerShell)
$env:NODE_ENV = "production"
$env:NODE_OPTIONS = "--max-old-space-size=4096"
```

### Performance Tuning
```bash
# Increase memory limit for large models
node --max-old-space-size=4096 test_onnx_model.js

# Enable V8 optimizations
node --optimize-for-size test_onnx_model.js

# Use specific V8 flags for performance
node --turbo-inlining --turbo-inline-array-builtins test_onnx_model.js
```

## 🎯 Usage Examples

### Basic Usage
```bash
# Default test suite
npm start

# Sports classification
npm start "France defeats Argentina in World Cup final"

# Health classification
npm start "New study reveals breakthrough in cancer treatment"

# Politics classification
npm start "President signs new legislation on healthcare reform"

# Technology classification
npm start "Apple announces new iPhone with revolutionary AI features"

# World news classification
npm start "Climate change summit begins in Paris"
```

### Performance Benchmarking
```bash
# Quick benchmark (10 iterations)
npm run benchmark-quick

# Standard benchmark (100 iterations)
npm run benchmark

# Comprehensive benchmark (1000 iterations)
npm run benchmark-comprehensive

# Custom benchmark
node test_onnx_model.js --benchmark 500
```

### Advanced Usage
```bash
# Run with specific Node.js version (using nvm)
nvm use 18 && npm start

# Run with memory profiling
node --inspect test_onnx_model.js

# Run with CPU profiling
node --prof test_onnx_model.js
```

## 🐛 Troubleshooting

### Windows Issues

**1. "'node' is not recognized as an internal or external command"**
```powershell
# Add Node.js to PATH
$env:PATH += ";C:\Program Files\nodejs"

# Or reinstall Node.js with PATH option checked
```

**2. "npm ERR! EACCES: permission denied"**
```powershell
# Run as Administrator or use different npm prefix
npm config set prefix %APPDATA%\npm
```

**3. "Error: Cannot find module 'onnxruntime-node'"**
```powershell
# Clear npm cache and reinstall
npm cache clean --force
npm install

# Or install globally
npm install -g onnxruntime-node
```

**4. "gyp ERR! stack Error: Can't find Python executable"**
```powershell
# Install Python and Windows Build Tools
npm install --global windows-build-tools
# Or install Visual Studio Build Tools
```

### Linux Issues

**1. "node: /lib/x86_64-linux-gnu/libc.so.6: version 'GLIBC_2.28' not found"**
```bash
# Update glibc or use older Node.js version
# For Ubuntu 18.04, use Node.js 16.x
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt-get install -y nodejs
```

**2. "npm WARN checkPermissions Missing write access"**
```bash
# Fix npm permissions
sudo chown -R $(whoami) ~/.npm
sudo chown -R $(whoami) /usr/local/lib/node_modules

# Or use nvm instead of system Node.js
```

**3. "Error: ENOSPC: System limit for number of file watchers reached"**
```bash
# Increase file watcher limit
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

**4. "Module did not self-register"**
```bash
# Rebuild native modules
npm rebuild

# Or clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### macOS Issues

**1. "gyp: No Xcode or CLT version detected!"**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Accept license
sudo xcodebuild -license accept
```

**2. "Error: EACCES: permission denied, mkdir '/usr/local/lib/node_modules'"**
```bash
# Use nvm instead of system Node.js
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.zshrc
nvm install 18
nvm use 18
```

**3. "Apple Silicon compatibility issues"**
```bash
# Use Rosetta for Intel-only packages
arch -x86_64 npm install

# Or use native ARM64 Node.js
brew install node
```

**4. "Certificate verification failed"**
```bash
# Fix npm SSL issues
npm config set strict-ssl false
# Or update certificates
npm config set ca ""
```

## 📊 Expected Output

```
🤖 ONNX MULTICLASS CLASSIFIER - JAVASCRIPT IMPLEMENTATION
========================================================

🔄 Processing: France defeats Argentina in World Cup final

💻 SYSTEM INFORMATION:
   Platform: darwin
   Processor: arm64
   CPU Cores: 8
   Total Memory: 16.0 GB
   Runtime: JavaScript Implementation
   Node.js Version: v18.17.0
   ONNX Runtime Version: 1.22.0

📊 MULTICLASS CLASSIFICATION RESULTS:
   🏆 Predicted Category: sports
   📈 Confidence: 92.34% (0.9234)
   📝 Input Text: "France defeats Argentina in World Cup final"
   📋 All Class Probabilities:
      health: 0.0123 (1.2%)
      politics: 0.0234 (2.3%)
      sports: 0.9234 (92.3%)
      world: 0.0409 (4.1%)

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 38.45ms
   ┣━ Preprocessing: 8.23ms (21.4%)
   ┣━ Model Inference: 25.67ms (66.8%)
   ┗━ Postprocessing: 4.55ms (11.8%)

🚀 THROUGHPUT:
   Texts per second: 26.0

💾 RESOURCE USAGE:
   Memory Start: 42.34 MB
   Memory End: 43.78 MB
   Memory Delta: +1.44 MB
   CPU Usage: 12.8% avg, 38.9% peak (10 samples)

🎯 PERFORMANCE RATING: ✅ GOOD
   (38.5ms total - Target: <100ms)
```

## 🚀 Features

- **News Classification**: Multiclass classification (health, politics, sports, world) using token-based preprocessing
- **Performance Monitoring**: Detailed timing breakdown, resource usage tracking, and throughput analysis
- **System Information**: Platform detection, CPU/memory specs, runtime versions
- **Benchmarking Mode**: Statistical performance analysis with multiple runs
- **CI/CD Ready**: Graceful handling when model files are missing
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 🎯 Performance Characteristics

- **Total Time**: ~38ms (good performance)
- **Memory Usage**: Low (~1.4MB additional)
- **CPU Efficiency**: Moderate CPU usage with good throughput
- **Platform**: Consistent performance across operating systems
- **Scalability**: Good for medium-throughput applications

## 🔧 Technical Details

### Preprocessing Pipeline
1. **Text Tokenization**: Split text into words and convert to lowercase
2. **Token Mapping**: Convert words to integer IDs using vocabulary
3. **Sequence Padding**: Pad/truncate to fixed length of 30 tokens
4. **OOV Handling**: Unknown words mapped to `<OOV>` token

### Model Architecture
- **Input**: Int32 tensor [1, 30] (token sequence)
- **Output**: Float32 tensor [1, 4] (class probabilities)
- **Classes**: health, politics, sports, world

### Performance Monitoring
- **High-Resolution Timing**: Uses `performance.now()` for microsecond precision
- **Memory Tracking**: Monitors heap usage before/after processing
- **CPU Monitoring**: Samples CPU usage during processing (approximation)
- **Statistical Analysis**: Mean, min, max calculations for benchmarking

## 🏗️ CI/CD Integration

The implementation includes CI-friendly features:

```javascript
// Graceful handling when model files are missing
if (!checkModelFiles()) {
    console.log('⚠️ Model files not found - exiting safely');
    console.log('✅ JavaScript implementation compiled successfully');
    return;
}
```

### GitHub Actions Integration
```yaml
- name: Test JavaScript Multiclass Classifier
  run: |
    cd tests/multiclass_classifier/nodejs
    npm install
    npm test
```

## 🎯 Classification Categories

The model classifies news articles into these categories:

- **🏥 Health**: Medical news, healthcare policies, disease outbreaks
- **🏛️ Politics**: Government actions, elections, political events
- **⚽ Sports**: Sports events, competitions, athlete news
- **🌍 World**: International news, global events, foreign affairs

## 📝 Notes

- **Good Performance**: Moderate processing time with reasonable resource usage
- **Easy Deployment**: Simple setup with npm and minimal dependencies
- **Cross-Platform**: Consistent behavior across operating systems
- **Developer Friendly**: Easy to integrate and modify

### When to Use Node.js Implementation
- ✅ **Web Applications**: Server-side text classification
- ✅ **API Services**: RESTful classification endpoints
- ✅ **Rapid Prototyping**: Quick development and testing
- ✅ **JavaScript Ecosystem**: Integration with existing JS/TS projects
- ✅ **Microservices**: Lightweight classification service
- ❌ **High Performance**: Not the fastest option available
- ❌ **Resource Constrained**: Higher memory usage than C/Rust

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 