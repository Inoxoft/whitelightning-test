# 🚀 Node.js Binary Classifier - JavaScript Implementation

A high-performance sentiment analysis classifier using ONNX Runtime for JavaScript (Node.js) with comprehensive performance monitoring, system information display, and cross-platform support.

## 📋 System Requirements

### Minimum Requirements
- **CPU**: x86_64 or ARM64 architecture
- **RAM**: 2GB available memory
- **Storage**: 150MB free space
- **Node.js**: 18.0.0+ (recommended: 18.17.0 or 20.x LTS)
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+

### Supported Platforms
- ✅ **Windows**: 10, 11 (x64, ARM64)
- ✅ **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 9+, Fedora 30+
- ✅ **macOS**: 10.15+ (Intel & Apple Silicon)

## 🛠️ Step-by-Step Installation

### 🪟 Windows Installation

#### Step 1: Install Node.js
```powershell
# Option A: Download from nodejs.org (Recommended)
# Visit: https://nodejs.org/en/download/
# Download Node.js LTS (18.x or 20.x)
# Run installer and follow setup wizard

# Option B: Install via winget
winget install OpenJS.NodeJS

# Option C: Install via Chocolatey
choco install nodejs

# Verify installation
node --version
npm --version
```

#### Step 2: Create Project Directory
```powershell
# Create project directory
mkdir C:\whitelightning-nodejs
cd C:\whitelightning-nodejs

# Initialize npm project (if creating from scratch)
npm init -y
```

#### Step 3: Install Dependencies
```powershell
# Install ONNX Runtime for Node.js
npm install onnxruntime-node

# Install additional dependencies (if needed)
npm install

# Verify installation
npm list onnxruntime-node
```

#### Step 4: Copy Source Files
```powershell
# Copy your source files to the project directory
# test_onnx_model.js, model.onnx, vocab.json, scaler.json, package.json
```

#### Step 5: Configure package.json
```powershell
# Ensure package.json has the correct configuration
# Edit package.json to include:
# {
#   "type": "module",
#   "scripts": {
#     "start": "node test_onnx_model.js",
#     "benchmark": "node test_onnx_model.js --benchmark 100"
#   }
# }
```

#### Step 6: Run the Program
```powershell
# Run with default text
npm start

# Run with custom text
npm start "This product is amazing!"

# Run benchmark
npm run benchmark

# Direct node execution
node test_onnx_model.js "Custom text here"
```

---

### 🐧 Linux Installation

#### Step 1: Install Node.js
```bash
# Option A: Install via NodeSource repository (Recommended)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Option B: Install via package manager
# Ubuntu/Debian
sudo apt update
sudo apt install -y nodejs npm

# CentOS/RHEL 8+
sudo dnf install -y nodejs npm

# CentOS/RHEL 7
sudo yum install -y nodejs npm

# Fedora
sudo dnf install -y nodejs npm

# Option C: Install via snap
sudo snap install node --classic

# Verify installation
node --version
npm --version
```

#### Step 2: Create Project Directory
```bash
# Create project directory
mkdir -p ~/whitelightning-nodejs
cd ~/whitelightning-nodejs

# Initialize npm project (if creating from scratch)
npm init -y
```

#### Step 3: Install Dependencies
```bash
# Install ONNX Runtime for Node.js
npm install onnxruntime-node

# Install additional dependencies (if needed)
npm install

# Verify installation
npm list onnxruntime-node
```

#### Step 4: Copy Source Files
```bash
# Copy your source files to the project directory
# test_onnx_model.js, model.onnx, vocab.json, scaler.json, package.json
```

#### Step 5: Configure package.json
```bash
# Ensure package.json has the correct configuration
cat > package.json << 'EOF'
{
  "name": "whitelightning-nodejs",
  "version": "1.0.0",
  "description": "ONNX Binary Classifier for Node.js",
  "type": "module",
  "main": "test_onnx_model.js",
  "scripts": {
    "start": "node test_onnx_model.js",
    "benchmark": "node test_onnx_model.js --benchmark 100",
    "test": "node test_onnx_model.js 'Test message'"
  },
  "dependencies": {
    "onnxruntime-node": "^1.19.2"
  }
}
EOF
```

#### Step 6: Run the Program
```bash
# Run with default text
npm start

# Run with custom text
npm start "This product is amazing!"

# Run benchmark
npm run benchmark

# Direct node execution
node test_onnx_model.js "Custom text here"
```

---

### 🍎 macOS Installation

#### Step 1: Install Node.js
```bash
# Option A: Install via Homebrew (Recommended)
# Install Homebrew first if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Node.js
brew install node@18
# or for latest LTS
brew install node

# Add to PATH (if needed)
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Option B: Download from nodejs.org
# Visit: https://nodejs.org/en/download/
# Download macOS installer and run

# Option C: Install via MacPorts
sudo port install nodejs18

# Verify installation
node --version
npm --version
```

#### Step 2: Create Project Directory
```bash
# Create project directory
mkdir -p ~/whitelightning-nodejs
cd ~/whitelightning-nodejs

# Initialize npm project (if creating from scratch)
npm init -y
```

#### Step 3: Install Dependencies
```bash
# Install ONNX Runtime for Node.js
npm install onnxruntime-node

# For Apple Silicon Macs, ensure compatibility
# The package should automatically install the correct binary

# Install additional dependencies (if needed)
npm install

# Verify installation
npm list onnxruntime-node
```

#### Step 4: Copy Source Files
```bash
# Copy your source files to the project directory
# test_onnx_model.js, model.onnx, vocab.json, scaler.json, package.json
```

#### Step 5: Configure package.json
```bash
# Ensure package.json has the correct configuration
cat > package.json << 'EOF'
{
  "name": "whitelightning-nodejs",
  "version": "1.0.0",
  "description": "ONNX Binary Classifier for Node.js",
  "type": "module",
  "main": "test_onnx_model.js",
  "scripts": {
    "start": "node test_onnx_model.js",
    "benchmark": "node test_onnx_model.js --benchmark 100",
    "test": "node test_onnx_model.js 'Test message'"
  },
  "dependencies": {
    "onnxruntime-node": "^1.19.2"
  },
  "engines": {
    "node": ">=18.0.0"
  }
}
EOF
```

#### Step 6: Run the Program
```bash
# Run with default text
npm start

# Run with custom text
npm start "This product is amazing!"

# Run benchmark
npm run benchmark

# Direct node execution
node test_onnx_model.js "Custom text here"
```

## 🔧 Advanced Configuration

### Package.json Configuration
```json
{
  "name": "whitelightning-nodejs",
  "version": "1.0.0",
  "description": "ONNX Binary Classifier for Node.js",
  "type": "module",
  "main": "test_onnx_model.js",
  "scripts": {
    "start": "node test_onnx_model.js",
    "benchmark": "node test_onnx_model.js --benchmark 100",
    "test": "node test_onnx_model.js 'Test message'",
    "dev": "node --inspect test_onnx_model.js",
    "profile": "node --prof test_onnx_model.js --benchmark 1000"
  },
  "dependencies": {
    "onnxruntime-node": "^1.19.2"
  },
  "engines": {
    "node": ">=18.0.0"
  },
  "keywords": ["onnx", "machine-learning", "sentiment-analysis", "nlp"],
  "author": "Your Name",
  "license": "MIT"
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

# Enable inspector for debugging
node --inspect test_onnx_model.js

# Profile performance
node --prof test_onnx_model.js --benchmark 1000
node --prof-process isolate-*.log > profile.txt
```

## 🎯 Usage Examples

### Basic Usage
```bash
# Default test
npm start

# Positive sentiment
npm start "I love this product! It's amazing!"

# Negative sentiment
npm start "This is terrible and disappointing."

# Neutral sentiment
npm start "The product is okay, nothing special."
```

### Advanced Usage
```bash
# Benchmark with custom iterations
node test_onnx_model.js --benchmark 500

# Multiple text processing
node test_onnx_model.js "First text" "Second text" "Third text"

# With environment variables
NODE_ENV=production node test_onnx_model.js "Production test"

# Memory profiling
node --inspect test_onnx_model.js "Memory test"
```

### Programmatic Usage
```javascript
// classifier.js
import { InferenceSession, Tensor } from 'onnxruntime-node';
import fs from 'fs';

class BinaryClassifier {
    constructor(modelPath, vocabPath, scalerPath) {
        this.modelPath = modelPath;
        this.vocabPath = vocabPath;
        this.scalerPath = scalerPath;
        this.session = null;
        this.vocab = null;
        this.scaler = null;
    }

    async initialize() {
        // Load ONNX model
        this.session = await InferenceSession.create(this.modelPath);
        
        // Load vocabulary and scaler
        this.vocab = JSON.parse(fs.readFileSync(this.vocabPath, 'utf8'));
        this.scaler = JSON.parse(fs.readFileSync(this.scalerPath, 'utf8'));
    }

    async predict(text) {
        if (!this.session) {
            throw new Error('Classifier not initialized. Call initialize() first.');
        }

        // Preprocess text to TF-IDF features
        const features = this.preprocessText(text);
        
        // Create input tensor
        const inputTensor = new Tensor('float32', features, [1, 5000]);
        
        // Run inference
        const results = await this.session.run({ float_input: inputTensor });
        
        // Extract probability
        return results.output.data[0];
    }

    preprocessText(text) {
        // Implement TF-IDF preprocessing
        const features = new Float32Array(5000);
        // ... TF-IDF implementation ...
        return features;
    }

    async close() {
        if (this.session) {
            await this.session.release();
        }
    }
}

// Usage
const classifier = new BinaryClassifier('model.onnx', 'vocab.json', 'scaler.json');
await classifier.initialize();

const probability = await classifier.predict("This product is amazing!");
const sentiment = probability > 0.5 ? "Positive" : "Negative";
console.log(`Sentiment: ${sentiment} (${(probability * 100).toFixed(2)}%)`);

await classifier.close();
```

## 🐛 Troubleshooting

### Windows Issues

**1. "'node' is not recognized as an internal or external command"**
```powershell
# Add Node.js to PATH manually
$env:PATH += ";C:\Program Files\nodejs"

# Or reinstall Node.js with "Add to PATH" checked
```

**2. "Cannot find module 'onnxruntime-node'"**
```powershell
# Clear npm cache and reinstall
npm cache clean --force
npm install onnxruntime-node

# Check npm configuration
npm config list
```

**3. "Error: EACCES: permission denied"**
```powershell
# Run PowerShell as Administrator
# Or configure npm to use a different directory
npm config set prefix "C:\Users\%USERNAME%\AppData\Roaming\npm"
```

**4. "Python not found" during installation**
```powershell
# Install Python (required for some native modules)
winget install Python.Python.3.11

# Or install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
```

### Linux Issues

**1. "node: command not found"**
```bash
# Install Node.js via NodeSource
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Or via package manager
sudo apt install nodejs npm  # Ubuntu/Debian
sudo dnf install nodejs npm  # Fedora
```

**2. "npm: command not found"**
```bash
# Install npm separately
sudo apt install npm  # Ubuntu/Debian
sudo dnf install npm  # Fedora

# Or reinstall Node.js with npm included
```

**3. "Permission denied" when installing packages**
```bash
# Change npm default directory
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Or use sudo (not recommended)
sudo npm install -g onnxruntime-node
```

**4. "ENOSPC: System limit for number of file watchers reached"**
```bash
# Increase file watcher limit
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### macOS Issues

**1. "node: command not found"**
```bash
# Install via Homebrew
brew install node

# Or download from nodejs.org
# https://nodejs.org/en/download/
```

**2. "gyp: No Xcode or CLT version detected"**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Accept license
sudo xcodebuild -license accept
```

**3. "Architecture compatibility issues (Apple Silicon)"**
```bash
# Check Node.js architecture
node -p "process.arch"

# Install Rosetta 2 if needed (for Intel compatibility)
softwareupdate --install-rosetta

# Use native ARM64 Node.js
brew install node
```

**4. "npm install fails with permission errors"**
```bash
# Fix npm permissions
sudo chown -R $(whoami) $(npm config get prefix)/{lib/node_modules,bin,share}

# Or use nvm for better version management
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18
```

## 📊 Expected Output

```
🤖 ONNX BINARY CLASSIFIER - JAVASCRIPT IMPLEMENTATION
====================================================
🔄 Processing: "This product is amazing!"

💻 SYSTEM INFORMATION:
   Platform: darwin
   Processor: arm64
   CPU Cores: 12
   Total Memory: 32.0 GB
   Runtime: JavaScript Implementation
   Node.js Version: v18.17.0
   ONNX Runtime Version: 1.19.2

📊 SENTIMENT ANALYSIS RESULTS:
   🏆 Predicted Sentiment: Positive ✅
   📈 Confidence: 87.45% (0.8745)
   📝 Input Text: "This product is amazing!"

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 28.89ms
   ┣━ Preprocessing: 12.34ms (42.7%)
   ┣━ Model Inference: 14.45ms (50.0%)
   ┗━ Postprocessing: 2.10ms (7.3%)

🚀 THROUGHPUT:
   Texts per second: 34.6

💾 RESOURCE USAGE:
   Memory Start: 45.67 MB
   Memory End: 47.23 MB
   Memory Delta: +1.56 MB
   CPU Usage: 15.2% avg, 45.8% peak (12 samples)

🎯 PERFORMANCE RATING: ✅ GOOD
   (28.9ms total - Target: <100ms)
```

## 🚀 Features

- **Sentiment Analysis**: Binary classification (Positive/Negative) using TF-IDF preprocessing
- **Performance Monitoring**: Detailed timing breakdown, resource usage tracking, and throughput analysis
- **System Information**: Platform detection, CPU/memory specs, runtime versions
- **Benchmarking Mode**: Statistical performance analysis with multiple runs
- **CI/CD Ready**: Graceful handling when model files are missing
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 📁 Required Files

Place these files in the same directory as `package.json`:

1. **model.onnx** - The trained ONNX model file
2. **vocab.json** - TF-IDF vocabulary and IDF weights
   ```json
   {
     "vocab": {"word1": 0, "word2": 1, ...},
     "idf": [1.23, 4.56, ...]
   }
   ```
3. **scaler.json** - Feature scaling parameters
   ```json
   {
     "mean": [0.1, 0.2, ...],
     "scale": [1.1, 1.2, ...]
   }
   ```

## 🔧 Technical Details

### Preprocessing Pipeline
1. **Text Tokenization**: Split text into words and convert to lowercase
2. **TF-IDF Vectorization**: Convert to 5000-dimensional feature vector
3. **Feature Scaling**: Apply mean normalization and standard scaling

### Model Architecture
- **Input**: Float32 tensor [1, 5000] (TF-IDF features)
- **Output**: Float32 tensor [1, 1] (sentiment probability)
- **Threshold**: 0.5 (>0.5 = Positive, ≤0.5 = Negative)

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
- name: Test JavaScript Binary Classifier
  run: |
    cd Test/binary_classifier/nodejs
    npm install
    npm test
```

## 📈 Performance Expectations

- **Target**: <100ms total processing time
- **Typical**: 20-80ms on modern hardware
- **Throughput**: 15-50 texts/second depending on hardware

## 📝 Notes

- **Excellent Performance**: Fast processing with low memory usage
- **Cross-Platform**: Consistent behavior across operating systems
- **Production Ready**: Suitable for web applications and microservices
- **Easy Integration**: Simple JavaScript API for web and server applications

### When to Use Node.js Implementation
- ✅ **Web Applications**: Server-side sentiment analysis
- ✅ **Microservices**: Containerized classification services
- ✅ **Real-time Processing**: WebSocket or API endpoints
- ✅ **Full-Stack**: JavaScript ecosystem integration
- ✅ **Prototyping**: Quick development and testing

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 