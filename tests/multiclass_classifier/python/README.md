# 🐍 Python Multiclass Classification ONNX Model

This directory contains a **Python implementation** for multiclass text classification using ONNX Runtime. The model performs **topic classification** on news articles and other text content, categorizing them into 10 predefined categories with comprehensive cross-platform support.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8+ (recommended: Python 3.9-3.11)
- **RAM**: 2GB available memory
- **Storage**: 200MB free space
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+

### Recommended Versions
- **Python**: 3.9.16 or 3.10.11 (best compatibility)
- **pip**: 21.0+ (comes with Python)
- **ONNX Runtime**: 1.22.0

### Supported Platforms
- ✅ **Windows**: 10, 11 (x64, ARM64)
- ✅ **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 9+, Fedora 30+
- ✅ **macOS**: 10.15+ (Intel & Apple Silicon)

## 📁 Directory Structure

```
python/
├── test_onnx_model.py         # Main Python implementation
├── model.onnx                 # Multiclass classification ONNX model
├── vocab.json                 # Vocabulary mapping for tokenization
├── scaler.json                # Label mapping for categories
├── requirements.txt           # Python dependencies
├── performance_results.json   # Performance test results
└── README.md                  # This file
```

## 🛠️ Step-by-Step Installation

### 🪟 Windows Installation

#### Step 1: Install Python
```powershell
# Option A: Download from official website (Recommended)
# Visit: https://www.python.org/downloads/windows/
# Download Python 3.9.16 or 3.10.11 (64-bit)
# During installation, check "Add Python to PATH"

# Option B: Using Microsoft Store
# Search "Python" in Microsoft Store and install

# Option C: Using winget
winget install Python.Python.3.10

# Option D: Using Chocolatey
choco install python

# Verify installation
python --version
pip --version
```

#### Step 2: Create Virtual Environment
```powershell
# Create project directory
mkdir C:\whitelightning-python-multiclass
cd C:\whitelightning-python-multiclass

# Copy project files
# test_onnx_model.py, model.onnx, vocab.json, scaler.json, requirements.txt

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If execution policy error occurs:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Step 3: Install Dependencies
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install from requirements.txt
pip install -r requirements.txt

# Or install manually
pip install onnxruntime==1.22.0 numpy==1.24.3 psutil==5.9.5

# Verify installation
pip list
```

#### Step 4: Run the Application
```powershell
# Run with default test
python test_onnx_model.py

# Run with custom text
python test_onnx_model.py "France defeats Argentina in World Cup final"

# Deactivate virtual environment when done
deactivate
```

---

### 🐧 Linux Installation

#### Step 1: Install Python
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-dev

# CentOS/RHEL 8+
sudo dnf install -y python3 python3-pip python3-venv python3-devel

# CentOS/RHEL 7
sudo yum install -y python3 python3-pip python3-venv python3-devel

# Fedora
sudo dnf install -y python3 python3-pip python3-venv python3-devel

# Alternative: Install specific version using pyenv
curl https://pyenv.run | bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc
pyenv install 3.10.11
pyenv global 3.10.11

# Verify installation
python3 --version
pip3 --version
```

#### Step 2: Create Virtual Environment
```bash
# Create project directory
mkdir -p ~/whitelightning-python-multiclass
cd ~/whitelightning-python-multiclass

# Copy project files
# test_onnx_model.py, model.onnx, vocab.json, scaler.json, requirements.txt

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show (venv) in prompt)
which python
```

#### Step 3: Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install from requirements.txt
pip install -r requirements.txt

# Or install manually
pip install onnxruntime==1.22.0 numpy==1.24.3 psutil==5.9.5

# For GPU support (optional)
# pip install onnxruntime-gpu==1.22.0

# Verify installation
pip list
```

#### Step 4: Run the Application
```bash
# Run with default test
python test_onnx_model.py

# Run with custom text
python test_onnx_model.py "France defeats Argentina in World Cup final"

# Deactivate virtual environment when done
deactivate
```

---

### 🍎 macOS Installation

#### Step 1: Install Python
```bash
# Option A: Using Homebrew (Recommended)
# Install Homebrew first if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH (Apple Silicon)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc

# Add to PATH (Intel)
echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc

# Install Python
brew install python@3.10

# Option B: Download from official website
# Visit: https://www.python.org/downloads/macos/
# Download Python 3.10.11 (Universal2)

# Option C: Using MacPorts
sudo port install python310

# Option D: Using pyenv
curl https://pyenv.run | bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
source ~/.zshrc
pyenv install 3.10.11
pyenv global 3.10.11

# Verify installation
python3 --version
pip3 --version
```

#### Step 2: Create Virtual Environment
```bash
# Create project directory
mkdir -p ~/whitelightning-python-multiclass
cd ~/whitelightning-python-multiclass

# Copy project files
# test_onnx_model.py, model.onnx, vocab.json, scaler.json, requirements.txt

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show (venv) in prompt)
which python
```

#### Step 3: Install Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install from requirements.txt
pip install -r requirements.txt

# Or install manually
pip install onnxruntime==1.22.0 numpy==1.24.3 psutil==5.9.5

# For Apple Silicon optimization (optional)
# pip install intel-numpy  # Intel MKL optimized numpy

# Verify installation
pip list
```

#### Step 4: Run the Application
```bash
# Run with default test
python test_onnx_model.py

# Run with custom text
python test_onnx_model.py "France defeats Argentina in World Cup final"

# Deactivate virtual environment when done
deactivate
```

## 🔧 Advanced Configuration

### requirements.txt Template
```txt
# Core dependencies
onnxruntime==1.22.0
numpy==1.24.3
psutil==5.9.5

# Optional: GPU support
# onnxruntime-gpu==1.22.0

# Optional: Performance optimization
# intel-numpy==1.24.3  # Intel MKL optimized numpy

# Development dependencies (optional)
# pytest==7.4.0
# black==23.7.0
# flake8==6.0.0
```

### Environment Variables
```bash
# Linux/macOS
export OMP_NUM_THREADS=4
export PYTHONPATH=$PYTHONPATH:.

# Windows (PowerShell)
$env:OMP_NUM_THREADS = "4"
$env:PYTHONPATH = "$env:PYTHONPATH;."
```

### Performance Optimization
```bash
# Use optimized BLAS libraries
pip install intel-numpy

# Set thread count for OpenMP
export OMP_NUM_THREADS=$(nproc)

# Use Intel MKL (if available)
export MKL_NUM_THREADS=4
```

## 🎯 Usage Examples

### Basic Usage
```bash
# Navigate to the Python directory
cd tests/multiclass_classifier/python

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\Activate.ps1  # Windows

# Default test suite
python test_onnx_model.py

# Sports classification
python test_onnx_model.py "NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"

# Health classification
python test_onnx_model.py "New study reveals breakthrough in cancer treatment"

# Politics classification
python test_onnx_model.py "President signs new legislation on healthcare reform"

# Technology classification
python test_onnx_model.py "Apple announces new iPhone with revolutionary AI features"

# Business classification
python test_onnx_model.py "Stock market reaches record high amid economic recovery"

# Science classification
python test_onnx_model.py "Scientists discover new species in Amazon rainforest"

# Environment classification
python test_onnx_model.py "Climate change summit begins in Paris"

# Education classification
python test_onnx_model.py "Universities announce new online learning programs"

# Entertainment classification
python test_onnx_model.py "Oscar nominations announced for this year's ceremony"

# World news classification
python test_onnx_model.py "International trade agreement signed between nations"
```

### Programmatic Usage
```python
# integration_example.py
import numpy as np
import onnxruntime as ort
import json
import time

class MulticlassClassifier:
    def __init__(self, model_path, vocab_path, scaler_path):
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        
        # Load vocabulary and label mapping
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        with open(scaler_path, 'r') as f:
            self.labels = json.load(f)
    
    def predict(self, text):
        # Tokenize and pad sequence
        tokens = self._tokenize(text)
        sequence = self._pad_sequence(tokens, max_len=30)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        result = self.session.run(None, {input_name: sequence})
        
        # Get probabilities and category
        probabilities = result[0][0]
        predicted_id = np.argmax(probabilities)
        predicted_category = self.labels[str(predicted_id)]
        
        return predicted_category, probabilities
    
    def predict_batch(self, texts):
        """Process multiple texts efficiently"""
        results = []
        for text in texts:
            category, probs = self.predict(text)
            results.append((category, probs))
        return results
    
    def _tokenize(self, text):
        words = text.lower().split()
        return [self.vocab.get(word, self.vocab.get('<OOV>', 1)) for word in words]
    
    def _pad_sequence(self, tokens, max_len):
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens.extend([0] * (max_len - len(tokens)))
        return np.array([tokens], dtype=np.int32)

# Usage example
if __name__ == "__main__":
    classifier = MulticlassClassifier(
        "model.onnx",
        "vocab.json", 
        "scaler.json"
    )
    
    # Single prediction
    category, probs = classifier.predict("France wins World Cup final")
    print(f"Category: {category}")
    
    # Batch prediction
    texts = [
        "New vaccine shows promising results",
        "Stock market hits new record",
        "Olympic games begin in Tokyo"
    ]
    results = classifier.predict_batch(texts)
    for text, (category, _) in zip(texts, results):
        print(f"'{text}' -> {category}")
```

## 🐛 Troubleshooting

### Windows Issues

**1. "'python' is not recognized as an internal or external command"**
```powershell
# Add Python to PATH manually
$env:PATH += ";C:\Users\YourUsername\AppData\Local\Programs\Python\Python310"

# Or reinstall Python with "Add to PATH" checked
```

**2. "Microsoft Visual C++ 14.0 is required"**
```powershell
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Or use pre-compiled wheels
pip install --only-binary=all onnxruntime
```

**3. "SSL: CERTIFICATE_VERIFY_FAILED"**
```powershell
# Upgrade certificates
pip install --upgrade certifi

# Or use trusted hosts
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org onnxruntime
```

**4. "Permission denied" errors****
```powershell
# Run as Administrator or use --user flag
pip install --user onnxruntime

# Or fix permissions
icacls "C:\Users\YourUsername\AppData\Local\Programs\Python" /grant YourUsername:F /t
```

### Linux Issues

**1. "python3: command not found"**
```bash
# Install Python 3
sudo apt update && sudo apt install -y python3 python3-pip

# Create symlink if needed
sudo ln -s /usr/bin/python3 /usr/bin/python
```

**2. "pip3: command not found"**
```bash
# Install pip
sudo apt install -y python3-pip

# Or download get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --user
```

**3. "error: Microsoft Visual C++ 14.0 is required" (WSL)**
```bash
# Install build tools
sudo apt install -y build-essential python3-dev

# Or use pre-compiled wheels
pip install --only-binary=all onnxruntime
```

**4. "ImportError: libgomp.so.1: cannot open shared object file"**
```bash
# Install OpenMP library
sudo apt install -y libgomp1  # Ubuntu/Debian
sudo dnf install -y libgomp   # CentOS/RHEL/Fedora
```

### macOS Issues

**1. "python3: command not found"**
```bash
# Install Python via Homebrew
brew install python@3.10

# Or use system Python (not recommended)
# macOS comes with Python 2.7, install Python 3 separately
```

**2. "SSL: CERTIFICATE_VERIFY_FAILED"**
```bash
# Update certificates
/Applications/Python\ 3.10/Install\ Certificates.command

# Or install certificates via pip
pip install --upgrade certifi
```

**3. "Apple Silicon compatibility issues"**
```bash
# Use universal2 wheels when available
pip install --platform macosx_11_0_universal2 --only-binary=all onnxruntime

# Or use Rosetta for Intel packages
arch -x86_64 pip install onnxruntime
```

**4. "xcrun: error: invalid active developer path"**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Accept license
sudo xcodebuild -license accept
```

## 📊 Expected Output

```
🤖 ONNX MULTICLASS CLASSIFIER - PYTHON IMPLEMENTATION
==================================================
🔄 Processing: NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship

💻 SYSTEM INFORMATION:
   Platform: Linux
   Processor: AMD EPYC 7763 64-Core Processor
   CPU Cores: 4 physical, 4 logical
   Total Memory: 15.6 GB
   Runtime: Python 3.9.2

📊 TOPIC CLASSIFICATION RESULTS:
⏱️  Processing Time: 510ms
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
   Total Processing Time: 510.01ms
   ┣━ Preprocessing: 0.04ms (0.0%)
   ┣━ Model Inference: 1.92ms (0.4%)
   ┗━ Post-processing: 0.05ms (0.0%)
   🧠 CPU Usage: 3.0% avg, 15.0% peak (5 readings)
   💾 Memory: 45.2MB → 46.3MB (Δ+1.1MB)
   🚀 Throughput: 2.0 texts/sec
```

## 🎯 Performance Characteristics

- **Total Time**: ~510ms (needs optimization)
- **Preprocessing**: Sequence tokenization (30 tokens)
- **Model Input**: Int32 array [1, 30]
- **Inference Engine**: ONNX Runtime CPU Provider
- **Memory Usage**: Low (~1MB additional)
- **CPU Usage**: Low (~3% average)

## 🔧 Technical Details

### Model Architecture
- **Type**: Multiclass Classification (10 categories)
- **Input**: Text string → Token sequence
- **Features**: Sequential tokens (max 30 tokens)
- **Output**: Probability distribution over 10 classes
- **Categories**: Business, Education, Entertainment, Environment, Health, Politics, Science, Sports, Technology, World

### Processing Pipeline
1. **Text Preprocessing**: Tokenization and sequence creation
2. **Vocabulary Mapping**: Convert words to token IDs
3. **Sequence Padding**: Pad/truncate to 30 tokens
4. **Model Inference**: ONNX Runtime execution
5. **Post-processing**: Softmax probabilities and category mapping

### Input Format
- **Max Sequence Length**: 30 tokens
- **Vocabulary Size**: Variable (from vocab.json)
- **Unknown Tokens**: Mapped to `<OOV>` token ID (usually 1)
- **Padding**: Zero-padding for shorter sequences

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

## 📊 Performance Analysis

### Current Performance Issues
- **Slow overall processing** (~510ms vs C++ 33ms)
- **Python overhead**: Interpreter and library loading costs
- **Optimization needed**: General performance improvements

### Performance Comparison
- **Python**: 510.01ms (this implementation)
- **C++**: 32.84ms (15x faster)
- **C**: 32.54ms (15x faster)
- **Rust**: 1.24ms (410x faster)
- **Swift**: 7.47ms (68x faster)

### Optimization Strategies
1. **Session Reuse**: Load ONNX model once for multiple predictions
2. **Batch Processing**: Process multiple texts together
3. **Caching**: Cache tokenizer and model components
4. **NumPy Optimization**: Use efficient array operations

## 📝 Notes

- **Comprehensive Categories**: Supports 10 different topic classifications
- **Easy Integration**: Simple Python API with minimal dependencies
- **Cross-Platform**: Consistent behavior across operating systems
- **Development Friendly**: Easy to modify and extend

### When to Use Python Implementation
- ✅ **Rapid Development**: Quick prototyping and experimentation
- ✅ **Data Science**: Integration with pandas, matplotlib, jupyter
- ✅ **Machine Learning**: Easy integration with scikit-learn, tensorflow
- ✅ **Scripting**: Automation and batch processing
- ✅ **Research**: Academic and research applications
- ❌ **High Performance**: Slower than compiled languages
- ❌ **Production Scale**: May need optimization for high-throughput

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 