# 🐍 Python Binary Classification ONNX Model

This directory contains a **Python implementation** for binary text classification using ONNX Runtime. The model performs **sentiment analysis** on text input using TF-IDF vectorization and a trained neural network with cross-platform support.

## 📋 System Requirements

### Minimum Requirements
- **CPU**: x86_64 or ARM64 architecture
- **RAM**: 2GB available memory
- **Storage**: 200MB free space
- **Python**: 3.8+ (recommended: 3.9, 3.10, or 3.11)
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+

### Supported Platforms
- ✅ **Windows**: 10, 11 (x64, ARM64)
- ✅ **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 9+, Fedora 30+
- ✅ **macOS**: 10.15+ (Intel & Apple Silicon)

## 📁 Directory Structure

```
python/
├── test_onnx_model.py         # Main Python implementation
├── model.onnx                 # Binary classification ONNX model
├── vocab.json                 # TF-IDF vocabulary and IDF weights
├── scaler.json                # Feature scaling parameters
├── requirements.txt           # Python dependencies
├── performance_results.json   # Performance test results
└── README.md                  # This file
```

## 🛠️ Step-by-Step Installation

### 🪟 Windows Installation

#### Step 1: Install Python
```powershell
# Option A: Download from python.org (Recommended)
# Visit: https://www.python.org/downloads/windows/
# Download Python 3.11.x (latest stable)
# During installation:
# - Check "Add Python to PATH"
# - Check "Install pip"

# Option B: Install via Microsoft Store
# Search for "Python 3.11" in Microsoft Store

# Option C: Install via winget
winget install Python.Python.3.11

# Verify installation
python --version
pip --version
```

#### Step 2: Create Virtual Environment
```powershell
# Navigate to project directory
mkdir C:\whitelightning-python
cd C:\whitelightning-python

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### Step 3: Install Dependencies
```powershell
# Install ONNX Runtime
pip install onnxruntime

# Install other dependencies
pip install numpy scikit-learn psutil

# Or install all from requirements.txt
pip install -r requirements.txt
```

#### Step 4: Copy Source Files
```powershell
# Copy your source files to the project directory
# test_onnx_model.py, model.onnx, vocab.json, scaler.json, requirements.txt
```

#### Step 5: Run the Program
```powershell
# Ensure virtual environment is activated
venv\Scripts\activate

# Run with default text
python test_onnx_model.py

# Run with custom text
python test_onnx_model.py "This product is amazing!"

# Deactivate virtual environment when done
deactivate
```

---

### 🐧 Linux Installation

#### Step 1: Install Python
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# CentOS/RHEL 8+
sudo dnf install -y python3 python3-pip

# CentOS/RHEL 7
sudo yum install -y python3 python3-pip

# Fedora
sudo dnf install -y python3 python3-pip

# Verify installation
python3 --version
pip3 --version
```

#### Step 2: Create Project Directory & Virtual Environment
```bash
# Create project directory
mkdir -p ~/whitelightning-python
cd ~/whitelightning-python

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### Step 3: Install Dependencies
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install ONNX Runtime
pip install onnxruntime

# Install other dependencies
pip install numpy scikit-learn psutil

# Or install all from requirements.txt
pip install -r requirements.txt
```

#### Step 4: Copy Source Files
```bash
# Copy your source files to the project directory
# test_onnx_model.py, model.onnx, vocab.json, scaler.json, requirements.txt
```

#### Step 5: Run the Program
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Run with default text
python test_onnx_model.py

# Run with custom text
python test_onnx_model.py "This product is amazing!"

# Deactivate virtual environment when done
deactivate
```

---

### 🍎 macOS Installation

#### Step 1: Install Python
```bash
# Option A: Install via Homebrew (Recommended)
# Install Homebrew first if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Add to PATH (if needed)
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Option B: Download from python.org
# Visit: https://www.python.org/downloads/macos/
# Download and install Python 3.11.x

# Verify installation
python3 --version
pip3 --version
```

#### Step 2: Create Project Directory & Virtual Environment
```bash
# Create project directory
mkdir -p ~/whitelightning-python
cd ~/whitelightning-python

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### Step 3: Install Dependencies
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# For Apple Silicon Macs, you might need specific versions
# Install ONNX Runtime
pip install onnxruntime

# Install other dependencies
pip install numpy scikit-learn psutil

# Or install all from requirements.txt
pip install -r requirements.txt
```

#### Step 4: Copy Source Files
```bash
# Copy your source files to the project directory
# test_onnx_model.py, model.onnx, vocab.json, scaler.json, requirements.txt
```

#### Step 5: Run the Program
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Run with default text
python test_onnx_model.py

# Run with custom text
python test_onnx_model.py "This product is amazing!"

# Deactivate virtual environment when done
deactivate
```

## 🔧 Advanced Configuration

### Requirements.txt
```txt
onnxruntime>=1.16.0
numpy>=1.21.0
scikit-learn>=1.0.0
psutil>=5.8.0
```

### Virtual Environment Management
```bash
# Create environment with specific Python version
python3.11 -m venv venv

# List installed packages
pip list

# Save current environment
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Remove environment
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows
```

### Performance Optimization
```bash
# Install optimized NumPy (Intel MKL)
pip install intel-numpy

# Install ONNX Runtime with GPU support (if available)
pip uninstall onnxruntime
pip install onnxruntime-gpu

# Install additional performance libraries
pip install numba cython
```

## 🎯 Usage Examples

### Basic Usage
```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Default test
python test_onnx_model.py

# Positive sentiment
python test_onnx_model.py "I love this product! It's amazing!"

# Negative sentiment
python test_onnx_model.py "This is terrible and disappointing."

# Neutral sentiment
python test_onnx_model.py "The product is okay, nothing special."
```

### Batch Testing
```bash
# Test multiple sentences
python test_onnx_model.py "Great service and fast delivery!"
python test_onnx_model.py "Poor quality, would not recommend."
python test_onnx_model.py "Average product, meets expectations."
```

### Programmatic Usage
```python
# test_script.py
import sys
sys.path.append('.')
from test_onnx_model import BinaryClassifier

# Initialize classifier
classifier = BinaryClassifier('model.onnx', 'vocab.json', 'scaler.json')

# Test multiple texts
texts = [
    "This product is amazing!",
    "Terrible quality, very disappointed.",
    "Good value for money.",
    "Not worth the price."
]

for text in texts:
    result = classifier.predict(text)
    sentiment = "Positive" if result > 0.5 else "Negative"
    print(f"'{text}' → {sentiment} ({result:.4f})")
```

## 🐛 Troubleshooting

### Windows Issues

**1. "'python' is not recognized as an internal or external command"**
```powershell
# Add Python to PATH manually
$env:PATH += ";C:\Python311;C:\Python311\Scripts"

# Or reinstall Python with "Add to PATH" checked
```

**2. "Microsoft Visual C++ 14.0 is required"**
```powershell
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Or install via pip
pip install --upgrade setuptools wheel
```

**3. "Access is denied" when installing packages**
```powershell
# Run PowerShell as Administrator
# Or install in user directory
pip install --user onnxruntime numpy scikit-learn psutil
```

**4. "SSL certificate verify failed"**
```powershell
# Upgrade certificates
pip install --upgrade certifi

# Or use trusted hosts
pip install --trusted-host pypi.org --trusted-host pypi.python.org onnxruntime
```

### Linux Issues

**1. "python3: command not found"**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip

# CentOS/RHEL
sudo yum install python3 python3-pip
```

**2. "pip3: command not found"**
```bash
# Install pip
sudo apt install python3-pip  # Ubuntu/Debian
sudo yum install python3-pip  # CentOS/RHEL

# Or use ensurepip
python3 -m ensurepip --upgrade
```

**3. "Permission denied" when installing packages**
```bash
# Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Or install in user directory
pip3 install --user onnxruntime numpy scikit-learn psutil
```

**4. "Failed building wheel for some-package"**
```bash
# Install development tools
sudo apt install build-essential python3-dev  # Ubuntu/Debian
sudo yum groupinstall "Development Tools" python3-devel  # CentOS/RHEL

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel
```

### macOS Issues

**1. "python3: command not found"**
```bash
# Install Python via Homebrew
brew install python@3.11

# Or download from python.org
# https://www.python.org/downloads/macos/
```

**2. "SSL: CERTIFICATE_VERIFY_FAILED"**
```bash
# Update certificates
/Applications/Python\ 3.11/Install\ Certificates.command

# Or install certifi
pip3 install --upgrade certifi
```

**3. "Architecture compatibility issues (Apple Silicon)"**
```bash
# Check Python architecture
python3 -c "import platform; print(platform.machine())"

# Install universal packages
pip install --no-deps onnxruntime
pip install numpy scikit-learn psutil

# Or use conda for better Apple Silicon support
brew install miniconda
conda install onnxruntime numpy scikit-learn psutil
```

**4. "Command line tools not installed"**
```bash
# Install Xcode command line tools
xcode-select --install
```

## 📊 Expected Output

```
🤖 ONNX BINARY CLASSIFIER - PYTHON IMPLEMENTATION
===============================================
🔄 Processing: "This product is amazing!"

💻 SYSTEM INFORMATION:
   Platform: macOS 14.0 (Darwin)
   Processor: Apple M2 Pro
   CPU Cores: 12 physical, 12 logical
   Total Memory: 32.0 GB
   Runtime: Python 3.11.2

📊 SENTIMENT ANALYSIS RESULTS:
   🏆 Predicted Sentiment: Positive ✅
   📈 Confidence: 99.8% (0.9982)
   📝 Input Text: "This product is amazing!"

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 332.33ms
   ┣━ Preprocessing: 0.85ms (0.3%)
   ┣━ Model Inference: 0.59ms (0.2%)
   ┗━ Postprocessing: 0.12ms (0.0%)

🚀 THROUGHPUT:
   Texts per second: 3.0

💾 RESOURCE USAGE:
   Memory Start: 45.20 MB
   Memory End: 45.49 MB
   Memory Delta: +0.29 MB
   CPU Usage: 15.0% avg, 25.0% peak (5 samples)
```

## 🎯 Performance Characteristics

- **Total Time**: ~332ms (needs optimization)
- **Preprocessing**: TF-IDF vectorization (5000 features) 
- **Model Input**: Float32 array [1, 5000]
- **Inference Engine**: ONNX Runtime CPU Provider
- **Memory Usage**: Low (~0.3MB additional)
- **CPU Usage**: Moderate (~15% average)

## 🔧 Technical Details

### Model Architecture
- **Type**: Binary Classification  
- **Input**: Text string
- **Features**: TF-IDF vectors (5000 dimensions)
- **Output**: Probability [0.0 - 1.0]
- **Threshold**: >0.5 = Positive, ≤0.5 = Negative

### Processing Pipeline
1. **Text Preprocessing**: Tokenization and lowercasing
2. **TF-IDF Vectorization**: Using scikit-learn and vocabulary
3. **Feature Scaling**: Standardization using mean/scale parameters  
4. **Model Inference**: ONNX Runtime execution
5. **Post-processing**: Probability interpretation

## 📊 Performance Analysis

### Current Performance Issues
- **Slow overall processing** (~332ms vs C++ 44ms)
- **Potential bottlenecks**: Python overhead, library initialization
- **Optimization needed**: Preprocessing and model loading

### Optimization Strategies
1. **Session Reuse**: Load ONNX model once for multiple predictions
2. **Vectorization**: Use NumPy operations instead of loops
3. **Caching**: Cache TF-IDF vectorizer and scaler
4. **Batch Processing**: Process multiple texts together

## 🚀 Integration Example

```python
import numpy as np
import onnxruntime as ort
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import time

class BinaryClassifier:
    def __init__(self, model_path, vocab_path, scaler_path):
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        
        # Load vocabulary and scaler
        with open(vocab_path, 'r') as f:
            self.vocab_data = json.load(f)
        with open(scaler_path, 'r') as f:
            self.scaler_data = json.load(f)
    
    def predict(self, text):
        # Preprocess text
        features = self._preprocess_text(text)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        result = self.session.run(None, {input_name: features})
        
        return result[0][0][0]  # Extract probability
    
    def predict_batch(self, texts):
        """Predict multiple texts at once for better performance"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
    
    def _preprocess_text(self, text):
        # TF-IDF vectorization logic here
        # This is a simplified version - implement full TF-IDF logic
        pass

# Usage
if __name__ == "__main__":
    classifier = BinaryClassifier('model.onnx', 'vocab.json', 'scaler.json')
    
    # Single prediction
    probability = classifier.predict("Your text here")
    sentiment = "Positive" if probability > 0.5 else "Negative"
    print(f"Sentiment: {sentiment} ({probability:.4f})")
    
    # Batch prediction
    texts = [
        "This is great!",
        "Not very good.",
        "Average quality."
    ]
    
    start_time = time.time()
    results = classifier.predict_batch(texts)
    end_time = time.time()
    
    print(f"\nBatch processing time: {(end_time - start_time)*1000:.2f}ms")
    for text, prob in zip(texts, results):
        sentiment = "Positive" if prob > 0.5 else "Negative"
        print(f"'{text}' → {sentiment} ({prob:.4f})")
```

## 📊 Model Files

- **`model.onnx`**: Binary classification neural network (~10MB)
- **`vocab.json`**: TF-IDF vocabulary mapping and IDF weights  
- **`scaler.json`**: Feature standardization parameters (mean/scale)
- **`requirements.txt`**: Python package dependencies

## 🔄 Testing

```bash
# Run with different test cases
python test_onnx_model.py "This is a great product!"
python test_onnx_model.py "Click here to win $1000 now!"
python test_onnx_model.py "Thank you for your service."

# Test with file input
echo "This product is amazing!" | python test_onnx_model.py

# Benchmark performance
python -c "
import time
from test_onnx_model import BinaryClassifier

classifier = BinaryClassifier('model.onnx', 'vocab.json', 'scaler.json')
text = 'This is a test message for performance measurement.'

# Warmup
for _ in range(5):
    classifier.predict(text)

# Benchmark
start = time.time()
for _ in range(100):
    classifier.predict(text)
end = time.time()

print(f'Average time per prediction: {(end-start)*10:.2f}ms')
print(f'Throughput: {100/(end-start):.1f} predictions/sec')
"
```

## 📝 Notes

- Python implementation shows good accuracy but needs performance optimization
- Consider this implementation for prototyping and development
- For production use, consider C++ or Rust implementations for better performance
- Memory usage is efficient, but processing time needs improvement

### When to Use Python Implementation
- ✅ **Prototyping**: Quick development and testing
- ✅ **Low Volume**: Occasional text classification
- ✅ **Integration**: Existing Python ecosystems
- ✅ **Data Science**: Research and experimentation
- ❌ **High Throughput**: Real-time or batch processing needs

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 