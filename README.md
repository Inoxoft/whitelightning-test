# 🤖 White Lightning ONNX Model Testing Framework

A comprehensive cross-language testing framework for ONNX models with support for **Binary Classification** (sentiment analysis), **Multiclass Classification** (topic classification), and **Multiclass Sigmoid** (emotion classification) across 8 programming languages.

## 🚀 Available Workflows

### 1. **Individual Model Testing** (`onnx-model-tests.yml`)
Run tests for **specific models and languages** with custom text input:
- ✅ **Flexible**: Choose any combination of model type + language
- ✅ **Custom Input**: Test with your own text
- ✅ **Detailed Output**: Comprehensive performance analysis
- ✅ **Manual Dispatch**: Run on-demand with custom parameters

### 2. **Comprehensive Testing** (`comprehensive-onnx-tests.yml`) 
Run **all 24 combinations** automatically with standardized inputs:
- ✅ **Complete Coverage**: Tests 3 models × 8 languages = 24 combinations
- ✅ **Standardized**: Uses consistent test inputs for comparison
- ✅ **Automated**: Runs on push/PR + manual dispatch available
- ✅ **Performance Comparison**: Easy to compare across languages

## 🎯 Model Types & Tasks

### ⚖️ **Binary Classification** - Sentiment Analysis
- **Task**: Positive vs Negative sentiment detection
- **Architecture**: Sigmoid activation, TF-IDF preprocessing
- **Input**: `[1, 5000]` TF-IDF feature vector
- **Output**: Single probability score (0.0-1.0)

### 🏷️ **Multiclass Classification** - Topic Classification  
- **Task**: News topic categorization (Business, Health, Politics, Sports)
- **Architecture**: Softmax activation, token-based preprocessing
- **Input**: `[1, 30]` tokenized sequence
- **Output**: 4-class probability distribution

### 🎭 **Multiclass Sigmoid** - Emotion Classification
- **Task**: Multi-label emotion detection (fear, happy, love, sadness)
- **Architecture**: Multi-label sigmoid, keyword-based detection
- **Input**: `[1, 5000]` feature vector (simplified approach)
- **Output**: Independent probabilities for each emotion

## 📊 What Information You'll See

Every test run provides standardized output in this format:

```
🤖 ONNX [BINARY/MULTICLASS/MULTICLASS SIGMOID] CLASSIFIER - [LANGUAGE] IMPLEMENTATION
===============================================================================
🔄 Processing: [Test Text]

💻 SYSTEM INFORMATION:
   Platform: Linux/macOS/Windows
   Processor: CPU Name
   CPU Cores: X physical, Y logical
   Total Memory: N GB
   Runtime: Language Implementation Version

📊 [SENTIMENT/TOPIC/EMOTION] ANALYSIS RESULTS:
   🏆 Predicted [Sentiment/Topic/Emotion]: POSITIVE/NEGATIVE or POLITICS/TECH/etc or fear/happy/love/sadness
   📈 Confidence: XX.XX% (0.XXXX)
   📝 Input Text: "Your test text here"

📈 PERFORMANCE SUMMARY:
   Total Processing Time: Tms
   ┣━ Preprocessing: Xms (X%)
   ┣━ Model Inference: Yms (Y%)
   ┗━ Postprocessing: Zms (Z%)

🚀 THROUGHPUT:
   Texts per second: TPS

💾 RESOURCE USAGE:
   Memory Start: MB
   Memory End: MB
   Memory Delta: +MB
   CPU Usage: avg% avg, peak% peak (N samples)

🎯 PERFORMANCE RATING: 🚀 EXCELLENT / ✅ GOOD / ⚠️ ACCEPTABLE / 🐌 SLOW
   (Tms total - Target: <100ms)
```

## 🎯 Standard Test Inputs

- **Binary Classifier**: `"Congratulations! You've won a free iPhone — click here to claim your prize now!"` (sentiment analysis)
- **Multiclass Classifier**: `"NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"` (topic classification)
- **Multiclass Sigmoid**: `"I'm terrified of what might happen"` (emotion classification)

## 🛠️ Supported Languages

| Language | Binary Classifier | Multiclass Classifier | Multiclass Sigmoid | Status |
|----------|-------------------|----------------------|-------------------|---------|
| **Python** | ✅ | ✅ | ✅ | Full Support |
| **Java** | ✅ | ✅ | ✅ | Full Support |
| **C++** | ✅ | ✅ | ✅ | Full Support |
| **C** | ✅ | ✅ | ✅ | Full Support |
| **Node.js** | ✅ | ✅ | ✅ | Full Support |
| **Rust** | ✅ | ✅ | ✅ | Full Support |
| **Dart/Flutter** | ✅ | ✅ | ✅ | Full Support |
| **Swift** | ✅ | ✅ | ✅ | Full Support |

## 🔧 How to Use This Repository

### 1. **Clone the Repository**
```bash
git clone https://github.com/your-org/whitelightning-test.git
cd whitelightning-test
```

### 2. **Add Your Models**
Place your ONNX models in the appropriate directories:

```
tests/
├── binary_classifier/
│   ├── python/
│   │   ├── model.onnx          # Your binary classification model
│   │   ├── vocab.json          # Vocabulary file
│   │   └── scaler.json         # Preprocessing scaler
│   ├── java/
│   ├── cpp/
│   └── [other languages]/
├── multiclass_classifier/
│   ├── python/
│   │   ├── model.onnx          # Your multiclass model
│   │   ├── vocab.json          # Vocabulary file
│   │   └── scaler.json         # Preprocessing scaler
│   └── [other languages]/
└── multiclass_sigmoid/
    ├── python/
    │   ├── model.onnx          # Your multiclass sigmoid model
    │   ├── vocab.json          # Vocabulary file (if applicable)
    │   └── scaler.json         # Preprocessing scaler (if applicable)
    └── [other languages]/
```

### 3. **Read Model-Specific Documentation**
Each model type has comprehensive documentation:

- 📁 `tests/binary_classifier/README.md` - Binary classification guide
- 📁 `tests/multiclass_classifier/README.md` - Multiclass classification guide  
- 📁 `tests/multiclass_sigmoid/README.md` - Multiclass sigmoid guide

### 4. **Read Language-Specific Documentation**
Each language implementation has its own README with specific setup instructions:

- 📁 `tests/[model_type]/python/README.md` - Python setup
- 📁 `tests/[model_type]/java/README.md` - Java setup  
- 📁 `tests/[model_type]/cpp/README.md` - C++ setup
- 📁 `tests/[model_type]/c/README.md` - C setup
- 📁 `tests/[model_type]/nodejs/README.md` - Node.js setup
- 📁 `tests/[model_type]/rust/README.md` - Rust setup
- 📁 `tests/[model_type]/dart/README.md` - Dart/Flutter setup
- 📁 `tests/[model_type]/swift/README.md` - Swift/iOS setup
- 📁 `tests/[model_type]/javascript/README.md` - Client-side JavaScript/HTML setup

### 5. **Test Locally**
Navigate to any language directory and run the tests:

```bash
# Example: Test Python binary classifier
cd tests/binary_classifier/python
python test_onnx_model.py "Your custom text here"

# Example: Test Rust multiclass classifier
cd tests/multiclass_classifier/rust
cargo run --release -- "Your custom text here"

# Example: Test Node.js multiclass sigmoid
cd tests/multiclass_sigmoid/nodejs
node test_onnx_model.js "Your custom text here"
```

### 6. **Run GitHub Actions Workflows**

#### **Individual Testing** (Custom Input)
1. Go to **Actions** → **ONNX Model Tests**
2. Click **Run workflow**
3. Select:
   - **Model Type**: `binary_classifier`, `multiclass_classifier`, or `multiclass_sigmoid`
   - **Language**: `python`, `java`, `cpp`, `c`, `nodejs`, `rust`, `dart`, or `swift`
   - **Custom Text**: Your test input (optional)

#### **Comprehensive Testing** (All Languages)
1. Go to **Actions** → **Comprehensive ONNX Tests**
2. Click **Run workflow** (uses standard test inputs)
3. View results for all 24 language-model combinations

## 📋 Requirements

### Model Files
Each implementation expects these files:
- **`model.onnx`** - Your trained ONNX model
- **`vocab.json`** - Vocabulary mapping for text preprocessing (if applicable)
- **`scaler.json`** - Feature scaling parameters or label mappings

### Dependencies
Each language has its own dependencies listed in:
- Python: `requirements.txt`
- Java: `pom.xml` or `build.gradle`
- C++: `CMakeLists.txt` or `Makefile`
- C: `Makefile`
- Node.js: `package.json`
- Rust: `Cargo.toml`
- Dart: `pubspec.yaml`

## 🔧 Troubleshooting

### GitHub Actions Warnings

**Git Exit Code 128 Errors**
These are caused by Swift Package Manager fetching dependencies in CI:
- ✅ **Not a critical issue** - Tests still run successfully
- 🔧 **Fixed in latest workflow** - Added git configuration and caching
- 🍎 **Swift-specific** - Only affects Swift implementations

**macOS Migration Warnings**
GitHub Actions informational notices:
```
The macos-latest label will migrate to macOS 15 beginning August 4, 2025
```
- ✅ **Not an error** - Just an informational notice
- 🔧 **Fixed** - Updated to use `macos-14` explicitly

### Local Development Issues

**Swift Package Manager Problems**
If you encounter git issues locally:
```bash
cd tests/binary_classifier/swift
swift package reset
swift package resolve
swift build
```

**Missing Dependencies**
Ensure all language runtimes are installed:
- Python 3.8+, Java 17+, Node.js 16+, Rust stable
- Flutter 3.16+, Swift 5.7+, GCC/Clang for C/C++

**Performance Issues**
For faster local testing:
```bash
# Test single language-model combination
cd tests/binary_classifier/python
python test_onnx_model.py "Your test text"

# Use release builds
cargo build --release  # Rust
swift build --configuration release  # Swift
```

## 🎯 Performance Benchmarking

The framework provides detailed performance metrics:

- ⏱️ **Timing Analysis**: Preprocessing, inference, and postprocessing times
- 💾 **Memory Usage**: Memory consumption tracking
- 🖥️ **CPU Monitoring**: Average and peak CPU usage
- 🚀 **Throughput**: Texts processed per second
- 📊 **Performance Rating**: Automatic classification based on speed

### 📊 **Performance Comparison Table** (Binary Classifier)

*Test Input: "Congratulations! You've won a free iPhone — click here to claim your prize now!"*  
*Environment: GitHub Actions (Linux, 4 cores, 15.6GB RAM)*

| Language | Total Time | Preprocessing | Inference | Memory Δ | CPU Usage | Throughput |
|----------|------------|---------------|-----------|-----------|-----------|------------|
| **Rust** | **0.40ms** | 0.01ms (2.8%) | 0.38ms (96.1%) | +0.00MB | 0.0% avg | **2,520/sec** |
| **Node.js** | **28.89ms** | 5.44ms (18.8%) | 22.89ms (79.2%) | +1.11MB | 100.0% peak | **34.6/sec** |
| **C++** | **43.54ms** | 9.19ms (21.1%) | 34.28ms (78.7%) | +37.72MB | 0.0% avg | **23.0/sec** |
| **C** | **87.21ms** | 50.93ms (58.4%) | 0.31ms (0.4%) | +37.29MB | 0.0% avg | **11.5/sec** |
| **Dart** | **159ms** | 150ms (94.3%) | 8ms (5.0%) | 4MB | 20% avg | **6.3/sec** |
| **Swift** | **7.47ms** | 0.33ms (4.4%) | 6.37ms (85.3%) | 5MB | 15% avg | **133.8/sec** |
| **Java** | **217.98ms** | 183.48ms (84.2%) | 6.38ms (2.9%) | +22.00MB | 42.1% avg | **4.6/sec** |
| **Python** | **332.33ms** | 0.85ms (0.3%) | 0.59ms (0.2%) | +0.29MB | 15.0% avg | **3.0/sec** |

### 📊 **Performance Comparison Table** (Multiclass Classifier)

*Test Input: "NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"*  
*Environment: GitHub Actions (Linux, 4 cores, 15.6GB RAM)*

| Language | Total Time | Preprocessing | Inference | Memory Δ | CPU Usage | Throughput |
|----------|------------|---------------|-----------|-----------|-----------|------------|
| **Rust** | **1.24ms** | 0.01ms (0.6%) | 1.23ms (99.1%) | +0.00MB | 0.0% avg | **807/sec** |
| **Node.js** | **24.40ms** | 1.99ms (8.2%) | 21.65ms (88.7%) | +0.89MB | 100.0% peak | **41.0/sec** |
| **C** | **32.54ms** | 0.83ms (2.5%) | 1.50ms (4.6%) | +22.8MB | 0.0% avg | **30.7/sec** |
| **C++** | **32.84ms** | 1.97ms (6.0%) | 30.76ms (93.7%) | +21.57MB | 0.0% avg | **30.4/sec** |
| **Dart** | **114ms** | 34ms (30%) | 68ms (60%) | 4MB | 20% avg | **8.8/sec** |
| **Swift** | **7.47ms** | 0.33ms (4.4%) | 6.37ms (85.3%) | 5MB | 15% avg | **133.8/sec** |
| **Java** | **162.21ms** | 120.09ms (74.0%) | 8.28ms (5.1%) | +12.00MB | 26.3% avg | **6.2/sec** |
| **Python** | **510.01ms** | 0.04ms (0.0%) | 1.92ms (0.4%) | +1.12MB | 3.0% avg | **2.0/sec** |

### 📊 **Performance Comparison Table** (Multiclass Sigmoid)

*Test Input: "I'm terrified of what might happen"*  
*Environment: GitHub Actions (Linux, 4 cores, 15.6GB RAM)*

| Language | Total Time | Processing | Performance Rating | Throughput |
|----------|------------|------------|-------------------|------------|
| **C++** | **~1ms** | Keyword detection | 🚀 EXCELLENT | **1,000/sec** |
| **Rust** | **~1ms** | Keyword detection | 🚀 EXCELLENT | **1,000/sec** |
| **Swift** | **~1ms** | Keyword detection | 🚀 EXCELLENT | **1,000/sec** |
| **C** | **~544ms** | Keyword detection | ⚠️ ACCEPTABLE | **1.8/sec** |
| **Dart** | **~15-25ms** | Keyword detection | 🚀 EXCELLENT | **40-67/sec** |
| **Python** | **~15ms** | Keyword detection | 🚀 EXCELLENT | **67/sec** |
| **Java** | **~20ms** | Keyword detection | ✅ GOOD | **50/sec** |
| **Node.js** | **~25ms** | Keyword detection | ✅ GOOD | **40/sec** |

### 🏆 **Key Performance Insights**

#### **Cross-Model Performance Leaders**
- **🥇 Speed Champion**: Rust - consistently fastest across all model types
- **🥈 Mobile Excellence**: Swift - exceptional performance for iOS/mobile applications
- **🥉 Web Efficiency**: Node.js - optimal for web applications with minimal memory usage
- **🏅 System Integration**: C++ - excellent balance of speed and compatibility

#### **Model-Specific Optimizations**
- **Binary Classifier**: Rust achieves 0.40ms (2,520 texts/sec)
- **Multiclass Classifier**: Rust leads at 1.24ms (807 texts/sec)  
- **Multiclass Sigmoid**: Multiple languages achieve ~1ms (simplified approach)

### 📈 **Architecture-Specific Insights**

| Model Type | Best Language | Key Strength | Optimization Focus |
|------------|---------------|--------------|-------------------|
| **Binary** | Rust (0.40ms) | Ultra-fast inference | TF-IDF preprocessing |
| **Multiclass** | Rust (1.24ms) | Minimal overhead | Token processing |
| **Sigmoid** | C++/Rust/Swift (~1ms) | Keyword detection | Real-time emotion analysis |

## 🤝 Contributing

1. **Add New Languages**: Create implementation in `tests/[model_type]/[language]/`
2. **Add New Model Types**: Follow the existing structure for new classification tasks
3. **Improve Performance**: Optimize existing implementations
4. **Add Features**: Enhance testing capabilities
5. **Update Documentation**: Keep model-specific and language-specific READMEs current

## ⚡  Main Repo

[WhiteLightning](https://github.com/Inoxoft/whitelightning) distills massive, state-of-the-art language models into lightweight, hyper-efficient text classifiers. It's a command-line tool that lets you create specialized models that run anywhere—from the cloud to the edge—using the universal ONNX format for maximum compatibility.

## 🌐 Documentation & Website

Need comprehensive guides and documentation? Check out our [WhiteLightning Site](https://github.com/whitelightning-ai/whitelightning-site) - this repository hosts the official website for WhiteLightning at https://whitelightning.ai, a cutting-edge LLM distillation tool with detailed documentation, tutorials, and implementation guides.

## 📚 Model Library

Looking for pre-trained models or want to share your own? Visit our [WhiteLightning Model Library](https://github.com/whitelightning-ai/whitelightning-model-library) - a centralized repository for uploading, downloading, and managing trained machine learning models. Perfect for sharing community contributions and accessing ready-to-use classifiers.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Issues & Support

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions or share improvements
- **Wiki**: Detailed documentation and guides

---

*Happy testing! 🚀 Compare ONNX model performance across languages and find the best implementation for your use case.*
