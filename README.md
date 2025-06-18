# 🤖 White Lightning ONNX Model Testing Framework

A comprehensive cross-language testing framework for ONNX models with support for **Binary Classification** (sentiment analysis) and **Multiclass Classification** (topic classification) across 7 programming languages.

## 🚀 Available Workflows

### 1. **Individual Model Testing** (`onnx-model-tests.yml`)
Run tests for **specific models and languages** with custom text input:
- ✅ **Flexible**: Choose any combination of model type + language
- ✅ **Custom Input**: Test with your own text
- ✅ **Detailed Output**: Comprehensive performance analysis
- ✅ **Manual Dispatch**: Run on-demand with custom parameters

### 2. **Comprehensive Testing** (`comprehensive-onnx-tests.yml`) 
Run **all 16 combinations** automatically with standardized inputs:
- ✅ **Complete Coverage**: Tests 2 models × 8 languages = 16 combinations
- ✅ **Standardized**: Uses consistent test inputs for comparison
- ✅ **Automated**: Runs on push/PR + manual dispatch available
- ✅ **Performance Comparison**: Easy to compare across languages

## 📊 What Information You'll See

Every test run provides standardized output in this format:

```
🤖 ONNX [BINARY/MULTICLASS] CLASSIFIER - [LANGUAGE] IMPLEMENTATION
===============================================================
🔄 Processing: [Test Text]

💻 SYSTEM INFORMATION:
   Platform: Linux/macOS/Windows
   Processor: CPU Name
   CPU Cores: X physical, Y logical
   Total Memory: N GB
   Runtime: Language Implementation Version

📊 [SENTIMENT/TOPIC] ANALYSIS RESULTS:
   🏆 Predicted [Sentiment/Topic]: POSITIVE/NEGATIVE or POLITICS/TECH/etc
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

- **Binary Classifier**: `"It was very bad purchase"` (spam analysis)
- **Multiclass Classifier**: `"President signs new legislation on healthcare reform"` (news topic classification)

## 🛠️ Supported Languages

| Language | Binary Classifier | Multiclass Classifier | Status |
|----------|-------------------|----------------------|---------|
| **Python** | ✅ | ✅ | Full Support |
| **Java** | ✅ | ✅ | Full Support |
| **C++** | ✅ | ✅ | Full Support |
| **C** | ✅ | ✅ | Full Support |
| **Node.js** | ✅ | ✅ | Full Support |
| **Rust** | ✅ | ✅ | Full Support |
| **Dart/Flutter** | ✅ | ✅ | Full Support |
| **Swift** | ✅ | ✅ | Full Support |

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
└── multiclass_classifier/
    ├── python/
    │   ├── model.onnx          # Your multiclass model
    │   ├── vocab.json          # Vocabulary file
    │   └── scaler.json         # Preprocessing scaler
    └── [other languages]/
```

### 3. **Read Language-Specific Documentation**
Each language implementation has its own README with specific setup instructions:

- 📁 `tests/binary_classifier/python/README.md` - Python setup
- 📁 `tests/binary_classifier/java/README.md` - Java setup  
- 📁 `tests/binary_classifier/cpp/README.md` - C++ setup
- 📁 `tests/binary_classifier/c/README.md` - C setup
- 📁 `tests/binary_classifier/nodejs/README.md` - Node.js setup
- 📁 `tests/binary_classifier/rust/README.md` - Rust setup
- 📁 `tests/binary_classifier/dart/README.md` - Dart/Flutter setup
- 📁 `tests/binary_classifier/swift/README.md` - Swift/iOS setup

*The same structure exists for `multiclass_classifier/`*

### 4. **Test Locally**
Navigate to any language directory and run the tests:

```bash
# Example: Test Python implementation
cd tests/binary_classifier/python
python test_onnx_model.py "Your custom text here"

# Example: Test Rust implementation  
cd tests/binary_classifier/rust
cargo run --release -- "Your custom text here"

# Example: Test Node.js implementation
cd tests/binary_classifier/nodejs
node test_onnx_model.js "Your custom text here"
```

### 5. **Run GitHub Actions Workflows**

#### **Individual Testing** (Custom Input)
1. Go to **Actions** → **ONNX Model Tests**
2. Click **Run workflow**
3. Select:
   - **Model Type**: `binary_classifier` or `multiclass_classifier`
   - **Language**: `python`, `java`, `cpp`, `c`, `nodejs`, `rust`, or `dart`
   - **Custom Text**: Your test input (optional)

#### **Comprehensive Testing** (All Languages)
1. Go to **Actions** → **Comprehensive ONNX Tests**
2. Click **Run workflow** (uses standard test inputs)
3. View results for all 14 language-model combinations

## 📋 Requirements

### Model Files
Each implementation expects these files:
- **`model.onnx`** - Your trained ONNX model
- **`vocab.json`** - Vocabulary mapping for text preprocessing
- **`scaler.json`** - Feature scaling parameters

### Dependencies
Each language has its own dependencies listed in:
- Python: `requirements.txt`
- Java: `pom.xml`
- C++: `CMakeLists.txt` or `Makefile`
- C: `Makefile`
- Node.js: `package.json`
- Rust: `Cargo.toml`
- Dart: `pubspec.yaml`

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
| **Dart** | **144ms** | - | - | - | - | **7.0/sec** |
| **Java** | **217.98ms** | 183.48ms (84.2%) | 6.38ms (2.9%) | +22.00MB | 42.1% avg | **4.6/sec** |
| **Python** | **332.33ms** | 0.85ms (0.3%) | 0.59ms (0.2%) | +0.29MB | 15.0% avg | **3.0/sec** |

### 🏆 **Key Performance Insights**

- **🥇 Fastest**: Rust (0.40ms) - 830x faster than slowest
- **🥈 Most Efficient**: Node.js (28.89ms) with lowest memory usage
- **🥉 Best Balance**: C++ (43.54ms) - excellent speed with reasonable memory
- **🔧 Optimization Needed**: Java (preprocessing bottleneck) and Python (overall performance)

### 📈 **Language-Specific Analysis**

| Metric | Best Performer | Worst Performer | Insight |
|--------|----------------|-----------------|---------|
| **Total Speed** | Rust (0.40ms) | Python (332ms) | Rust is 830x faster |
| **Preprocessing** | Rust (0.01ms) | Java (183ms) | Java needs TF-IDF optimization |
| **Inference** | C (0.31ms) | C++ (34ms) | C has optimized inference |
| **Memory Usage** | Node.js (+1MB) | C++ (+38MB) | Node.js most memory efficient |
| **CPU Efficiency** | Rust/C++ (0%) | Node.js (100%) | Native code more efficient |
| **Throughput** | Rust (2,520/s) | Python (3/s) | Rust handles 840x more volume |

### 📊 **Performance Comparison Table** (Multiclass Classifier)

*Test Input: "NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"*  
*Environment: GitHub Actions (Linux, 4 cores, 15.6GB RAM)*

| Language | Total Time | Preprocessing | Inference | Memory Δ | CPU Usage | Throughput |
|----------|------------|---------------|-----------|-----------|-----------|------------|
| **Rust** | **1.24ms** | 0.01ms (0.6%) | 1.23ms (99.1%) | +0.00MB | 0.0% avg | **807/sec** |
| **Node.js** | **24.40ms** | 1.99ms (8.2%) | 21.65ms (88.7%) | +0.89MB | 100.0% peak | **41.0/sec** |
| **C** | **32.54ms** | 0.83ms (2.5%) | 1.50ms (4.6%) | +22.8MB | 0.0% avg | **30.7/sec** |
| **C++** | **32.84ms** | 1.97ms (6.0%) | 30.76ms (93.7%) | +21.57MB | 0.0% avg | **30.4/sec** |
| **Dart** | **124ms** | ~37ms (30%) | ~74ms (60%) | ~4MB | ~20% avg | **8.1/sec** |
| **Java** | **162.21ms** | 120.09ms (74.0%) | 8.28ms (5.1%) | +12.00MB | 26.3% avg | **6.2/sec** |
| **Python** | **510.01ms** | 0.04ms (0.0%) | 1.92ms (0.4%) | +1.12MB | 3.0% avg | **2.0/sec** |

### 🏆 **Key Performance Insights** (Multiclass)

- **🥇 Speed Champion**: Rust (1.24ms) - 410x faster than slowest
- **🥈 Efficiency Leader**: Node.js (24.40ms) with minimal memory footprint  
- **🥉 Native Excellence**: C and C++ both under 33ms with excellent performance
- **🔧 Optimization Targets**: Java (preprocessing bottleneck) and Python (overall performance)

### 📈 **Multiclass vs Binary Performance**

| Language | Binary Time | Multiclass Time | Speedup/Slowdown | Best Use Case |
|----------|-------------|-----------------|-------------------|---------------|
| **Rust** | 0.40ms | 1.24ms | 3.1x slower | High-volume processing |
| **Node.js** | 28.89ms | 24.40ms | 1.2x faster | Web applications |
| **C++** | 43.54ms | 32.84ms | 1.3x faster | System integration |
| **C** | 87.21ms | 32.54ms | 2.7x faster | Embedded systems |
| **Java** | 217.98ms | 162.21ms | 1.3x faster | Enterprise applications |
| **Python** | 332.33ms | 510.01ms | 1.5x slower | Prototyping/research |
| **Dart** | 144ms | 124ms | 1.2x faster | Mobile applications |

### 🔍 **Architecture-Specific Insights**

| Model Type | Preprocessing Bottleneck | Inference Champion | Memory Efficient |
|------------|-------------------------|-------------------|------------------|
| **Binary** | Java (84% preprocessing) | C (0.31ms) | Node.js (+1MB) |
| **Multiclass** | Java (74% preprocessing) | C (1.50ms) | Node.js (+0.89MB) |
| **Winner** | C consistently faster | C wins both | Node.js wins both |

### 🎯 **Language Selection Guide**

| Priority | Binary Classifier | Multiclass Classifier | Recommendation |
|----------|------------------|----------------------|----------------|
| **Ultra-High Performance** | Rust (0.40ms) | Rust (1.24ms) | Choose Rust for maximum speed |
| **Web Development** | Node.js (28.89ms) | Node.js (24.40ms) | Node.js excels in both |
| **System Integration** | C++ (43.54ms) | C++ (32.84ms) | C++ solid for native apps |
| **Mobile Apps** | Dart (144ms) | Dart (124ms) | Dart/Flutter consistent |
| **Enterprise** | Java (218ms) | Java (162ms) | Java needs preprocessing optimization |
| **Research/Prototyping** | Python (332ms) | Python (510ms) | Python needs performance work |

## 🤝 Contributing

1. **Add New Languages**: Create implementation in `tests/[model_type]/[language]/`
2. **Improve Performance**: Optimize existing implementations
3. **Add Features**: Enhance testing capabilities
4. **Update Documentation**: Keep language-specific READMEs current

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Issues & Support

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions or share improvements
- **Wiki**: Detailed documentation and guides

---

*Happy testing! 🚀 Compare ONNX model performance across languages and find the best implementation for your use case.*
