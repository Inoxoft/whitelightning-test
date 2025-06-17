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
Run **all 14 combinations** automatically with standardized inputs:
- ✅ **Complete Coverage**: Tests 2 models × 7 languages = 14 combinations
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

- **Binary Classifier**: `"It was very bad purchase"` (sentiment analysis)
- **Multiclass Classifier**: `"President signs new legislation on healthcare reform"` (topic classification)

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
| **Swift** | ⚠️ | ⚠️ | Coming Soon |

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
