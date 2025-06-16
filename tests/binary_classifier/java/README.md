# Binary Classifier - Java Implementation

🤖 **ONNX Binary Classifier for Sentiment Analysis**

This Java implementation provides comprehensive sentiment analysis using ONNX Runtime with advanced performance monitoring and system information collection.

## 🚀 Features

- **TF-IDF Text Preprocessing** - Converts text to 5000-dimensional feature vectors
- **ONNX Runtime Integration** - Uses Microsoft ONNX Runtime for Java
- **Dynamic Input Detection** - Automatically detects model input/output names
- **Comprehensive Performance Monitoring** - System info, timing breakdown, CPU/memory tracking
- **Sentiment Analysis** - Binary classification (Positive/Negative)
- **Benchmarking Mode** - Performance testing with detailed statistics

## 📋 Requirements

- **Java 17+** (OpenJDK or Oracle JDK)
- **Maven 3.6+** for dependency management
- **Model Files**:
  - `model.onnx` - ONNX binary classification model
  - `vocab.json` - TF-IDF vocabulary and IDF weights
  - `scaler.json` - Feature scaling parameters (mean/scale arrays)

## 🔧 Dependencies

- **ONNX Runtime Java** (1.22.0) - Model inference
- **Jackson** (2.15.2) - JSON processing
- **SLF4J + Logback** - Logging framework
- **JUnit 5** - Testing framework

## 🏗️ Building

```bash
# Compile the project
mvn clean compile

# Run with default test texts
mvn exec:java

# Test custom text
mvn exec:java -Dexec.args="\"Your custom text here\""

# Run performance benchmark
mvn exec:java -Dexec.args="--benchmark 100"
```

## 📊 Usage Examples

### Basic Sentiment Analysis
```bash
mvn exec:java -Dexec.args="\"This product is amazing!\""
```

### Performance Benchmarking
```bash
mvn exec:java -Dexec.args="--benchmark 50"
```

### Default Test Suite
```bash
mvn exec:java
```

## 📈 Performance Monitoring

The implementation provides comprehensive performance metrics:

### System Information
- Platform and processor details
- CPU cores (physical/logical)
- Total memory and Java version
- ONNX Runtime version

### Timing Breakdown
- **Preprocessing** - Text tokenization and TF-IDF conversion
- **Model Inference** - ONNX model execution
- **Postprocessing** - Result interpretation
- **Total Time** - End-to-end processing

### Resource Usage
- Memory consumption (start/end/delta)
- CPU usage monitoring (average/peak)
- Throughput calculations (texts per second)

### Performance Classification
- 🚀 **EXCELLENT** - < 50ms
- ✅ **GOOD** - 50-100ms  
- ⚠️ **ACCEPTABLE** - 100-200ms
- ❌ **POOR** - > 200ms

## 🔍 Model Details

### Input Format
- **Shape**: [1, 5000] (batch_size=1, features=5000)
- **Type**: Float32
- **Preprocessing**: TF-IDF vectorization with scaling

### Output Format
- **Shape**: [1] (single probability value)
- **Type**: Float32
- **Range**: 0.0 to 1.0 (probability of positive sentiment)
- **Threshold**: > 0.5 = Positive, ≤ 0.5 = Negative

## 🧪 Testing

The implementation includes multiple test scenarios:

1. **Positive Sentiment**: "This product is amazing!"
2. **Negative Sentiment**: "Terrible service, would not recommend."
3. **Neutral Sentiment**: "It's okay, nothing special."
4. **Strong Positive**: "Best purchase ever!"
5. **Strong Negative**: "The product broke after just two days — total waste of money."

## 🔧 CI/CD Integration

The implementation is designed for GitHub Actions:

- **Safe CI Execution** - Graceful handling when model files are missing
- **Build Verification** - Confirms compilation and startup
- **Artifact Upload** - Saves compiled classes and logs
- **Custom Text Testing** - Supports parameterized text input

## 📝 Example Output

```
🤖 ONNX BINARY CLASSIFIER - JAVA IMPLEMENTATION
===============================================

🔄 Processing: This product is amazing!

💻 SYSTEM INFORMATION:
   Platform: Linux
   Processor: amd64
   CPU Cores: 4
   Total Memory: 8.0 GB
   Runtime: Java Implementation
   Java Version: 17.0.7
   ONNX Runtime Version: 1.22.0

📊 SENTIMENT ANALYSIS RESULTS:
   🏆 Predicted Sentiment: Positive
   📈 Confidence: 87.34% (0.8734)
   📝 Input Text: "This product is amazing!"

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 45.23ms
   ┣━ Preprocessing: 12.45ms (27.5%)
   ┣━ Model Inference: 28.67ms (63.4%)
   ┗━ Postprocessing: 4.11ms (9.1%)

🚀 THROUGHPUT:
   Texts per second: 22.1

💾 RESOURCE USAGE:
   Memory Start: 156.78 MB
   Memory End: 162.34 MB
   Memory Delta: +5.56 MB
   CPU Usage: 15.2% avg, 45.7% peak (12 samples)

🎯 PERFORMANCE RATING: ✅ GOOD
   (45.2ms total - Target: <100ms)
```

## 🚀 GitHub Actions

To run in GitHub Actions:

1. **Model Type**: Select `binary_classifier(Customer feedback classifier)`
2. **Language**: Select `java`
3. **Custom Text**: (Optional) Enter your text to analyze

The workflow will automatically build, test, and benchmark the implementation! 