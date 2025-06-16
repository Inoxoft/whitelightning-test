# Multiclass Classifier - Java Implementation

🤖 **ONNX Multiclass Classifier for News Category Classification**

This Java implementation provides comprehensive news text classification using ONNX Runtime with advanced performance monitoring and system information collection.

## 🚀 Features

- **Token-based Text Preprocessing** - Converts text to 30-token sequences with vocabulary mapping
- **ONNX Runtime Integration** - Uses Microsoft ONNX Runtime for Java
- **Dynamic Input Detection** - Automatically detects model input/output names
- **Comprehensive Performance Monitoring** - System info, timing breakdown, CPU/memory tracking
- **News Classification** - 4-class classification (health, politics, sports, world)
- **Benchmarking Mode** - Performance testing with detailed statistics

## 📋 Requirements

- **Java 17+** (OpenJDK or Oracle JDK)
- **Maven 3.6+** for dependency management
- **Model Files**:
  - `model.onnx` - ONNX multiclass classification model
  - `vocab.json` - Token vocabulary mapping (word → token_id)
  - `scaler.json` - Label mapping (class_index → class_name)

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
mvn exec:java -Dexec.args="\"Your custom news text here\""

# Run performance benchmark
mvn exec:java -Dexec.args="--benchmark 100"
```

## 📊 Usage Examples

### Basic News Classification
```bash
mvn exec:java -Dexec.args="\"France Defeats Argentina in Thrilling World Cup Final\""
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
- **Preprocessing** - Text tokenization and padding
- **Model Inference** - ONNX model execution
- **Postprocessing** - Result interpretation and label mapping
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
- **Shape**: [1, 30] (batch_size=1, sequence_length=30)
- **Type**: Int32
- **Preprocessing**: Tokenization with vocabulary mapping and zero-padding

### Output Format
- **Shape**: [4] (4 class probabilities)
- **Type**: Float32
- **Classes**: 
  - **0**: health
  - **1**: politics  
  - **2**: sports
  - **3**: world

### Text Preprocessing
1. **Tokenization** - Split text into words and convert to lowercase
2. **Vocabulary Mapping** - Convert words to token IDs using vocab.json
3. **OOV Handling** - Unknown words mapped to `<OOV>` token (ID: 1)
4. **Padding** - Sequences padded/truncated to exactly 30 tokens with zeros

## 🧪 Testing

The implementation includes multiple test scenarios:

1. **Sports**: "France Defeats Argentina in Thrilling World Cup Final"
2. **Health**: "New Healthcare Policy Announced by Government"
3. **Politics**: "Stock Market Reaches Record High"
4. **World**: "Climate Change Summit Begins in Paris"
5. **Science**: "Scientists Discover New Species in Amazon"

## 🔧 CI/CD Integration

The implementation is designed for GitHub Actions:

- **Safe CI Execution** - Graceful handling when model files are missing
- **Build Verification** - Confirms compilation and startup
- **Artifact Upload** - Saves compiled classes and logs
- **Custom Text Testing** - Supports parameterized text input

## 📝 Example Output

```
🤖 ONNX MULTICLASS CLASSIFIER - JAVA IMPLEMENTATION
==================================================

🔄 Processing: France Defeats Argentina in Thrilling World Cup Final

💻 SYSTEM INFORMATION:
   Platform: Linux
   Processor: amd64
   CPU Cores: 4
   Total Memory: 8.0 GB
   Runtime: Java Implementation
   Java Version: 17.0.7
   ONNX Runtime Version: 1.22.0

📊 MULTICLASS CLASSIFICATION RESULTS:
   🏆 Predicted Category: sports
   📈 Confidence: 94.67% (0.9467)
   📝 Input Text: "France Defeats Argentina in Thrilling World Cup Final"
   📋 All Class Probabilities:
      health: 0.0123 (1.2%)
      politics: 0.0234 (2.3%)
      sports: 0.9467 (94.7%)
      world: 0.0176 (1.8%)

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 38.45ms
   ┣━ Preprocessing: 8.23ms (21.4%)
   ┣━ Model Inference: 25.67ms (66.8%)
   ┗━ Postprocessing: 4.55ms (11.8%)

🚀 THROUGHPUT:
   Texts per second: 26.0

💾 RESOURCE USAGE:
   Memory Start: 142.34 MB
   Memory End: 147.89 MB
   Memory Delta: +5.55 MB
   CPU Usage: 12.8% avg, 38.9% peak (10 samples)

🎯 PERFORMANCE RATING: 🚀 EXCELLENT
   (38.5ms total - Target: <100ms)
```

## ⚠️ Known Issues

**Model Training Bias**: The current model has documented training issues where it tends to classify most text as "sports" regardless of content. This is due to corrupted/mislabeled training data where the model learned artificial token patterns rather than real text semantics.

**Recommendations**:
- Model needs retraining with properly balanced and labeled dataset
- Current implementation serves as infrastructure testing
- Classification results should be interpreted with caution

## 🚀 GitHub Actions

To run in GitHub Actions:

1. **Model Type**: Select `multiclass_classifier(News classifier)`
2. **Language**: Select `java`
3. **Custom Text**: (Optional) Enter your news text to classify

The workflow will automatically build, test, and benchmark the implementation!

## 🔄 Comparison with Other Languages

| Language | Preprocessing | Inference | Total Time | Memory Usage |
|----------|---------------|-----------|------------|--------------|
| **Java** | ~8ms | ~26ms | ~38ms | ~6MB |
| **Python** | ~15ms | ~35ms | ~55ms | ~12MB |
| **C++** | ~5ms | ~20ms | ~28ms | ~3MB |
| **C** | ~4ms | ~18ms | ~25ms | ~2MB |

*Performance may vary based on system specifications and model complexity.* 