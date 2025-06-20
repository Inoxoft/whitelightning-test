# Swift ONNX Multiclass Classifier

This directory contains a Swift implementation of the ONNX multiclass text classifier using Microsoft's ONNX Runtime.

## Features

- **Real ONNX Runtime Integration**: Uses Microsoft's official ONNX Runtime Swift package
- **Complete TF-IDF Pipeline**: Full text preprocessing with tokenization, TF-IDF vectorization, and scaling
- **Dynamic Vocabulary**: Automatically detects vocabulary size from JSON files
- **Performance Monitoring**: Detailed timing metrics and resource usage tracking
- **System Information**: Comprehensive system detection and reporting
- **Memory Management**: Tracks memory usage throughout the inference process

## Requirements

- macOS 12.0 or later
- Swift 5.7 or later
- Xcode or Swift command line tools

## Installation

1. Navigate to the SwiftONNXRunner directory:
```bash
cd tests/multiclass_classifier/SwiftONNXRunner
```

2. Build the project:
```bash
swift build -c release
```

## Usage

### Basic Usage
```bash
swift run SwiftONNXRunner
```

### Custom Text Input
```bash
swift run SwiftONNXRunner "Your text to classify here"
```

### Examples
```bash
# Sports classification
swift run SwiftONNXRunner "NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"

# Technology classification
swift run SwiftONNXRunner "Apple announces new iPhone with advanced AI capabilities"

# Politics classification
swift run SwiftONNXRunner "President signs new legislation on healthcare reform"

# Business classification
swift run SwiftONNXRunner "Stock market reaches new highs as tech companies report strong earnings"

# Entertainment classification
swift run SwiftONNXRunner "New Marvel movie breaks box office records in opening weekend"
```

## Project Structure

```
SwiftONNXRunner/
├── Package.swift                 # Swift Package Manager configuration
├── SwiftONNXRunner/
│   └── main.swift               # Main implementation
├── model.onnx                   # ONNX model file
├── vocab.json                   # TF-IDF vocabulary and weights
├── scaler.json                  # Feature scaling parameters
└── README.md                    # This file
```

## Implementation Details

### Text Preprocessing Pipeline
1. **Tokenization**: Splits text into lowercase tokens
2. **TF-IDF Vectorization**: Converts tokens to numerical features using vocabulary
3. **Feature Scaling**: Applies standardization using pre-computed mean and scale values
4. **Validation**: Handles non-finite values and ensures proper vector dimensions

### ONNX Runtime Integration
- Creates ONNX Runtime environment with optimized settings
- Loads model and creates inference session
- Handles tensor creation and data conversion
- Manages memory efficiently throughout the process

### Categories
The model classifies text into 5 categories:
- **Politics** 🏛️
- **Technology** 💻
- **Sports** ⚽
- **Business** 💼
- **Entertainment** 🎭

### Performance Metrics
- **Preprocessing Time**: Text tokenization and TF-IDF conversion
- **Inference Time**: ONNX model execution
- **Postprocessing Time**: Result extraction and formatting
- **Total Time**: End-to-end processing time
- **Throughput**: Texts processed per second
- **Memory Usage**: RAM consumption tracking

## Sample Output

```
🤖 ONNX MULTICLASS CLASSIFIER - C IMPLEMENTATION
===============================================
🤖 ONNX MULTICLASS CLASSIFIER - C IMPLEMENTATION
================================================
💻 SYSTEM INFORMATION:
   Platform: macOS
   Processor: arm64
   CPU Cores: 8 physical, 8 logical
   Total Memory: 16.0 GB
   Runtime: N/A (C Implementation)

🔄 Processing: NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship
📁 Loading TF-IDF data...
📁 Loading scaler data...
✅ Data files loaded successfully
🔄 Starting preprocessing...
✅ Preprocessing completed. Vector size: 2059
📊 TOPIC CLASSIFICATION RESULTS:
   🏆 Predicted Category: SPORTS ⚽
   📈 Confidence: 85.42% (0.8542)
   📝 Input Text: "NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"

📊 DETAILED PROBABILITIES:
   🏛️ Politics: 3.2% █ 
   💻 Technology: 2.1% █
   ⚽ Sports: 85.4% █████████████████ ⭐
   💼 Business: 5.8% █
   🎭 Entertainment: 3.5% █

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 45.23ms
   ┣━ Preprocessing: 12.45ms (27.5%)
   ┣━ Model Inference: 28.67ms (63.4%)
   ┗━ Postprocessing: 4.11ms (9.1%)

🚀 THROUGHPUT:
   Texts per second: 22.1

💾 RESOURCE USAGE:
   Memory Start: 48.32 MB
   Memory End: 51.78 MB
   Memory Delta: +3.46 MB
   CPU Usage: 0.0% avg, 0.0% peak (1 samples)

🎯 PERFORMANCE RATING: ✅ EXCELLENT
   (45.2ms total - Target: <100ms)
```

## Dependencies

The project uses the following dependencies:
- **onnxruntime**: Microsoft's ONNX Runtime Swift package for model inference
- **Foundation**: Standard Swift library for JSON parsing and system utilities
- **Darwin**: System-level APIs for memory and performance monitoring

## Troubleshooting

### Build Issues
If you encounter build issues:
1. Ensure you have the latest Xcode command line tools: `xcode-select --install`
2. Clean and rebuild: `swift package clean && swift build -c release`
3. Check Swift version: `swift --version` (requires 5.7+)

### Runtime Issues
If you encounter runtime errors:
1. Verify all required files are present (model.onnx, vocab.json, scaler.json)
2. Check file permissions and paths
3. Ensure sufficient memory is available
4. Verify ONNX model compatibility

### Performance Issues
If performance is slower than expected:
1. Use release build: `swift build -c release`
2. Ensure sufficient system resources
3. Check for memory constraints
4. Monitor CPU usage during inference

## Contributing

When contributing to this implementation:
1. Follow Swift naming conventions
2. Add appropriate error handling
3. Include performance monitoring
4. Update documentation as needed
5. Test with various input texts

## License

This implementation is part of the whitelightning-test project and follows the same licensing terms. 