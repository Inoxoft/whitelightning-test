# 🍎 Running Real ONNX Runtime with Swift Locally

This guide shows how to run the **real ONNX Runtime** (not simulation) on your Mac.

## 🚀 Quick Setup

```bash
# 1. Navigate to Swift directory
cd tests/binary_classifier/swift

# 2. Install CocoaPods dependencies
pod install

# 3. Build the project
swift build --configuration release

# 4. Run with real ONNX Runtime
swift run binary-classifier "Your text here"

# 5. Run performance benchmark
swift run binary-classifier "Test text" --benchmark 50
```

## 📋 Prerequisites

- ✅ **macOS 12.0+**
- ✅ **Xcode 14.0+** (for Swift toolchain)
- ✅ **CocoaPods** installed (`gem install cocoapods`)

## 🔧 Troubleshooting

### Issue: "Pod not found"
```bash
# Install CocoaPods
sudo gem install cocoapods
pod --version
```

### Issue: "Framework not found"
```bash
# Clean and rebuild
rm -rf .build
swift build --configuration release
```

### Issue: "Model file not found"
Make sure these files exist:
- `model.onnx` (4.5MB)
- `vocab.json` (71KB)  
- `scaler.json` (90KB)

## 🎯 Expected Output (Real ONNX)

```
🤖 ONNX BINARY CLASSIFIER - SWIFT WITH REAL ONNX RUNTIME
=====================================================
🔄 Processing: "Congratulations! You've won a free iPhone — click here to claim your prize now!"

✅ ONNX Runtime initialized successfully
   📊 Vocabulary size: 2059
   🔢 IDF weights loaded: 2059
   ⚖️ Scaler parameters loaded: 2059

📊 RUNNING SINGLE PREDICTION:
   🏆 Predicted Sentiment: NEGATIVE 😞
   📈 Confidence: 87.2%
   📊 Probability: 0.128
   ⏱️  Processing Time: 23.4ms

✅ SWIFT ONNX RUNTIME INFERENCE COMPLETED SUCCESSFULLY!
```

## 🆚 Simulation vs Real ONNX

| Feature | Simulation Mode | Real ONNX Runtime |
|---------|----------------|-------------------|
| **Dependencies** | None | CocoaPods + onnxruntime-objc |
| **Performance** | ~0.9ms (fake) | ~25ms (real) |
| **Accuracy** | Basic heuristics | Full ML model |
| **Model Loading** | No files needed | Real model.onnx |
| **CI/CD** | ✅ Always works | ❌ Complex setup |
| **Local Dev** | ✅ Quick testing | ✅ Production-ready |

## 💡 Why Simulation in CI?

- **Reliability**: No complex dependencies to fail
- **Speed**: Faster CI runs
- **Consistency**: Same results every time
- **Maintenance**: Less setup complexity

The simulation provides the same **structure validation** and **API testing** as the real implementation, just with simplified logic.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Swift App     │    │  ONNX Runtime    │    │   Model Files   │
│                 │───▶│                  │───▶│                 │
│ • Text Input    │    │ • Tensor Ops     │    │ • model.onnx    │
│ • CLI Interface │    │ • Inference      │    │ • vocab.json    │
│ • Results       │    │ • Performance    │    │ • scaler.json   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

This gives you the **best of both worlds**: reliable CI testing with the option for full ONNX Runtime locally! 