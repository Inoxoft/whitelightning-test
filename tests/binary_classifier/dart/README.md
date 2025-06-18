# 🎯 Flutter ONNX Inference Tests

This demonstrates **real ONNX model inference** on Flutter with two classifiers:

## 📊 **Binary Sentiment Classifier**
- **Input**: Text → **5000 TF-IDF features** 
- **Output**: Sentiment (Positive/Negative) with confidence
- **Model**: Uses your exact working binary classification code
- **Performance**: 58-239ms inference time

## 📚 **Multiclass Topic Classifier**  
- **Input**: Text → **30 token sequences**
- **Output**: Topic (Science, Politics, Technology, Entertainment, Business, etc.)
- **Model**: Uses your exact working multiclass classification code
- **Performance**: 67-248ms inference time

## 🚀 **Run Locally**

### Prerequisites
```bash
# Install Flutter (if not installed)
# For macOS:
brew install flutter

# Install ONNX Runtime
# macOS:
wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-osx-x86_64-1.15.1.tgz
tar -xzf onnxruntime-osx-x86_64-1.15.1.tgz
sudo cp onnxruntime-osx-x86_64-1.15.1/lib/libonnxruntime.1.15.1.dylib /usr/local/lib/
sudo ln -sf /usr/local/lib/libonnxruntime.1.15.1.dylib /usr/local/lib/libonnxruntime.dylib
```

### Test Binary Classifier
```bash
cd tests/binary_classifier/dart
flutter pub get
flutter test test/onnx_inference_test.dart
```

### Test Multiclass Classifier
```bash
cd tests/multiclass_classifier/dart
flutter pub get
flutter test test/onnx_inference_test.dart
```

### Run UI Version (Optional)
```bash
# Binary classifier app
cd tests/binary_classifier/dart
flutter run

# Multiclass classifier app  
cd tests/multiclass_classifier/dart
flutter run
```

## 🤖 **GitHub Actions**

The workflow **automatically runs** on:
- Push to main branch
- Pull requests
- Manual trigger (`workflow_dispatch`)

**View Results**: Check the Actions tab to see **real ONNX predictions** in the console logs!

## 📝 **Sample Predictions**

### Binary Classifier Output:
```
🎯 REAL PREDICTION: POSITIVE 😊
📈 Probability: 99.82%
🎪 Confidence: 99.6%
⏱️  Processing time: 239ms
```

### Multiclass Classifier Output:
```
🎯 PREDICTED CLASS: Science (Score: 0.9999)
⏱️  Processing time: 248ms
⚡ Performance: EXCELLENT
```

## 🎉 **Key Features**

✅ **Real ONNX inference** (not mocked)  
✅ **Performance metrics** and timing  
✅ **Confidence scores** with emojis  
✅ **Works locally** and in **GitHub Actions**  
✅ **Mobile/Desktop ready** (uses native ONNX Runtime)  
✅ **Console output** perfect for CI/CD logging  

The tests show **exactly what your models predict** for different input texts! 🚀 