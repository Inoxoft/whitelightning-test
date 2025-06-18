# 🍎 Swift Multiclass Classifier ONNX Implementation

This directory contains the Swift/iOS implementation for multiclass topic classification using ONNX Runtime.

## 📋 Prerequisites

- **Xcode** (latest version)
- **CocoaPods** installed
- **macOS** (for iOS development)

## 🚀 Quick Setup

### 1. Create Xcode Project
```bash
cd tests/multiclass_classifier/swift
```

Open Xcode and create a new iOS project:
- **File** → **New** → **Project** → **iOS** → **App**
- **Product Name**: `ONNXTest`
- **Language**: Swift
- **Save in**: current directory

### 2. Initialize CocoaPods
```bash
pod init
```

Edit the generated `Podfile`:
```ruby
platform :ios, '13.0'

target 'ONNXTest' do
  use_frameworks!
  pod 'onnxruntime-objc'
end
```

Install dependencies:
```bash
pod install
```

**Important**: Always open `ONNXTest.xcworkspace` (not `.xcodeproj`)

### 3. Add Model Files
Copy these files to your project and add them to Xcode:
- `model.onnx` - Your multiclass classification model
- `vocab.json` - Vocabulary mapping
- `label_map.json` - Topic label mapping

Make sure to:
- Drag files into Xcode Navigator
- Check "Copy items if needed"
- Set Target Membership ✅

### 4. Add Swift Code

Create `Utils.swift` with the provided ONNX implementation code (see main README).

### 5. Test Implementation

Create a test or main function that calls:
```swift
let runner = try ONNXModelRunner()
let results = try runner.predict(text: "Your test text here")
```

## 🎯 Expected Output Format

When running tests, output should match this format:

```
🤖 ONNX MULTICLASS CLASSIFIER - SWIFT IMPLEMENTATION
==================================================
🔄 Processing: President signs new legislation on healthcare reform

💻 SYSTEM INFORMATION:
   Platform: Darwin
   Processor: arm64
   CPU Cores: X physical, Y logical
   Total Memory: N GB
   Runtime: Swift 5.x

📊 TOPIC CLASSIFICATION RESULTS:
   🏆 Predicted Topic: POLITICS/HEALTH/TECH/etc
   📈 Confidence: XX.XX% (0.XXXX)
   📝 Input Text: "President signs new legislation on healthcare reform"

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
   CPU Usage: avg% avg, peak% peak

🎯 PERFORMANCE RATING: 🚀 EXCELLENT / ✅ GOOD / ⚠️ ACCEPTABLE / 🐌 SLOW
```

## 📁 File Structure

```
tests/multiclass_classifier/swift/
├── README.md
├── Podfile
├── Podfile.lock
├── ONNXTest.xcworkspace/
├── ONNXTest.xcodeproj/
├── ONNXTest/
│   ├── Utils.swift
│   ├── ContentView.swift
│   └── ...
├── model.onnx
├── vocab.json
└── label_map.json
```

## 🔧 Troubleshooting

### Build Issues
- Ensure you're opening `.xcworkspace` not `.xcodeproj`
- Clean build folder: **Product** → **Clean Build Folder**
- Update CocoaPods: `pod update`

### Model Loading Issues
- Verify model files are added to Xcode target
- Check file paths in Bundle.main
- Ensure correct file extensions (.onnx, .json)

### Runtime Issues
- Test on iOS Simulator first
- Check iOS deployment target (13.0+)
- Verify ONNX Runtime pod installation

## 📱 Testing

### iOS Simulator
```bash
xcodebuild -workspace ONNXTest.xcworkspace \
           -scheme ONNXTest \
           -destination 'platform=iOS Simulator,name=iPhone 15,OS=latest' \
           test
```

### Device Testing
Connect iOS device and select as run destination in Xcode.

## 🎯 Model Differences

This multiclass implementation differs from binary classifier:
- **Multiple output classes** (politics, tech, health, sports, etc.)
- **Softmax activation** for probability distribution
- **argmax** for final prediction
- **label_map.json** instead of simple positive/negative 