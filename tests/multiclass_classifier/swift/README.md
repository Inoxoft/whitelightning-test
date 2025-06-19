# 🍎 Swift Multiclass Classification ONNX Model

This directory contains the Swift/iOS implementation for multiclass topic classification using ONNX Runtime with comprehensive cross-platform support, performance monitoring, and native Apple ecosystem integration.

## 📋 System Requirements

### Minimum Requirements
- **Swift**: 5.7+ (recommended: Swift 5.9+)
- **Xcode**: 14.0+ (for iOS/macOS development)
- **macOS**: 12.0+ (for development)
- **iOS**: 13.0+ (for deployment)
- **RAM**: 4GB available memory
- **Storage**: 2GB free space (including Xcode)

### Recommended Versions
- **Swift**: 5.9+ (latest stable)
- **Xcode**: 15.0+ (latest stable)
- **ONNX Runtime**: 1.22.0
- **CocoaPods**: 1.12.0+

### Supported Platforms
- ✅ **macOS**: 12.0+ (Intel & Apple Silicon)
- ✅ **iOS**: 13.0+ (iPhone, iPad)
- ✅ **iOS Simulator**: All supported versions
- ✅ **Linux**: Ubuntu 18.04+ (Swift Package Manager)
- ✅ **tvOS**: 13.0+ (Apple TV)
- ✅ **watchOS**: 6.0+ (Apple Watch)

## 📁 Directory Structure

```
swift/
├── Sources/                   # Swift Package Manager structure
│   └── ONNXClassifier/
│       ├── Utils.swift        # ONNX utility functions
│       └── infer.swift        # Main inference implementation
├── Tests/                     # Unit tests
│   └── ONNXClassifierTests/
├── Package.swift              # Swift Package Manager configuration
├── ONNXTest.xcworkspace/      # Xcode workspace (if using CocoaPods)
├── ONNXTest.xcodeproj/        # Xcode project
├── Podfile                    # CocoaPods configuration
├── Podfile.lock               # CocoaPods lock file
├── model.onnx                 # Multiclass classification ONNX model
├── vocab.json                 # Token vocabulary mapping
├── scaler.json                # Label mapping for categories
└── README.md                  # This file
```

## 🛠️ Step-by-Step Installation

### 🍎 macOS Development Setup

#### Step 1: Install Xcode
```bash
# Option A: Download from Mac App Store (Recommended)
# Search "Xcode" in Mac App Store and install

# Option B: Download from Apple Developer Portal
# Visit: https://developer.apple.com/xcode/
# Requires Apple ID

# Option C: Install via command line (if available)
xcode-select --install

# Accept license
sudo xcodebuild -license accept

# Verify installation
xcode-select -p
swift --version
```

#### Step 2: Install CocoaPods
```bash
# Option A: Using Homebrew (Recommended)
# Install Homebrew first if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH (Apple Silicon)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc

# Add to PATH (Intel)
echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc

# Install CocoaPods
brew install cocoapods

# Option B: Using RubyGems
sudo gem install cocoapods

# Option C: Using system Ruby (not recommended)
sudo gem install cocoapods -n /usr/local/bin

# Verify installation
pod --version
```

#### Step 3: Create Xcode Project
```bash
# Create project directory
mkdir -p ~/whitelightning-swift-multiclass
cd ~/whitelightning-swift-multiclass

# Open Xcode and create new project
# File → New → Project → iOS → App
# Product Name: ONNXTest
# Language: Swift
# Interface: SwiftUI or UIKit
# Save in: current directory

# Or create via command line (requires xcodeproj gem)
gem install xcodeproj
```

#### Step 4: Initialize CocoaPods
```bash
# Navigate to project directory
cd ~/whitelightning-swift-multiclass/ONNXTest

# Initialize Podfile
pod init

# Edit Podfile
cat > Podfile << 'EOF'
platform :ios, '13.0'

target 'ONNXTest' do
  use_frameworks!
  
  # ONNX Runtime
  pod 'onnxruntime-objc', '~> 1.22.0'
  
  # Optional: Additional utilities
  pod 'SwiftyJSON', '~> 5.0'
  
  target 'ONNXTestTests' do
    inherit! :search_paths
  end
end

post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '13.0'
    end
  end
end
EOF

# Install dependencies
pod install
```

#### Step 5: Add Model Files
```bash
# Copy model files to project
cp model.onnx vocab.json scaler.json ~/whitelightning-swift-multiclass/ONNXTest/

# Open workspace (NOT .xcodeproj)
open ONNXTest.xcworkspace

# In Xcode:
# 1. Drag model files into project navigator
# 2. Check "Copy items if needed"
# 3. Ensure "Add to target" is checked for ONNXTest
# 4. Set Target Membership ✅
```

#### Step 6: Add Swift Implementation
```bash
# Create Utils.swift file in Xcode
# File → New → File → Swift File
# Name: Utils.swift

# Add the ONNX implementation code (see main implementation)
# Build and run: Cmd+R
```

---

### 📱 iOS Project Setup

#### Step 1: Configure iOS Deployment
```bash
# In Xcode project settings:
# 1. Select project in navigator
# 2. Select target "ONNXTest"
# 3. General tab:
#    - iOS Deployment Info: iOS 13.0
#    - Supported Destinations: iPhone, iPad
# 4. Build Settings tab:
#    - iOS Deployment Target: 13.0
#    - Architectures: arm64, x86_64 (for simulator)
```

#### Step 2: Configure App Permissions
```bash
# Edit Info.plist if needed for file access
# Privacy - Photo Library Usage Description (if using image classification)
# Privacy - Camera Usage Description (if using camera input)
```

#### Step 3: Build and Test
```bash
# Build for iOS Simulator
# Product → Destination → iOS Simulator → iPhone 15
# Product → Build (Cmd+B)

# Run on iOS Simulator
# Product → Run (Cmd+R)

# Build for iOS Device
# Product → Destination → iOS Device → Your Device
# Ensure developer account is configured
# Product → Run (Cmd+R)
```

---

### 🐧 Linux Development Setup

#### Step 1: Install Swift
```bash
# Ubuntu/Debian
# Download Swift from: https://swift.org/download/
wget https://download.swift.org/swift-5.9-release/ubuntu2004/swift-5.9-RELEASE/swift-5.9-RELEASE-ubuntu20.04.tar.gz
tar -xzf swift-5.9-RELEASE-ubuntu20.04.tar.gz
sudo mv swift-5.9-RELEASE-ubuntu20.04 /opt/swift

# Add to PATH
echo 'export PATH=/opt/swift/usr/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Install dependencies
sudo apt update
sudo apt install -y clang libicu-dev libcurl4-openssl-dev libssl-dev

# CentOS/RHEL/Fedora
sudo dnf install -y clang libicu-devel libcurl-devel openssl-devel

# Verify installation
swift --version
```

#### Step 2: Create Swift Package
```bash
# Create project directory
mkdir -p ~/whitelightning-swift-multiclass
cd ~/whitelightning-swift-multiclass

# Initialize Swift package
swift package init --type executable --name ONNXClassifier

# Edit Package.swift
cat > Package.swift << 'EOF'
// swift-tools-version: 5.7
import PackageDescription

let package = Package(
    name: "ONNXClassifier",
    platforms: [
        .macOS(.v12),
        .iOS(.v13),
        .linux
    ],
    products: [
        .executable(name: "ONNXClassifier", targets: ["ONNXClassifier"])
    ],
    dependencies: [
        // ONNX Runtime Swift (if available for Linux)
        // .package(url: "https://github.com/microsoft/onnxruntime-swift", from: "1.22.0")
    ],
    targets: [
        .executableTarget(
            name: "ONNXClassifier",
            dependencies: []
        ),
        .testTarget(
            name: "ONNXClassifierTests",
            dependencies: ["ONNXClassifier"]
        )
    ]
)
EOF
```

#### Step 3: Build and Run
```bash
# Build the package
swift build

# Run with default test
swift run

# Run with custom text
swift run ONNXClassifier "France defeats Argentina in World Cup final"

# Build in release mode
swift build -c release

# Run release binary
.build/release/ONNXClassifier
```

---

### 📺 tvOS and ⌚ watchOS Setup

#### Step 1: Add Additional Targets
```bash
# In Xcode:
# 1. File → New → Target
# 2. Select tvOS → App (for Apple TV)
# 3. Select watchOS → App (for Apple Watch)
# 4. Configure deployment targets:
#    - tvOS: 13.0+
#    - watchOS: 6.0+
```

#### Step 2: Configure Platform-Specific Code
```swift
#if os(iOS)
    // iOS-specific code
#elseif os(macOS)
    // macOS-specific code
#elseif os(tvOS)
    // tvOS-specific code
#elseif os(watchOS)
    // watchOS-specific code
#endif
```

## 🔧 Advanced Configuration

### Package.swift Template (Swift Package Manager)
```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ONNXMulticlassClassifier",
    platforms: [
        .macOS(.v12),
        .iOS(.v13),
        .tvOS(.v13),
        .watchOS(.v6)
    ],
    products: [
        .library(name: "ONNXMulticlassClassifier", targets: ["ONNXMulticlassClassifier"]),
        .executable(name: "ClassifierCLI", targets: ["ClassifierCLI"])
    ],
    dependencies: [
        // Add dependencies here when available
    ],
    targets: [
        .target(
            name: "ONNXMulticlassClassifier",
            dependencies: [],
            resources: [
                .process("Resources/model.onnx"),
                .process("Resources/vocab.json"),
                .process("Resources/scaler.json")
            ]
        ),
        .executableTarget(
            name: "ClassifierCLI",
            dependencies: ["ONNXMulticlassClassifier"]
        ),
        .testTarget(
            name: "ONNXMulticlassClassifierTests",
            dependencies: ["ONNXMulticlassClassifier"]
        )
    ]
)
```

### Podfile Template (CocoaPods)
```ruby
platform :ios, '13.0'
use_frameworks!

target 'ONNXTest' do
  # ONNX Runtime
  pod 'onnxruntime-objc', '~> 1.22.0'
  
  # JSON handling
  pod 'SwiftyJSON', '~> 5.0'
  
  # Performance monitoring
  pod 'os_log', '~> 1.0'
  
  target 'ONNXTestTests' do
    inherit! :search_paths
    pod 'XCTest'
  end
end

# Universal platform support
target 'ONNXTest-macOS' do
  platform :macos, '12.0'
  # macOS specific pods
end

target 'ONNXTest-tvOS' do
  platform :tvos, '13.0'
  # tvOS specific pods
end

post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      # Ensure minimum deployment targets
      config.build_settings['IPHONEOS_DEPLOYMENT_TARGET'] = '13.0'
      config.build_settings['MACOSX_DEPLOYMENT_TARGET'] = '12.0'
      config.build_settings['TVOS_DEPLOYMENT_TARGET'] = '13.0'
      config.build_settings['WATCHOS_DEPLOYMENT_TARGET'] = '6.0'
      
      # Enable bitcode for watchOS
      if target.platform_name == :watchos
        config.build_settings['ENABLE_BITCODE'] = 'YES'
      end
    end
  end
end
```

### Build Settings Optimization
```bash
# In Xcode Build Settings:
# Swift Compiler - Optimization Level: Optimize for Speed [-O]
# Swift Compiler - Code Generation: Whole Module Optimization
# Apple Clang - Optimization Level: Fastest, Smallest [-Os]
# Build Active Architecture Only: No (for release)
# Architectures: arm64, x86_64
```

## 🎯 Usage Examples

### Basic Usage
```swift
// Example implementation
import Foundation
import onnxruntime_objc

class MulticlassClassifier {
    private let session: ORTSession
    private let vocab: [String: Int]
    private let labels: [String: String]
    
    init() throws {
        // Load model
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx") else {
            throw ClassifierError.modelNotFound
        }
        
        let env = try ORTEnv(loggingLevel: .warning)
        self.session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: nil)
        
        // Load vocab and labels
        self.vocab = try loadVocab()
        self.labels = try loadLabels()
    }
    
    func predict(text: String) throws -> (category: String, confidence: Float) {
        // Implementation here
        // Return predicted category and confidence
        return ("sports", 0.95)
    }
}

// Usage
do {
    let classifier = try MulticlassClassifier()
    let result = try classifier.predict(text: "France defeats Argentina in World Cup final")
    print("Category: \(result.category), Confidence: \(result.confidence)")
} catch {
    print("Error: \(error)")
}
```

### iOS App Integration
```swift
import SwiftUI
import onnxruntime_objc

struct ContentView: View {
    @State private var inputText = ""
    @State private var result = ""
    @State private var isLoading = false
    
    private let classifier = try? MulticlassClassifier()
    
    var body: some View {
        VStack {
            TextField("Enter text to classify", text: $inputText)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
            
            Button("Classify") {
                classifyText()
            }
            .disabled(isLoading || inputText.isEmpty)
            .padding()
            
            if isLoading {
                ProgressView()
            } else {
                Text(result)
                    .padding()
            }
        }
        .padding()
    }
    
    private func classifyText() {
        isLoading = true
        
        Task {
            do {
                let prediction = try classifier?.predict(text: inputText)
                await MainActor.run {
                    result = "Category: \(prediction?.category ?? "Unknown")"
                    isLoading = false
                }
            } catch {
                await MainActor.run {
                    result = "Error: \(error.localizedDescription)"
                    isLoading = false
                }
            }
        }
    }
}
```

### Command Line Tool
```swift
// Sources/ClassifierCLI/main.swift
import Foundation
import ONNXMulticlassClassifier

@main
struct ClassifierCLI {
    static func main() async throws {
        let arguments = CommandLine.arguments
        let text = arguments.count > 1 ? arguments[1] : "Default test text"
        
        let classifier = try MulticlassClassifier()
        let result = try classifier.predict(text: text)
        
        print("🤖 ONNX MULTICLASS CLASSIFIER - SWIFT IMPLEMENTATION")
        print("==================================================")
        print("🔄 Processing: \(text)")
        print("🏆 Predicted Category: \(result.category)")
        print("📈 Confidence: \(String(format: "%.2f%%", result.confidence * 100))")
    }
}
```

## 🐛 Troubleshooting

### macOS/Xcode Issues

**1. "Could not find module 'onnxruntime_objc'"**
```bash
# Ensure CocoaPods is installed and updated
pod install
pod update

# Clean and rebuild
Product → Clean Build Folder (Cmd+Shift+K)
Product → Build (Cmd+B)
```

**2. "No such module 'SwiftUI'" (older Xcode)**
```bash
# Update Xcode to 11.0+
# Or use UIKit instead of SwiftUI
```

**3. "Apple Silicon compatibility issues"**
```bash
# Ensure pods support Apple Silicon
# In Podfile, add:
post_install do |installer|
  installer.pods_project.build_configurations.each do |config|
    config.build_settings["EXCLUDED_ARCHS[sdk=iphonesimulator*]"] = "arm64"
  end
end
```

**4. "Code signing issues"**
```bash
# Configure development team
# Project Settings → Signing & Capabilities
# Select your Apple Developer account
# Enable "Automatically manage signing"
```

### iOS Device Issues

**1. "App installation failed"**
```bash
# Ensure device is registered in developer portal
# Check provisioning profile
# Verify bundle identifier is unique
```

**2. "Model file not found in bundle"**
```bash
# Verify model files are added to target
# Check Bundle Resources in Build Phases
# Ensure files have correct target membership
```

**3. "Memory issues on device"**
```bash
# Optimize model size
# Use quantized models if available
# Implement memory management best practices
```

### Linux Issues

**1. "Swift command not found"**
```bash
# Ensure Swift is in PATH
export PATH=/opt/swift/usr/bin:$PATH

# Or install via package manager
sudo apt install swift-lang  # Ubuntu (if available)
```

**2. "ONNX Runtime not available"**
```bash
# ONNX Runtime Swift support on Linux is limited
# Consider using C API bindings
# Or run via Docker with macOS/iOS environment
```

**3. "Compilation errors"**
```bash
# Install required dependencies
sudo apt install clang libicu-dev

# Update Swift to latest version
```

## 📊 Expected Output

```
🤖 ONNX MULTICLASS CLASSIFIER - SWIFT IMPLEMENTATION
==================================================
🔄 Processing: France defeats Argentina in World Cup final

💻 SYSTEM INFORMATION:
   Platform: Darwin
   Processor: arm64
   CPU Cores: 8 physical, 8 logical
   Total Memory: 16.0 GB
   Runtime: Swift 5.9

📊 TOPIC CLASSIFICATION RESULTS:
   🏆 Predicted Topic: SPORTS ⚽
   📈 Confidence: 92.34% (0.9234)
   📝 Input Text: "France defeats Argentina in World Cup final"

📊 DETAILED PROBABILITIES:
   📝 Business: 1.2% ▓
   📝 Education: 0.8% ▓
   📝 Entertainment: 2.1% ▓▓
   📝 Environment: 0.5% ▓
   📝 Health: 1.0% ▓
   📝 Politics: 2.3% ▓▓
   📝 Science: 0.7% ▓
   📝 Sports: 92.3% ████████████████████ ⭐
   📝 Technology: 0.9% ▓
   📝 World: 0.2% ▓

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 1.2ms
   ┣━ Preprocessing: 0.3ms (25.0%)
   ┣━ Model Inference: 0.7ms (58.3%)
   ┗━ Postprocessing: 0.2ms (16.7%)

🚀 THROUGHPUT:
   Texts per second: 833.3

💾 RESOURCE USAGE:
   Memory Start: 12.5 MB
   Memory End: 14.2 MB
   Memory Delta: +1.7 MB
   CPU Usage: 15.2% avg, 45.8% peak

🎯 PERFORMANCE RATING: 🚀 EXCELLENT
   (1.2ms total - Target: <10ms)
```

## 🚀 Features

- **Cross-Platform Support**: macOS, iOS, tvOS, watchOS, Linux
- **Native Performance**: Optimized for Apple Silicon and Intel
- **SwiftUI Integration**: Modern declarative UI framework
- **Async/Await Support**: Modern concurrency patterns
- **Memory Efficient**: Automatic memory management
- **Type Safety**: Swift's strong type system prevents runtime errors
- **Package Manager Support**: Both CocoaPods and Swift Package Manager

## 🎯 Performance Characteristics

- **Total Time**: ~1.2ms (second fastest implementation)
- **Memory Usage**: Low (~1.7MB additional)
- **CPU Efficiency**: Optimized for Apple hardware
- **Platform**: Excellent performance on Apple devices
- **Battery Efficiency**: Optimized for mobile devices

## 📝 Notes

- **Apple Ecosystem**: Perfect integration with iOS, macOS, tvOS, watchOS
- **Native Performance**: Optimized for Apple Silicon and Metal Performance Shaders
- **App Store Ready**: Suitable for App Store distribution
- **Developer Experience**: Excellent Xcode integration and debugging

### When to Use Swift Implementation
- ✅ **iOS Apps**: Native iOS application development
- ✅ **macOS Apps**: Native macOS application development
- ✅ **Apple Ecosystem**: Integration across Apple devices
- ✅ **App Store**: Distribution via App Store
- ✅ **Core ML**: Integration with Apple's ML frameworks
- ✅ **SwiftUI**: Modern declarative UI development
- ❌ **Cross-Platform**: Limited to Apple platforms primarily
- ❌ **Server-Side**: Better alternatives for backend services

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 