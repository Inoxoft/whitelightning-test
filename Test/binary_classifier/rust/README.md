# 🦀 Rust Binary Classification ONNX Model

A high-performance sentiment analysis classifier using ONNX Runtime for Rust with comprehensive performance monitoring, system information display, and cross-platform support.

## 📋 System Requirements

### Minimum Requirements
- **CPU**: x86_64 or ARM64 architecture
- **RAM**: 2GB available memory
- **Storage**: 1GB free space
- **Rust**: 1.70.0+ (recommended: latest stable)
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+

### Supported Platforms
- ✅ **Windows**: 10, 11 (x64, ARM64)
- ✅ **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 9+, Fedora 30+
- ✅ **macOS**: 10.15+ (Intel & Apple Silicon)

## 📁 Directory Structure

```
rust/
├── src/
│   └── main.rs                # Main Rust implementation
├── model.onnx                 # Binary classification ONNX model
├── vocab.json                 # TF-IDF vocabulary and IDF weights
├── scaler.json                # Feature scaling parameters
├── Cargo.toml                 # Rust dependencies and configuration
├── Cargo.lock                 # Dependency lock file
└── README.md                  # This file
```

## 🛠️ Step-by-Step Installation

### 🪟 Windows Installation

#### Step 1: Install Rust
```powershell
# Option A: Download from rustup.rs (Recommended)
# Visit: https://rustup.rs/
# Download and run rustup-init.exe

# Option B: Install via winget
winget install Rustlang.Rustup

# Option C: Install via Chocolatey
choco install rustup.install

# Verify installation
rustc --version
cargo --version
```

#### Step 2: Install Build Tools
```powershell
# Install Visual Studio Build Tools (required for linking)
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
# During installation, select "C++ build tools"

# Alternative: Install Visual Studio Community
# Download from: https://visualstudio.microsoft.com/vs/community/
# Select "Desktop development with C++"
```

#### Step 3: Install Git (if not installed)
```powershell
# Download from: https://git-scm.com/download/win
# Or install via package manager
winget install Git.Git
```

#### Step 4: Create Project Directory
```powershell
# Create project directory
mkdir C:\whitelightning-rust
cd C:\whitelightning-rust

# Initialize Rust project
cargo init binary_classifier
cd binary_classifier
```

#### Step 5: Configure Dependencies
```powershell
# Edit Cargo.toml (see configuration section below)
# Add ONNX Runtime and other dependencies

# Build the project
cargo build --release
```

#### Step 6: Copy Source Files & Run
```powershell
# Copy your source files to the project
# src/main.rs, model.onnx, vocab.json, scaler.json

# Run with default text
cargo run --release

# Run with custom text
cargo run --release "This product is amazing!"

# Run benchmark
cargo run --release -- --benchmark 100
```

---

### 🐧 Linux Installation

#### Step 1: Install Rust
```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Source the environment
source ~/.cargo/env

# Or add to shell profile
echo 'source ~/.cargo/env' >> ~/.bashrc

# Verify installation
rustc --version
cargo --version
```

#### Step 2: Install Build Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev

# CentOS/RHEL 8+
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y pkg-config openssl-devel

# CentOS/RHEL 7
sudo yum groupinstall -y "Development Tools"
sudo yum install -y pkg-config openssl-devel

# Fedora
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y pkg-config openssl-devel
```

#### Step 3: Install Additional Tools (Optional)
```bash
# Install Git (if not installed)
sudo apt install git  # Ubuntu/Debian
sudo dnf install git  # CentOS/RHEL/Fedora

# Install VS Code (optional)
sudo snap install code --classic
```

#### Step 4: Create Project Directory
```bash
# Create project directory
mkdir -p ~/whitelightning-rust
cd ~/whitelightning-rust

# Initialize Rust project
cargo init binary_classifier
cd binary_classifier
```

#### Step 5: Configure Dependencies
```bash
# Edit Cargo.toml (see configuration section below)
# Build the project
cargo build --release
```

#### Step 6: Copy Source Files & Run
```bash
# Copy your source files to the project
# src/main.rs, model.onnx, vocab.json, scaler.json

# Run with default text
cargo run --release

# Run with custom text
cargo run --release "This product is amazing!"

# Run benchmark
cargo run --release -- --benchmark 100
```

---

### 🍎 macOS Installation

#### Step 1: Install Rust
```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Source the environment
source ~/.cargo/env

# Or add to shell profile
echo 'source ~/.cargo/env' >> ~/.zshrc

# Verify installation
rustc --version
cargo --version
```

#### Step 2: Install Xcode Command Line Tools
```bash
# Install Xcode Command Line Tools (required for compilation)
xcode-select --install

# Accept license if needed
sudo xcodebuild -license accept
```

#### Step 3: Install Homebrew (Optional)
```bash
# Install Homebrew for additional tools
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install additional tools (optional)
brew install git
```

#### Step 4: Create Project Directory
```bash
# Create project directory
mkdir -p ~/whitelightning-rust
cd ~/whitelightning-rust

# Initialize Rust project
cargo init binary_classifier
cd binary_classifier
```

#### Step 5: Configure Dependencies
```bash
# Edit Cargo.toml (see configuration section below)
# Build the project
cargo build --release
```

#### Step 6: Copy Source Files & Run
```bash
# Copy your source files to the project
# src/main.rs, model.onnx, vocab.json, scaler.json

# Run with default text
cargo run --release

# Run with custom text
cargo run --release "This product is amazing!"

# Run benchmark
cargo run --release -- --benchmark 100
```

## 🔧 Advanced Configuration

### Cargo.toml Configuration
```toml
[package]
name = "binary_classifier"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <your.email@example.com>"]
description = "ONNX Binary Classifier for Rust"
license = "MIT"

[dependencies]
# ONNX Runtime
ort = "2.0.0-rc.4"

# JSON handling
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Async runtime
tokio = { version = "1.0", features = ["full"] }

# System information
sysinfo = "0.30"

# Command line argument parsing
clap = { version = "4.0", features = ["derive"] }

# Error handling
anyhow = "1.0"

# Math operations
nalgebra = "0.32"

# Performance timing
chrono = { version = "0.4", features = ["serde"] }

[profile.release]
# Optimize for performance
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"

[profile.dev]
# Fast compilation for development
opt-level = 0
debug = true
```

### Environment Variables
```bash
# Linux/macOS
export RUST_LOG=info
export CARGO_TARGET_DIR=target

# Windows (PowerShell)
$env:RUST_LOG = "info"
$env:CARGO_TARGET_DIR = "target"
```

### Rust Toolchain Management
```bash
# Update Rust to latest stable
rustup update stable

# Install specific toolchain
rustup install 1.75.0
rustup default 1.75.0

# Add additional targets
rustup target add x86_64-pc-windows-gnu  # Windows
rustup target add x86_64-apple-darwin    # macOS Intel
rustup target add aarch64-apple-darwin   # macOS Apple Silicon
```

## 🎯 Usage Examples

### Basic Usage
```bash
# Default test
cargo run --release

# Positive sentiment
cargo run --release "I love this product! It's amazing!"

# Negative sentiment
cargo run --release "This is terrible and disappointing."

# Neutral sentiment
cargo run --release "The product is okay, nothing special."
```

### Performance Benchmarking
```bash
# Quick benchmark (10 iterations)
cargo run --release -- --benchmark 10

# Comprehensive benchmark (1000 iterations)
cargo run --release -- --benchmark 1000

# Save results to file
cargo run --release -- --benchmark 100 > benchmark_results.txt
```

### Development Mode
```bash
# Run in debug mode (faster compilation, slower execution)
cargo run "This is a test"

# Run with logging
RUST_LOG=debug cargo run --release "Test with logging"

# Check for issues
cargo clippy
cargo fmt
```

## 🐛 Troubleshooting

### Windows Issues

**1. "linker 'link.exe' not found"**
```powershell
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Or install Visual Studio Community
# Download from: https://visualstudio.microsoft.com/vs/community/
```

**2. "'cargo' is not recognized as an internal or external command"**
```powershell
# Add Cargo to PATH
$env:PATH += ";$env:USERPROFILE\.cargo\bin"

# Or reinstall Rust and ensure PATH is updated
```

**3. "failed to run custom build command for openssl-sys"**
```powershell
# Install OpenSSL for Windows
# Download from: https://slproweb.com/products/Win32OpenSSL.html

# Or use vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install openssl:x64-windows
```

**4. "Access is denied" when building**
```powershell
# Run PowerShell as Administrator
# Or exclude Cargo target directory from antivirus scanning
```

### Linux Issues

**1. "error: Microsoft Visual C++ 14.0 is required"**
```bash
# This error shouldn't occur on Linux, but if it does:
# Install build essentials
sudo apt install build-essential  # Ubuntu/Debian
sudo dnf groupinstall "Development Tools"  # CentOS/RHEL/Fedora
```

**2. "error: failed to run custom build command for openssl-sys"**
```bash
# Install OpenSSL development headers
sudo apt install libssl-dev pkg-config  # Ubuntu/Debian
sudo dnf install openssl-devel pkg-config  # CentOS/RHEL/Fedora
```

**3. "Permission denied" when installing Rust**
```bash
# Don't use sudo with rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# If already installed with sudo, remove and reinstall
sudo rm -rf /usr/local/cargo /usr/local/rustup
```

**4. "error: linker 'cc' not found"**
```bash
# Install GCC
sudo apt install gcc  # Ubuntu/Debian
sudo dnf install gcc  # CentOS/RHEL/Fedora
```

### macOS Issues

**1. "xcrun: error: invalid active developer path"**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Reset developer directory
sudo xcode-select --reset
```

**2. "error: failed to run custom build command for ring"**
```bash
# Update Xcode Command Line Tools
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install
```

**3. "Apple Silicon compatibility issues"**
```bash
# Check Rust target
rustc --print target-list | grep apple

# Install Apple Silicon target
rustup target add aarch64-apple-darwin

# Build for specific target
cargo build --target aarch64-apple-darwin --release
```

**4. "dyld: Library not loaded"**
```bash
# Check library dependencies
otool -L target/release/binary_classifier

# Fix library paths if needed
install_name_tool -change old_path new_path target/release/binary_classifier
```

## 📊 Expected Output

```
🤖 ONNX BINARY CLASSIFIER - RUST IMPLEMENTATION
==================================================
🔄 Processing: "This product is amazing!"

💻 SYSTEM INFORMATION:
   Platform: macOS 14.0 (Darwin)
   Processor: aarch64
   CPU Cores: 12 physical, 12 logical
   Total Memory: 32.0 GB
   Runtime: Rust Implementation
   Rust Version: 1.75.0
   ONNX Runtime Version: 2.0.0-rc.4

📊 SENTIMENT ANALYSIS RESULTS:
   🏆 Predicted Sentiment: Positive ✅
   📈 Confidence: 87.45% (0.8745)
   📝 Input Text: "This product is amazing!"

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 0.40ms
   ┣━ Preprocessing: 0.15ms (37.5%)
   ┣━ Model Inference: 0.20ms (50.0%)
   ┗━ Postprocessing: 0.05ms (12.5%)

🚀 THROUGHPUT:
   Texts per second: 2500.0

💾 RESOURCE USAGE:
   Memory Start: 12.45 MB
   Memory End: 13.78 MB
   Memory Delta: +1.33 MB
   CPU Usage: 25.3% avg, 67.8% peak (15 samples)

🎯 PERFORMANCE RATING: 🚀 EXCELLENT
   (0.4ms total - Target: <100ms)
```

## 🚀 Features

- **Blazing Fast Performance**: Sub-millisecond inference times
- **Memory Safe**: Zero-cost abstractions with Rust's safety guarantees
- **Async Processing**: Tokio-based async runtime for optimal performance
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Comprehensive Monitoring**: Detailed timing and resource tracking
- **Benchmarking Mode**: Statistical performance analysis with multiple runs
- **CI/CD Ready**: Graceful handling when model files are missing

## 🎯 Performance Characteristics

- **Total Time**: ~0.4ms (fastest implementation)
- **Zero-Copy Operations**: Minimal memory allocations during inference
- **SIMD Optimizations**: Leverages CPU vector instructions when available
- **Link-Time Optimization**: Enabled in release builds for maximum performance
- **Memory Safety**: No runtime overhead from garbage collection

## 🔧 Technical Details

### Model Architecture
- **Type**: Binary Classification
- **Input**: Text string
- **Features**: TF-IDF vectors (5000 dimensions)
- **Output**: Probability [0.0 - 1.0]
- **Threshold**: >0.5 = Positive, ≤0.5 = Negative

### Processing Pipeline
1. **Text Tokenization**: Split text into words and convert to lowercase
2. **TF-IDF Vectorization**: Convert to 5000-dimensional feature vector
3. **Feature Scaling**: Apply mean normalization and standard scaling
4. **Model Inference**: ONNX Runtime execution
5. **Post-processing**: Probability interpretation

### Rust-Specific Optimizations
- **Zero-Copy Operations**: Minimal memory allocations during inference
- **SIMD Optimizations**: Leverages CPU vector instructions when available
- **Link-Time Optimization**: Enabled in release builds for maximum performance
- **Memory Safety**: No runtime overhead from garbage collection
- **Async CPU Monitoring**: Non-blocking CPU sampling using Tokio tasks

## 🚀 Integration Example

```rust
use anyhow::Result;
use ort::{Environment, ExecutionProvider, Session, SessionBuilder, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio;

#[derive(Debug, Serialize, Deserialize)]
struct VocabData {
    vocab: HashMap<String, usize>,
    idf: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ScalerData {
    mean: Vec<f32>,
    scale: Vec<f32>,
}

struct BinaryClassifier {
    session: Session,
    vocab: VocabData,
    scaler: ScalerData,
}

impl BinaryClassifier {
    pub async fn new(
        model_path: &str,
        vocab_path: &str,
        scaler_path: &str,
    ) -> Result<Self> {
        // Initialize ONNX Runtime
        let environment = Environment::builder()
            .with_name("BinaryClassifier")
            .build()?;

        // Load ONNX model
        let session = SessionBuilder::new(&environment)?
            .with_execution_providers([ExecutionProvider::CPU(Default::default())])?
            .with_model_from_file(model_path)?;

        // Load vocabulary and scaler
        let vocab_content = tokio::fs::read_to_string(vocab_path).await?;
        let vocab: VocabData = serde_json::from_str(&vocab_content)?;

        let scaler_content = tokio::fs::read_to_string(scaler_path).await?;
        let scaler: ScalerData = serde_json::from_str(&scaler_content)?;

        Ok(Self {
            session,
            vocab,
            scaler,
        })
    }

    pub async fn predict(&self, text: &str) -> Result<f32> {
        // Preprocess text to TF-IDF features
        let features = self.preprocess_text(text).await?;

        // Create input tensor
        let input_tensor = Value::from_array(
            self.session.allocator(),
            &[features]
        )?;

        // Run inference
        let outputs = self.session.run(vec![input_tensor])?;

        // Extract probability
        let output: &[f32] = outputs[0].try_extract()?;
        Ok(output[0])
    }

    async fn preprocess_text(&self, text: &str) -> Result<Vec<f32>> {
        let mut features = vec![0.0f32; 5000];

        // Tokenize text
        let tokens: Vec<&str> = text
            .to_lowercase()
            .split_whitespace()
            .collect();

        // Calculate TF-IDF
        for token in tokens {
            if let Some(&index) = self.vocab.vocab.get(token) {
                if index < features.len() {
                    features[index] += 1.0 * self.vocab.idf[index];
                }
            }
        }

        // Apply scaling
        for (i, feature) in features.iter_mut().enumerate() {
            *feature = (*feature - self.scaler.mean[i]) / self.scaler.scale[i];
        }

        Ok(features)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let classifier = BinaryClassifier::new(
        "model.onnx",
        "vocab.json",
        "scaler.json"
    ).await?;

    let probability = classifier.predict("This product is amazing!").await?;
    let sentiment = if probability > 0.5 { "Positive" } else { "Negative" };

    println!("Sentiment: {} ({:.2}%)", sentiment, probability * 100.0);

    Ok(())
}
```

## 📈 Performance Tips

1. **Release Builds**: Always use `cargo build --release` for production
2. **Link-Time Optimization**: Enabled in Cargo.toml for maximum performance
3. **Target CPU**: Use `RUSTFLAGS="-C target-cpu=native"` for CPU-specific optimizations
4. **Profile-Guided Optimization**: Use PGO for even better performance
5. **Memory Allocation**: Consider using custom allocators for specific workloads

## 🏗️ CI/CD Integration

```rust
// Graceful handling when model files are missing
if !check_model_files() {
    println!("⚠️ Model files not found - exiting safely");
    println!("✅ Rust implementation compiled successfully");
    return Ok(());
}
```

### GitHub Actions Integration
```yaml
- name: Test Rust Binary Classifier
  run: |
    cd Test/binary_classifier/rust
    cargo build --release
    cargo run --release
```

## 📝 Notes

- **Fastest Implementation**: Sub-millisecond performance with zero-cost abstractions
- **Memory Safe**: Rust's ownership system prevents memory-related bugs
- **Production Ready**: Suitable for high-throughput, low-latency applications
- **Cross-Platform**: Consistent performance across operating systems

### When to Use Rust Implementation
- ✅ **High Performance**: Real-time or high-throughput requirements
- ✅ **System Programming**: Low-level control and optimization
- ✅ **Memory Safety**: Critical applications requiring reliability
- ✅ **WebAssembly**: Compile to WASM for web deployment
- ✅ **Microservices**: Lightweight, fast-starting services
- ❌ **Rapid Prototyping**: Longer development time vs. Python/JavaScript

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 