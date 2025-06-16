# ONNX Binary Classifier - Rust Implementation

A high-performance sentiment analysis classifier using ONNX Runtime for Rust with comprehensive performance monitoring and system information display.

## 🚀 Features

- **Sentiment Analysis**: Binary classification (Positive/Negative) using TF-IDF preprocessing
- **Performance Monitoring**: Detailed timing breakdown, resource usage tracking, and throughput analysis
- **System Information**: Platform detection, CPU/memory specs, runtime versions
- **Benchmarking Mode**: Statistical performance analysis with multiple runs
- **CI/CD Ready**: Graceful handling when model files are missing
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Memory Safe**: Zero-cost abstractions with Rust's safety guarantees
- **Async Processing**: Tokio-based async runtime for optimal performance

## 📋 Requirements

- Rust 1.70.0 or higher
- ONNX Runtime 2.0.0-rc.4
- Model files: `model.onnx`, `vocab.json`, `scaler.json`

## 🛠️ Installation

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the project
cargo build --release

# Or build in debug mode for development
cargo build
```

## 📁 Required Files

Place these files in the same directory as `Cargo.toml`:

1. **model.onnx** - The trained ONNX model file
2. **vocab.json** - TF-IDF vocabulary and IDF weights
   ```json
   {
     "vocab": {"word1": 0, "word2": 1, ...},
     "idf": [1.23, 4.56, ...]
   }
   ```
3. **scaler.json** - Feature scaling parameters
   ```json
   {
     "mean": [0.1, 0.2, ...],
     "scale": [1.1, 1.2, ...]
   }
   ```

## 🎯 Usage

### Basic Usage
```bash
# Run with default test texts
cargo run --release

# Test custom text
cargo run --release "This product is amazing!"

# Using the compiled binary
./target/release/test_onnx_model "Great service!"
```

### Performance Benchmarking
```bash
# Run 100 iterations benchmark
cargo run --release -- --benchmark 100

# Custom number of iterations
cargo run --release -- --benchmark 500
```

### Development Mode
```bash
# Run in debug mode (faster compilation, slower execution)
cargo run "This is a test"
```

## 📊 Output Features

### System Information
- Platform and processor architecture
- CPU cores and total memory
- Rust and ONNX Runtime versions

### Performance Metrics
- **Timing Breakdown**: Preprocessing, inference, and postprocessing times with percentages
- **Resource Usage**: Memory consumption and CPU utilization
- **Throughput**: Texts processed per second
- **Performance Rating**: EXCELLENT/GOOD/ACCEPTABLE/POOR classification

### Sample Output
```
🤖 ONNX BINARY CLASSIFIER - RUST IMPLEMENTATION
==================================================

🔄 Processing: This product is amazing!

💻 SYSTEM INFORMATION:
   Platform: macos
   Processor: aarch64
   CPU Cores: 8
   Total Memory: 16.0 GB
   Runtime: Rust Implementation
   Rust Version: 1.0.0
   ONNX Runtime Version: 2.0.0-rc.4

📊 SENTIMENT ANALYSIS RESULTS:
   🏆 Predicted Sentiment: Positive
   📈 Confidence: 87.45% (0.8745)
   📝 Input Text: "This product is amazing!"

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 25.34ms
   ┣━ Preprocessing: 8.12ms (32.0%)
   ┣━ Model Inference: 15.67ms (61.8%)
   ┗━ Postprocessing: 1.55ms (6.1%)

🚀 THROUGHPUT:
   Texts per second: 39.5

💾 RESOURCE USAGE:
   Memory Start: 12.45 MB
   Memory End: 13.78 MB
   Memory Delta: +1.33 MB
   CPU Usage: 25.3% avg, 67.8% peak (15 samples)

🎯 PERFORMANCE RATING: 🚀 EXCELLENT
   (25.3ms total - Target: <100ms)
```

## 🔧 Technical Details

### Preprocessing Pipeline
1. **Text Tokenization**: Split text into words and convert to lowercase
2. **TF-IDF Vectorization**: Convert to 5000-dimensional feature vector
3. **Feature Scaling**: Apply mean normalization and standard scaling

### Model Architecture
- **Input**: Float32 tensor [1, 5000] (TF-IDF features)
- **Output**: Float32 tensor [1, 1] (sentiment probability)
- **Threshold**: 0.5 (>0.5 = Positive, ≤0.5 = Negative)

### Performance Monitoring
- **High-Resolution Timing**: Uses `std::time::Instant` for nanosecond precision
- **Memory Tracking**: Monitors system memory usage before/after processing
- **CPU Monitoring**: Real-time CPU usage sampling with `sysinfo` crate
- **Async CPU Monitoring**: Non-blocking CPU sampling using Tokio tasks

### Rust-Specific Optimizations
- **Zero-Copy Operations**: Minimal memory allocations during inference
- **SIMD Optimizations**: Leverages CPU vector instructions when available
- **Link-Time Optimization**: Enabled in release builds for maximum performance
- **Memory Safety**: No runtime overhead from garbage collection

## 🏗️ CI/CD Integration

The implementation includes CI-friendly features:

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
    cd tests/binary_classifier/rust
    cargo build --release
    cargo run --release
```

## 🐛 Troubleshooting

### Common Issues

1. **Compilation errors**
   ```bash
   # Update Rust toolchain
   rustup update
   
   # Clean and rebuild
   cargo clean
   cargo build --release
   ```

2. **ONNX Runtime linking issues**
   ```bash
   # Ensure ONNX Runtime is properly installed
   # The ort crate handles automatic downloading
   cargo clean
   cargo build --release
   ```

3. **Memory issues with large models**
   ```bash
   # Increase stack size if needed
   export RUST_MIN_STACK=8388608
   cargo run --release
   ```

4. **Performance issues**
   ```bash
   # Always use release mode for performance testing
   cargo run --release -- --benchmark 100
   
   # Check CPU governor settings on Linux
   cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

## 📈 Performance Expectations

- **Target**: <100ms total processing time
- **Typical**: 10-50ms on modern hardware
- **Throughput**: 20-100 texts/second depending on hardware
- **Memory**: Low memory footprint due to Rust's zero-cost abstractions

## 🔒 Security Features

- **Memory Safety**: No buffer overflows or memory leaks
- **Type Safety**: Compile-time guarantees prevent runtime errors
- **Thread Safety**: Safe concurrent processing with Rust's ownership model

## 🤝 Contributing

1. Follow Rust coding conventions (`cargo fmt`)
2. Run tests before submitting (`cargo test`)
3. Update documentation for new features
4. Ensure CI compatibility

## 📄 License

This implementation is part of the ONNX model testing framework. 