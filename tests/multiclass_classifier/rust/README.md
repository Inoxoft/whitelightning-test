# ONNX Multiclass Classifier - Rust Implementation

A high-performance news category classifier using ONNX Runtime for Rust with comprehensive performance monitoring and system information display.

## 🚀 Features

- **News Classification**: Multiclass classification (health, politics, sports, world) using token-based preprocessing
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
2. **vocab.json** - Tokenizer vocabulary mapping
   ```json
   {
     "word1": 1,
     "word2": 2,
     "<OOV>": 1,
     ...
   }
   ```
3. **scaler.json** - Label mapping for classes
   ```json
   {
     "0": "health",
     "1": "politics", 
     "2": "sports",
     "3": "world"
   }
   ```

## 🎯 Usage

### Basic Usage
```bash
# Run with default test texts
cargo run --release

# Test custom text
cargo run --release "France defeats Argentina in World Cup final"

# Using the compiled binary
./target/release/test_onnx_model "New healthcare policy announced"
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
🤖 ONNX MULTICLASS CLASSIFIER - RUST IMPLEMENTATION
======================================================

🔄 Processing: France defeats Argentina in World Cup final

💻 SYSTEM INFORMATION:
   Platform: macos
   Processor: aarch64
   CPU Cores: 8
   Total Memory: 16.0 GB
   Runtime: Rust Implementation
   Rust Version: 1.0.0
   ONNX Runtime Version: 2.0.0-rc.4

📊 MULTICLASS CLASSIFICATION RESULTS:
   🏆 Predicted Category: sports
   📈 Confidence: 92.34% (0.9234)
   📝 Input Text: "France defeats Argentina in World Cup final"
   📋 All Class Probabilities:
      health: 0.0123 (1.2%)
      politics: 0.0234 (2.3%)
      sports: 0.9234 (92.3%)
      world: 0.0409 (4.1%)

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 18.45ms
   ┣━ Preprocessing: 3.23ms (17.5%)
   ┣━ Model Inference: 13.67ms (74.1%)
   ┗━ Postprocessing: 1.55ms (8.4%)

🚀 THROUGHPUT:
   Texts per second: 54.2

💾 RESOURCE USAGE:
   Memory Start: 11.45 MB
   Memory End: 12.78 MB
   Memory Delta: +1.33 MB
   CPU Usage: 28.3% avg, 72.8% peak (12 samples)

🎯 PERFORMANCE RATING: 🚀 EXCELLENT
   (18.5ms total - Target: <100ms)
```

## 🔧 Technical Details

### Preprocessing Pipeline
1. **Text Tokenization**: Split text into words and convert to lowercase
2. **Token Mapping**: Convert words to integer IDs using vocabulary
3. **Sequence Padding**: Pad/truncate to fixed length of 30 tokens
4. **OOV Handling**: Unknown words mapped to `<OOV>` token

### Model Architecture
- **Input**: Int32 tensor [1, 30] (token sequence)
- **Output**: Float32 tensor [1, 4] (class probabilities)
- **Classes**: health, politics, sports, world

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
- name: Test Rust Multiclass Classifier
  run: |
    cd tests/multiclass_classifier/rust
    cargo build --release
    cargo run --release
```

## 🎯 Classification Categories

The model classifies news articles into these categories:

- **🏥 Health**: Medical news, healthcare policies, disease outbreaks
- **🏛️ Politics**: Government actions, elections, political events
- **⚽ Sports**: Sports events, competitions, athlete news
- **🌍 World**: International news, global events, foreign affairs

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

4. **Tensor shape mismatches**
   ```bash
   # Ensure input sequence length is exactly 30
   # Check vocab.json format matches expected structure
   ```

## 📈 Performance Expectations

- **Target**: <100ms total processing time
- **Typical**: 10-40ms on modern hardware
- **Throughput**: 25-100 texts/second depending on hardware
- **Memory**: Low memory footprint due to Rust's zero-cost abstractions

## 🧪 Testing Examples

```bash
# Sports news
cargo run --release "Lakers win championship game"

# Politics news  
cargo run --release "President announces new economic policy"

# Health news
cargo run --release "New vaccine shows promising results"

# World news
cargo run --release "Climate summit reaches historic agreement"
```

## 🔒 Security Features

- **Memory Safety**: No buffer overflows or memory leaks
- **Type Safety**: Compile-time guarantees prevent runtime errors
- **Thread Safety**: Safe concurrent processing with Rust's ownership model
- **Input Validation**: Robust handling of malformed input data

## 🚀 Performance Optimizations

### Compile-Time Optimizations
- **Link-Time Optimization (LTO)**: Whole-program optimization
- **Single Codegen Unit**: Maximum optimization at link time
- **Target CPU**: Optimized for the build machine's CPU features

### Runtime Optimizations
- **CPU Thread Pool**: Utilizes all available CPU cores
- **Memory Pool**: Efficient memory allocation for tensors
- **SIMD Instructions**: Vectorized operations when available

## 🤝 Contributing

1. Follow Rust coding conventions (`cargo fmt`)
2. Run tests before submitting (`cargo test`)
3. Update documentation for new features
4. Ensure CI compatibility
5. Add benchmarks for performance-critical changes

## 📄 License

This implementation is part of the ONNX model testing framework. 