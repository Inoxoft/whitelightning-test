# C Implementation GitHub Actions Integration

This document describes how to run the C implementations of the ONNX model tests in GitHub Actions.

## 🚀 Quick Start

### Running C Tests via GitHub Actions

1. **Go to the Actions tab** in your GitHub repository
2. **Click "Run workflow"** on the "ONNX Model Tests" workflow
3. **Select the following options**:
   - **Model type**: Choose either:
     - `binary_classifier(Customer feedback classifier)`
     - `multiclass_classifier(News classifier)`
   - **Language**: Select `c`
   - **Custom text** (optional): Enter text to test

### Example Workflow Runs

#### Binary Classifier Test
```yaml
Model type: binary_classifier(Customer feedback classifier)
Language: c
Custom text: "This product is amazing!"
```

#### Multiclass Classifier Test
```yaml
Model type: multiclass_classifier(News classifier)
Language: c
Custom text: "шляк би тебе трафив"
```

## 🔧 Technical Implementation

### Workflow Configuration

The C tests are integrated into `.github/workflows/onnx-model-tests.yml` with the following features:

#### Dependencies Installation
- **Build tools**: `build-essential`
- **JSON parsing**: `libcjson-dev`
- **Download tools**: `wget`

#### ONNX Runtime Setup
- Downloads ONNX Runtime v1.22.0 for Linux
- Creates symlinks for consistent paths
- Sets up library paths for runtime execution

#### Build Process
- Uses Makefile-based compilation
- Platform-specific optimizations (Linux/macOS)
- Comprehensive error handling

#### Test Execution
- **Model file detection**: Automatically detects if model files are present
- **Full tests**: Runs complete test suite when models are available
- **CI verification**: Runs build verification when models are missing
- **Custom text support**: Tests user-provided text input
- **Performance benchmarking**: Runs 50-iteration benchmarks

### Build Targets

#### Standard Targets
```bash
make            # Compile the implementation
make clean      # Clean build artifacts
make test       # Run tests (requires model files)
make benchmark  # Run performance benchmark
```

#### CI-Specific Targets
```bash
make test-ci    # Build verification for CI (no model files required)
```

### Environment Setup

#### Library Path Configuration
```bash
export LD_LIBRARY_PATH=$PWD/onnxruntime-linux-x64-1.22.0/lib:$LD_LIBRARY_PATH
```

#### Platform Detection
- **Linux**: Adds `-ldl` and `-D_GNU_SOURCE` flags
- **macOS**: Adds CoreFoundation and IOKit frameworks

## 📊 Expected Output

### Successful Build (No Model Files)
```
🔨 Building binary classifier C implementation...
📍 Working directory: /home/runner/work/repo/repo/tests/binary_classifier/c
🔗 Library path: /home/runner/work/repo/repo/onnxruntime-linux-x64-1.22.0/lib:
✅ Binary classifier C implementation compiled successfully
🏗️ Build artifacts:
-rwxr-xr-x 1 runner docker 45632 Nov 15 10:30 test_onnx_model
```

### Successful Test (With Model Files)
```
🚀 Running binary classifier C tests...
💻 SYSTEM INFORMATION:
   Platform: Linux
   CPU: Intel(R) Xeon(R) CPU @ 2.20GHz
   CPU Cores: 2 physical, 2 logical
   Total Memory: 7.0 GB
   Implementation: C with ONNX Runtime

📊 SENTIMENT ANALYSIS RESULTS:
   🏆 Predicted Sentiment: Positive
   📈 Confidence: 87.34% (0.8734)

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 12.45ms
   ┣━ Preprocessing: 8.23ms (66.1%)
   ┣━ Model Inference: 3.12ms (25.1%)
   ┗━ Post-processing: 1.10ms (8.8%)
   🧠 CPU Usage: 45.2% avg, 78.9% peak (124 readings)
   💾 Memory: 12.3MB → 12.8MB (Δ+0.5MB)
   🚀 Throughput: 80.3 texts/sec
   Performance Rating: 🚀 EXCELLENT
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Build Failures
**Symptom**: Compilation errors
**Solution**: Check dependency installation and library paths

#### 2. Library Not Found
**Symptom**: `libonnxruntime.so: cannot open shared object file`
**Solution**: Verify LD_LIBRARY_PATH is set correctly

#### 3. cJSON Missing
**Symptom**: `cjson/cJSON.h: No such file or directory`
**Solution**: Ensure `libcjson-dev` is installed

#### 4. Model Files Missing
**Symptom**: Tests skip execution
**Expected**: This is normal for CI - build verification runs instead

### Debug Information

The workflow provides comprehensive debug output:
- System information
- Library paths
- Build process details
- File existence checks
- Performance metrics

### Test Scripts

Both implementations include test scripts for local verification:
- `tests/binary_classifier/c/.github-test.sh`
- `tests/multiclass_classifier/c/.github-test.sh`

## 🎯 Performance Expectations

### Build Times
- **Compilation**: ~30-60 seconds
- **Dependency installation**: ~60-120 seconds
- **ONNX Runtime download**: ~30-60 seconds

### Test Execution
- **Build verification**: ~5-10 seconds
- **Full test suite**: ~30-60 seconds (with models)
- **Performance benchmark**: ~60-120 seconds (50 iterations)

### Resource Usage
- **Memory**: ~50-100MB during compilation
- **Disk**: ~200MB for ONNX Runtime + build artifacts
- **CPU**: Moderate usage during compilation and testing

## 🔍 Monitoring and Artifacts

### Uploaded Artifacts
- Compiled executables (`test_onnx_model`)
- Log files (if generated)
- Build verification results

### Performance Metrics
- Compilation time
- Test execution time
- Memory usage
- CPU utilization
- Throughput measurements

## 🤝 Integration Benefits

### Advantages of C Implementation
1. **High Performance**: Native compilation for maximum speed
2. **Low Resource Usage**: Minimal memory footprint
3. **Production Ready**: Suitable for deployment environments
4. **Cross-platform**: Works on Linux and macOS
5. **Comprehensive Monitoring**: Same performance analytics as Python

### CI/CD Integration
- **Automated Testing**: Runs on every push/PR
- **Build Verification**: Ensures code compiles correctly
- **Performance Benchmarking**: Tracks performance metrics
- **Artifact Generation**: Produces deployable binaries
- **Multi-platform Support**: Tests on different environments

This integration provides a robust testing framework for C implementations while maintaining the same comprehensive performance monitoring capabilities as the Python versions. 