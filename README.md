# ONNX Model Testing Tool

A comprehensive testing framework for ONNX models (binary and multiclass classification) before production deployment. This tool validates model performance, robustness, and deployment readiness through extensive automated testing.

## 🆕 Multiclass Classification Testing

This repository includes dedicated testing infrastructure for multiclass ONNX models with advanced debugging and validation capabilities. See the [Multiclass Testing](#multiclass-classifier-testing) section for detailed information.

## Features

### Core Testing Capabilities
- **Model Loading Validation** - Verify ONNX model loads correctly and inspect architecture
- **Performance Benchmarking** - Measure inference speed and throughput
- **Input Validation Testing** - Test model robustness with various input types and edge values
- **Training Data Performance** - Validate accuracy on training/validation datasets
- **Edge Case Testing** - Test model behavior on edge cases and corner scenarios
- **H5 vs ONNX Comparison** - Compare ONNX model outputs with original TensorFlow/Keras model

### Deployment Readiness Assessment
- **Automated Scoring System** - 100-point scoring system across multiple criteria
- **Deployment Status Classification** - Ready, Caution, or Not Ready recommendations
- **Issue Detection** - Identifies critical problems that could affect production
- **Actionable Recommendations** - Specific suggestions for improvement

## Installation

```bash
# Clone or download the testing tool
git clone <repository-url>
cd onnx-model-tester

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Super Simple - Just Point to Your Folder! 🎯
```bash
# This is the easiest way - just specify your folder path
python onnx_model_tester.py --folder /path/to/your/model/folder

# Or for current directory
python onnx_model_tester.py --folder .
```

The tool will automatically discover:
- `*.onnx` files (your ONNX model)
- `*.h5` files (original Keras/TensorFlow model)
- `*training*.csv` or `*train*.csv` (training data)
- `*edge*.csv` (edge case data)
- `scaler.json`, `vocab.json` (configuration files)

### Individual File Specification (Advanced)
```bash
python onnx_model_tester.py \
    --model model.onnx \
    --config-dir . \
    --training-data training_data.csv \
    --edge-cases edge_case_data.csv \
    --h5-model model.h5 \
    --output deployment_report.json
```

## File Structure Requirements

Your project directory should contain:
```
your_project/
├── model.onnx                 # Your ONNX model (required)
├── model.h5                   # Original Keras/TF model (optional)
├── scaler.json               # Preprocessing scaler config (optional)
├── vocab.json                # Vocabulary mapping (optional)
├── training_data.csv         # Training/validation data (optional)
├── edge_case_data.csv        # Edge case test data (optional)
└── api_requests/             # API request examples (optional)
```

### Data Format Requirements

#### Training Data CSV
- Last column should contain target labels
- All other columns should be numerical features
- Example:
```csv
feature1,feature2,feature3,target
0.5,1.2,-0.3,1
-0.1,0.8,2.1,0
```

#### Edge Case Data CSV
- Same format as training data
- Should contain challenging/boundary cases
- Missing values, extreme values, etc.

#### Scaler JSON
```json
{
  "mean": [0.5, 1.0, 0.2],
  "scale": [0.3, 0.5, 0.1]
}
```

## Command Line Arguments

### Folder-Based (Recommended)
| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--folder` | Path to folder containing model files | Yes* | - |
| `--output` | Output report file path | No | "deployment_report.json" |

### Individual Files (Advanced)
| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--model` | Path to ONNX model file | Yes* | - |
| `--config-dir` | Directory containing config files | No | "." |
| `--training-data` | Path to training data CSV | No | - |
| `--edge-cases` | Path to edge case data CSV | No | - |
| `--h5-model` | Path to original H5 model | No | - |
| `--output` | Output report file path | No | "deployment_report.json" |

*Either `--folder` OR `--model` is required

## Test Categories

### 1. Model Loading Test (20 points)
- ✅ Model loads successfully
- ✅ Architecture inspection
- ✅ Input/output shape validation

### 2. Input Validation Test (15 points)
- ✅ Normal values
- ✅ Zero values
- ✅ Large values
- ✅ Small values
- ✅ Negative values
- ✅ NaN/Inf detection

### 3. Performance Test (15 points)
- 🚀 **Excellent**: < 100ms average inference
- ✅ **Good**: < 500ms average inference
- ⚠️ **Acceptable**: < 1s average inference
- ❌ **Poor**: > 1s average inference

### 4. Training Data Performance (20 points)
- 🚀 **Excellent**: ≥ 90% accuracy
- ✅ **Good**: ≥ 80% accuracy
- ⚠️ **Acceptable**: ≥ 70% accuracy
- ❌ **Poor**: < 70% accuracy

### 5. Edge Case Robustness (15 points)
- 🚀 **Excellent**: ≥ 95% success rate
- ✅ **Good**: ≥ 80% success rate
- ⚠️ **Needs Improvement**: ≥ 60% success rate
- ❌ **Poor**: < 60% success rate

### 6. H5 Comparison (15 points)
- ✅ **Match**: Max difference < 1e-5
- ✅ **Close Match**: Max difference < 1e-3
- ⚠️ **Significant Differences**: Larger differences

## Deployment Status

| Score | Status | Description |
|-------|--------|-------------|
| ≥ 85% | ✅ **Ready for Deployment** | All critical tests passed |
| 70-84% | ⚠️ **Deploy with Caution** | Some issues identified |
| < 70% | ❌ **Not Ready** | Critical issues must be resolved |

## Example Output

```
🚀 Starting Comprehensive ONNX Model Testing
============================================================

=== Model Loading Test ===
✓ Model size: 2.34 MB
✓ Input layers: 1
✓ Output layers: 1
  Input 0: input_1 - Shape: [-1, 10]
  Output 0: dense_2 - Shape: [-1, 1]

=== Inference Speed Test (100 samples) ===
✓ Average inference time: 12.34 ms
✓ Throughput: 81.05 samples/sec

=== Input Validation Test ===
✓ normal_values: Passed
✓ zeros: Passed
✓ ones: Passed
✓ large_values: Passed
✓ small_values: Passed
✓ negative_values: Passed
✓ Input validation: 6/6 tests passed

=== Training Data Test ===
✓ Loaded training data: 1000 samples
✓ Accuracy: 0.8750
✓ Classification type: Binary

============================================================
📊 DEPLOYMENT READINESS REPORT
============================================================
✓ Model Loading: PASS (20/20 points)
✓ Input Validation: PASS (15/15 points)
✓ Performance: EXCELLENT (15/15 points)
✓ Training Data Performance: GOOD (15/20 points)
? Edge Cases: NOT TESTED (0/15 points)
? H5 Comparison: NOT TESTED (0/15 points)

📈 OVERALL SCORE: 65/100 (65.0%)
⚠️ STATUS: DEPLOYMENT WITH CAUTION

💡 RECOMMENDATIONS (2):
  • Test with edge cases for robustness validation
  • Compare with original H5 model for consistency validation

📄 Report saved to: deployment_report.json
```

## Programmatic Usage

```python
from onnx_model_tester import ONNXModelTester

# Initialize tester
tester = ONNXModelTester("model.onnx", config_dir=".")

# Run individual tests
tester.test_model_loading()
tester.test_inference_speed()
tester.test_input_validation()

# Run comprehensive testing
results = tester.run_comprehensive_test(
    training_data_path="training_data.csv",
    edge_case_data_path="edge_case_data.csv",
    h5_model_path="model.h5"
)

# Generate deployment report
report = tester.generate_deployment_report()
print(f"Deployment ready: {report['is_ready']}")
```

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check ONNX model file path and format
   - Ensure ONNX Runtime is properly installed

2. **Shape Mismatch Errors**
   - Verify input data dimensions match model expectations
   - Check if preprocessing (scaling) is applied correctly

3. **Performance Issues**
   - Consider model optimization techniques
   - Check if running on appropriate hardware (CPU/GPU)

4. **Low Accuracy on Training Data**
   - Verify data preprocessing steps
   - Check if the correct target column is used

### Dependencies

- Python 3.7+
- NumPy ≥ 1.21.0
- Pandas ≥ 1.3.0
- ONNX Runtime ≥ 1.10.0
- TensorFlow ≥ 2.7.0 (for H5 comparison)
- Scikit-learn ≥ 1.0.0

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if needed
5. Submit a pull request

## Multiclass Classifier Testing

### Overview

The repository includes specialized testing infrastructure for multiclass ONNX models located in `tests/multiclass_classifier/python/`. This testing framework provides comprehensive validation, debugging, and performance analysis for multiclass classification models.

### Directory Structure

```
tests/multiclass_classifier/python/
├── test_onnx_model.py          # Main test suite with debugging features
├── requirements.txt            # Python dependencies
├── model.onnx                  # Your multiclass ONNX model
├── vocab.json                  # Vocabulary/tokenization mapping
├── scaler.json                 # Label mapping (class names)
└── performance_results.json    # Generated performance metrics
```

### Features

#### 🔍 Advanced Debugging
- **Step-by-step preprocessing analysis** - Shows tokenization, padding, and vocabulary coverage
- **Model architecture inspection** - Validates input/output shapes and types
- **Comprehensive label mapping diagnosis** - Tests all categories systematically
- **Performance profiling** - Measures inference time and memory usage

#### 🧪 Systematic Testing
- **Multi-category validation** - Tests with examples from each class
- **Confusion matrix generation** - Identifies classification patterns and biases
- **Real-world text examples** - Uses actual text samples for validation
- **Edge case detection** - Identifies model limitations and training issues

#### 📊 Performance Monitoring
- **Automated benchmarking** - Measures speed and resource usage
- **Results persistence** - Saves metrics to JSON for CI/CD integration
- **Threshold validation** - Ensures performance meets requirements

### Quick Start

#### Local Testing

```bash
# Navigate to the multiclass classifier test directory
cd tests/multiclass_classifier/python

# Install dependencies
pip install -r requirements.txt

# Run comprehensive tests
python -m pytest test_onnx_model.py -v -s

# Test with custom text
python -c "from test_onnx_model import test_custom_text; test_custom_text('Your custom text here')"
```

#### GitHub Actions Integration

The repository includes automated testing via GitHub Actions:

1. **Manual Trigger**: Go to Actions → "ONNX Model Tests" → "Run workflow"
2. **Select Model Type**: Choose "multiclass_classifier" 
3. **Choose Language**: Select "python"
4. **Optional Custom Text**: Add specific text to test

The workflow automatically:
- Sets up the Python environment
- Installs dependencies
- Runs comprehensive tests
- Generates performance reports
- Provides debugging output

### Model Requirements

Your multiclass model should follow this structure:

#### Input Format
- **Sequence tokenization**: Text → token IDs → padded arrays
- **Fixed length**: 30 tokens (configurable)
- **Data type**: `int32`
- **Shape**: `[batch_size, sequence_length]` = `[1, 30]`

#### Output Format
- **Probabilities**: One probability per class
- **Data type**: `float32`
- **Shape**: `[batch_size, num_classes]`

#### Required Files
- **`vocab.json`**: Word-to-token mapping `{"word": token_id, ...}`
- **`scaler.json`**: Class index to label mapping `{"0": "health", "1": "politics", ...}`

### Example Usage

#### Basic Classification Test

```python
from test_onnx_model import ONNXMulticlassModelTester
from pathlib import Path

# Initialize tester
tester = ONNXMulticlassModelTester(Path("model.onnx"))

# Load model
assert tester.test_model_loading()

# Test inference
results = tester.test_inference([
    "The government announced new policies",
    "Doctor recommends surgery for patient", 
    "Team wins championship game",
    "Earthquake strikes coastal region"
])

# Print results
for result in results:
    print(f"Text: {result['text']}")
    print(f"Prediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence_score']:.4f}")
```

#### Advanced Debugging

```python
# Run comprehensive diagnosis
tester.diagnose_label_mapping()

# Analyze model architecture
tester.analyze_model_architecture()

# Test multiple political texts
tester.test_multiple_political_texts()
```

### Known Issues & Solutions

#### 🚨 Training Bias Detection

The testing framework can detect systematic training issues:

**Issue**: Model classifies all text as one category (e.g., "sports")
```
⚠️ WARNING: Model has training bias issues
🔧 RECOMMENDATION: Model needs retraining with proper balanced dataset
```

**Diagnosis**: The framework provides detailed analysis:
- Confusion matrix showing misclassification patterns
- Accuracy breakdown by category
- Vocabulary and tokenization validation

**Solutions**:
1. **Retrain the model** with balanced, properly labeled data
2. **Verify training data** quality and label accuracy
3. **Check tokenization** consistency between training and inference

#### 🔧 Common Problems

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Vocabulary Mismatch** | Unknown words → poor accuracy | Ensure vocab.json matches training |
| **Label Mapping Error** | Wrong categories predicted | Verify scaler.json class mappings |
| **Training Data Bias** | Always predicts same class | Retrain with balanced dataset |
| **Architecture Mismatch** | Shape errors | Check input/output dimensions |

### Performance Expectations

#### Acceptable Metrics
- **Inference Time**: < 100ms per text
- **Memory Usage**: < 500MB
- **Accuracy**: > 70% on test data
- **Category Balance**: Each class should be predictable

#### Warning Signs
- 🚨 **All predictions same class**: Training bias
- 🚨 **100% confidence always**: Model overconfident
- 🚨 **High inference time**: Model too complex
- 🚨 **Memory leaks**: Resource management issues

### Workflow Configuration

The GitHub Actions workflow supports multiclass testing:

```yaml
# .github/workflows/onnx-model-tests.yml
inputs:
  model_type:
    description: 'Model type to test (includes customer feedback classifier)'
    options:
      - binary_classifier
      - multiclass_classifier  # ← Your option
  language:
    description: 'Programming language to test'
    options:
      - python
  custom_text:
    description: 'Custom text to test (optional)'
```

### Advanced Features

#### Custom Text Analysis
Provides detailed breakdown of model behavior:
- **Preprocessing steps**: Shows tokenization process
- **Model outputs**: Raw probabilities and predictions  
- **Performance metrics**: Timing and memory usage
- **Vocabulary analysis**: Coverage and unknown words

#### Systematic Category Testing
Tests examples from each category:
- **Health**: Medical terms and scenarios
- **Politics**: Government and policy texts
- **Sports**: Athletic events and competitions
- **World**: International news and events

#### Diagnostic Tools
- **Token pattern analysis**: Tests artificial inputs
- **Confusion matrix generation**: Visual classification patterns
- **Performance profiling**: Detailed timing breakdown
- **Error detection**: Identifies training issues

### Next Steps

1. **Test your model** using the provided framework
2. **Analyze the results** and identify any issues
3. **Fix training problems** if systematic bias detected
4. **Integrate into CI/CD** using GitHub Actions
5. **Monitor performance** in production

## Support

For issues and questions:
- Create an issue in the repository
- Check existing documentation
- Review troubleshooting section 
- For multiclass classifier issues, include performance_results.json 