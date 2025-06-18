# 🐍 Python Binary Classification ONNX Model

This directory contains a **Python implementation** for binary text classification using ONNX Runtime. The model performs **sentiment analysis** on text input using TF-IDF vectorization and a trained neural network.

## 📁 Directory Structure

```
python/
├── test_onnx_model.py         # Main Python implementation
├── model.onnx                 # Binary classification ONNX model
├── vocab.json                 # TF-IDF vocabulary and IDF weights
├── scaler.json                # Feature scaling parameters
├── requirements.txt           # Python dependencies
├── performance_results.json   # Performance test results
└── README.md                  # This file
```

## 🛠️ Prerequisites

### Required Python Version
- **Python 3.8+** (recommended: Python 3.9 or 3.10)

### Installation

```bash
# Navigate to the Python directory
cd tests/binary_classifier/python

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install onnxruntime numpy scikit-learn psutil
```

### Dependencies
- **onnxruntime**: ONNX model inference
- **numpy**: Numerical computations
- **scikit-learn**: TF-IDF vectorization
- **psutil**: System monitoring

## 🚀 Usage

### Basic Usage
```bash
# Navigate to the Python directory
cd tests/binary_classifier/python

# Run with default text
python test_onnx_model.py

# Run with custom text
python test_onnx_model.py "Congratulations! You've won a free iPhone — click here to claim your prize now!"
```

### Expected Output
```
🤖 ONNX BINARY CLASSIFIER - PYTHON IMPLEMENTATION
===============================================
🔄 Processing: Your text here

💻 SYSTEM INFORMATION:
   Platform: Linux
   Processor: AMD EPYC 7763 64-Core Processor
   CPU Cores: 4 physical, 4 logical
   Total Memory: 15.6 GB
   Runtime: Python 3.9.2

📊 SENTIMENT ANALYSIS RESULTS:
   🏆 Predicted Sentiment: Positive
   📈 Confidence: 99.98% (0.9998)
   📝 Input Text: "Your text here"

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 332.33ms
   ┣━ Preprocessing: 0.85ms (0.3%)
   ┣━ Model Inference: 0.59ms (0.2%)
   ┗━ Postprocessing: 0.00ms (0.0%)

🚀 THROUGHPUT:
   Texts per second: 3.0

💾 RESOURCE USAGE:
   Memory Start: 45.20 MB
   Memory End: 45.49 MB
   Memory Delta: +0.29 MB
   CPU Usage: 15.0% avg, 25.0% peak (5 samples)
```

## 🎯 Performance Characteristics

- **Total Time**: ~332ms (needs optimization)
- **Preprocessing**: TF-IDF vectorization (5000 features) 
- **Model Input**: Float32 array [1, 5000]
- **Inference Engine**: ONNX Runtime CPU Provider
- **Memory Usage**: Low (~0.3MB additional)
- **CPU Usage**: Moderate (~15% average)

## 🔧 Technical Details

### Model Architecture
- **Type**: Binary Classification  
- **Input**: Text string
- **Features**: TF-IDF vectors (5000 dimensions)
- **Output**: Probability [0.0 - 1.0]
- **Threshold**: >0.5 = Positive, ≤0.5 = Negative

### Processing Pipeline
1. **Text Preprocessing**: Tokenization and lowercasing
2. **TF-IDF Vectorization**: Using scikit-learn and vocabulary
3. **Feature Scaling**: Standardization using mean/scale parameters  
4. **Model Inference**: ONNX Runtime execution
5. **Post-processing**: Probability interpretation

## 📊 Performance Analysis

### Current Performance Issues
- **Slow overall processing** (~332ms vs C++ 44ms)
- **Potential bottlenecks**: Python overhead, library initialization
- **Optimization needed**: Preprocessing and model loading

### Optimization Strategies
1. **Session Reuse**: Load ONNX model once for multiple predictions
2. **Vectorization**: Use NumPy operations instead of loops
3. **Caching**: Cache TF-IDF vectorizer and scaler
4. **Batch Processing**: Process multiple texts together

## 🚀 Integration Example

```python
import numpy as np
import onnxruntime as ort
from sklearn.feature_extraction.text import TfidfVectorizer
import json

class BinaryClassifier:
    def __init__(self, model_path, vocab_path, scaler_path):
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        
        # Load vocabulary and scaler
        with open(vocab_path, 'r') as f:
            self.vocab_data = json.load(f)
        with open(scaler_path, 'r') as f:
            self.scaler_data = json.load(f)
    
    def predict(self, text):
        # Preprocess text
        features = self._preprocess_text(text)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        result = self.session.run(None, {input_name: features})
        
        return result[0][0][0]  # Extract probability
    
    def _preprocess_text(self, text):
        # TF-IDF vectorization logic here
        pass

# Usage
classifier = BinaryClassifier('model.onnx', 'vocab.json', 'scaler.json')
probability = classifier.predict("Your text here")
sentiment = "Positive" if probability > 0.5 else "Negative"
print(f"Sentiment: {sentiment} ({probability:.4f})")
```

## 🐛 Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'onnxruntime'"**
```bash
pip install onnxruntime
```

**2. "FileNotFoundError: model.onnx not found"**
```bash
# Make sure you're in the correct directory
cd tests/binary_classifier/python
ls -la  # Should show model.onnx, vocab.json, scaler.json
```

**3. "Performance is very slow"**
- Consider using `onnxruntime-gpu` if you have a compatible GPU
- Check if running in a virtual environment with limited resources
- Monitor system resources during execution

**4. "Memory issues with large texts"**
```python
# Limit text length before processing
text = text[:1000]  # Truncate to 1000 characters
```

## 📈 Performance Optimization

### Current vs Target Performance
- **Current**: 332ms total time
- **Target**: <100ms (similar to other languages)
- **Bottleneck**: Likely in preprocessing or model loading overhead

### Optimization Opportunities
1. **Model Loading**: Load once, reuse session
2. **TF-IDF Optimization**: Use optimized vectorization
3. **Memory Management**: Efficient array operations
4. **Caching**: Cache preprocessed components

## 📊 Model Files

- **`model.onnx`**: Binary classification neural network (~10MB)
- **`vocab.json`**: TF-IDF vocabulary mapping and IDF weights  
- **`scaler.json`**: Feature standardization parameters (mean/scale)
- **`requirements.txt`**: Python package dependencies

## 🔄 Testing

```bash
# Run with different test cases
python test_onnx_model.py "This is a great product!"
python test_onnx_model.py "Click here to win $1000 now!"
python test_onnx_model.py "Thank you for your service."
```

## 📝 Notes

- Python implementation shows good accuracy but needs performance optimization
- Consider this implementation for prototyping and development
- For production use, consider C++ or Rust implementations for better performance
- Memory usage is efficient, but processing time needs improvement

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 