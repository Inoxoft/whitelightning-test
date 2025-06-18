# 🐍 Python Multiclass Classification ONNX Model

This directory contains a **Python implementation** for multiclass text classification using ONNX Runtime. The model performs **topic classification** on news articles and other text content, categorizing them into 10 predefined categories.

## 📁 Directory Structure

```
python/
├── test_onnx_model.py         # Main Python implementation
├── model.onnx                 # Multiclass classification ONNX model
├── vocab.json                 # Vocabulary mapping for tokenization
├── scaler.json                # Label mapping for categories
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
cd tests/multiclass_classifier/python

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install onnxruntime numpy psutil
```

### Dependencies
- **onnxruntime**: ONNX model inference
- **numpy**: Numerical computations
- **psutil**: System monitoring

## 🚀 Usage

### Basic Usage
```bash
# Navigate to the Python directory
cd tests/multiclass_classifier/python

# Run with default text
python test_onnx_model.py

# Run with custom text
python test_onnx_model.py "NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"
```

### Expected Output
```
🤖 ONNX MULTICLASS CLASSIFIER - PYTHON IMPLEMENTATION
==================================================
🔄 Processing: NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship

💻 SYSTEM INFORMATION:
   Platform: Linux
   Processor: AMD EPYC 7763 64-Core Processor
   CPU Cores: 4 physical, 4 logical
   Total Memory: 15.6 GB
   Runtime: Python 3.9.2

📊 TOPIC CLASSIFICATION RESULTS:
⏱️  Processing Time: 510ms
   🏆 Predicted Category: SPORTS 📝
   📈 Confidence: 100.0%
   📝 Input Text: "NBA Finals: Celtics Defeat Mavericks in Game 5 to Win Championship"

📊 DETAILED PROBABILITIES:
   📝 Business: 0.0% 
   📝 Education: 0.0% 
   📝 Entertainment: 0.0% 
   📝 Environment: 0.0% 
   📝 Health: 0.0% 
   📝 Politics: 0.0% 
   📝 Science: 0.0% 
   📝 Sports: 100.0% ████████████████████ ⭐
   📝 Technology: 0.0% 
   📝 World: 0.0% 

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 510.01ms
   ┣━ Preprocessing: 0.04ms (0.0%)
   ┣━ Model Inference: 1.92ms (0.4%)
   ┗━ Post-processing: 0.05ms (0.0%)
   🧠 CPU Usage: 3.0% avg, 15.0% peak (5 readings)
   💾 Memory: 45.2MB → 46.3MB (Δ+1.1MB)
   🚀 Throughput: 2.0 texts/sec
```

## 🎯 Performance Characteristics

- **Total Time**: ~510ms (needs optimization)
- **Preprocessing**: Sequence tokenization (30 tokens)
- **Model Input**: Int32 array [1, 30]
- **Inference Engine**: ONNX Runtime CPU Provider
- **Memory Usage**: Low (~1MB additional)
- **CPU Usage**: Low (~3% average)

## 🔧 Technical Details

### Model Architecture
- **Type**: Multiclass Classification (10 categories)
- **Input**: Text string → Token sequence
- **Features**: Sequential tokens (max 30 tokens)
- **Output**: Probability distribution over 10 classes
- **Categories**: Business, Education, Entertainment, Environment, Health, Politics, Science, Sports, Technology, World

### Processing Pipeline
1. **Text Preprocessing**: Tokenization and sequence creation
2. **Vocabulary Mapping**: Convert words to token IDs
3. **Sequence Padding**: Pad/truncate to 30 tokens
4. **Model Inference**: ONNX Runtime execution
5. **Post-processing**: Softmax probabilities and category mapping

### Input Format
- **Max Sequence Length**: 30 tokens
- **Vocabulary Size**: Variable (from vocab.json)
- **Unknown Tokens**: Mapped to `<OOV>` token ID (usually 1)
- **Padding**: Zero-padding for shorter sequences

## 🎯 Categories Supported

1. **Business** - Financial news, corporate announcements, market reports
2. **Education** - Academic research, educational policies, school news
3. **Entertainment** - Movies, music, celebrity news, TV shows
4. **Environment** - Climate change, conservation, environmental policies
5. **Health** - Medical research, health policies, disease outbreaks
6. **Politics** - Government news, elections, political events
7. **Science** - Scientific discoveries, research breakthroughs
8. **Sports** - Athletic events, sports news, player updates
9. **Technology** - Tech innovations, gadget reviews, software updates
10. **World** - International news, global events, foreign affairs

## 📊 Performance Analysis

### Current Performance Issues
- **Slow overall processing** (~510ms vs C++ 33ms)
- **Python overhead**: Interpreter and library loading costs
- **Optimization needed**: General performance improvements

### Performance Comparison
- **Python**: 510.01ms (this implementation)
- **C++**: 32.84ms (15x faster)
- **C**: 32.54ms (15x faster)
- **Rust**: 1.24ms (410x faster)
- **Swift**: 7.47ms (68x faster)

### Optimization Strategies
1. **Session Reuse**: Load ONNX model once for multiple predictions
2. **Batch Processing**: Process multiple texts together
3. **Caching**: Cache tokenizer and model components
4. **NumPy Optimization**: Use efficient array operations

## 🚀 Integration Example

```python
import numpy as np
import onnxruntime as ort
import json

class MulticlassClassifier:
    def __init__(self, model_path, vocab_path, scaler_path):
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        
        # Load vocabulary and label mapping
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        with open(scaler_path, 'r') as f:
            self.labels = json.load(f)
    
    def predict(self, text):
        # Tokenize and pad sequence
        tokens = self._tokenize(text)
        sequence = self._pad_sequence(tokens, max_len=30)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        result = self.session.run(None, {input_name: sequence})
        
        # Get probabilities and category
        probabilities = result[0][0]
        predicted_id = np.argmax(probabilities)
        predicted_category = self.labels[str(predicted_id)]
        
        return predicted_category, probabilities
    
    def _tokenize(self, text):
        words = text.lower().split()
        return [self.vocab.get(word, self.vocab.get('<OOV>', 1)) for word in words]
    
    def _pad_sequence(self, tokens, max_len):
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [0] * (max_len - len(tokens))
        return np.array([tokens], dtype=np.int32)

# Usage
classifier = MulticlassClassifier('model.onnx', 'vocab.json', 'scaler.json')
category, probabilities = classifier.predict("Breaking tech news about AI")
print(f"Category: {category}")
print(f"Confidence: {max(probabilities) * 100:.1f}%")
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
cd tests/multiclass_classifier/python
ls -la  # Should show model.onnx, vocab.json, scaler.json
```

**3. "Performance is very slow"**
- Consider using `onnxruntime-gpu` if you have a compatible GPU
- Use session reuse for multiple predictions
- Consider batch processing for multiple texts

**4. "KeyError with vocabulary"**
```python
# Handle unknown words gracefully
token_id = vocab.get(word, vocab.get('<OOV>', 1))
```

**5. "Array dimension mismatch"**
```python
# Ensure proper array shape
sequence = np.array([tokens], dtype=np.int32)  # Shape: [1, seq_len]
```

## 📈 Performance Optimization

### Current vs Target Performance
- **Current**: 510ms total time
- **Target**: <100ms (similar to other languages)
- **Bottleneck**: Python interpreter overhead and library loading

### Optimization Opportunities
1. **Model Session Reuse**: Load once, predict many times
2. **Preprocessing Optimization**: Faster tokenization
3. **Memory Management**: Efficient array operations
4. **Batch Processing**: Multiple texts at once

### Optimized Usage Pattern
```python
# Initialize once
classifier = MulticlassClassifier('model.onnx', 'vocab.json', 'scaler.json')

# Reuse for multiple predictions
texts = [
    "Apple reports strong quarterly earnings",
    "Scientists discover new exoplanet",
    "Local team wins championship game"
]

for text in texts:
    category, probs = classifier.predict(text)
    print(f"{text[:30]}... → {category}")
```

## 📊 Model Files

- **`model.onnx`**: Multiclass classification neural network (~2.8MB)
- **`vocab.json`**: Word-to-token ID mapping for text tokenization
- **`scaler.json`**: Category ID to label mapping (0="Business", 1="Education", etc.)
- **`requirements.txt`**: Python package dependencies

## 🔄 Testing Examples

```bash
# Technology news
python test_onnx_model.py "Apple announces new iPhone with revolutionary AI features"

# Health news  
python test_onnx_model.py "Researchers discover new treatment for rare genetic disease"

# Politics news
python test_onnx_model.py "President signs new climate legislation into law"

# Science news
python test_onnx_model.py "Scientists detect gravitational waves from black hole merger"

# Sports news
python test_onnx_model.py "World Cup final attracts record global television audience"
```

## 📝 Notes

- **High Accuracy**: Reliable category predictions despite performance issues
- **Easy Integration**: Simple Python API for embedding in applications
- **Development Friendly**: Great for prototyping and experimentation
- **Production Considerations**: Consider faster languages for high-throughput scenarios

### When to Use Python Implementation
- ✅ **Prototyping**: Quick development and testing
- ✅ **Low Volume**: Occasional text classification
- ✅ **Integration**: Existing Python ecosystems
- ❌ **High Throughput**: Real-time or batch processing needs

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 