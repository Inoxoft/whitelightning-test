# ONNX Model Testing Suite

A comprehensive enterprise-level tool for testing and validating ONNX models before deployment. Features automatic class detection, LLM-powered text generation, and detailed deployment readiness scoring.

## 🚀 Features

### 📦 **Structural Analysis**
- ONNX model integrity validation
- Opset version compatibility checking
- Graph architecture analysis
- Parameter counting and distribution

### 🔍 **Input/Output Analysis**
- Dynamic shape detection
- Data type validation
- Input/output metadata extraction

### 🧪 **Comprehensive Testing**
- Inference smoke tests
- Input validation with edge cases
- Performance profiling (latency, throughput, memory)
- Training data performance evaluation

### 🤖 **AI-Powered Testing**
- **Automatic Class Detection**: Intelligently identifies model classes and their meanings
- **LLM Text Generation**: Creates realistic test examples for each class (2 samples per class)
- **Balanced Testing**: Ensures equal representation across all model capabilities
- **Model-Specific Content**: Generates appropriate content based on model type (sentiment, hate speech, news, etc.)

### ⚡ **Performance & Compatibility**
- Cross-platform compatibility analysis
- Mobile/web deployment readiness
- Memory usage monitoring
- Deployment scoring system (0-100%)

## 🛠 Installation

### Prerequisites
- Python 3.8+
- ONNX Runtime
- OpenRouter API key (for LLM testing)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file in the project root:
```bash
# OpenRouter API key for LLM text generation
OPENROUTER_API_KEY=your_api_key_here
```

## 📋 Usage

### Basic Usage (Auto-Discovery)
```bash
# Test model with automatic file discovery
python onnx_model_tester.py --folder /path/to/model/folder

# Include LLM-powered testing (automatic sample calculation)
python onnx_model_tester.py --folder /path/to/model/folder --use-llm-text
```

### Advanced Usage
```bash
# Manual file specification
python onnx_model_tester.py \
  --model model.onnx \
  --config-dir ./configs \
  --training-data training.csv \
  --edge-cases edge_cases.csv \
  --use-llm-text

# API testing integration
python onnx_model_tester.py \
  --folder /path/to/model \
  --api-endpoint https://api.example.com/preprocess \
  --api-headers '{"Authorization": "Bearer token"}' \
  --use-llm-text
```

## 🎯 Automatic Class Detection

The tool automatically detects model types and classes:

### Binary Classification (2 classes)
- **Sentiment Analysis**: `positive_sentiment`, `negative_sentiment`
- **Hate Speech**: `hate_speech`, `normal_speech`
- **Spam Detection**: `spam`, `legitimate`

### Multiclass Classification
- **News Classification**: `politics`, `sports`, `technology`, `business`, `entertainment`, `health`, `science`, `world`, `opinion`, `local`
- **Emotion Detection**: `anger`, `fear`, `joy`, `love`, `sadness`, `surprise`, `neutral`
- **Custom Models**: Automatically inferred from model architecture

### Sample Generation
- **Automatic Calculation**: 2 samples per class (no manual configuration needed)
- **Targeted Content**: Each sample generated specifically for its intended class
- **Realistic Examples**: LLM creates contextually appropriate test cases

## 📊 Model Support

### Supported Model Types
- **Text Classification**: Sentiment, hate speech, spam detection, news categorization
- **NLP Models**: Token-based (int32) and feature-based (float32) inputs
- **Binary & Multiclass**: Automatic detection and appropriate testing

### Input Formats
- **Token-based**: `tensor(int32)` with vocabulary support
- **Feature-based**: `tensor(float32)` with preprocessing pipelines
- **Dynamic Shapes**: Full support for variable input dimensions

## 📈 Deployment Scoring

### Scoring Categories (100 points total)
- **Model Loading** (20 pts): Basic functionality
- **Input Validation** (15 pts): Edge case handling
- **Performance** (15 pts): Speed and efficiency
- **Training Data** (20 pts): Accuracy on known data
- **Edge Cases** (15 pts): Robustness testing
- **LLM Testing** (15 pts): Real-world content validation

### Deployment Status
- **85-100%**: ✅ Ready for Deployment
- **70-84%**: ⚠️ Deployment with Caution
- **0-69%**: ❌ Not Ready for Deployment

## 📁 Project Structure

```
onnx-model-tester/
├── onnx_model_tester.py      # Main testing tool
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .env.example             # Environment template
├── .gitignore              # Git ignore rules
└── examples/               # Example usage scripts
    ├── test_sentiment.py   # Sentiment model example
    ├── test_news.py       # News classifier example
    └── test_hate_speech.py # Hate speech example
```

## 🔧 Configuration Files

### Model Configuration
The tool automatically discovers and loads:
- `scaler.json`: StandardScaler parameters
- `vocab.json`: Vocabulary mappings
- `generation_config.json`: Model generation settings

### Example Folder Structure
```
your_model_folder/
├── model.onnx              # ONNX model file
├── training_data.csv       # Training dataset
├── edge_case_data.csv      # Edge cases
├── scaler.json            # Preprocessing config
└── vocab.json             # Vocabulary
```

## 🧪 Testing Examples

### Binary Sentiment Model
```bash
python onnx_model_tester.py \
  --folder /models/sentiment_binary \
  --use-llm-text
```
**Output**: 4 samples (2 positive + 2 negative sentiment examples)

### 10-Class News Classifier
```bash
python onnx_model_tester.py \
  --folder /models/news_classifier \
  --use-llm-text
```
**Output**: 20 samples (2 examples per news category)

### Hate Speech Detection
```bash
python onnx_model_tester.py \
  --folder /models/hate_speech \
  --use-llm-text
```
**Output**: 4 samples (2 normal + 2 potentially offensive examples)

## 📊 Output Reports

### JSON Report
Detailed deployment report saved as `deployment_report.json`:
```json
{
  "score": 85,
  "max_score": 100,
  "percentage": 85.0,
  "deployment_status": "READY FOR DEPLOYMENT",
  "is_ready": true,
  "issues": [],
  "recommendations": [],
  "test_results": {
    "model_loading": true,
    "inference_speed": {...},
    "llm_generated_text": {...}
  }
}
```

### Console Output
Real-time testing progress with detailed analysis:
```
🚀 Starting Comprehensive ONNX Model Testing
✓ Detected 4 classes: {0: 'politics', 1: 'sports', 2: 'technology', 3: 'business'}
✓ Sample 1/8: 'Breaking: Congress passes...' → Class 0 (0.95)
📊 LLM Generated Text Test Summary:
✓ Success rate: 100.0%
📈 Prediction distribution: {0: 2, 1: 2, 2: 2, 3: 2}
```

## 🔗 API Integration

### LLM Router API Testing
```bash
python onnx_model_tester.py \
  --folder /path/to/model \
  --api-endpoint https://your-api.com/preprocess \
  --api-headers '{"Authorization": "Bearer your-token"}'
```

### Custom Preprocessing Pipeline
The tool supports integration with external preprocessing APIs for feature extraction and text normalization.

## 🚨 Troubleshooting

### Common Issues

**Missing API Key**
```
⚠ OpenRouter API key not found
💡 Add to .env file: OPENROUTER_API_KEY=your-api-key
```

**Text Data Detection**
```
⚠ Detected text data - cannot test directly
💡 Provide preprocessed numerical features
```

**Model Loading Errors**
```
✗ Model loading failed
💡 Check ONNX model compatibility and file path
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ONNX Runtime** for model execution
- **OpenRouter** for LLM API access
- **Meta Llama** for text generation capabilities

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review example usage scripts

---

**Built for enterprise-level ONNX model validation and deployment readiness assessment.** 