# 🌐 Client-Side JavaScript Binary Classifier - Web Implementation

A modern web-based sentiment analysis classifier using ONNX Runtime Web for client-side inference. This implementation runs entirely in the browser without requiring a server, providing fast and private sentiment analysis.

## 🚀 Features

- **🌐 Client-Side Inference**: Runs entirely in the browser using ONNX Runtime Web
- **🔒 Privacy-First**: No data sent to servers - all processing happens locally
- **⚡ Fast Performance**: Optimized for real-time sentiment analysis
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices
- **🎨 Modern UI**: Beautiful, intuitive interface with animations
- **📊 Performance Metrics**: Real-time monitoring of processing times
- **🧪 Benchmark Mode**: Performance testing with statistical analysis
- **📝 Example Texts**: Pre-loaded examples for quick testing

## 📋 System Requirements

### Browser Support
- **Chrome**: 78+ (recommended)
- **Firefox**: 72+
- **Safari**: 14+
- **Edge**: 79+

### Minimum Requirements
- **JavaScript**: ES2018+ support
- **WebAssembly**: WASM support required
- **Memory**: 1GB available RAM
- **Storage**: ~20MB for model files

## 🛠️ Installation & Setup

### Option 1: Local Web Server (Recommended)

Due to CORS restrictions, you need to serve the files through a web server:

#### Using Python (Built-in)
```bash
# Navigate to the directory
cd tests/binary_classifier/javascript

# Python 3
python -m http.server 8000

# Python 2 (legacy)
python -m SimpleHTTPServer 8000

# Open browser
open http://localhost:8000
```

#### Using Node.js (http-server)
```bash
# Install http-server globally
npm install -g http-server

# Navigate to directory and serve
cd tests/binary_classifier/javascript
http-server -p 8000

# Open browser
open http://localhost:8000
```

#### Using PHP (Built-in)
```bash
cd tests/binary_classifier/javascript
php -S localhost:8000

# Open browser
open http://localhost:8000
```

#### Using Live Server (VS Code Extension)
1. Install "Live Server" extension in VS Code
2. Right-click on `index.html`
3. Select "Open with Live Server"

### Option 2: Online Deployment

Deploy to any static hosting service:

#### Netlify
```bash
# Drag and drop the folder to netlify.com
# Or use Netlify CLI
npm install -g netlify-cli
netlify deploy --dir=tests/binary_classifier/javascript
```

#### GitHub Pages
```bash
# Push to GitHub repository
# Enable GitHub Pages in repository settings
# Access via: https://yourusername.github.io/repo/tests/binary_classifier/javascript
```

#### Vercel
```bash
npm install -g vercel
cd tests/binary_classifier/javascript
vercel
```

## 🎯 Usage

### Basic Web Interface

1. **Open the Application**
   - Navigate to `http://localhost:8000` (or your hosted URL)
   - Wait for the model to load (progress shown in status bar)

2. **Analyze Text**
   - Enter text in the textarea
   - Click "🔍 Analyze Sentiment" or press Enter
   - View results with confidence scores and performance metrics

3. **Use Example Texts**
   - Click any example button to load pre-written text
   - Examples include positive, negative, and neutral sentiments

4. **Run Benchmarks**
   - Click "⚡ Benchmark" to test performance
   - Runs 50 iterations and shows statistical analysis

### Programmatic Usage

You can also use the classifier programmatically:

```javascript
// Initialize classifier
const classifier = new BinarySentimentClassifier();
await classifier.initialize();

// Analyze sentiment
const result = await classifier.predict("This product is amazing!");

console.log(result);
// Output:
// {
//   sentiment: "Positive",
//   confidence: 0.8745,
//   metrics: {
//     totalTime: 15.2,
//     preprocessTime: 8.1,
//     inferenceTime: 6.8,
//     throughput: 65.8
//   }
// }

// Clean up
await classifier.release();
```

### Integration in Your Web App

```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js"></script>
</head>
<body>
    <script>
        class SentimentAnalyzer {
            constructor() {
                this.classifier = new BinarySentimentClassifier();
            }

            async init() {
                await this.classifier.initialize();
            }

            async analyze(text) {
                const result = await this.classifier.predict(text);
                return {
                    isPositive: result.sentiment === 'Positive',
                    confidence: result.confidence,
                    processingTime: result.metrics.totalTime
                };
            }
        }

        // Usage
        const analyzer = new SentimentAnalyzer();
        await analyzer.init();

        const result = await analyzer.analyze("I love this!");
        console.log(result); // { isPositive: true, confidence: 0.89, processingTime: 12.3 }
    </script>
</body>
</html>
```

## 📊 Expected Performance

### Typical Performance Metrics

| Metric | Desktop | Mobile | Target |
|--------|---------|---------|---------|
| **Total Time** | 8-25ms | 15-40ms | <50ms |
| **Preprocessing** | 3-12ms | 8-20ms | <25ms |
| **Inference** | 4-12ms | 6-18ms | <25ms |
| **Throughput** | 40-125 texts/sec | 25-65 texts/sec | >20/sec |

### Browser Performance Comparison

| Browser | Avg Time | Throughput | Notes |
|---------|----------|------------|-------|
| **Chrome** | 12ms | 83/sec | Best performance |
| **Firefox** | 15ms | 67/sec | Good performance |
| **Safari** | 18ms | 56/sec | Solid performance |
| **Edge** | 14ms | 71/sec | Good performance |

## 🔧 Technical Details

### Architecture
- **Frontend**: HTML5, CSS3, Modern JavaScript (ES2018+)
- **ML Runtime**: ONNX Runtime Web 1.19.2
- **Model**: Binary classification with sigmoid activation
- **Features**: TF-IDF vectorization (5000 dimensions)
- **Preprocessing**: Tokenization → TF-IDF → Standardization

### File Structure
```
javascript/
├── index.html              # Main web interface
├── classifier.js           # Core classifier implementation
├── model.onnx             # ONNX model (binary sentiment)
├── vocab.json             # TF-IDF vocabulary and IDF weights
├── scaler.json            # Feature scaling parameters
└── README.md              # This file
```

### Model Information
- **Input Shape**: [1, 5000] (TF-IDF features)
- **Output Shape**: [1, 1] (probability score)
- **Activation**: Sigmoid (0.0-1.0 range)
- **Threshold**: 0.5 (>0.5 = Positive, ≤0.5 = Negative)

### Preprocessing Pipeline
1. **Tokenization**: Split text into words, convert to lowercase
2. **TF-IDF Calculation**: 
   - Term Frequency: `count / total_words`
   - Multiply by IDF weights from vocabulary
3. **Feature Scaling**: Apply mean normalization and standard scaling
4. **Vector Creation**: Generate 5000-dimensional float32 array

## 🐛 Troubleshooting

### Common Issues

**1. "Failed to load model" Error**
```
❌ Problem: CORS policy blocking file access
✅ Solution: Use a web server (not file:// protocol)

# Quick fix:
python -m http.server 8000
```

**2. "Model files not found" Error**
```
❌ Problem: Missing model.onnx, vocab.json, or scaler.json
✅ Solution: Ensure all files are in the same directory

Required files:
- index.html
- classifier.js  
- model.onnx
- vocab.json
- scaler.json
```

**3. Slow Performance on Mobile**
```
❌ Problem: High processing times on mobile devices
✅ Solutions:
- Use Chrome on mobile (best WebAssembly performance)
- Reduce text length for real-time analysis
- Consider caching results for repeated texts
```

**4. Memory Issues with Large Texts**
```
❌ Problem: Browser crashes with very long texts
✅ Solution: Limit input text length

// Add text length validation
if (text.length > 10000) {
    text = text.substring(0, 10000);
    console.warn('Text truncated to 10,000 characters');
}
```

**5. WebAssembly Not Supported**
```
❌ Problem: Older browser without WASM support
✅ Solutions:
- Update browser to latest version
- Use Chrome, Firefox, Safari, or Edge
- Provide fallback message for unsupported browsers
```

### Browser-Specific Issues

**Safari**
- May show security warnings for local files
- Use HTTPS for production deployment
- Performance slightly slower than Chrome

**Firefox**
- Excellent WebAssembly support
- May require enabling certain flags in about:config for local development

**Mobile Browsers**
- iOS Safari: Works well, slightly slower than desktop
- Chrome Mobile: Best mobile performance
- Firefox Mobile: Good compatibility

## 🔒 Security & Privacy

### Privacy Benefits
- **No Data Transmission**: All processing happens locally in the browser
- **No Server Logs**: No text data is sent to any server
- **Offline Capable**: Works without internet after initial load
- **GDPR Compliant**: No personal data processing on servers

### Security Considerations
- **Content Security Policy**: Implement CSP headers for production
- **HTTPS Required**: Use HTTPS for production deployments
- **Input Validation**: Sanitize user inputs if integrating with forms

```html
<!-- Example CSP header -->
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net;">
```

## 🚀 Deployment

### Production Deployment Checklist

1. **Optimize Files**
   ```bash
   # Minify JavaScript (optional)
   uglifyjs classifier.js -o classifier.min.js
   
   # Enable gzip compression on server
   # Use CDN for ONNX Runtime Web library
   ```

2. **Configure Web Server**
   ```nginx
   # Nginx configuration
   location ~* \.(onnx|json)$ {
       expires 1y;
       add_header Cache-Control "public, immutable";
   }
   
   # Enable gzip for better performance
   gzip on;
   gzip_types application/octet-stream application/json;
   ```

3. **Performance Monitoring**
   ```javascript
   // Add analytics (optional)
   function trackPerformance(metrics) {
       // Send to your analytics service
       analytics.track('sentiment_analysis', {
           totalTime: metrics.totalTime,
           throughput: metrics.throughput
       });
   }
   ```

## 🆚 Comparison with Other Implementations

| Feature | JavaScript (Web) | Node.js | Python | Native |
|---------|------------------|---------|---------|---------|
| **Performance** | 8-25ms | 20-40ms | 15-30ms | 5-15ms |
| **Deployment** | Static hosting | Server required | Server required | Binary distribution |
| **Privacy** | Excellent (local) | Server-side | Server-side | Local |
| **Scalability** | Client-side | High | High | Medium |
| **Setup** | Very easy | Easy | Easy | Complex |
| **Dependencies** | Browser only | Node.js + npm | Python + pip | Compiled binary |

### When to Use JavaScript Implementation

✅ **Best For:**
- **Privacy-Sensitive Applications**: No data leaves the client
- **Static Websites**: No server infrastructure needed
- **Real-Time UI**: Immediate feedback without network latency
- **Demos & Prototypes**: Easy to share and demonstrate
- **Offline Applications**: Works without internet connection
- **Cost-Effective Solutions**: No server costs for inference

❌ **Not Ideal For:**
- **Batch Processing**: Limited by browser memory and performance
- **Server-Side Integration**: Use Node.js implementation instead
- **Very High Volume**: Server implementations are more scalable
- **Resource-Constrained Devices**: Native implementations are faster

## 📈 Example Output

```
🤖 SENTIMENT CLASSIFIER - WEB DEMO
===================================

📝 Input: "This product is absolutely amazing! I love it!"

📊 RESULTS:
   🏆 Sentiment: Positive 😊
   📈 Confidence: 87.4% (0.8742)
   
📈 PERFORMANCE METRICS:
   ⏱️ Total Time: 12.3ms
   🔧 Preprocessing: 6.8ms (55.3%)
   🧠 Inference: 4.9ms (39.8%)
   ⚡ Throughput: 81.3 texts/sec

💻 SYSTEM INFO:
   🌐 Browser: Chrome 119.0.0.0
   🔧 ONNX Runtime: 1.19.2
   💾 Memory Usage: +2.1MB
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Performance Optimization**
   - WebAssembly SIMD optimizations
   - Better caching strategies
   - Memory usage optimization

2. **UI/UX Enhancements**
   - Dark mode support
   - Better mobile responsiveness
   - Accessibility improvements

3. **Features**
   - Batch processing support
   - Export/import functionality
   - Integration examples

4. **Testing**
   - Unit tests for preprocessing
   - Cross-browser testing
   - Performance regression tests

---

*For more information about the sentiment analysis models and other implementations, see the main [README.md](../../../README.md) in the project root.* 