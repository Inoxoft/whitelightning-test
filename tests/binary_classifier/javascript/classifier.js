/**
 * Binary Sentiment Classifier - Web Implementation
 * Uses ONNX Runtime Web for client-side inference
 */

class BinarySentimentClassifier {
    constructor() {
        this.session = null;
        this.vocab = null;
        this.scaler = null;
        this.isLoaded = false;
    }

    async initialize() {
        try {
            updateStatus('loading', '🔄 Loading ONNX model...');
            
            // Load all resources in parallel
            const [modelBuffer, vocabData, scalerData] = await Promise.all([
                fetch('model.onnx').then(r => r.arrayBuffer()),
                fetch('vocab.json').then(r => r.json()),
                fetch('scaler.json').then(r => r.json())
            ]);

            updateStatus('loading', '🧠 Initializing ONNX session...');
            
            // Create ONNX session
            this.session = await ort.InferenceSession.create(modelBuffer);
            this.vocab = vocabData;
            this.scaler = scalerData;
            
            this.isLoaded = true;
            updateStatus('success', '✅ Model loaded successfully! Ready to analyze sentiment.');
            
            // Enable UI elements
            document.getElementById('textInput').disabled = false;
            document.getElementById('analyzeBtn').disabled = false;
            document.getElementById('clearBtn').disabled = false;
            document.getElementById('benchmarkBtn').disabled = false;
            
            console.log('🎉 Classifier initialized successfully');
            console.log('📊 Model info:', {
                inputNames: this.session.inputNames,
                outputNames: this.session.outputNames,
                vocabSize: Object.keys(this.vocab.vocab).length
            });
            
        } catch (error) {
            console.error('❌ Failed to initialize classifier:', error);
            updateStatus('error', `❌ Failed to load model: ${error.message}`);
            throw error;
        }
    }

    preprocessText(text) {
        const startTime = performance.now();
        
        // Get vocabulary and parameters
        const { vocab, idf } = this.vocab;
        const { mean, scale } = this.scaler;
        const vocabSize = idf.length;
        
        // Initialize feature vector
        const vector = new Float32Array(vocabSize);
        
        // Tokenize text
        const textLower = text.toLowerCase();
        const words = textLower.split(/\s+/).filter(word => word.length > 0);
        const totalWords = words.length;
        const wordCounts = {};
        
        // Count word frequencies
        for (const word of words) {
            wordCounts[word] = (wordCounts[word] || 0) + 1;
        }
        
        // Apply TF-IDF
        if (totalWords > 0) {
            for (const [word, count] of Object.entries(wordCounts)) {
                if (vocab.hasOwnProperty(word)) {
                    const idx = vocab[word];
                    if (idx < vocabSize) {
                        const tf = count / totalWords;
                        vector[idx] = tf * idf[idx];
                    }
                }
            }
        }
        
        // Apply scaling
        for (let i = 0; i < vocabSize; i++) {
            vector[i] = (vector[i] - mean[i]) / scale[i];
        }
        
        const preprocessTime = performance.now() - startTime;
        return { vector, preprocessTime };
    }

    async predict(text) {
        if (!this.isLoaded) {
            throw new Error('Classifier not initialized. Call initialize() first.');
        }

        const totalStart = performance.now();
        
        // Preprocessing
        const { vector, preprocessTime } = this.preprocessText(text);
        
        // Model inference
        const inferenceStart = performance.now();
        
        // Create input tensor
        const inputTensor = new ort.Tensor('float32', vector, [1, vector.length]);
        
        // Get input name dynamically
        const inputName = this.session.inputNames[0];
        
        // Run inference
        const feeds = { [inputName]: inputTensor };
        const results = await this.session.run(feeds);
        
        const inferenceTime = performance.now() - inferenceStart;
        
        // Extract results
        const outputTensor = results[this.session.outputNames[0]];
        const prediction = outputTensor.data[0];
        const sentiment = prediction > 0.5 ? 'Positive' : 'Negative';
        
        const totalTime = performance.now() - totalStart;
        const throughput = 1000 / totalTime;
        
        return {
            sentiment,
            confidence: prediction,
            metrics: {
                totalTime,
                preprocessTime,
                inferenceTime,
                throughput
            }
        };
    }

    async release() {
        if (this.session) {
            await this.session.release();
            this.session = null;
            this.isLoaded = false;
        }
    }
}

// Global classifier instance
let classifier = null;

// UI Helper Functions
function updateStatus(type, message) {
    const statusEl = document.getElementById('status');
    statusEl.className = `status ${type}`;
    statusEl.textContent = message;
    
    if (type === 'success') {
        setTimeout(() => {
            statusEl.style.display = 'none';
        }, 3000);
    }
}

function setExample(text) {
    document.getElementById('textInput').value = text;
}

function clearResults() {
    document.getElementById('textInput').value = '';
    document.getElementById('results').style.display = 'none';
}

function displayResults(result, inputText) {
    const resultsEl = document.getElementById('results');
    
    // Update sentiment
    const sentimentEl = document.getElementById('sentimentResult');
    sentimentEl.textContent = `${result.sentiment} ${result.sentiment === 'Positive' ? '😊' : '😔'}`;
    sentimentEl.className = `result-value sentiment-${result.sentiment.toLowerCase()}`;
    
    // Update confidence
    const confidence = (result.confidence * 100).toFixed(1);
    document.getElementById('confidenceResult').textContent = `${confidence}% (${result.confidence.toFixed(4)})`;
    document.getElementById('confidenceFill').style.width = `${confidence}%`;
    
    // Update input text
    document.getElementById('inputTextResult').textContent = `"${inputText}"`;
    
    // Update performance metrics
    const metrics = result.metrics;
    document.getElementById('totalTime').textContent = metrics.totalTime.toFixed(1);
    document.getElementById('preprocessTime').textContent = metrics.preprocessTime.toFixed(1);
    document.getElementById('inferenceTime').textContent = metrics.inferenceTime.toFixed(1);
    document.getElementById('throughput').textContent = metrics.throughput.toFixed(1);
    
    // Show results
    resultsEl.style.display = 'block';
    resultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function analyzeSentiment() {
    const textInput = document.getElementById('textInput');
    const text = textInput.value.trim();
    
    if (!text) {
        alert('Please enter some text to analyze.');
        return;
    }
    
    if (!classifier || !classifier.isLoaded) {
        alert('Classifier is not loaded yet. Please wait...');
        return;
    }
    
    try {
        updateStatus('loading', '🔄 Analyzing sentiment...');
        
        const result = await classifier.predict(text);
        
        updateStatus('success', '✅ Analysis complete!');
        displayResults(result, text);
        
        console.log('📊 Analysis result:', result);
        
    } catch (error) {
        console.error('❌ Analysis failed:', error);
        updateStatus('error', `❌ Analysis failed: ${error.message}`);
    }
}

async function runBenchmark() {
    if (!classifier || !classifier.isLoaded) {
        alert('Classifier is not loaded yet. Please wait...');
        return;
    }
    
    const numRuns = 50;
    const testText = "This is a sample text for performance testing.";
    
    try {
        updateStatus('loading', `🚀 Running benchmark (${numRuns} iterations)...`);
        
        const times = [];
        const preprocessTimes = [];
        const inferenceTimes = [];
        
        // Warmup
        for (let i = 0; i < 3; i++) {
            await classifier.predict(testText);
        }
        
        // Benchmark runs
        for (let i = 0; i < numRuns; i++) {
            const result = await classifier.predict(testText);
            times.push(result.metrics.totalTime);
            preprocessTimes.push(result.metrics.preprocessTime);
            inferenceTimes.push(result.metrics.inferenceTime);
        }
        
        // Calculate statistics
        const avgTime = times.reduce((a, b) => a + b, 0) / numRuns;
        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);
        const avgThroughput = 1000 / avgTime;
        
        const avgPreprocess = preprocessTimes.reduce((a, b) => a + b, 0) / numRuns;
        const avgInference = inferenceTimes.reduce((a, b) => a + b, 0) / numRuns;
        
        // Display benchmark results
        const benchmarkResult = {
            sentiment: 'Benchmark',
            confidence: 0.5,
            metrics: {
                totalTime: avgTime,
                preprocessTime: avgPreprocess,
                inferenceTime: avgInference,
                throughput: avgThroughput
            }
        };
        
        displayResults(benchmarkResult, `Benchmark Results (${numRuns} runs)`);
        
        updateStatus('success', `✅ Benchmark complete! Avg: ${avgTime.toFixed(1)}ms (${minTime.toFixed(1)}-${maxTime.toFixed(1)}ms)`);
        
        console.log('📈 Benchmark results:', {
            runs: numRuns,
            avgTime: avgTime.toFixed(2),
            minTime: minTime.toFixed(2),
            maxTime: maxTime.toFixed(2),
            avgThroughput: avgThroughput.toFixed(1),
            avgPreprocess: avgPreprocess.toFixed(2),
            avgInference: avgInference.toFixed(2)
        });
        
    } catch (error) {
        console.error('❌ Benchmark failed:', error);
        updateStatus('error', `❌ Benchmark failed: ${error.message}`);
    }
}

// Initialize classifier when page loads
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Binary Sentiment Classifier - Web Demo');
    console.log('🌐 ONNX Runtime Web version:', ort.version);
    
    try {
        classifier = new BinarySentimentClassifier();
        await classifier.initialize();
    } catch (error) {
        console.error('❌ Failed to initialize:', error);
    }
});

// Handle Enter key in textarea
document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('textInput');
    textInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            analyzeSentiment();
        }
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', async () => {
    if (classifier) {
        await classifier.release();
    }
}); 