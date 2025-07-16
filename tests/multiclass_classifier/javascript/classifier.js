/**
 * Multiclass Topic Classifier - Web Implementation
 * Uses ONNX Runtime Web for client-side inference
 * Categories: Business, Health, Politics, Sports
 */

class MulticlassTopicClassifier {
    constructor() {
        this.session = null;
        this.vocab = null;
        this.scaler = null;
        this.isLoaded = false;
        this.categories = ['Business', 'Health', 'Politics', 'Sports'];
        this.categoryEmojis = ['💼', '🏥', '🏛️', '⚽'];
        this.maxSequenceLength = 30; // For tokenized sequences
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
            updateStatus('success', '✅ Model loaded successfully! Ready to classify topics.');
            
            // Enable UI elements
            document.getElementById('textInput').disabled = false;
            document.getElementById('analyzeBtn').disabled = false;
            document.getElementById('clearBtn').disabled = false;
            document.getElementById('benchmarkBtn').disabled = false;
            
            console.log('🎉 Classifier initialized successfully');
            console.log('📊 Model info:', {
                inputNames: this.session.inputNames,
                outputNames: this.session.outputNames,
                vocabSize: Object.keys(this.vocab).length,
                categories: this.categories
            });
            
        } catch (error) {
            console.error('❌ Failed to initialize classifier:', error);
            updateStatus('error', `❌ Failed to load model: ${error.message}`);
            throw error;
        }
    }

    preprocessText(text) {
        const startTime = performance.now();
        
        // Tokenize text into sequence of token IDs
        const tokens = this.tokenizeText(text);
        
        // Pad or truncate to fixed length
        const sequence = new Array(this.maxSequenceLength).fill(0);
        for (let i = 0; i < Math.min(tokens.length, this.maxSequenceLength); i++) {
            sequence[i] = tokens[i];
        }
        
        const preprocessTime = performance.now() - startTime;
        return { 
            sequence: new Float32Array(sequence), 
            preprocessTime 
        };
    }

    tokenizeText(text) {
        // Simple tokenization: convert text to lowercase, split by spaces and punctuation
        const tokens = [];
        const words = text.toLowerCase()
            .replace(/[^\w\s]/g, ' ')  // Replace punctuation with spaces
            .split(/\s+/)              // Split by whitespace
            .filter(word => word.length > 0);
        
        for (const word of words) {
            // Map word to token ID, default to 1 (unknown token) if not in vocab
            const tokenId = this.vocab[word] || 1;
            tokens.push(tokenId);
        }
        
        return tokens;
    }

    softmax(logits) {
        // Apply softmax to convert logits to probabilities
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(x => Math.exp(x - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        return expLogits.map(x => x / sumExp);
    }

    async predict(text) {
        if (!this.isLoaded) {
            throw new Error('Classifier not initialized. Call initialize() first.');
        }

        const totalStart = performance.now();
        
        // Preprocessing
        const { sequence, preprocessTime } = this.preprocessText(text);
        
        // Model inference
        const inferenceStart = performance.now();
        
        // Create input tensor
        const inputTensor = new ort.Tensor('float32', sequence, [1, this.maxSequenceLength]);
        
        // Get input name dynamically
        const inputName = this.session.inputNames[0];
        
        // Run inference
        const feeds = { [inputName]: inputTensor };
        const results = await this.session.run(feeds);
        
        const inferenceTime = performance.now() - inferenceStart;
        
        // Extract and process results
        const outputTensor = results[this.session.outputNames[0]];
        const logits = Array.from(outputTensor.data);
        
        // Apply softmax to get probabilities
        const probabilities = this.softmax(logits);
        
        // Find predicted class
        const predictedIndex = probabilities.indexOf(Math.max(...probabilities));
        const predictedTopic = this.categories[predictedIndex];
        const confidence = probabilities[predictedIndex];
        
        const totalTime = performance.now() - totalStart;
        const throughput = 1000 / totalTime;
        
        return {
            topic: predictedTopic,
            topicIndex: predictedIndex,
            confidence,
            probabilities,
            categories: this.categories,
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
    
    // Update topic result
    const topicEl = document.getElementById('topicResult');
    const emoji = classifier.categoryEmojis[result.topicIndex];
    const topicClass = `topic-${result.topic.toLowerCase()}`;
    topicEl.innerHTML = `<span class="topic-result ${topicClass}">${emoji} ${result.topic}</span>`;
    
    // Update input text
    document.getElementById('inputTextResult').textContent = `"${inputText}"`;
    
    // Update probability bars and values
    const categories = ['business', 'health', 'politics', 'sports'];
    result.probabilities.forEach((prob, index) => {
        const category = categories[index];
        const percentage = (prob * 100).toFixed(1);
        
        // Update bar width
        document.getElementById(`${category}Fill`).style.width = `${percentage}%`;
        
        // Update percentage value
        document.getElementById(`${category}Value`).textContent = `${percentage}%`;
    });
    
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

async function classifyTopic() {
    const textInput = document.getElementById('textInput');
    const text = textInput.value.trim();
    
    if (!text) {
        alert('Please enter some text to classify.');
        return;
    }
    
    if (!classifier || !classifier.isLoaded) {
        alert('Classifier is not loaded yet. Please wait...');
        return;
    }
    
    try {
        updateStatus('loading', '🔄 Classifying topic...');
        
        const result = await classifier.predict(text);
        
        updateStatus('success', '✅ Classification complete!');
        displayResults(result, text);
        
        console.log('📊 Classification result:', result);
        
    } catch (error) {
        console.error('❌ Classification failed:', error);
        updateStatus('error', `❌ Classification failed: ${error.message}`);
    }
}

async function runBenchmark() {
    if (!classifier || !classifier.isLoaded) {
        alert('Classifier is not loaded yet. Please wait...');
        return;
    }
    
    const numRuns = 50;
    const testText = "Apple announces new iPhone with advanced AI capabilities.";
    
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
            topic: 'Benchmark',
            topicIndex: 0, // Default to Business for display
            confidence: 0.25, // Equal probability for display
            probabilities: [0.25, 0.25, 0.25, 0.25], // Equal for all categories
            categories: classifier.categories,
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
    console.log('🚀 Multiclass Topic Classifier - Web Demo');
    console.log('🌐 ONNX Runtime Web version:', ort.version);
    
    try {
        classifier = new MulticlassTopicClassifier();
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
            classifyTopic();
        }
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', async () => {
    if (classifier) {
        await classifier.release();
    }
}); 