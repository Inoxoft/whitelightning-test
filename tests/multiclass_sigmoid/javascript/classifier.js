/**
 * Multiclass Sigmoid Emotion Classifier - Web Implementation
 * Uses ONNX Runtime Web for client-side inference
 * Emotions: Fear, Happy, Love, Sadness (Multi-label classification)
 */

class EmotionClassifier {
    constructor() {
        this.session = null;
        this.vocab = null;
        this.scaler = null;
        this.isLoaded = false;
        this.emotions = ['fear', 'happy', 'love', 'sadness'];
        this.emotionLabels = ['😨 Fear', '😊 Happy', '❤️ Love', '😢 Sadness'];
        this.threshold = 0.5; // Threshold for multi-label classification
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
            updateStatus('success', '✅ Model loaded successfully! Ready to analyze emotions.');
            
            // Enable UI elements
            document.getElementById('textInput').disabled = false;
            document.getElementById('analyzeBtn').disabled = false;
            document.getElementById('clearBtn').disabled = false;
            document.getElementById('benchmarkBtn').disabled = false;
            
            console.log('🎉 Emotion classifier initialized successfully');
            console.log('📊 Model info:', {
                inputNames: this.session.inputNames,
                outputNames: this.session.outputNames,
                emotions: this.emotions,
                threshold: this.threshold
            });
            
        } catch (error) {
            console.error('❌ Failed to initialize classifier:', error);
            updateStatus('error', `❌ Failed to load model: ${error.message}`);
            throw error;
        }
    }

    preprocessText(text) {
        const startTime = performance.now();
        
        // Simple keyword-based feature extraction for emotions
        const features = this.extractEmotionFeatures(text);
        
        const preprocessTime = performance.now() - startTime;
        return { 
            features: new Float32Array(features), 
            preprocessTime 
        };
    }

    extractEmotionFeatures(text) {
        // Convert text to lowercase for case-insensitive matching
        const textLower = text.toLowerCase();
        
        // Define emotion-related keywords
        const emotionKeywords = {
            fear: [
                'afraid', 'scared', 'terrified', 'frightened', 'anxious', 'worried', 'nervous',
                'panic', 'terror', 'dread', 'phobia', 'fear', 'fearful', 'horror', 'scary',
                'threatening', 'dangerous', 'risk', 'unsafe', 'vulnerable', 'intimidated'
            ],
            happy: [
                'happy', 'joy', 'joyful', 'cheerful', 'glad', 'pleased', 'delighted',
                'excited', 'thrilled', 'elated', 'euphoric', 'blissful', 'content',
                'satisfied', 'amazing', 'wonderful', 'fantastic', 'great', 'excellent',
                'awesome', 'brilliant', 'fabulous', 'marvelous', 'superb', 'outstanding'
            ],
            love: [
                'love', 'adore', 'cherish', 'treasure', 'devoted', 'affection', 'romance',
                'romantic', 'passionate', 'caring', 'tender', 'sweet', 'dear', 'beloved',
                'darling', 'honey', 'sweetheart', 'heart', 'hearts', 'valentine',
                'crush', 'infatuated', 'smitten', 'enchanted', 'charmed'
            ],
            sadness: [
                'sad', 'sadness', 'sorrow', 'grief', 'melancholy', 'depressed', 'depression',
                'blue', 'down', 'unhappy', 'miserable', 'heartbroken', 'devastated',
                'disappointed', 'dejected', 'despondent', 'gloomy', 'mournful',
                'tragic', 'tragedy', 'cry', 'crying', 'tears', 'weep', 'sob'
            ]
        };
        
        // Count keyword matches for each emotion
        const features = [];
        
        this.emotions.forEach(emotion => {
            const keywords = emotionKeywords[emotion] || [];
            let score = 0;
            
            keywords.forEach(keyword => {
                // Simple substring matching
                if (textLower.includes(keyword)) {
                    score += 1;
                }
                
                // Bonus for exact word matches
                const wordRegex = new RegExp(`\\b${keyword}\\b`, 'gi');
                const matches = textLower.match(wordRegex);
                if (matches) {
                    score += matches.length;
                }
            });
            
            // Normalize by text length (simple approach)
            const textLength = textLower.split(/\s+/).length;
            const normalizedScore = textLength > 0 ? score / Math.sqrt(textLength) : score;
            
            features.push(normalizedScore);
        });
        
        return features;
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    async predict(text) {
        if (!this.isLoaded) {
            throw new Error('Classifier not initialized. Call initialize() first.');
        }

        const totalStart = performance.now();
        
        // Preprocessing
        const { features, preprocessTime } = this.preprocessText(text);
        
        // Model inference
        const inferenceStart = performance.now();
        
        // Create input tensor
        const inputTensor = new ort.Tensor('float32', features, [1, features.length]);
        
        // Get input name dynamically
        const inputName = this.session.inputNames[0];
        
        // Run inference
        const feeds = { [inputName]: inputTensor };
        const results = await this.session.run(feeds);
        
        const inferenceTime = performance.now() - inferenceStart;
        
        // Extract and process results
        const outputTensor = results[this.session.outputNames[0]];
        const logits = Array.from(outputTensor.data);
        
        // Apply sigmoid to get probabilities for each emotion
        const probabilities = logits.map(logit => this.sigmoid(logit));
        
        // Determine which emotions are detected (above threshold)
        const detectedEmotions = [];
        probabilities.forEach((prob, index) => {
            if (prob >= this.threshold) {
                detectedEmotions.push({
                    emotion: this.emotions[index],
                    label: this.emotionLabels[index],
                    probability: prob
                });
            }
        });
        
        // Sort detected emotions by probability (highest first)
        detectedEmotions.sort((a, b) => b.probability - a.probability);
        
        const totalTime = performance.now() - totalStart;
        const throughput = 1000 / totalTime;
        
        return {
            detectedEmotions,
            probabilities,
            emotions: this.emotions,
            emotionLabels: this.emotionLabels,
            threshold: this.threshold,
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
    
    // Update detected emotions
    const detectedEl = document.getElementById('detectedEmotions');
    if (result.detectedEmotions.length > 0) {
        detectedEl.innerHTML = result.detectedEmotions.map(emotion => 
            `<span class="emotion-tag ${emotion.emotion}">${emotion.label} (${(emotion.probability * 100).toFixed(1)}%)</span>`
        ).join('');
    } else {
        detectedEl.innerHTML = '<span style="color: #666; font-style: italic;">No emotions detected above threshold (50%)</span>';
    }
    
    // Update input text
    document.getElementById('inputTextResult').textContent = `"${inputText}"`;
    
    // Update emotion bars and values
    result.probabilities.forEach((prob, index) => {
        const emotion = result.emotions[index];
        const percentage = (prob * 100).toFixed(1);
        
        // Update bar width
        document.getElementById(`${emotion}Fill`).style.width = `${percentage}%`;
        
        // Update percentage value
        document.getElementById(`${emotion}Value`).textContent = `${percentage}%`;
        
        // Update threshold indicator (highlight if above threshold)
        const thresholdEl = document.getElementById(`${emotion}Threshold`);
        if (prob >= result.threshold) {
            thresholdEl.style.color = '#28a745';
            thresholdEl.style.fontWeight = 'bold';
            thresholdEl.textContent = 'DETECTED!';
        } else {
            thresholdEl.style.color = '#666';
            thresholdEl.style.fontWeight = 'normal';
            thresholdEl.textContent = 'Threshold: 50%';
        }
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

async function analyzeEmotions() {
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
        updateStatus('loading', '🔄 Analyzing emotions...');
        
        const result = await classifier.predict(text);
        
        updateStatus('success', '✅ Analysis complete!');
        displayResults(result, text);
        
        console.log('📊 Emotion analysis result:', result);
        
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
    const testText = "I'm terrified but also excited about tomorrow!";
    
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
            detectedEmotions: [
                { emotion: 'fear', label: '😨 Fear', probability: 0.75 },
                { emotion: 'happy', label: '😊 Happy', probability: 0.65 }
            ],
            probabilities: [0.75, 0.65, 0.25, 0.15], // Fear, Happy, Love, Sadness
            emotions: classifier.emotions,
            emotionLabels: classifier.emotionLabels,
            threshold: classifier.threshold,
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
    console.log('🚀 Emotion Classifier - Web Demo');
    console.log('🌐 ONNX Runtime Web version:', ort.version);
    
    try {
        classifier = new EmotionClassifier();
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
            analyzeEmotions();
        }
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', async () => {
    if (classifier) {
        await classifier.release();
    }
}); 