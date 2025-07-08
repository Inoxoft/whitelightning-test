const ort = require('onnxruntime-node');
const fs = require('fs');
const os = require('os');

async function loadModelArtifacts() {
    console.log('🔧 Loading components...');
    
    // Load ONNX model
    const session = await ort.InferenceSession.create('model.onnx');
    console.log('✅ ONNX model loaded');
    
    // Load vectorizer data
    const vectorizerData = JSON.parse(fs.readFileSync('vocab.json', 'utf8'));
    console.log(`✅ Vectorizer loaded (vocab: ${Object.keys(vectorizerData.vocabulary || vectorizerData.vocab).length} words)`);
    
    // Load classes
    const classes = JSON.parse(fs.readFileSync('scaler.json', 'utf8'));
    console.log(`✅ Classes loaded: ${Object.keys(classes).map(k => classes[k]).join(', ')}`);
    
    return { session, vectorizer: vectorizerData, classes };
}

function getSystemInfo() {
    return {
        platform: os.platform(),
        arch: os.arch(),
        cpus: os.cpus().length,
        memory: Math.round(os.totalmem() / (1024 * 1024 * 1024) * 10) / 10,
        nodeVersion: process.version
    };
}

async function preprocessText(text, vectorizer) {
    const startTime = Date.now();
    
    try {
        // Handle both vocabulary formats
        const vocabulary = vectorizer.vocabulary || vectorizer.vocab;
        const idf = vectorizer.idf;
        const maxFeatures = vectorizer.max_features || 5000;
        
        if (!vocabulary || !idf) {
            throw new Error('Invalid vectorizer: missing vocabulary or idf');
        }

        // Tokenize text (match sklearn's TfidfVectorizer exactly)
        const tokens = text.toLowerCase().match(/\b\w\w+\b/g) || [];
        console.log(`📊 Tokens found: ${tokens.length}, First 10: ${tokens.slice(0, 10).join(', ')}`);
        
        // Count term frequencies
        const termCounts = {};
        tokens.forEach(token => {
            termCounts[token] = (termCounts[token] || 0) + 1;
        });

        // Create TF-IDF vector
        const vector = new Float32Array(maxFeatures).fill(0);
        let foundInVocab = 0;
        
        // Apply TF-IDF: raw term frequency * IDF weight
        for (const [term, count] of Object.entries(termCounts)) {
            const termIndex = vocabulary[term];
            if (termIndex !== undefined && termIndex < maxFeatures) {
                vector[termIndex] = count * idf[termIndex];
                foundInVocab++;
            }
        }
        
        console.log(`📊 Found ${foundInVocab} terms in vocabulary out of ${tokens.length} total tokens`);
        
        // L2 normalization (crucial for sklearn compatibility)
        let norm = 0;
        for (let i = 0; i < vector.length; i++) {
            norm += vector[i] * vector[i];
        }
        norm = Math.sqrt(norm);
        
        // Normalize only if norm > 0
        if (norm > 0) {
            for (let i = 0; i < vector.length; i++) {
                vector[i] = vector[i] / norm;
            }
        }
        
        console.log(`📊 TF-IDF: ${vector.filter(v => v !== 0).length} non-zero, max: ${Math.max(...vector).toFixed(4)}, norm: ${norm.toFixed(4)}`);

        const preprocessingTime = Date.now() - startTime;
        return { vector, preprocessingTime };
    } catch (error) {
        console.error('❌ Preprocessing error:', error);
        throw error;
    }
}

async function runInference(session, vector) {
    const startTime = Date.now();
    
    const inputName = session.inputNames[0];
    const tensor = new ort.Tensor('float32', vector, [1, vector.length]);
    
    const feeds = {};
    feeds[inputName] = tensor;
    
    const results = await session.run(feeds);
    const outputTensor = results[Object.keys(results)[0]];
    const predictions = Array.from(outputTensor.data);
    
    const inferenceTime = Date.now() - startTime;
    
    return { predictions, inferenceTime };
}

async function main() {
    // Get command line argument for custom text
    const testText = process.argv[2] || 
        "I'm about to give birth, and I'm terrified. What if something goes wrong? What if I can't handle the pain? Received an unexpected compliment at work today. Small moments of happiness can make a big difference.";
    
    console.log('🤖 ONNX MULTICLASS SIGMOID CLASSIFIER - NODE.JS IMPLEMENTATION');
    console.log('='.repeat(65));
    console.log(`🔄 Processing: ${testText}`);
    console.log();
    
    // System information
    const systemInfo = getSystemInfo();
    console.log('💻 SYSTEM INFORMATION:');
    console.log(`   Platform: ${systemInfo.platform}`);
    console.log(`   Architecture: ${systemInfo.arch}`);
    console.log(`   CPU Cores: ${systemInfo.cpus}`);
    console.log(`   Total Memory: ${systemInfo.memory} GB`);
    console.log(`   Runtime: Node.js ${systemInfo.nodeVersion}`);
    console.log();
    
    // Track memory usage
    const memoryStart = process.memoryUsage().heapUsed / 1024 / 1024; // MB
    const totalStartTime = Date.now();
    
    try {
        // Load model and artifacts
        const { session, vectorizer, classes } = await loadModelArtifacts();
        
        // Preprocess text
        const { vector, preprocessingTime } = await preprocessText(testText, vectorizer);
        
        console.log(`📊 TF-IDF shape: [1, ${vector.length}]`);
        console.log(`📊 Non-zero features: ${vector.filter(v => v !== 0).length}`);
        console.log(`📊 Max TF-IDF value: ${Math.max(...vector).toFixed(4)}`);
        console.log(`📊 Min TF-IDF value: ${Math.min(...vector).toFixed(4)}`);
        console.log();
        
        // Run inference
        const { predictions, inferenceTime } = await runInference(session, vector);
        
        // Post-processing
        const postprocessingStart = Date.now();
        
        console.log('📊 EMOTION ANALYSIS RESULTS:');
        const emotionResults = [];
        predictions.forEach((prob, idx) => {
            const className = classes[idx.toString()] || `Class ${idx}`;
            emotionResults.push({ class: className, probability: prob });
            console.log(`   ${className}: ${prob.toFixed(3)}`);
        });
        
        // Find dominant emotion
        const dominantEmotion = emotionResults.reduce((max, current) => 
            current.probability > max.probability ? current : max
        );
        console.log(`   🏆 Dominant Emotion: ${dominantEmotion.class} (${dominantEmotion.probability.toFixed(3)})`);
        
        const postprocessingTime = Date.now() - postprocessingStart;
        
        console.log(`   📝 Input Text: "${testText}"`);
        console.log();
        
        // Performance metrics
        const totalTime = Date.now() - totalStartTime;
        const memoryEnd = process.memoryUsage().heapUsed / 1024 / 1024; // MB
        const memoryDelta = memoryEnd - memoryStart;
        
        console.log('📈 PERFORMANCE SUMMARY:');
        console.log(`   Total Processing Time: ${totalTime}ms`);
        console.log(`   ┣━ Preprocessing: ${preprocessingTime}ms (${(preprocessingTime/totalTime*100).toFixed(1)}%)`);
        console.log(`   ┣━ Model Inference: ${inferenceTime}ms (${(inferenceTime/totalTime*100).toFixed(1)}%)`);
        console.log(`   ┗━ Postprocessing: ${postprocessingTime}ms (${(postprocessingTime/totalTime*100).toFixed(1)}%)`);
        console.log();
        
        // Throughput
        const throughput = 1000 / totalTime;
        console.log('🚀 THROUGHPUT:');
        console.log(`   Texts per second: ${throughput.toFixed(1)}`);
        console.log();
        
        // Memory usage
        console.log('💾 RESOURCE USAGE:');
        console.log(`   Memory Start: ${memoryStart.toFixed(2)}MB`);
        console.log(`   Memory End: ${memoryEnd.toFixed(2)}MB`);
        console.log(`   Memory Delta: ${memoryDelta >= 0 ? '+' : ''}${memoryDelta.toFixed(2)}MB`);
        console.log();
        
        // Performance rating
        let rating;
        if (totalTime < 50) {
            rating = '🚀 EXCELLENT';
        } else if (totalTime < 100) {
            rating = '✅ GOOD';
        } else if (totalTime < 500) {
            rating = '⚠️ ACCEPTABLE';
        } else {
            rating = '🐌 SLOW';
        }
        
        console.log(`🎯 PERFORMANCE RATING: ${rating}`);
        console.log(`   (${totalTime}ms total - Target: <100ms)`);
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        process.exit(1);
    }
}

if (require.main === module) {
    main().catch(console.error);
} 