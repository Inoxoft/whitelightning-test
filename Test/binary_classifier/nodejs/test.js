const ort = require('onnxruntime-node');
const path = require('path');

class SpamDetectorTest {
    constructor() {
        this.modelPath = path.join(__dirname, '../../../models/spam_detector/model.onnx');
        this.session = null;
    }

    async initialize() {
        try {
            this.session = await ort.InferenceSession.create(this.modelPath);
            console.log('✓ Model loaded successfully');
        } catch (error) {
            console.error('✗ Failed to load model:', error);
            throw error;
        }
    }

    async testModelLoading() {
        console.log('\nTesting model loading...');
        if (this.session) {
            const inputNames = this.session.inputNames;
            const outputNames = this.session.outputNames;
            console.log('Input names:', inputNames);
            console.log('Output names:', outputNames);
        }
    }

    async testSpamDetection() {
        console.log('\nTesting spam detection...');
        const testTexts = [
            'Buy now! Limited time offer!',
            'Hello, how are you doing today?',
            'URGENT: Your account needs verification',
            'Meeting at 2 PM in the conference room'
        ];

        for (const text of testTexts) {
            const startTime = process.hrtime();
            
            // Prepare input tensor
            const inputTensor = this.preprocessText(text);
            
            // Run inference
            const results = await this.session.run({ input: inputTensor });
            const output = results.output.data;
            
            const [seconds, nanoseconds] = process.hrtime(startTime);
            const inferenceTime = seconds * 1000 + nanoseconds / 1000000; // Convert to milliseconds
            
            console.log('Text:', text);
            console.log('Spam probability:', output[0]);
            console.log('Is spam:', output[0] > 0.5);
            console.log('Inference time:', inferenceTime.toFixed(2), 'ms');
            console.log('---');
        }
    }

    async testPerformance() {
        console.log('\nTesting performance...');
        const numIterations = 100;
        let totalTime = 0;
        let maxMemory = 0;

        for (let i = 0; i < numIterations; i++) {
            const text = `Sample text for performance testing ${i}`;
            
            const startTime = process.hrtime();
            const startMemory = process.memoryUsage().heapUsed;
            
            // Run inference
            const inputTensor = this.preprocessText(text);
            await this.session.run({ input: inputTensor });
            
            const [seconds, nanoseconds] = process.hrtime(startTime);
            const endMemory = process.memoryUsage().heapUsed;
            
            totalTime += seconds * 1000 + nanoseconds / 1000000;
            maxMemory = Math.max(maxMemory, endMemory - startMemory);
        }

        const avgTime = totalTime / numIterations;
        const maxMemoryMB = maxMemory / (1024 * 1024);
        
        console.log('Average inference time:', avgTime.toFixed(2), 'ms');
        console.log('Maximum memory usage:', maxMemoryMB.toFixed(2), 'MB');
    }

    preprocessText(text) {
        // TODO: Implement actual text preprocessing based on your model's requirements
        // This is a placeholder that returns a dummy tensor
        return new ort.Tensor('float32', new Float32Array(512).fill(0), [1, 512]);
    }
}

async function runTests() {
    try {
        const test = new SpamDetectorTest();
        await test.initialize();
        await test.testModelLoading();
        await test.testSpamDetection();
        await test.testPerformance();
    } catch (error) {
        console.error('Test failed:', error);
        process.exit(1);
    }
}

runTests(); 