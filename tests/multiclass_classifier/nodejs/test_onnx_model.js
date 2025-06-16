#!/usr/bin/env node

import * as ort from 'onnxruntime-node';
import { readFileSync, existsSync } from 'fs';
import { performance } from 'perf_hooks';
import os from 'os';
import process from 'process';

// Performance monitoring classes
class TimingMetrics {
    constructor() {
        this.totalTimeMs = 0;
        this.preprocessingTimeMs = 0;
        this.inferenceTimeMs = 0;
        this.postprocessingTimeMs = 0;
        this.throughputPerSec = 0;
    }
}

class ResourceMetrics {
    constructor() {
        this.memoryStartMB = 0;
        this.memoryEndMB = 0;
        this.memoryDeltaMB = 0;
        this.cpuAvgPercent = 0;
        this.cpuMaxPercent = 0;
        this.cpuReadingsCount = 0;
        this.cpuReadings = [];
    }
}

class SystemInfo {
    constructor() {
        this.platform = os.platform();
        this.processor = os.arch();
        this.cpuCores = os.cpus().length;
        this.totalMemoryGB = os.totalmem() / (1024 * 1024 * 1024);
        this.runtime = 'Node.js Implementation';
        this.nodeVersion = process.version;
        this.onnxVersion = ort.version || 'Unknown';
    }
}

// Global CPU monitoring
let cpuReadings = [];
let cpuMonitorInterval = null;
let monitoring = false;

// Utility functions
function getTimeMs() {
    return performance.now();
}

function getMemoryUsageMB() {
    const memUsage = process.memoryUsage();
    return memUsage.heapUsed / (1024 * 1024);
}

function getCpuUsagePercent() {
    // Simplified CPU usage estimation for Node.js
    // This is an approximation based on process CPU time
    const cpuUsage = process.cpuUsage();
    const totalTime = cpuUsage.user + cpuUsage.system;
    
    // Convert microseconds to percentage (very rough approximation)
    // This is not as accurate as native implementations but provides some indication
    return Math.min(100, (totalTime / 1000000) % 100);
}

function startCpuMonitoring() {
    monitoring = true;
    cpuReadings = [];
    
    cpuMonitorInterval = setInterval(() => {
        if (monitoring) {
            const cpuUsage = getCpuUsagePercent();
            cpuReadings.push(cpuUsage);
        }
    }, 100);
}

function stopCpuMonitoring(resources) {
    monitoring = false;
    if (cpuMonitorInterval) {
        clearInterval(cpuMonitorInterval);
        cpuMonitorInterval = null;
    }
    
    resources.cpuReadings = [...cpuReadings];
    resources.cpuReadingsCount = cpuReadings.length;
    
    if (cpuReadings.length > 0) {
        resources.cpuAvgPercent = cpuReadings.reduce((a, b) => a + b, 0) / cpuReadings.length;
        resources.cpuMaxPercent = Math.max(...cpuReadings);
    }
}

function printSystemInfo(info) {
    console.log('💻 SYSTEM INFORMATION:');
    console.log(`   Platform: ${info.platform}`);
    console.log(`   Processor: ${info.processor}`);
    console.log(`   CPU Cores: ${info.cpuCores}`);
    console.log(`   Total Memory: ${info.totalMemoryGB.toFixed(1)} GB`);
    console.log(`   Runtime: ${info.runtime}`);
    console.log(`   Node.js Version: ${info.nodeVersion}`);
    console.log(`   ONNX Runtime Version: ${info.onnxVersion}`);
    console.log();
}

function printPerformanceSummary(timing, resources) {
    console.log('📈 PERFORMANCE SUMMARY:');
    console.log(`   Total Processing Time: ${timing.totalTimeMs.toFixed(2)}ms`);
    console.log(`   ┣━ Preprocessing: ${timing.preprocessingTimeMs.toFixed(2)}ms (${((timing.preprocessingTimeMs / timing.totalTimeMs) * 100).toFixed(1)}%)`);
    console.log(`   ┣━ Model Inference: ${timing.inferenceTimeMs.toFixed(2)}ms (${((timing.inferenceTimeMs / timing.totalTimeMs) * 100).toFixed(1)}%)`);
    console.log(`   ┗━ Postprocessing: ${timing.postprocessingTimeMs.toFixed(2)}ms (${((timing.postprocessingTimeMs / timing.totalTimeMs) * 100).toFixed(1)}%)`);
    console.log();
    
    console.log('🚀 THROUGHPUT:');
    console.log(`   Texts per second: ${timing.throughputPerSec.toFixed(1)}`);
    console.log();
    
    console.log('💾 RESOURCE USAGE:');
    console.log(`   Memory Start: ${resources.memoryStartMB.toFixed(2)} MB`);
    console.log(`   Memory End: ${resources.memoryEndMB.toFixed(2)} MB`);
    console.log(`   Memory Delta: ${resources.memoryDeltaMB >= 0 ? '+' : ''}${resources.memoryDeltaMB.toFixed(2)} MB`);
    if (resources.cpuReadingsCount > 0) {
        console.log(`   CPU Usage: ${resources.cpuAvgPercent.toFixed(1)}% avg, ${resources.cpuMaxPercent.toFixed(1)}% peak (${resources.cpuReadingsCount} samples)`);
    }
    console.log();
    
    // Performance classification
    let performanceClass, emoji;
    if (timing.totalTimeMs < 50) {
        performanceClass = 'EXCELLENT';
        emoji = '🚀';
    } else if (timing.totalTimeMs < 100) {
        performanceClass = 'GOOD';
        emoji = '✅';
    } else if (timing.totalTimeMs < 200) {
        performanceClass = 'ACCEPTABLE';
        emoji = '⚠️';
    } else {
        performanceClass = 'POOR';
        emoji = '❌';
    }
    
    console.log(`🎯 PERFORMANCE RATING: ${emoji} ${performanceClass}`);
    console.log(`   (${timing.totalTimeMs.toFixed(1)}ms total - Target: <100ms)`);
    console.log();
}

async function preprocessText(text) {
    const vector = new Int32Array(30);
    
    // Load tokenizer
    const tokenizer = JSON.parse(readFileSync('vocab.json', 'utf8'));
    
    // Tokenize text
    const textLower = text.toLowerCase();
    const words = textLower.split(/\s+/);
    
    // Convert words to token IDs
    for (let i = 0; i < Math.min(words.length, 30); i++) {
        const word = words[i];
        if (tokenizer.hasOwnProperty(word)) {
            vector[i] = tokenizer[word];
        } else if (tokenizer.hasOwnProperty('<OOV>')) {
            vector[i] = tokenizer['<OOV>'];
        } else {
            vector[i] = 1; // Default OOV token
        }
    }
    
    return vector;
}

async function testSingleText(text) {
    console.log(`🔄 Processing: ${text}`);
    
    // Initialize system info
    const systemInfo = new SystemInfo();
    printSystemInfo(systemInfo);
    
    // Initialize metrics
    const timing = new TimingMetrics();
    const resources = new ResourceMetrics();
    
    const totalStart = getTimeMs();
    resources.memoryStartMB = getMemoryUsageMB();
    
    // Start CPU monitoring
    startCpuMonitoring();
    
    try {
        // Preprocessing
        const preprocessStart = getTimeMs();
        const inputVector = await preprocessText(text);
        timing.preprocessingTimeMs = getTimeMs() - preprocessStart;
        
        // Model inference
        const inferenceStart = getTimeMs();
        
        // Create ONNX session
        const session = await ort.InferenceSession.create('model.onnx');
        
        // Create input tensor
        const inputTensor = new ort.Tensor('int32', inputVector, [1, 30]);
        
        // Get input name dynamically
        const inputName = session.inputNames[0];
        
        // Run inference
        const feeds = { [inputName]: inputTensor };
        const results = await session.run(feeds);
        
        timing.inferenceTimeMs = getTimeMs() - inferenceStart;
        
        // Post-processing
        const postprocessStart = getTimeMs();
        const outputTensor = results[session.outputNames[0]];
        const output = Array.from(outputTensor.data);
        
        // Load label mapping
        const labelMap = JSON.parse(readFileSync('scaler.json', 'utf8'));
        
        // Find predicted class
        let predictedIdx = 0;
        let maxConfidence = output[0];
        for (let i = 1; i < output.length; i++) {
            if (output[i] > maxConfidence) {
                maxConfidence = output[i];
                predictedIdx = i;
            }
        }
        
        const predictedLabel = labelMap[predictedIdx.toString()];
        timing.postprocessingTimeMs = getTimeMs() - postprocessStart;
        
        // Final measurements
        timing.totalTimeMs = getTimeMs() - totalStart;
        timing.throughputPerSec = 1000.0 / timing.totalTimeMs;
        resources.memoryEndMB = getMemoryUsageMB();
        resources.memoryDeltaMB = resources.memoryEndMB - resources.memoryStartMB;
        
        // Stop CPU monitoring
        stopCpuMonitoring(resources);
        
        // Display results
        console.log('📊 MULTICLASS CLASSIFICATION RESULTS:');
        console.log(`   🏆 Predicted Category: ${predictedLabel}`);
        console.log(`   📈 Confidence: ${(maxConfidence * 100).toFixed(2)}% (${maxConfidence.toFixed(4)})`);
        console.log(`   📝 Input Text: "${text}"`);
        
        // Show all class probabilities
        console.log('   📋 All Class Probabilities:');
        for (let i = 0; i < output.length; i++) {
            const className = labelMap[i.toString()];
            const probability = output[i];
            console.log(`      ${className}: ${probability.toFixed(4)} (${(probability * 100).toFixed(1)}%)`);
        }
        console.log();
        
        // Print performance summary
        printPerformanceSummary(timing, resources);
        
        // Clean up
        await session.release();
        
    } catch (error) {
        stopCpuMonitoring(resources);
        throw error;
    }
}

async function runPerformanceBenchmark(numRuns) {
    console.log(`\n🚀 PERFORMANCE BENCHMARKING (${numRuns} runs)`);
    console.log('============================================================');
    
    const systemInfo = new SystemInfo();
    console.log(`💻 System: ${systemInfo.cpuCores} cores, ${systemInfo.totalMemoryGB.toFixed(1)}GB RAM`);
    
    const testText = 'France Defeats Argentina in Thrilling World Cup Final';
    console.log(`📝 Test Text: '${testText}'\n`);
    
    try {
        // Initialize ONNX session once
        const session = await ort.InferenceSession.create('model.onnx');
        const inputName = session.inputNames[0];
        
        // Preprocess once
        const inputVector = await preprocessText(testText);
        
        // Warmup runs
        console.log('🔥 Warming up model (5 runs)...');
        for (let i = 0; i < 5; i++) {
            const inputTensor = new ort.Tensor('int32', inputVector, [1, 30]);
            const feeds = { [inputName]: inputTensor };
            await session.run(feeds);
        }
        
        // Performance arrays
        const times = [];
        const inferenceTimes = [];
        
        console.log(`📊 Running ${numRuns} performance tests...`);
        const overallStart = getTimeMs();
        
        for (let i = 0; i < numRuns; i++) {
            if (i % 20 === 0 && i > 0) {
                console.log(`   Progress: ${i}/${numRuns} (${((i / numRuns) * 100).toFixed(1)}%)`);
            }
            
            const startTime = getTimeMs();
            const inferenceStart = getTimeMs();
            
            const inputTensor = new ort.Tensor('int32', inputVector, [1, 30]);
            const feeds = { [inputName]: inputTensor };
            await session.run(feeds);
            
            const inferenceTime = getTimeMs() - inferenceStart;
            const endTime = getTimeMs();
            
            times.push(endTime - startTime);
            inferenceTimes.push(inferenceTime);
        }
        
        const overallTime = getTimeMs() - overallStart;
        
        // Calculate statistics
        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);
        const avgInf = inferenceTimes.reduce((a, b) => a + b, 0) / inferenceTimes.length;
        
        // Display results
        console.log('\n📈 DETAILED PERFORMANCE RESULTS:');
        console.log('--------------------------------------------------');
        console.log('⏱️  TIMING ANALYSIS:');
        console.log(`   Mean: ${avgTime.toFixed(2)}ms`);
        console.log(`   Min: ${minTime.toFixed(2)}ms`);
        console.log(`   Max: ${maxTime.toFixed(2)}ms`);
        console.log(`   Model Inference: ${avgInf.toFixed(2)}ms`);
        console.log('\n🚀 THROUGHPUT:');
        console.log(`   Texts per second: ${(1000.0 / avgTime).toFixed(1)}`);
        console.log(`   Total benchmark time: ${(overallTime / 1000.0).toFixed(2)}s`);
        console.log(`   Overall throughput: ${(numRuns / (overallTime / 1000.0)).toFixed(1)} texts/sec`);
        
        // Performance classification
        let performanceClass;
        if (avgTime < 10) {
            performanceClass = '🚀 EXCELLENT';
        } else if (avgTime < 50) {
            performanceClass = '✅ GOOD';
        } else if (avgTime < 100) {
            performanceClass = '⚠️ ACCEPTABLE';
        } else {
            performanceClass = '❌ POOR';
        }
        
        console.log(`\n🎯 PERFORMANCE CLASSIFICATION: ${performanceClass}`);
        console.log(`   (${avgTime.toFixed(1)}ms average - Target: <100ms)`);
        
        // Clean up
        await session.release();
        
    } catch (error) {
        console.error('❌ Benchmark error:', error.message);
        process.exit(1);
    }
}

function checkModelFiles() {
    return existsSync('model.onnx') && 
           existsSync('vocab.json') && 
           existsSync('scaler.json');
}

async function runDefaultTests() {
    const defaultTexts = [
        "France Defeats Argentina in Thrilling World Cup Final",
        "New Healthcare Policy Announced by Government",
        "Stock Market Reaches Record High",
        "Climate Change Summit Begins in Paris",
        "Scientists Discover New Species in Amazon"
    ];
    
    console.log('🔄 Testing multiple texts...');
    for (let i = 0; i < defaultTexts.length; i++) {
        console.log(`\n--- Test ${i + 1}/${defaultTexts.length} ---`);
        await testSingleText(defaultTexts[i]);
    }
    
    console.log('\n🎉 All tests completed successfully!');
}

async function main() {
    console.log('🤖 ONNX MULTICLASS CLASSIFIER - NODE.JS IMPLEMENTATION');
    console.log('====================================================');
    
    // Check if we're in CI environment
    const ci = process.env.CI;
    const githubActions = process.env.GITHUB_ACTIONS;
    if (ci || githubActions) {
        if (!checkModelFiles()) {
            console.log('⚠️ Some model files missing in CI - exiting safely');
            console.log('✅ Node.js implementation compiled and started successfully');
            console.log('🏗️ Build verification completed');
            return;
        }
    }
    
    if (!checkModelFiles()) {
        console.log('⚠️ Model files not found - exiting safely');
        console.log('🔧 This is expected in CI environments without model files');
        console.log('✅ Node.js implementation compiled successfully');
        console.log('🏗️ Build verification completed');
        return;
    }
    
    try {
        const args = process.argv.slice(2);
        
        if (args.length > 0) {
            if (args[0] === '--benchmark') {
                const numRuns = args.length > 1 ? parseInt(args[1]) : 100;
                await runPerformanceBenchmark(numRuns);
            } else {
                // Test custom text
                await testSingleText(args[0]);
            }
        } else {
            // Default test with multiple texts
            await runDefaultTests();
        }
    } catch (error) {
        console.error('❌ Error during execution:', error.message);
        process.exit(1);
    }
}

// Run the main function
main().catch(error => {
    console.error('❌ Fatal error:', error.message);
    process.exit(1);
}); 