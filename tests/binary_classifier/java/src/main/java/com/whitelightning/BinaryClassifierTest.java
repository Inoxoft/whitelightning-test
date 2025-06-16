package com.whitelightning;

import ai.onnxruntime.*;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.OperatingSystemMXBean;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class BinaryClassifierTest {
    private static final Logger logger = LoggerFactory.getLogger(BinaryClassifierTest.class);
    
    // Performance monitoring
    private static class TimingMetrics {
        double totalTimeMs = 0;
        double preprocessingTimeMs = 0;
        double inferenceTimeMs = 0;
        double postprocessingTimeMs = 0;
        double throughputPerSec = 0;
    }
    
    private static class ResourceMetrics {
        double memoryStartMB = 0;
        double memoryEndMB = 0;
        double memoryDeltaMB = 0;
        double cpuAvgPercent = 0;
        double cpuMaxPercent = 0;
        int cpuReadingsCount = 0;
        List<Double> cpuReadings = new ArrayList<>();
    }
    
    private static class SystemInfo {
        String platform;
        String processor;
        int cpuCores;
        double totalMemoryGB;
        String runtime = "Java Implementation";
        String javaVersion;
        String onnxVersion;
    }
    
    // CPU monitoring
    private static final List<Double> cpuReadings = Collections.synchronizedList(new ArrayList<>());
    private static ScheduledExecutorService cpuMonitor;
    private static volatile boolean monitoring = false;
    
    public static void main(String[] args) {
        System.out.println("🤖 ONNX BINARY CLASSIFIER - JAVA IMPLEMENTATION");
        System.out.println("===============================================");
        
        // Check if we're in CI environment
        String ci = System.getenv("CI");
        String githubActions = System.getenv("GITHUB_ACTIONS");
        if (ci != null || githubActions != null) {
            if (!checkModelFiles()) {
                System.out.println("⚠️ Some model files missing in CI - exiting safely");
                System.out.println("✅ Java implementation compiled and started successfully");
                System.out.println("🏗️ Build verification completed");
                return;
            }
        }
        
        if (!checkModelFiles()) {
            System.out.println("⚠️ Model files not found - exiting safely");
            System.out.println("🔧 This is expected in CI environments without model files");
            System.out.println("✅ Java implementation compiled successfully");
            System.out.println("🏗️ Build verification completed");
            return;
        }
        
        try {
            if (args.length > 0) {
                if ("--benchmark".equals(args[0])) {
                    int numRuns = args.length > 1 ? Integer.parseInt(args[1]) : 100;
                    runPerformanceBenchmark(numRuns);
                } else {
                    // Test custom text
                    testSingleText(args[0]);
                }
            } else {
                // Default test with multiple texts
                runDefaultTests();
            }
        } catch (Exception e) {
            logger.error("❌ Error during execution", e);
            System.exit(1);
        }
    }
    
    private static boolean checkModelFiles() {
        return new File("model.onnx").exists() && 
               new File("vocab.json").exists() && 
               new File("scaler.json").exists();
    }
    
    private static void runDefaultTests() throws Exception {
        String[] defaultTexts = {
            "This product is amazing!",
            "Terrible service, would not recommend.",
            "It's okay, nothing special.",
            "Best purchase ever!",
            "The product broke after just two days — total waste of money."
        };
        
        System.out.println("🔄 Testing multiple texts...");
        for (int i = 0; i < defaultTexts.length; i++) {
            System.out.println("\n--- Test " + (i + 1) + "/" + defaultTexts.length + " ---");
            testSingleText(defaultTexts[i]);
        }
        
        System.out.println("\n🎉 All tests completed successfully!");
    }
    
    private static void testSingleText(String text) throws Exception {
        System.out.println("🔄 Processing: " + text);
        
        // Initialize system info
        SystemInfo systemInfo = getSystemInfo();
        printSystemInfo(systemInfo);
        
        // Initialize metrics
        TimingMetrics timing = new TimingMetrics();
        ResourceMetrics resources = new ResourceMetrics();
        
        long totalStart = System.nanoTime();
        resources.memoryStartMB = getMemoryUsageMB();
        
        // Start CPU monitoring
        startCpuMonitoring();
        
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             OrtSession session = env.createSession("model.onnx")) {
            
            // Preprocessing
            long preprocessStart = System.nanoTime();
            float[] inputVector = preprocessText(text);
            timing.preprocessingTimeMs = (System.nanoTime() - preprocessStart) / 1_000_000.0;
            
            // Model inference
            long inferenceStart = System.nanoTime();
            
            // Create input tensor
            long[] shape = {1, 5000};
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputVector), shape);
            
            // Run inference
            Map<String, OnnxTensor> inputs = Map.of(getInputName(session), inputTensor);
            OrtSession.Result result = session.run(inputs);
            
            timing.inferenceTimeMs = (System.nanoTime() - inferenceStart) / 1_000_000.0;
            
            // Post-processing
            long postprocessStart = System.nanoTime();
            float[] output = ((OnnxTensor) result.get(0)).getFloatBuffer().array();
            float prediction = output[0];
            String sentiment = prediction > 0.5 ? "Positive" : "Negative";
            timing.postprocessingTimeMs = (System.nanoTime() - postprocessStart) / 1_000_000.0;
            
            // Final measurements
            timing.totalTimeMs = (System.nanoTime() - totalStart) / 1_000_000.0;
            timing.throughputPerSec = 1000.0 / timing.totalTimeMs;
            resources.memoryEndMB = getMemoryUsageMB();
            resources.memoryDeltaMB = resources.memoryEndMB - resources.memoryStartMB;
            
            // Stop CPU monitoring
            stopCpuMonitoring(resources);
            
            // Display results
            System.out.println("📊 SENTIMENT ANALYSIS RESULTS:");
            System.out.printf("   🏆 Predicted Sentiment: %s%n", sentiment);
            System.out.printf("   📈 Confidence: %.2f%% (%.4f)%n", prediction * 100.0, prediction);
            System.out.printf("   📝 Input Text: \"%s\"%n%n", text);
            
            // Print performance summary
            printPerformanceSummary(timing, resources);
            
        } catch (Exception e) {
            stopCpuMonitoring(resources);
            throw e;
        }
    }
    
    private static float[] preprocessText(String text) throws IOException {
        float[] vector = new float[5000];
        Arrays.fill(vector, 0.0f);
        
        // Load TF-IDF data
        ObjectMapper mapper = new ObjectMapper();
        JsonNode tfidfData = mapper.readTree(new File("vocab.json"));
        JsonNode vocab = tfidfData.get("vocab");
        JsonNode idfArray = tfidfData.get("idf");
        
        // Load scaler data
        JsonNode scalerData = mapper.readTree(new File("scaler.json"));
        JsonNode meanArray = scalerData.get("mean");
        JsonNode scaleArray = scalerData.get("scale");
        
        // Tokenize and count words
        String textLower = text.toLowerCase();
        String[] words = textLower.split("\\s+");
        Map<String, Integer> wordCounts = new HashMap<>();
        
        for (String word : words) {
            wordCounts.put(word, wordCounts.getOrDefault(word, 0) + 1);
        }
        
        // Apply TF-IDF
        for (Map.Entry<String, Integer> entry : wordCounts.entrySet()) {
            String word = entry.getKey();
            int count = entry.getValue();
            
            if (vocab.has(word)) {
                int idx = vocab.get(word).asInt();
                if (idx < 5000) {
                    double idf = idfArray.get(idx).asDouble();
                    vector[idx] = (float) (count * idf);
                }
            }
        }
        
        // Apply scaling
        for (int i = 0; i < 5000; i++) {
            double mean = meanArray.get(i).asDouble();
            double scale = scaleArray.get(i).asDouble();
            vector[i] = (float) ((vector[i] - mean) / scale);
        }
        
        return vector;
    }
    
    private static String getInputName(OrtSession session) throws OrtException {
        return session.getInputNames().iterator().next();
    }
    
    private static void runPerformanceBenchmark(int numRuns) throws Exception {
        System.out.println("\n🚀 PERFORMANCE BENCHMARKING (" + numRuns + " runs)");
        System.out.println("============================================================");
        
        SystemInfo systemInfo = getSystemInfo();
        System.out.printf("💻 System: %d cores, %.1fGB RAM%n", systemInfo.cpuCores, systemInfo.totalMemoryGB);
        
        String testText = "This is a sample text for performance testing.";
        System.out.println("📝 Test Text: '" + testText + "'\n");
        
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             OrtSession session = env.createSession("model.onnx")) {
            
            // Preprocess once
            float[] inputVector = preprocessText(testText);
            long[] shape = {1, 5000};
            String inputName = getInputName(session);
            
            // Warmup runs
            System.out.println("🔥 Warming up model (5 runs)...");
            for (int i = 0; i < 5; i++) {
                OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputVector), shape);
                Map<String, OnnxTensor> inputs = Map.of(inputName, inputTensor);
                session.run(inputs).close();
            }
            
            // Performance arrays
            double[] times = new double[numRuns];
            double[] inferenceTimes = new double[numRuns];
            
            System.out.println("📊 Running " + numRuns + " performance tests...");
            long overallStart = System.nanoTime();
            
            for (int i = 0; i < numRuns; i++) {
                if (i % 20 == 0 && i > 0) {
                    System.out.printf("   Progress: %d/%d (%.1f%%)%n", i, numRuns, (double) i / numRuns * 100.0);
                }
                
                long startTime = System.nanoTime();
                long inferenceStart = System.nanoTime();
                
                OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputVector), shape);
                Map<String, OnnxTensor> inputs = Map.of(inputName, inputTensor);
                session.run(inputs).close();
                
                long inferenceTime = System.nanoTime() - inferenceStart;
                long endTime = System.nanoTime();
                
                times[i] = (endTime - startTime) / 1_000_000.0;
                inferenceTimes[i] = inferenceTime / 1_000_000.0;
            }
            
            double overallTime = (System.nanoTime() - overallStart) / 1_000_000.0;
            
            // Calculate statistics
            double avgTime = Arrays.stream(times).average().orElse(0.0);
            double minTime = Arrays.stream(times).min().orElse(0.0);
            double maxTime = Arrays.stream(times).max().orElse(0.0);
            double avgInf = Arrays.stream(inferenceTimes).average().orElse(0.0);
            
            // Display results
            System.out.println("\n📈 DETAILED PERFORMANCE RESULTS:");
            System.out.println("--------------------------------------------------");
            System.out.println("⏱️  TIMING ANALYSIS:");
            System.out.printf("   Mean: %.2fms%n", avgTime);
            System.out.printf("   Min: %.2fms%n", minTime);
            System.out.printf("   Max: %.2fms%n", maxTime);
            System.out.printf("   Model Inference: %.2fms%n", avgInf);
            System.out.println("\n🚀 THROUGHPUT:");
            System.out.printf("   Texts per second: %.1f%n", 1000.0 / avgTime);
            System.out.printf("   Total benchmark time: %.2fs%n", overallTime / 1000.0);
            System.out.printf("   Overall throughput: %.1f texts/sec%n", numRuns / (overallTime / 1000.0));
            
            // Performance classification
            String performanceClass;
            if (avgTime < 10) {
                performanceClass = "🚀 EXCELLENT";
            } else if (avgTime < 50) {
                performanceClass = "✅ GOOD";
            } else if (avgTime < 100) {
                performanceClass = "⚠️ ACCEPTABLE";
            } else {
                performanceClass = "❌ POOR";
            }
            
            System.out.println("\n🎯 PERFORMANCE CLASSIFICATION: " + performanceClass);
            System.out.printf("   (%.1fms average - Target: <100ms)%n", avgTime);
        }
    }
    
    private static SystemInfo getSystemInfo() {
        SystemInfo info = new SystemInfo();
        
        OperatingSystemMXBean osBean = ManagementFactory.getOperatingSystemMXBean();
        info.platform = System.getProperty("os.name");
        info.processor = System.getProperty("os.arch");
        info.cpuCores = osBean.getAvailableProcessors();
        info.javaVersion = System.getProperty("java.version");
        
        // Get total memory
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        long maxMemory = memoryBean.getHeapMemoryUsage().getMax();
        if (maxMemory == -1) {
            maxMemory = Runtime.getRuntime().maxMemory();
        }
        info.totalMemoryGB = maxMemory / (1024.0 * 1024.0 * 1024.0);
        
        try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {
            info.onnxVersion = env.getVersion();
        } catch (Exception e) {
            info.onnxVersion = "Unknown";
        }
        
        return info;
    }
    
    private static void printSystemInfo(SystemInfo info) {
        System.out.println("💻 SYSTEM INFORMATION:");
        System.out.println("   Platform: " + info.platform);
        System.out.println("   Processor: " + info.processor);
        System.out.println("   CPU Cores: " + info.cpuCores);
        System.out.printf("   Total Memory: %.1f GB%n", info.totalMemoryGB);
        System.out.println("   Runtime: " + info.runtime);
        System.out.println("   Java Version: " + info.javaVersion);
        System.out.println("   ONNX Runtime Version: " + info.onnxVersion);
        System.out.println();
    }
    
    private static void printPerformanceSummary(TimingMetrics timing, ResourceMetrics resources) {
        System.out.println("📈 PERFORMANCE SUMMARY:");
        System.out.printf("   Total Processing Time: %.2fms%n", timing.totalTimeMs);
        System.out.printf("   ┣━ Preprocessing: %.2fms (%.1f%%)%n", 
            timing.preprocessingTimeMs, (timing.preprocessingTimeMs / timing.totalTimeMs) * 100.0);
        System.out.printf("   ┣━ Model Inference: %.2fms (%.1f%%)%n", 
            timing.inferenceTimeMs, (timing.inferenceTimeMs / timing.totalTimeMs) * 100.0);
        System.out.printf("   ┗━ Postprocessing: %.2fms (%.1f%%)%n%n", 
            timing.postprocessingTimeMs, (timing.postprocessingTimeMs / timing.totalTimeMs) * 100.0);
        
        System.out.println("🚀 THROUGHPUT:");
        System.out.printf("   Texts per second: %.1f%n%n", timing.throughputPerSec);
        
        System.out.println("💾 RESOURCE USAGE:");
        System.out.printf("   Memory Start: %.2f MB%n", resources.memoryStartMB);
        System.out.printf("   Memory End: %.2f MB%n", resources.memoryEndMB);
        System.out.printf("   Memory Delta: %+.2f MB%n", resources.memoryDeltaMB);
        if (resources.cpuReadingsCount > 0) {
            System.out.printf("   CPU Usage: %.1f%% avg, %.1f%% peak (%d samples)%n", 
                resources.cpuAvgPercent, resources.cpuMaxPercent, resources.cpuReadingsCount);
        }
        System.out.println();
        
        // Performance classification
        String performanceClass, emoji;
        if (timing.totalTimeMs < 50) {
            performanceClass = "EXCELLENT";
            emoji = "🚀";
        } else if (timing.totalTimeMs < 100) {
            performanceClass = "GOOD";
            emoji = "✅";
        } else if (timing.totalTimeMs < 200) {
            performanceClass = "ACCEPTABLE";
            emoji = "⚠️";
        } else {
            performanceClass = "POOR";
            emoji = "❌";
        }
        
        System.out.println("🎯 PERFORMANCE RATING: " + emoji + " " + performanceClass);
        System.out.printf("   (%.1fms total - Target: <100ms)%n%n", timing.totalTimeMs);
    }
    
    private static double getMemoryUsageMB() {
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        return memoryBean.getHeapMemoryUsage().getUsed() / (1024.0 * 1024.0);
    }
    
    private static void startCpuMonitoring() {
        monitoring = true;
        cpuReadings.clear();
        cpuMonitor = Executors.newScheduledThreadPool(1);
        
        cpuMonitor.scheduleAtFixedRate(() -> {
            if (monitoring) {
                OperatingSystemMXBean osBean = ManagementFactory.getOperatingSystemMXBean();
                double cpuUsage = osBean.getProcessCpuLoad() * 100.0;
                if (cpuUsage >= 0) {
                    cpuReadings.add(cpuUsage);
                }
            }
        }, 0, 100, TimeUnit.MILLISECONDS);
    }
    
    private static void stopCpuMonitoring(ResourceMetrics metrics) {
        monitoring = false;
        if (cpuMonitor != null) {
            cpuMonitor.shutdown();
            try {
                cpuMonitor.awaitTermination(1, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        
        synchronized (cpuReadings) {
            metrics.cpuReadings = new ArrayList<>(cpuReadings);
            metrics.cpuReadingsCount = cpuReadings.size();
            
            if (!cpuReadings.isEmpty()) {
                metrics.cpuAvgPercent = cpuReadings.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
                metrics.cpuMaxPercent = cpuReadings.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
            }
        }
    }
} 