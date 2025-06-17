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
import java.nio.IntBuffer;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class MulticlassClassifierTest {
    private static final Logger logger = LoggerFactory.getLogger(MulticlassClassifierTest.class);
    
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
        System.out.println("🤖 ONNX MULTICLASS CLASSIFIER - JAVA IMPLEMENTATION");
        System.out.println("==================================================");
        
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
            "France Defeats Argentina in Thrilling World Cup Final",
            "New Healthcare Policy Announced by Government",
            "Stock Market Reaches Record High",
            "Climate Change Summit Begins in Paris",
            "Scientists Discover New Species in Amazon"
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
            int[] inputVector = preprocessText(text);
            timing.preprocessingTimeMs = (System.nanoTime() - preprocessStart) / 1_000_000.0;
            
            // Model inference
            long inferenceStart = System.nanoTime();
            
            // Create input tensor
            long[] shape = {1, 30};
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, IntBuffer.wrap(inputVector), shape);
            
            // Run inference
            Map<String, OnnxTensor> inputs = Map.of(getInputName(session), inputTensor);
            OrtSession.Result result = session.run(inputs);
            
            timing.inferenceTimeMs = (System.nanoTime() - inferenceStart) / 1_000_000.0;
            
            // Post-processing
            long postprocessStart = System.nanoTime();
            float[] output = ((OnnxTensor) result.get(0)).getFloatBuffer().array();
            
            // Load label mapping
            ObjectMapper mapper = new ObjectMapper();
            JsonNode labelMap = mapper.readTree(new File("scaler.json"));
            
            // Find predicted class
            int predictedIdx = 0;
            float maxConfidence = output[0];
            for (int i = 1; i < output.length; i++) {
                if (output[i] > maxConfidence) {
                    maxConfidence = output[i];
                    predictedIdx = i;
                }
            }
            
            String predictedLabel = labelMap.get(String.valueOf(predictedIdx)).asText();
            timing.postprocessingTimeMs = (System.nanoTime() - postprocessStart) / 1_000_000.0;
            
            // Final measurements
            timing.totalTimeMs = (System.nanoTime() - totalStart) / 1_000_000.0;
            timing.throughputPerSec = 1000.0 / timing.totalTimeMs;
            resources.memoryEndMB = getMemoryUsageMB();
            resources.memoryDeltaMB = resources.memoryEndMB - resources.memoryStartMB;
            
            // Stop CPU monitoring
            stopCpuMonitoring(resources);
            
            // Display results
            System.out.println("📊 TOPIC CLASSIFICATION RESULTS:");
            System.out.printf("⏱️  Processing Time: %.1fms%n", timing.totalTimeMs);
            
            // Category emojis
            Map<String, String> categoryEmojis = Map.of(
                "politics", "🏛️",
                "technology", "💻",
                "sports", "⚽",
                "business", "💼",
                "entertainment", "🎭"
            );
            
            String emoji = categoryEmojis.getOrDefault(predictedLabel, "📝");
            System.out.printf("   🏆 Predicted Category: %s %s%n", predictedLabel.toUpperCase(), emoji);
            System.out.printf("   📈 Confidence: %.1f%%%n", maxConfidence * 100.0);
            System.out.printf("   📝 Input Text: \"%s\"%n", text);
            System.out.println();
            
            // Show all class probabilities
            System.out.println("📊 DETAILED PROBABILITIES:");
            for (int i = 0; i < output.length; i++) {
                String className = labelMap.get(String.valueOf(i)).asText();
                float probability = output[i];
                String classEmoji = categoryEmojis.getOrDefault(className, "📝");
                String bar = "█".repeat((int)(probability * 20));
                String star = (i == predictedIdx) ? " ⭐" : "";
                System.out.printf("   %s %s: %.1f%% %s%s%n", 
                    classEmoji, 
                    className.substring(0, 1).toUpperCase() + className.substring(1),
                    probability * 100.0, 
                    bar,
                    star);
            }
            System.out.println();
            
            // Print performance summary
            printPerformanceSummary(timing, resources);
            
        } catch (Exception e) {
            stopCpuMonitoring(resources);
            throw e;
        }
    }
    
    private static int[] preprocessText(String text) throws IOException {
        int[] vector = new int[30];
        Arrays.fill(vector, 0);
        
        // Load tokenizer
        ObjectMapper mapper = new ObjectMapper();
        JsonNode tokenizer = mapper.readTree(new File("vocab.json"));
        
        // Tokenize text
        String textLower = text.toLowerCase();
        String[] words = textLower.split("\\s+");
        
        // Convert words to token IDs
        for (int i = 0; i < Math.min(words.length, 30); i++) {
            String word = words[i];
            if (tokenizer.has(word)) {
                vector[i] = tokenizer.get(word).asInt();
            } else if (tokenizer.has("<OOV>")) {
                vector[i] = tokenizer.get("<OOV>").asInt();
            } else {
                vector[i] = 1; // Default OOV token
            }
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
        
        String testText = "France Defeats Argentina in Thrilling World Cup Final";
        System.out.println("📝 Test Text: '" + testText + "'\n");
        
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             OrtSession session = env.createSession("model.onnx")) {
            
            // Preprocess once
            int[] inputVector = preprocessText(testText);
            long[] shape = {1, 30};
            String inputName = getInputName(session);
            
            // Warmup runs
            System.out.println("🔥 Warming up model (5 runs)...");
            for (int i = 0; i < 5; i++) {
                OnnxTensor inputTensor = OnnxTensor.createTensor(env, IntBuffer.wrap(inputVector), shape);
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
                
                OnnxTensor inputTensor = OnnxTensor.createTensor(env, IntBuffer.wrap(inputVector), shape);
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
                // Use a simplified CPU monitoring approach that works across all JVMs
                // This is less accurate but more portable
                double cpuUsage = getCpuUsagePortable();
                if (cpuUsage >= 0) {
                    cpuReadings.add(cpuUsage);
                }
            }
        }, 0, 100, TimeUnit.MILLISECONDS);
    }
    
    private static double getCpuUsagePortable() {
        // Simplified CPU usage estimation using thread timing
        // This is less accurate but works on all JVM implementations
        try {
            OperatingSystemMXBean osBean = ManagementFactory.getOperatingSystemMXBean();
            
            // Try to use Sun-specific method if available (for better accuracy)
            if (osBean instanceof com.sun.management.OperatingSystemMXBean) {
                com.sun.management.OperatingSystemMXBean sunBean = 
                    (com.sun.management.OperatingSystemMXBean) osBean;
                return sunBean.getProcessCpuLoad() * 100.0;
            }
            
            // Fallback: Use system load average as approximation
            double loadAverage = osBean.getSystemLoadAverage();
            if (loadAverage >= 0) {
                // Convert load average to approximate CPU percentage
                int processors = osBean.getAvailableProcessors();
                return Math.min(100.0, (loadAverage / processors) * 100.0);
            }
            
            // Final fallback: return 0 (no CPU monitoring)
            return 0.0;
            
        } catch (Exception e) {
            // If any error occurs, return 0 (no CPU monitoring)
            return 0.0;
        }
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