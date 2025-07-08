package com.whitelightning;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import ai.onnxruntime.*;
import java.io.*;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.OperatingSystemMXBean;
import java.lang.management.RuntimeMXBean;
import java.util.*;
import java.util.regex.Pattern;

public class MulticlassSigmoidTest {
    private static final Pattern TOKEN_PATTERN = Pattern.compile("\\b\\w\\w+\\b");
    
    public static class VectorizerData {
        public Map<String, Integer> vocabulary;
        public Map<String, Integer> vocab;
        public double[] idf;
        public int max_features;
    }
    
    public static class SystemInfo {
        public String platform;
        public String processor;
        public int cpuCores;
        public double memoryGB;
        public String javaVersion;
        public String jvmName;
        
        public SystemInfo() {
            OperatingSystemMXBean osBean = ManagementFactory.getOperatingSystemMXBean();
            RuntimeMXBean runtimeBean = ManagementFactory.getRuntimeMXBean();
            
            this.platform = System.getProperty("os.name");
            this.processor = osBean.getArch();
            this.cpuCores = osBean.getAvailableProcessors();
            this.memoryGB = Runtime.getRuntime().maxMemory() / (1024.0 * 1024.0 * 1024.0);
            this.javaVersion = System.getProperty("java.version");
            this.jvmName = runtimeBean.getVmName();
        }
    }
    
    public static class ResourceMetrics {
        public double memoryStartMB;
        public double memoryEndMB;
        public double memoryDeltaMB;
        
        public ResourceMetrics(double start, double end) {
            this.memoryStartMB = start;
            this.memoryEndMB = end;
            this.memoryDeltaMB = end - start;
        }
    }
    
    public static class PreprocessingResult {
        public float[] vector;
        public long preprocessingTime;
        public int nonZeroFeatures;
        public float maxValue;
        public float minValue;
        
        public PreprocessingResult(float[] vector, long time) {
            this.vector = vector;
            this.preprocessingTime = time;
            this.nonZeroFeatures = 0;
            this.maxValue = Float.MIN_VALUE;
            this.minValue = Float.MAX_VALUE;
            
            for (float value : vector) {
                if (value != 0.0f) {
                    this.nonZeroFeatures++;
                    if (value > this.maxValue) this.maxValue = value;
                    if (value < this.minValue) this.minValue = value;
                }
            }
        }
    }
    
    public static VectorizerData loadVectorizer(String path) throws IOException {
        Gson gson = new Gson();
        return gson.fromJson(new FileReader(path), VectorizerData.class);
    }
    
    public static Map<String, String> loadClasses(String path) throws IOException {
        Gson gson = new Gson();
        return gson.fromJson(new FileReader(path), new TypeToken<Map<String, String>>(){}.getType());
    }
    
    public static float[] preprocessText(String text, VectorizerData vectorizer) {
        long startTime = System.currentTimeMillis();
        
        try {
            Map<String, Integer> vocabulary = vectorizer.vocabulary != null ? 
                vectorizer.vocabulary : vectorizer.vocab;
            
            if (vocabulary == null || vectorizer.idf == null) {
                throw new IllegalArgumentException("Invalid vectorizer: missing vocabulary or idf");
            }
            
            int maxFeatures = vectorizer.max_features > 0 ? vectorizer.max_features : 5000;
            
            // Tokenize text
            String lowerText = text.toLowerCase();
            java.util.regex.Matcher matcher = TOKEN_PATTERN.matcher(lowerText);
            List<String> tokens = new ArrayList<>();
            while (matcher.find()) {
                tokens.add(matcher.group());
            }
            
            System.out.println("📊 Tokens found: " + tokens.size() + 
                ", First 10: " + tokens.subList(0, Math.min(10, tokens.size())));
            
            // Count term frequencies
            Map<String, Integer> termCounts = new HashMap<>();
            for (String token : tokens) {
                termCounts.put(token, termCounts.getOrDefault(token, 0) + 1);
            }
            
            // Create TF-IDF vector
            float[] vector = new float[maxFeatures];
            Arrays.fill(vector, 0.0f);
            int foundInVocab = 0;
            
            // Apply TF-IDF
            for (Map.Entry<String, Integer> entry : termCounts.entrySet()) {
                String term = entry.getKey();
                int count = entry.getValue();
                Integer termIndex = vocabulary.get(term);
                
                if (termIndex != null && termIndex < maxFeatures) {
                    vector[termIndex] = (float) (count * vectorizer.idf[termIndex]);
                    foundInVocab++;
                }
            }
            
            System.out.println("📊 Found " + foundInVocab + " terms in vocabulary out of " + tokens.size() + " total tokens");
            
            // L2 normalization
            double norm = 0.0;
            for (float value : vector) {
                norm += value * value;
            }
            norm = Math.sqrt(norm);
            
            if (norm > 0) {
                for (int i = 0; i < vector.length; i++) {
                    vector[i] = (float) (vector[i] / norm);
                }
            }
            
            System.out.println("📊 TF-IDF: " + foundInVocab + " non-zero, max: " + 
                String.format("%.4f", Arrays.stream(vector).max().orElse(0.0f)) + 
                ", norm: " + String.format("%.4f", norm));
            
            return vector;
            
        } catch (Exception e) {
            System.err.println("❌ Preprocessing error: " + e.getMessage());
            throw new RuntimeException("Preprocessing failed", e);
        }
    }
    
    public static float[] runInference(OrtSession session, float[] vector) throws OrtException {
        String inputName = session.getInputNames().iterator().next();
        OnnxTensor inputTensor = OnnxTensor.createTensor(session.getEnvironment(), 
            new float[][]{vector});
        
        OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor));
        OnnxTensor outputTensor = (OnnxTensor) result.get(0);
        float[][] output = (float[][]) outputTensor.getValue();
        
        inputTensor.close();
        result.close();
        
        return output[0];
    }
    
    public static void main(String[] args) {
        String testText = args.length > 0 ? args[0] : 
            "I'm about to give birth, and I'm terrified. What if something goes wrong? What if I can't handle the pain? Received an unexpected compliment at work today. Small moments of happiness can make a big difference.";
        
        System.out.println("🤖 ONNX MULTICLASS SIGMOID CLASSIFIER - JAVA IMPLEMENTATION");
        System.out.println("=".repeat(64));
        System.out.println("🔄 Processing: " + testText);
        System.out.println();
        
        SystemInfo systemInfo = new SystemInfo();
        System.out.println("💻 SYSTEM INFORMATION:");
        System.out.println("   Platform: " + systemInfo.platform);
        System.out.println("   Processor: " + systemInfo.processor);
        System.out.println("   CPU Cores: " + systemInfo.cpuCores);
        System.out.println("   Total Memory: " + String.format("%.1f", systemInfo.memoryGB) + " GB");
        System.out.println("   Runtime: " + systemInfo.jvmName + " " + systemInfo.javaVersion);
        System.out.println();
        
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        double memoryStart = memoryBean.getHeapMemoryUsage().getUsed() / (1024.0 * 1024.0);
        
        long totalStartTime = System.currentTimeMillis();
        
        try {
            System.out.println("🔧 Loading components...");
            
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession session = env.createSession("model.onnx");
            System.out.println("✅ ONNX model loaded");
            
            VectorizerData vectorizer = loadVectorizer("vocab.json");
            System.out.println("✅ Vectorizer loaded");
            
            Map<String, String> classes = loadClasses("scaler.json");
            System.out.println("✅ Classes loaded: " + String.join(", ", classes.values()));
            System.out.println();
            
            float[] vector = preprocessText(testText, vectorizer);
            System.out.println("📊 TF-IDF shape: [1, " + vector.length + "]");
            System.out.println();
            
            float[] predictions = runInference(session, vector);
            
            System.out.println("📊 EMOTION ANALYSIS RESULTS:");
            List<Map.Entry<String, Float>> emotionResults = new ArrayList<>();
            
            for (int i = 0; i < predictions.length; i++) {
                String className = classes.getOrDefault(String.valueOf(i), "Class " + i);
                float probability = predictions[i];
                emotionResults.add(new AbstractMap.SimpleEntry<>(className, probability));
                System.out.println("   " + className + ": " + String.format("%.3f", probability));
            }
            
            Map.Entry<String, Float> dominantEmotion = emotionResults.stream()
                .max(Map.Entry.comparingByValue())
                .orElse(emotionResults.get(0));
            
            System.out.println("   🏆 Dominant Emotion: " + dominantEmotion.getKey() + 
                " (" + String.format("%.3f", dominantEmotion.getValue()) + ")");
            
            System.out.println("   📝 Input Text: \"" + testText + "\"");
            System.out.println();
            
            long totalTime = System.currentTimeMillis() - totalStartTime;
            double memoryEnd = memoryBean.getHeapMemoryUsage().getUsed() / (1024.0 * 1024.0);
            double memoryDelta = memoryEnd - memoryStart;
            
            System.out.println("📈 PERFORMANCE SUMMARY:");
            System.out.println("   Total Processing Time: " + totalTime + "ms");
            System.out.println();
            
            double throughput = 1000.0 / totalTime;
            System.out.println("🚀 THROUGHPUT:");
            System.out.println("   Texts per second: " + String.format("%.1f", throughput));
            System.out.println();
            
            System.out.println("💾 RESOURCE USAGE:");
            System.out.println("   Memory Start: " + String.format("%.2f", memoryStart) + "MB");
            System.out.println("   Memory End: " + String.format("%.2f", memoryEnd) + "MB");
            System.out.println("   Memory Delta: " + (memoryDelta >= 0 ? "+" : "") + 
                String.format("%.2f", memoryDelta) + "MB");
            System.out.println();
            
            String rating;
            if (totalTime < 50) {
                rating = "🚀 EXCELLENT";
            } else if (totalTime < 100) {
                rating = "✅ GOOD";
            } else if (totalTime < 500) {
                rating = "⚠️ ACCEPTABLE";
            } else {
                rating = "🐌 SLOW";
            }
            
            System.out.println("🎯 PERFORMANCE RATING: " + rating);
            System.out.println("   (" + totalTime + "ms total - Target: <100ms)");
            
            session.close();
            env.close();
            
        } catch (Exception e) {
            System.err.println("❌ Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
} 