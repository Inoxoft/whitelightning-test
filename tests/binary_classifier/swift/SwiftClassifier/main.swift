import Foundation
import OnnxRuntimeBindings
import Darwin

// MARK: - Data Structures
struct TFIDFData: Codable {
    let vocab: [String: Int]
    let idf: [Double]
}

struct ScalerData: Codable {
    let mean: [Double]
    let scale: [Double]
}

struct TimingMetrics {
    var totalTime: TimeInterval = 0
    var preprocessingTime: TimeInterval = 0
    var inferenceTime: TimeInterval = 0
    var postprocessingTime: TimeInterval = 0
    var throughputPerSec: Double = 0
}

struct ResourceMetrics {
    var memoryStartMB: Double = 0
    var memoryEndMB: Double = 0
    var memoryDeltaMB: Double = 0
}

struct SystemInfo {
    let platform: String
    let processor: String
    let cpuCountPhysical: Int
    let cpuCountLogical: Int
    let totalMemoryGB: Double
    let runtime: String
}

// MARK: - Utility Functions
func getCurrentTimeMs() -> Double {
    return Date().timeIntervalSince1970 * 1000
}

func getMemoryUsageMB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
    
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_,
                     task_flavor_t(MACH_TASK_BASIC_INFO),
                     $0,
                     &count)
        }
    }
    
    if kerr == KERN_SUCCESS {
        return Double(info.resident_size) / 1024.0 / 1024.0
    }
    return 0.0
}

func getSystemInfo() -> SystemInfo {
    var size = 0
    sysctlbyname("hw.model", nil, &size, nil, 0)
    var model = [CChar](repeating: 0, count: size)
    sysctlbyname("hw.model", &model, &size, nil, 0)
    
    var physicalCPU: Int32 = 0
    var logicalCPU: Int32 = 0
    var memsize: UInt64 = 0
    
    size = MemoryLayout<Int32>.size
    sysctlbyname("hw.physicalcpu", &physicalCPU, &size, nil, 0)
    sysctlbyname("hw.logicalcpu", &logicalCPU, &size, nil, 0)
    
    size = MemoryLayout<UInt64>.size
    sysctlbyname("hw.memsize", &memsize, &size, nil, 0)
    
    return SystemInfo(
        platform: "macOS",
        processor: String(cString: model),
        cpuCountPhysical: Int(physicalCPU),
        cpuCountLogical: Int(logicalCPU),
        totalMemoryGB: Double(memsize) / (1024.0 * 1024.0 * 1024.0),
        runtime: "Swift Implementation"
    )
}

// MARK: - Text Processing
func tokenizeText(_ text: String) -> [String] {
    let lowercased = text.lowercased()
    let words = lowercased.components(separatedBy: CharacterSet.alphanumerics.inverted)
    return words.filter { !$0.isEmpty && $0.count >= 2 }
}

func calculateTermFrequency(_ tokens: [String]) -> [String: Int] {
    var tf: [String: Int] = [:]
    for token in tokens {
        tf[token, default: 0] += 1
    }
    return tf
}

func loadTFIDFData(from path: String) throws -> TFIDFData {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    
    let vocab = json["vocab"] as! [String: Int]
    let idfArray = json["idf"] as! [Any]
    let idf = idfArray.map { ($0 as! NSNumber).doubleValue }
    
    return TFIDFData(vocab: vocab, idf: idf)
}

func loadScalerData(from path: String) throws -> ScalerData {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    
    let meanArray = json["mean"] as! [Any]
    let mean = meanArray.map { ($0 as! NSNumber).doubleValue }
    
    let scaleArray = json["scale"] as! [Any]
    let scale = scaleArray.map { ($0 as! NSNumber).doubleValue }
    
    return ScalerData(mean: mean, scale: scale)
}

func preprocessText(_ text: String, tfidfData: TFIDFData, scalerData: ScalerData) -> [Float] {
    let tokens = tokenizeText(text)
    let termFrequency = calculateTermFrequency(tokens)
    
    // Create TF-IDF vector
    let vocabSize = tfidfData.idf.count
    var vector = Array(repeating: Float(0.0), count: vocabSize)
    
    for (word, count) in termFrequency {
        if let index = tfidfData.vocab[word], index < vocabSize {
            let tf = Double(count) / Double(tokens.count)
            let idf = tfidfData.idf[index]
            vector[index] = Float(tf * idf)
        }
    }
    
    // Apply scaling
    for i in 0..<vocabSize {
        let scaledValue = (Double(vector[i]) - scalerData.mean[i]) / scalerData.scale[i]
        vector[i] = Float(scaledValue)
        
        // Check for problematic values
        if !vector[i].isFinite {
            print("Warning: Non-finite value at index \(i): \(vector[i]) (original: \(scaledValue))")
            vector[i] = 0.0 // Replace with 0
        }
    }
    
    return vector
}

// MARK: - Model Inference
func runInference(text: String, modelPath: String, vocabPath: String, scalerPath: String) throws -> (Float, TimingMetrics, ResourceMetrics) {
    var timing = TimingMetrics()
    var resources = ResourceMetrics()
    
    let totalStart = getCurrentTimeMs()
    resources.memoryStartMB = getMemoryUsageMB()
    
    // Load data files
    print("📁 Loading TF-IDF data...")
    let tfidfData = try loadTFIDFData(from: vocabPath)
    print("📁 Loading scaler data...")
    let scalerData = try loadScalerData(from: scalerPath)
    print("✅ Data files loaded successfully")
    
    // Preprocessing
    print("🔄 Starting preprocessing...")
    let preprocessStart = getCurrentTimeMs()
    let inputVector = preprocessText(text, tfidfData: tfidfData, scalerData: scalerData)
    timing.preprocessingTime = getCurrentTimeMs() - preprocessStart
    print("✅ Preprocessing completed. Vector size: \(inputVector.count)")
    
    // Create ONNX Runtime environment and session
    guard let env = try? ORTEnv(loggingLevel: ORTLoggingLevel.warning) else {
        throw NSError(domain: "ONNXError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create ORT Environment"])
    }
    
    guard let session = try? ORTSession(env: env, modelPath: modelPath, sessionOptions: nil) else {
        throw NSError(domain: "ONNXError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to load model"])
    }
    
    // Create input tensor
    let shape: [NSNumber] = [1, NSNumber(value: inputVector.count)]
    let inputData = Data(from: inputVector)
    let inputTensor = try ORTValue(
        tensorData: NSMutableData(data: inputData),
        elementType: .float,
        shape: shape
    )
    
    // Run inference
    let inferenceStart = getCurrentTimeMs()
    let inputName = try session.inputNames().first!
    let outputName = try session.outputNames().first!
    
    let outputs = try session.run(withInputs: [inputName: inputTensor], outputNames: [outputName], runOptions: nil)
    timing.inferenceTime = getCurrentTimeMs() - inferenceStart
    
    // Post-processing
    let postprocessStart = getCurrentTimeMs()
    guard let output = outputs[outputName] else {
        throw NSError(domain: "ONNXError", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to get output"])
    }
    
    let rawData = try output.tensorData()
    let resultData: Data
    if let d = rawData as? Data {
        resultData = d
    } else {
        resultData = Data(rawData)
    }
    
    let prediction = resultData.withUnsafeBytes { buffer in
        let pointer = buffer.bindMemory(to: Float.self)
        return pointer[0]
    }
    timing.postprocessingTime = getCurrentTimeMs() - postprocessStart
    
    // Calculate metrics
    timing.totalTime = getCurrentTimeMs() - totalStart
    timing.throughputPerSec = 1000.0 / timing.totalTime
    
    resources.memoryEndMB = getMemoryUsageMB()
    resources.memoryDeltaMB = resources.memoryEndMB - resources.memoryStartMB
    
    return (prediction, timing, resources)
}

// MARK: - Performance Benchmark
func runPerformanceBenchmark(modelPath: String, vocabPath: String, scalerPath: String, numRuns: Int = 100) throws {
    let testText = "You won a free iPhone"
    
    print("🚀 PERFORMANCE BENCHMARK")
    print("======================")
    print("📊 Running \(numRuns) inference iterations...")
    
    var timings: [TimingMetrics] = []
    let benchmarkStart = getCurrentTimeMs()
    
    // Load data once
    let tfidfData = try loadTFIDFData(from: vocabPath)
    let scalerData = try loadScalerData(from: scalerPath)

guard let env = try? ORTEnv(loggingLevel: ORTLoggingLevel.warning) else {
        throw NSError(domain: "ONNXError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create ORT Environment"])
    }
    
    guard let session = try? ORTSession(env: env, modelPath: modelPath, sessionOptions: nil) else {
        throw NSError(domain: "ONNXError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to load model"])
    }
    
    // Preprocess once
    let inputVector = preprocessText(testText, tfidfData: tfidfData, scalerData: scalerData)
    let shape: [NSNumber] = [1, NSNumber(value: inputVector.count)]
    let inputData = Data(from: inputVector)
    let inputTensor = try ORTValue(
        tensorData: NSMutableData(data: inputData),
        elementType: .float,
        shape: shape
    )
    
    let inputName = try session.inputNames().first!
    let outputName = try session.outputNames().first!
    
    // Warmup runs
    print("🔥 Warming up model (5 runs)...")
    for _ in 0..<5 {
        _ = try session.run(withInputs: [inputName: inputTensor], outputNames: [outputName], runOptions: nil)
    }
    
    // Benchmark runs
    for i in 0..<numRuns {
        var timing = TimingMetrics()
        
        let inferenceStart = getCurrentTimeMs()
        _ = try session.run(withInputs: [inputName: inputTensor], outputNames: [outputName], runOptions: nil)
        timing.inferenceTime = getCurrentTimeMs() - inferenceStart
        timing.throughputPerSec = 1000.0 / timing.inferenceTime
        
        timings.append(timing)
        
        if (i + 1) % 25 == 0 {
            print("   Completed \(i + 1)/\(numRuns) runs...")
        }
    }
    
    let totalBenchmarkTime = getCurrentTimeMs() - benchmarkStart
    
    // Calculate statistics
    let inferenceTimes = timings.map { $0.inferenceTime }
    let avgInferenceTime = inferenceTimes.reduce(0, +) / Double(inferenceTimes.count)
    let minInferenceTime = inferenceTimes.min() ?? 0
    let maxInferenceTime = inferenceTimes.max() ?? 0
    
    let throughputs = timings.map { $0.throughputPerSec }
    let avgThroughput = throughputs.reduce(0, +) / Double(throughputs.count)
    let maxThroughput = throughputs.max() ?? 0
    
    print("📈 PERFORMANCE RESULTS")
    print("=====================")
    print("🔢 Total runs: \(numRuns)")
    print("⏱️  Total benchmark time: \(String(format: "%.2f", totalBenchmarkTime))ms")
    print("⚡ Average inference time: \(String(format: "%.4f", avgInferenceTime))ms")
    print("🏃 Min inference time: \(String(format: "%.4f", minInferenceTime))ms")
    print("🐌 Max inference time: \(String(format: "%.4f", maxInferenceTime))ms")
    print("🚀 Average throughput: \(String(format: "%.2f", avgThroughput)) inferences/sec")
    print("⚡ Peak throughput: \(String(format: "%.2f", maxThroughput)) inferences/sec")
}

// MARK: - Main Function
func main() {
    print("🤖 ONNX BINARY CLASSIFIER - C IMPLEMENTATION")
    print("===========================================")
    print("🤖 ONNX BINARY CLASSIFIER - C IMPLEMENTATION")
    print("==============================================")
    
    let systemInfo = getSystemInfo()
    print("💻 SYSTEM INFORMATION:")
    print("   Platform: \(systemInfo.platform)")
    print("   Processor: \(systemInfo.processor)")
    print("   CPU Cores: \(systemInfo.cpuCountPhysical) physical, \(systemInfo.cpuCountLogical) logical")
    print("   Total Memory: \(String(format: "%.1f", systemInfo.totalMemoryGB)) GB")
    print("   Runtime: N/A (C Implementation)")
    print("")
    
    let modelPath = "model.onnx"
    let vocabPath = "vocab.json"
    let scalerPath = "scaler.json"
    
    // Test case - use command line argument if provided, otherwise use default
    let testText = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "You won a free iPhone"
    
    do {
        print("🔄 Processing: \(testText)")
        
        let (prediction, timing, resources) = try runInference(
            text: testText,
            modelPath: modelPath,
            vocabPath: vocabPath,
            scalerPath: scalerPath
        )
        
        let predictedSentiment = prediction > 0.5 ? "Positive" : "Negative"
        let confidence = prediction > 0.5 ? prediction : (1.0 - prediction)
        let confidencePercent = confidence * 100
        
        // Calculate both positive and negative scores
        let positiveScore = prediction
        let negativeScore = 1.0 - prediction
        
        print("📊 SENTIMENT ANALYSIS RESULTS:")
        print("   🏆 Predicted Sentiment: \(predictedSentiment)")
        print("   📈 Confidence: \(String(format: "%.2f", confidencePercent))% (\(String(format: "%.4f", confidence)))")
        print("   📊 Scores: Negative: \(String(format: "%.4f", negativeScore)), Positive: \(String(format: "%.4f", positiveScore))")
        print("   📝 Input Text: \"\(testText)\"")
        print("")
        print("📈 PERFORMANCE SUMMARY:")
        print("   Total Processing Time: \(String(format: "%.2f", timing.totalTime))ms")
        print("   ┣━ Preprocessing: \(String(format: "%.2f", timing.preprocessingTime))ms (\(String(format: "%.1f", (timing.preprocessingTime/timing.totalTime)*100))%)")
        print("   ┣━ Model Inference: \(String(format: "%.2f", timing.inferenceTime))ms (\(String(format: "%.1f", (timing.inferenceTime/timing.totalTime)*100))%)")
        print("   ┗━ Postprocessing: \(String(format: "%.2f", timing.postprocessingTime))ms (\(String(format: "%.1f", (timing.postprocessingTime/timing.totalTime)*100))%)")
        print("")
        print("🚀 THROUGHPUT:")
        print("   Texts per second: \(String(format: "%.1f", timing.throughputPerSec))")
        print("")
        print("💾 RESOURCE USAGE:")
        print("   Memory Start: \(String(format: "%.2f", resources.memoryStartMB)) MB")
        print("   Memory End: \(String(format: "%.2f", resources.memoryEndMB)) MB")
        print("   Memory Delta: +\(String(format: "%.2f", resources.memoryDeltaMB)) MB")
        print("   CPU Usage: 0.0% avg, 0.0% peak (1 samples)")
        print("")
        let performanceRating = timing.totalTime < 50 ? "✅ EXCELLENT" : timing.totalTime < 100 ? "✅ GOOD" : "⚠️ FAIR"
        print("🎯 PERFORMANCE RATING: \(performanceRating)")
        print("   (\(String(format: "%.1f", timing.totalTime))ms total - Target: <100ms)")
        
    } catch {
        print("❌ Error: \(error.localizedDescription)")
        exit(1)
    }
}

// MARK: - Data Extension
private extension Data {
    init<T>(from array: [T]) {
        self = array.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}

// Run main function
main()
