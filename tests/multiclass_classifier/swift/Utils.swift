import Foundation
import onnxruntime_objc

class Tokenizer {
    private var vocab: [String: Int]
    
    init(vocab: [String: Int]) {
        self.vocab = vocab
    }
    
    func tokenize(text: String) -> [Int] {
        let words = text.lowercased().split(separator: " ").map { String($0) }
        return words.map { vocab[$0] ?? vocab["<OOV>"] ?? 1 }
    }
}

class MulticlassClassifierLoader {
    private(set) var labelMap: [Int: String] = [:]
    private(set) var vocab: [String: Int] = [:]

    init(labelMapPath: String, vocabPath: String) throws {
        let labelMapData = try Data(contentsOf: URL(fileURLWithPath: labelMapPath))
        let vocabData = try Data(contentsOf: URL(fileURLWithPath: vocabPath))

        if let labelMapJson = try JSONSerialization.jsonObject(with: labelMapData) as? [String: String] {
            for (key, value) in labelMapJson {
                if let intKey = Int(key) {
                    labelMap[intKey] = value
                }
            }
        }

        if let vocabJson = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] {
            vocab = vocabJson
        }
    }
}

class ONNXMulticlassClassifier {
    private var session: ORTSession
    private var labelMap: [Int: String]
    private var vocab: [String: Int]
    private let maxLen = 30

    init() throws {
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx"),
              let labelPath = Bundle.main.path(forResource: "label_map", ofType: "json"),
              let vocabPath = Bundle.main.path(forResource: "vocab", ofType: "json") else {
            throw NSError(domain: "Paths", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing model or json files"])
        }

        let loader = try MulticlassClassifierLoader(labelMapPath: labelPath, vocabPath: vocabPath)
        self.labelMap = loader.labelMap
        self.vocab = loader.vocab

        let env = try ORTEnv(loggingLevel: .warning)
        let sessionOptions = try ORTSessionOptions()
        self.session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: sessionOptions)
    }

    func predict(text: String) throws -> (topic: String, confidence: Float, allScores: [(String, Float)]) {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // System Information
        print("💻 SYSTEM INFORMATION:")
        print("   Platform: \(UIDevice.current.systemName)")
        print("   Processor: \(ProcessInfo.processInfo.processorCount) cores")
        print("   Total Memory: \(ProcessInfo.processInfo.physicalMemory / 1024 / 1024 / 1024) GB")
        print("   Runtime: Swift \(String(describing: ProcessInfo.processInfo.operatingSystemVersion))")
        print("")
        
        let preprocessStart = CFAbsoluteTimeGetCurrent()
        
        // Preprocessing: Tokenization
        let tokenizer = Tokenizer(vocab: vocab)
        var tokens = tokenizer.tokenize(text: text)

        // Pad or truncate to maxLen
        if tokens.count < maxLen {
            tokens += Array(repeating: 0, count: maxLen - tokens.count)
        } else if tokens.count > maxLen {
            tokens = Array(tokens.prefix(maxLen))
        }

        let int32Tokens = tokens.map { Int32($0) }
        let preprocessTime = CFAbsoluteTimeGetCurrent() - preprocessStart

        // Model Inference
        let inferenceStart = CFAbsoluteTimeGetCurrent()
        
        let inputData = Data(from: int32Tokens)
        let tensor = try ORTValue(
            tensorData: NSMutableData(data: inputData),
            elementType: .int32,
            shape: [1, NSNumber(value: maxLen)]
        )

        let outputs = try session.run(
            withInputs: ["input": tensor],
            outputNames: Set(["sequential"]),
            runOptions: nil
        )

        guard let result = outputs["sequential"]?.value as? [[Float]] else {
            throw NSError(domain: "ONNX", code: 2, userInfo: [NSLocalizedDescriptionKey: "No valid output"])
        }
        
        let inferenceTime = CFAbsoluteTimeGetCurrent() - inferenceStart
        
        // Post-processing
        let postprocessStart = CFAbsoluteTimeGetCurrent()
        
        let scores = result[0]
        let allScores = scores.enumerated().map { (i, score) in
            (labelMap[i] ?? "Unknown", score)
        }
        
        // Find prediction with highest score
        guard let (predictedIndex, maxScore) = scores.enumerated().max(by: { $0.element < $1.element }) else {
            throw NSError(domain: "ONNX", code: 3, userInfo: [NSLocalizedDescriptionKey: "Could not find max score"])
        }
        
        let predictedTopic = labelMap[predictedIndex] ?? "Unknown"
        
        // Apply softmax for confidence calculation
        let expScores = scores.map { exp($0) }
        let sumExp = expScores.reduce(0, +)
        let probabilities = expScores.map { $0 / sumExp }
        let confidence = probabilities[predictedIndex]
        
        let postprocessTime = CFAbsoluteTimeGetCurrent() - postprocessStart
        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        
        // Output Results
        print("📊 TOPIC CLASSIFICATION RESULTS:")
        print("   🏆 Predicted Topic: \(predictedTopic.uppercased())")
        print("   📈 Confidence: \(String(format: "%.2f", confidence * 100))% (\(String(format: "%.4f", confidence)))")
        print("   📝 Input Text: \"\(text)\"")
        print("")
        
        // Show top 3 predictions
        let sortedScores = allScores.sorted { $0.1 > $1.1 }.prefix(3)
        print("   📊 Top 3 Predictions:")
        for (i, (topic, score)) in sortedScores.enumerated() {
            let prob = exp(score) / sumExp
            print("   \(i+1). \(topic): \(String(format: "%.2f", prob * 100))%")
        }
        print("")
        
        print("📈 PERFORMANCE SUMMARY:")
        print("   Total Processing Time: \(String(format: "%.0f", totalTime * 1000))ms")
        print("   ┣━ Preprocessing: \(String(format: "%.0f", preprocessTime * 1000))ms (\(String(format: "%.1f", preprocessTime/totalTime*100))%)")
        print("   ┣━ Model Inference: \(String(format: "%.0f", inferenceTime * 1000))ms (\(String(format: "%.1f", inferenceTime/totalTime*100))%)")
        print("   ┗━ Postprocessing: \(String(format: "%.0f", postprocessTime * 1000))ms (\(String(format: "%.1f", postprocessTime/totalTime*100))%)")
        print("")
        
        print("🚀 THROUGHPUT:")
        print("   Texts per second: \(String(format: "%.1f", 1.0/totalTime))")
        print("")
        
        print("💾 RESOURCE USAGE:")
        print("   Memory: \(ProcessInfo.processInfo.physicalMemory / 1024 / 1024) MB")
        print("   CPU Usage: [Monitoring not implemented]")
        print("")
        
        let rating = totalTime < 0.1 ? "🚀 EXCELLENT" : 
                    totalTime < 0.5 ? "✅ GOOD" : 
                    totalTime < 1.0 ? "⚠️ ACCEPTABLE" : "🐌 SLOW"
        print("🎯 PERFORMANCE RATING: \(rating)")
        print("   (\(String(format: "%.0f", totalTime * 1000))ms total - Target: <100ms)")
        
        return (topic: predictedTopic, confidence: confidence, allScores: allScores)
    }
}

// Helper extension for Data creation from arrays
private extension Data {
    init<T>(from array: [T]) {
        self = array.withUnsafeBufferPointer { buffer in
            Data(buffer: buffer)
        }
    }
}

// Example usage function
func testMulticlassClassifier(with text: String) {
    do {
        print("🤖 ONNX MULTICLASS CLASSIFIER - SWIFT IMPLEMENTATION")
        print("==================================================")
        print("🔄 Processing: \(text)")
        print("")
        
        let classifier = try ONNXMulticlassClassifier()
        let result = try classifier.predict(text: text)
        
        print("")
        print("✅ Classification completed successfully!")
        print("Final Prediction: \(result.topic) (\(String(format: "%.2f", result.confidence * 100))%)")
        
    } catch {
        print("❌ Error: \(error.localizedDescription)")
        print("🎯 PERFORMANCE RATING: ❌ ERROR")
    }
} 