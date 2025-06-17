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

class BinaryClassifierLoader {
    private(set) var vocab: [String: Int] = [:]
    private(set) var scaler: [String: [Double]] = [:]

    init(vocabPath: String, scalerPath: String) throws {
        let vocabData = try Data(contentsOf: URL(fileURLWithPath: vocabPath))
        let scalerData = try Data(contentsOf: URL(fileURLWithPath: scalerPath))

        if let vocabJson = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] {
            vocab = vocabJson
        }
        
        if let scalerJson = try JSONSerialization.jsonObject(with: scalerData) as? [String: [Double]] {
            scaler = scalerJson
        }
    }
}

class ONNXBinaryClassifier {
    private var session: ORTSession
    private var vocab: [String: Int]
    private var scaler: [String: [Double]]
    private let maxLen = 30

    init() throws {
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx"),
              let vocabPath = Bundle.main.path(forResource: "vocab", ofType: "json"),
              let scalerPath = Bundle.main.path(forResource: "scaler", ofType: "json") else {
            throw NSError(domain: "Paths", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing model or json files"])
        }

        let loader = try BinaryClassifierLoader(vocabPath: vocabPath, scalerPath: scalerPath)
        self.vocab = loader.vocab
        self.scaler = loader.scaler

        let env = try ORTEnv(loggingLevel: .warning)
        let sessionOptions = try ORTSessionOptions()
        self.session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: sessionOptions)
    }

    func predict(text: String) throws -> (sentiment: String, confidence: Float, probability: Float) {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // System Information
        print("💻 SYSTEM INFORMATION:")
        print("   Platform: \(UIDevice.current.systemName)")
        print("   Processor: \(ProcessInfo.processInfo.processorCount) cores")
        print("   Total Memory: \(ProcessInfo.processInfo.physicalMemory / 1024 / 1024 / 1024) GB")
        print("   Runtime: Swift \(String(describing: ProcessInfo.processInfo.operatingSystemVersion))")
        print("")
        
        let preprocessStart = CFAbsoluteTimeGetCurrent()
        
        // Preprocessing: Tokenization + TF-IDF + Scaling
        let tokenizer = Tokenizer(vocab: vocab)
        var tokens = tokenizer.tokenize(text: text)

        // Pad or truncate to maxLen
        if tokens.count < maxLen {
            tokens += Array(repeating: 0, count: maxLen - tokens.count)
        } else if tokens.count > maxLen {
            tokens = Array(tokens.prefix(maxLen))
        }

        // Convert to TF-IDF (simplified - you might need to implement full TF-IDF)
        var tfidfVector = Array(repeating: Float(0), count: vocab.count)
        let termFreqs = Dictionary(grouping: tokens, by: { $0 })
            .mapValues { Float($0.count) / Float(tokens.count) }
        
        for (term, freq) in termFreqs {
            if term < tfidfVector.count {
                tfidfVector[term] = freq
            }
        }
        
        // Apply scaling if available
        if let mean = scaler["mean"], let scale = scaler["scale"] {
            for i in 0..<min(tfidfVector.count, min(mean.count, scale.count)) {
                tfidfVector[i] = (tfidfVector[i] - Float(mean[i])) / Float(scale[i])
            }
        }
        
        let preprocessTime = CFAbsoluteTimeGetCurrent() - preprocessStart

        // Model Inference
        let inferenceStart = CFAbsoluteTimeGetCurrent()
        
        let inputData = Data(bytes: &tfidfVector, count: tfidfVector.count * MemoryLayout<Float>.size)
        let tensor = try ORTValue(
            tensorData: NSMutableData(data: inputData),
            elementType: .float,
            shape: [1, NSNumber(value: tfidfVector.count)]
        )

        let outputs = try session.run(
            withInputs: ["input": tensor],
            outputNames: Set(["output"]),
            runOptions: nil
        )

        guard let result = outputs["output"]?.value as? [[Float]],
              let prediction = result.first?.first else {
            throw NSError(domain: "ONNX", code: 2, userInfo: [NSLocalizedDescriptionKey: "No valid output"])
        }
        
        let inferenceTime = CFAbsoluteTimeGetCurrent() - inferenceStart
        
        // Post-processing
        let postprocessStart = CFAbsoluteTimeGetCurrent()
        let probability = prediction // Assuming sigmoid output
        let sentiment = probability > 0.5 ? "POSITIVE" : "NEGATIVE"
        let confidence = probability > 0.5 ? probability : (1.0 - probability)
        let postprocessTime = CFAbsoluteTimeGetCurrent() - postprocessStart
        
        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        
        // Output Results
        print("📊 SENTIMENT ANALYSIS RESULTS:")
        print("   🏆 Predicted Sentiment: \(sentiment)")
        print("   📈 Confidence: \(String(format: "%.2f", confidence * 100))% (\(String(format: "%.4f", confidence)))")
        print("   📝 Input Text: \"\(text)\"")
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
        
        return (sentiment: sentiment, confidence: confidence, probability: probability)
    }
}

// Example usage function
func testBinaryClassifier(with text: String) {
    do {
        print("🤖 ONNX BINARY CLASSIFIER - SWIFT IMPLEMENTATION")
        print("==============================================")
        print("🔄 Processing: \(text)")
        print("")
        
        let classifier = try ONNXBinaryClassifier()
        let result = try classifier.predict(text: text)
        
        print("")
        print("✅ Classification completed successfully!")
        
    } catch {
        print("❌ Error: \(error.localizedDescription)")
        print("🎯 PERFORMANCE RATING: ❌ ERROR")
    }
} 