import Foundation
import onnxruntime_objc

// Vocabulary data structure for TF-IDF
struct VocabData: Codable {
    let vocab: [String: Int]
    let idf: [Float]
}

// Scaler data structure for feature normalization
struct ScalerData: Codable {
    let mean: [Float]
    let scale: [Float]
}

// Binary classification result
struct ClassificationResult {
    let sentiment: String
    let confidence: Float
    let probability: Float
    let processingTime: TimeInterval
}

class ONNXBinaryClassifier {
    private let session: ORTSession
    private let vocabData: VocabData
    private let scalerData: ScalerData
    private let vocabSize: Int
    
    init(modelPath: String, vocabPath: String, scalerPath: String) throws {
        // Initialize ONNX Runtime environment
        let env = try ORTEnv(loggingLevel: .warning)
        let sessionOptions = try ORTSessionOptions()
        
        // Create session
        self.session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: sessionOptions)
        
        // Load vocabulary data
        let vocabData = try Data(contentsOf: URL(fileURLWithPath: vocabPath))
        self.vocabData = try JSONDecoder().decode(VocabData.self, from: vocabData)
        
        // Load scaler data
        let scalerData = try Data(contentsOf: URL(fileURLWithPath: scalerPath))
        self.scalerData = try JSONDecoder().decode(ScalerData.self, from: scalerData)
        
        self.vocabSize = self.vocabData.vocab.count
        
        print("✅ ONNX Runtime initialized successfully")
        print("   📊 Vocabulary size: \(vocabSize)")
        print("   🔢 IDF weights loaded: \(self.vocabData.idf.count)")
        print("   ⚖️ Scaler parameters loaded: \(self.scalerData.mean.count)")
    }
    
    func predict(text: String) throws -> ClassificationResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Preprocess text to TF-IDF features
        let features = preprocessText(text)
        
        // Create input tensor
        let inputData = Data(from: features)
        let inputTensor = try ORTValue(
            tensorData: NSMutableData(data: inputData),
            elementType: .float,
            shape: [1, NSNumber(value: vocabSize)]
        )
        
        // Run inference
        let outputs = try session.run(
            withInputs: ["float_input": inputTensor],
            outputNames: Set(["output_0"]),
            runOptions: nil
        )
        
        guard let result = outputs["output_0"],
              let outputData = result.tensorData as? Data else {
            throw NSError(domain: "ONNXInference", code: 1, 
                         userInfo: [NSLocalizedDescriptionKey: "Failed to get output tensor"])
        }
        
        // Extract probability
        let probability = outputData.withUnsafeBytes { buffer in
            return buffer.bindMemory(to: Float.self)[0]
        }
        
        let endTime = CFAbsoluteTimeGetCurrent()
        let processingTime = endTime - startTime
        
        let sentiment = probability > 0.5 ? "POSITIVE" : "NEGATIVE"
        let confidence = probability > 0.5 ? probability : (1.0 - probability)
        
        return ClassificationResult(
            sentiment: sentiment,
            confidence: confidence,
            probability: probability,
            processingTime: processingTime
        )
    }
    
    private func preprocessText(_ text: String) -> [Float] {
        // Initialize TF-IDF vector
        var vector = Array(repeating: Float(0.0), count: vocabSize)
        
        // Tokenize text (simple whitespace splitting)
        let tokens = text.lowercased()
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
        
        // Count term frequencies
        var termFreqs: [String: Int] = [:]
        for token in tokens {
            termFreqs[token, default: 0] += 1
        }
        
        // Calculate TF-IDF
        for (word, tf) in termFreqs {
            if let idx = vocabData.vocab[word], idx < vocabSize {
                let tfValue = Float(tf)
                let idfValue = vocabData.idf[idx]
                vector[idx] = tfValue * idfValue
            }
        }
        
        // Apply scaling
        for i in 0..<vocabSize {
            vector[i] = (vector[i] - scalerData.mean[i]) / scalerData.scale[i]
        }
        
        return vector
    }
}

// Helper extension for Data creation from array
private extension Data {
    init<T>(from array: [T]) {
        self = array.withUnsafeBufferPointer { buffer in
            Data(buffer: buffer)
        }
    }
} 