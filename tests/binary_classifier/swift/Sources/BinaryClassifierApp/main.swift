import Foundation
import BinaryClassifierLib

func printSystemInfo() {
    print("💻 SYSTEM INFORMATION:")
    print("   Platform: \(ProcessInfo.processInfo.operatingSystemVersionString)")
    print("   Processor: \(ProcessInfo.processInfo.processorCount) cores")
    
    #if os(macOS)
    do {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/sysctl")
        task.arguments = ["-n", "hw.memsize"]
        let pipe = Pipe()
        task.standardOutput = pipe
        try task.run()
        task.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        if let memString = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
           let memBytes = Int64(memString) {
            let memGB = memBytes / (1024 * 1024 * 1024)
            print("   Total Memory: \(memGB) GB")
        }
    } catch {
        print("   Total Memory: [Unable to detect]")
    }
    #endif
    
    print("   Runtime: Swift with ONNX Runtime")
    print("")
}

func runBenchmark(classifier: ONNXBinaryClassifier, text: String, iterations: Int = 10) {
    print("📊 RUNNING PERFORMANCE BENCHMARK:")
    print("   Iterations: \(iterations)")
    print("   Input: \"\(text)\"")
    print("")
    
    var totalTime: TimeInterval = 0
    var results: [ClassificationResult] = []
    
    // Warmup runs
    print("🔥 Warming up (3 runs)...")
    for _ in 0..<3 {
        do {
            _ = try classifier.predict(text: text)
        } catch {
            print("❌ Warmup failed: \(error)")
            return
        }
    }
    
    print("⏱️  Running benchmark...")
    let startTime = CFAbsoluteTimeGetCurrent()
    
    for i in 0..<iterations {
        do {
            let result = try classifier.predict(text: text)
            results.append(result)
            totalTime += result.processingTime
            
            if (i + 1) % 5 == 0 {
                print("   Completed \(i + 1)/\(iterations) iterations...")
            }
        } catch {
            print("❌ Benchmark iteration \(i + 1) failed: \(error)")
            return
        }
    }
    
    let endTime = CFAbsoluteTimeGetCurrent()
    let wallTime = endTime - startTime
    
    let avgProcessingTime = totalTime / Double(iterations) * 1000  // Convert to ms
    let throughput = Double(iterations) / wallTime
    
    print("")
    print("📈 BENCHMARK RESULTS:")
    print("   Average Processing Time: \(String(format: "%.1f", avgProcessingTime))ms")
    print("   Wall Clock Time: \(String(format: "%.1f", wallTime * 1000))ms")
    print("   Throughput: \(String(format: "%.1f", throughput)) texts/second")
    print("   Total Iterations: \(iterations)")
    
    if let lastResult = results.last {
        let emoji = lastResult.sentiment == "POSITIVE" ? "😊" : "😞"
        print("")
        print("🎯 FINAL RESULT:")
        print("   Sentiment: \(lastResult.sentiment) \(emoji)")
        print("   Confidence: \(String(format: "%.1f", lastResult.confidence * 100))%")
        print("   Probability: \(String(format: "%.3f", lastResult.probability))")
    }
}

func main() {
    let arguments = CommandLine.arguments
    let inputText = arguments.count > 1 ? arguments[1] : "Congratulations! You've won a free iPhone — click here to claim your prize now!"
    
    print("🤖 ONNX BINARY CLASSIFIER - SWIFT WITH REAL ONNX RUNTIME")
    print("=====================================================")
    print("🔄 Processing: \"\(inputText)\"")
    print("")
    
    printSystemInfo()
    
    // File paths (adjust for your project structure)
    let modelPath = "model.onnx"
    let vocabPath = "vocab.json"
    let scalerPath = "scaler.json"
    
    do {
        // Initialize ONNX classifier
        print("🚀 Initializing ONNX Runtime...")
        let classifier = try ONNXBinaryClassifier(
            modelPath: modelPath,
            vocabPath: vocabPath,
            scalerPath: scalerPath
        )
        
        print("")
        print("📊 RUNNING SINGLE PREDICTION:")
        
        // Single prediction
        let result = try classifier.predict(text: inputText)
        
        let emoji = result.sentiment == "POSITIVE" ? "😊" : "😞"
        print("   🏆 Predicted Sentiment: \(result.sentiment) \(emoji)")
        print("   📈 Confidence: \(String(format: "%.1f", result.confidence * 100))%")
        print("   📊 Probability: \(String(format: "%.3f", result.probability))")
        print("   ⏱️  Processing Time: \(String(format: "%.1f", result.processingTime * 1000))ms")
        print("")
        
        // Check if benchmark requested
        if arguments.contains("--benchmark") {
            if let benchmarkIndex = arguments.firstIndex(of: "--benchmark"),
               benchmarkIndex + 1 < arguments.count,
               let iterations = Int(arguments[benchmarkIndex + 1]) {
                runBenchmark(classifier: classifier, text: inputText, iterations: iterations)
            } else {
                runBenchmark(classifier: classifier, text: inputText, iterations: 50)
            }
        }
        
        print("✅ SWIFT ONNX RUNTIME INFERENCE COMPLETED SUCCESSFULLY!")
        
    } catch {
        print("❌ Error: \(error.localizedDescription)")
        
        if error.localizedDescription.contains("model.onnx") {
            print("")
            print("💡 TROUBLESHOOTING:")
            print("   • Make sure model.onnx is in the current directory")
            print("   • Verify vocab.json and scaler.json are present")
            print("   • Check file permissions")
        }
        
        exit(1)
    }
}

main() 