import Foundation
import BinaryClassifierLib

func main() {
    let arguments = CommandLine.arguments
    let inputText = arguments.count > 1 ? arguments[1] : "Congratulations! You've won a free iPhone — click here to claim your prize now!"
    
    print("🤖 SWIFT ONNX BINARY CLASSIFIER")
    print("================================")
    print("")
    
    // Print system information
    SystemInfo.printSystemInfo()
    
    do {
        // Initialize classifier with model files
        let classifier = try ONNXBinaryClassifier(
            modelPath: "model.onnx",
            vocabPath: "vocab.json",
            scalerPath: "scaler.json"
        )
        
        // Perform prediction
        let result = try classifier.predict(text: inputText)
        
        // Performance rating
        let rating = result.confidence > 0.8 ? "🎯 HIGH CONFIDENCE" :
                     result.confidence > 0.6 ? "🎯 MEDIUM CONFIDENCE" : "🎯 LOW CONFIDENCE"
        
        print("")
        print("🎯 PERFORMANCE RATING: ✅ \(rating)")
        
        #if os(Linux)
        print("   (Swift on Linux with ONNX Runtime support)")
        #elseif os(macOS)
        print("   (Swift on macOS with ONNX Runtime)")
        #elseif os(iOS)
        print("   (Swift on iOS with ONNX Runtime)")
        #else
        print("   (Swift cross-platform ONNX Runtime)")
        #endif
        
        print("")
        print("✅ Classification completed successfully!")
        
    } catch {
        print("❌ Error: \(error.localizedDescription)")
        print("🎯 PERFORMANCE RATING: ❌ ERROR")
        
        // Fallback demonstration
        print("")
        print("📝 FALLBACK DEMONSTRATION:")
        let words = inputText.lowercased().components(separatedBy: .whitespacesAndNewlines)
        let hasSpam = words.contains { ["free", "win", "prize", "click"].contains($0) }
        let sentiment = hasSpam ? "NEGATIVE" : "POSITIVE"
        print("   Basic analysis suggests: \(sentiment)")
    }
}

main() 