#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <regex>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct SystemInfo {
    std::string platform;
    int cpuCores;
    std::string cppVersion;
    
    SystemInfo() {
        #ifdef _WIN32
        platform = "Windows";
        #elif __APPLE__
        platform = "macOS";
        #elif __linux__
        platform = "Linux";
        #else
        platform = "Unknown";
        #endif
        
        cpuCores = std::thread::hardware_concurrency();
        cppVersion = std::to_string(__cplusplus);
    }
};

struct VectorizerData {
    std::map<std::string, int> vocabulary;
    std::vector<double> idf;
    int maxFeatures;
};

VectorizerData loadVectorizer(const std::string& path) {
    std::ifstream file(path);
    json j;
    file >> j;
    
    VectorizerData vectorizer;
    
    // Handle both vocabulary field names
    if (j.contains("vocabulary")) {
        vectorizer.vocabulary = j["vocabulary"];
    } else if (j.contains("vocab")) {
        vectorizer.vocabulary = j["vocab"];
    }
    
    vectorizer.idf = j["idf"];
    vectorizer.maxFeatures = j.contains("max_features") ? j["max_features"] : 5000;
    
    return vectorizer;
}

std::map<std::string, std::string> loadClasses(const std::string& path) {
    std::ifstream file(path);
    json j;
    file >> j;
    return j;
}

std::vector<float> preprocessText(const std::string& text, const VectorizerData& vectorizer) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Tokenize text (match sklearn's pattern)
    std::regex tokenRegex("\\b\\w\\w+\\b");
    std::string lowerText = text;
    std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(), ::tolower);
    
    std::vector<std::string> tokens;
    std::sregex_iterator iter(lowerText.begin(), lowerText.end(), tokenRegex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        tokens.push_back(iter->str());
    }
    
    std::cout << "📊 Tokens found: " << tokens.size() << std::endl;
    
    // Count term frequencies
    std::map<std::string, int> termCounts;
    for (const auto& token : tokens) {
        termCounts[token]++;
    }
    
    // Create TF-IDF vector
    std::vector<float> vector(vectorizer.maxFeatures, 0.0f);
    int foundInVocab = 0;
    
    // Apply TF-IDF
    for (const auto& [term, count] : termCounts) {
        auto it = vectorizer.vocabulary.find(term);
        if (it != vectorizer.vocabulary.end() && it->second < vectorizer.maxFeatures) {
            vector[it->second] = count * vectorizer.idf[it->second];
            foundInVocab++;
        }
    }
    
    std::cout << "📊 Found " << foundInVocab << " terms in vocabulary out of " 
              << tokens.size() << " total tokens" << std::endl;
    
    // L2 normalization
    double norm = 0.0;
    for (float value : vector) {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    
    if (norm > 0) {
        for (float& value : vector) {
            value /= norm;
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    std::cout << "📊 TF-IDF: " << foundInVocab << " non-zero, norm: " 
              << std::fixed << std::setprecision(4) << norm << std::endl;
    
    return vector;
}

std::vector<float> runInference(Ort::Session& session, const std::vector<float>& vector) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    std::string inputName = session.GetInputName(0, allocator);
    std::string outputName = session.GetOutputName(0, allocator);
    
    // Create input tensor
    std::vector<int64_t> inputShape = {1, static_cast<int64_t>(vector.size())};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        const_cast<float*>(vector.data()), vector.size(), inputShape.data(), inputShape.size());
    
    // Run inference
    std::vector<Ort::Value> outputs = session.Run(
        Ort::RunOptions{nullptr}, &inputName, &inputTensor, 1, &outputName, 1);
    
    // Get output
    float* outputData = outputs[0].GetTensorMutableData<float>();
    std::vector<int64_t> outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    
    std::vector<float> predictions(outputData, outputData + outputShape[1]);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    std::cout << "📊 Inference completed in " << duration.count() << "ms" << std::endl;
    
    return predictions;
}

int main(int argc, char* argv[]) {
    std::string testText = (argc > 1) ? argv[1] : 
        "I'm about to give birth, and I'm terrified. What if something goes wrong? What if I can't handle the pain? Received an unexpected compliment at work today. Small moments of happiness can make a big difference.";
    
    std::cout << "🤖 ONNX MULTICLASS SIGMOID CLASSIFIER - C++ IMPLEMENTATION" << std::endl;
    std::cout << std::string(63, '=') << std::endl;
    std::cout << "🔄 Processing: " << testText << std::endl << std::endl;
    
    // System information
    SystemInfo systemInfo;
    std::cout << "💻 SYSTEM INFORMATION:" << std::endl;
    std::cout << "   Platform: " << systemInfo.platform << std::endl;
    std::cout << "   CPU Cores: " << systemInfo.cpuCores << std::endl;
    std::cout << "   Runtime: C++ " << systemInfo.cppVersion << std::endl << std::endl;
    
    auto totalStartTime = std::chrono::high_resolution_clock::now();
    
    try {
        // Load components
        std::cout << "🔧 Loading components..." << std::endl;
        
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MulticlassSigmoidTest");
        Ort::SessionOptions sessionOptions;
        Ort::Session session(env, "model.onnx", sessionOptions);
        std::cout << "✅ ONNX model loaded" << std::endl;
        
        VectorizerData vectorizer = loadVectorizer("vocab.json");
        std::cout << "✅ Vectorizer loaded (vocab: " << vectorizer.vocabulary.size() << " words)" << std::endl;
        
        std::map<std::string, std::string> classes = loadClasses("scaler.json");
        std::cout << "✅ Classes loaded" << std::endl << std::endl;
        
        // Preprocess text
        std::vector<float> vector = preprocessText(testText, vectorizer);
        std::cout << "📊 TF-IDF shape: [1, " << vector.size() << "]" << std::endl << std::endl;
        
        // Run inference
        std::vector<float> predictions = runInference(session, vector);
        
        // Display results
        std::cout << "📊 EMOTION ANALYSIS RESULTS:" << std::endl;
        std::vector<std::pair<std::string, float>> emotionResults;
        
        for (size_t i = 0; i < predictions.size(); i++) {
            std::string className = classes.count(std::to_string(i)) ? 
                classes[std::to_string(i)] : ("Class " + std::to_string(i));
            float probability = predictions[i];
            emotionResults.push_back({className, probability});
            std::cout << "   " << className << ": " << std::fixed << std::setprecision(3) 
                      << probability << std::endl;
        }
        
        // Find dominant emotion
        auto dominantEmotion = *std::max_element(emotionResults.begin(), emotionResults.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        
        std::cout << "   🏆 Dominant Emotion: " << dominantEmotion.first 
                  << " (" << std::fixed << std::setprecision(3) << dominantEmotion.second << ")" << std::endl;
        
        std::cout << "   📝 Input Text: \"" << testText << "\"" << std::endl << std::endl;
        
        // Performance metrics
        auto totalEndTime = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(totalEndTime - totalStartTime);
        
        std::cout << "📈 PERFORMANCE SUMMARY:" << std::endl;
        std::cout << "   Total Processing Time: " << totalDuration.count() << "ms" << std::endl << std::endl;
        
        // Throughput
        double throughput = 1000.0 / totalDuration.count();
        std::cout << "🚀 THROUGHPUT:" << std::endl;
        std::cout << "   Texts per second: " << std::fixed << std::setprecision(1) 
                  << throughput << std::endl << std::endl;
        
        // Performance rating
        std::string rating;
        if (totalDuration.count() < 50) {
            rating = "🚀 EXCELLENT";
        } else if (totalDuration.count() < 100) {
            rating = "✅ GOOD";
        } else if (totalDuration.count() < 500) {
            rating = "⚠️ ACCEPTABLE";
        } else {
            rating = "🐌 SLOW";
        }
        
        std::cout << "🎯 PERFORMANCE RATING: " << rating << std::endl;
        std::cout << "   (" << totalDuration.count() << "ms total - Target: <100ms)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 