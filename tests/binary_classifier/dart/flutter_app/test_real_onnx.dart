import 'dart:convert';
import 'dart:typed_data';
import 'dart:io';
import 'package:onnxruntime/onnxruntime.dart' as ort;

Future<void> main(List<String> args) async {
  print("🚀 Real ONNX Desktop Inference Test");
  print("═══════════════════════════════════");
  
  // Test inputs
  final testTexts = args.isNotEmpty ? args : [
    "This product is absolutely amazing and works perfectly!",
    "Terrible experience, worst purchase ever, totally disappointed",
    "The weather is okay today, nothing special but decent",
    "I love this item, excellent quality and fast delivery!",
    "Poor quality, broke after one day, very disappointed"
  ];
  
  for (int i = 0; i < testTexts.length; i++) {
    final text = testTexts[i];
    print("\n📝 Test ${i + 1}: \"$text\"");
    print("─" * 60);
    
    try {
      final totalStart = DateTime.now();
      
      // 1. Load and preprocess
      print("📚 Loading model files...");
      final vocab = await loadVocab();
      final scaler = await loadScaler();
      
      print("⚙️  Preprocessing text...");
      final processedVector = preprocessText(text, vocab, scaler);
      print("📊 Vector size: ${processedVector.length} features");
      
      // 2. Initialize ONNX
      print("🧠 Initializing ONNX Runtime...");
      ort.OrtEnv.instance.init();
      
      // 3. Load model
      print("📦 Loading ONNX model...");
      final modelBytes = await File('assets/models/model.onnx').readAsBytes();
      final sessionOptions = ort.OrtSessionOptions();
      final session = ort.OrtSession.fromBuffer(modelBytes, sessionOptions);
      
      print("📊 Model size: ${(modelBytes.length / (1024 * 1024)).toStringAsFixed(2)} MB");
      
      // 4. Run inference
      print("🚀 Running inference...");
      final inferenceStart = DateTime.now();
      
      final inputNames = session.inputNames;
      final inputName = inputNames.first;
      
      final inputTensor = ort.OrtValueTensor.createTensorWithDataList(
        processedVector,
        [1, processedVector.length],
      );
      
      final result = await session.runAsync(ort.OrtRunOptions(), {
        inputName: inputTensor,
      });
      
      final inferenceEnd = DateTime.now();
      final inferenceTime = inferenceEnd.difference(inferenceStart);
      
      // 5. Extract results
      double probability = 0.5;
      if (result != null && result.isNotEmpty) {
        final outputTensor = result.values.first as ort.OrtValueTensor;
        final List<dynamic> probabilities = outputTensor.value as List<dynamic>;
        final List<dynamic> flatProbs = (probabilities.isNotEmpty && probabilities.first is List)
            ? probabilities.first as List<dynamic>
            : probabilities;
        if (flatProbs.isNotEmpty) {
          probability = (flatProbs[0] as num).toDouble().clamp(0.0, 1.0);
        }
      }
      
      // 6. Calculate metrics
      final confidence = (probability - 0.5).abs() * 2;
      String sentiment;
      String emoji;
      
      if (probability > 0.65) {
        sentiment = "POSITIVE";
        emoji = "😊";
      } else if (probability < 0.35) {
        sentiment = "NEGATIVE";
        emoji = "😞";
      } else {
        sentiment = "NEUTRAL";
        emoji = "😐";
      }
      
      final totalTime = DateTime.now().difference(totalStart);
      
      // 7. Display results
      print("⏱️  Inference time: ${inferenceTime.inMicroseconds}μs (${inferenceTime.inMilliseconds}ms)");
      print("⏱️  Total time: ${totalTime.inMilliseconds}ms");
      print("🎯 REAL PREDICTION: $sentiment $emoji");
      print("📈 Probability: ${(probability * 100).toStringAsFixed(2)}%");
      print("🎪 Confidence: ${(confidence * 100).toStringAsFixed(1)}%");
      
      if (inferenceTime.inMilliseconds < 10) {
        print("⚡ Performance: EXCELLENT");
      } else if (inferenceTime.inMilliseconds < 50) {
        print("🚀 Performance: VERY GOOD");
      } else {
        print("✅ Performance: ACCEPTABLE");
      }
      
      // Cleanup
      inputTensor.release();
      session.release();
      ort.OrtEnv.instance.release();
      
      print("✅ REAL ONNX INFERENCE SUCCESSFUL!");
      
    } catch (e) {
      print("❌ ONNX Error: $e");
      print("💡 Make sure ONNX Runtime is installed on your system");
    }
  }
  
  print("\n🏆 Real Desktop ONNX Test Complete!");
  print("🎯 This shows actual ML predictions with your model!");
}

Future<Map<String, dynamic>> loadVocab() async {
  final vocabData = await File('assets/models/vocab.json').readAsString();
  return json.decode(vocabData);
}

Future<Map<String, dynamic>> loadScaler() async {
  final scalerData = await File('assets/models/scaler.json').readAsString();
  return json.decode(scalerData);
}

Float32List preprocessText(String text, Map<String, dynamic> vocab, Map<String, dynamic> scaler) {
  final wordToIndex = Map<String, int>.from(vocab['vocab']);
  final idf = (vocab['idf'] as List).map((e) => (e as num).toDouble()).toList();
  final mean = (scaler['mean'] as List).map((e) => (e as num).toDouble()).toList();
  final scale = (scaler['scale'] as List).map((e) => (e as num).toDouble()).toList();
  
  // Tokenize and compute TF
  final tf = List<double>.filled(wordToIndex.length, 0.0);
  final words = text.toLowerCase().split(' ');
  
  for (final word in words) {
    final index = wordToIndex[word];
    if (index != null) {
      tf[index] += 1.0;
    }
  }
  
  // Normalize TF
  final tfSum = tf.reduce((a, b) => a + b);
  if (tfSum > 0) {
    for (int i = 0; i < tf.length; i++) {
      tf[i] = tf[i] / tfSum;
    }
  }
  
  // Compute TF-IDF
  final tfidf = List<double>.generate(tf.length, (i) => tf[i] * idf[i]);
  
  // Scale features
  final tfidfScaled = List<double>.generate(
    tfidf.length,
    (i) => (tfidf[i] - mean[i]) / scale[i],
  );
  
  return Float32List.fromList(tfidfScaled);
} 