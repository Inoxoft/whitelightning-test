import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:onnxruntime/onnxruntime.dart';

class VectorizerData {
  final Map<String, int> vocabulary;
  final List<double> idf;
  final int maxFeatures;
  
  VectorizerData({
    required this.vocabulary,
    required this.idf,
    required this.maxFeatures,
  });
  
  factory VectorizerData.fromJson(Map<String, dynamic> json) {
    final vocab = json['vocabulary'] as Map<String, dynamic>? ?? 
                  json['vocab'] as Map<String, dynamic>;
    final vocabulary = Map<String, int>.from(vocab);
    final idf = List<double>.from(json['idf']);
    final maxFeatures = json['max_features'] as int? ?? 5000;
    
    return VectorizerData(
      vocabulary: vocabulary,
      idf: idf,
      maxFeatures: maxFeatures,
    );
  }
}

class SystemInfo {
  final String platform;
  final int processorCount;
  final String dartVersion;
  
  SystemInfo()
      : platform = Platform.operatingSystem,
        processorCount = Platform.numberOfProcessors,
        dartVersion = Platform.version;
}

Future<VectorizerData> loadVectorizer(String path) async {
  final file = File(path);
  final contents = await file.readAsString();
  final json = jsonDecode(contents) as Map<String, dynamic>;
  return VectorizerData.fromJson(json);
}

Future<Map<String, String>> loadClasses(String path) async {
  final file = File(path);
  final contents = await file.readAsString();
  final json = jsonDecode(contents) as Map<String, dynamic>;
  return Map<String, String>.from(json);
}

List<double> preprocessText(String text, VectorizerData vectorizer) {
  final stopwatch = Stopwatch()..start();
  
  // Tokenize text (match sklearn's pattern)
  final tokenRegex = RegExp(r'\b\w\w+\b');
  final textLower = text.toLowerCase();
  final tokens = tokenRegex.allMatches(textLower).map((m) => m.group(0)!).toList();
  
  print('📊 Tokens found: ${tokens.length}, First 10: ${tokens.take(10).join(', ')}');
  
  // Count term frequencies
  final termCounts = <String, int>{};
  for (final token in tokens) {
    termCounts[token] = (termCounts[token] ?? 0) + 1;
  }
  
  // Create TF-IDF vector
  final vector = List<double>.filled(vectorizer.maxFeatures, 0.0);
  int foundInVocab = 0;
  
  // Apply TF-IDF
  for (final entry in termCounts.entries) {
    final term = entry.key;
    final count = entry.value;
    final termIndex = vectorizer.vocabulary[term];
    
    if (termIndex != null && termIndex < vectorizer.maxFeatures) {
      vector[termIndex] = count * vectorizer.idf[termIndex];
      foundInVocab++;
    }
  }
  
  print('📊 Found $foundInVocab terms in vocabulary out of ${tokens.length} total tokens');
  
  // L2 normalization
  double norm = 0.0;
  for (final value in vector) {
    norm += value * value;
  }
  norm = sqrt(norm);
  
  if (norm > 0) {
    for (int i = 0; i < vector.length; i++) {
      vector[i] /= norm;
    }
  }
  
  stopwatch.stop();
  print('📊 TF-IDF: $foundInVocab non-zero, norm: ${norm.toStringAsFixed(4)}');
  print('📊 Preprocessing completed in ${stopwatch.elapsedMilliseconds}ms');
  
  return vector;
}

Future<List<double>> runInference(OrtSession session, List<double> vector) async {
  final stopwatch = Stopwatch()..start();
  
  // Create input tensor
  final inputTensor = OrtValueTensor.createTensorWithDataList(
    Float32List.fromList(vector.map((v) => v.toDouble()).toList()),
    [1, vector.length],
  );
  
  // Run inference
  final outputs = await session.runAsync(
    {'input': inputTensor},
    ['output'],
  );
  
  // Get output
  final outputTensor = outputs['output'] as OrtValueTensor;
  final predictions = outputTensor.value as List<double>;
  
  stopwatch.stop();
  print('📊 Inference completed in ${stopwatch.elapsedMilliseconds}ms');
  
  return predictions;
}

void main(List<String> args) async {
  final testText = args.isNotEmpty ? args[0] : 
      "I'm about to give birth, and I'm terrified. What if something goes wrong? What if I can't handle the pain? Received an unexpected compliment at work today. Small moments of happiness can make a big difference.";
  
  print('🤖 ONNX MULTICLASS SIGMOID CLASSIFIER - DART IMPLEMENTATION');
  print('=' * 62);
  print('🔄 Processing: $testText');
  print('');
  
  // System information
  final systemInfo = SystemInfo();
  print('💻 SYSTEM INFORMATION:');
  print('   Platform: ${systemInfo.platform}');
  print('   CPU Cores: ${systemInfo.processorCount}');
  print('   Runtime: Dart ${systemInfo.dartVersion}');
  print('');
  
  final totalStopwatch = Stopwatch()..start();
  
  try {
    // Load components
    print('🔧 Loading components...');
    
    OrtEnv.instance.init();
    final sessionOptions = OrtSessionOptions();
    final session = OrtSession.fromFile('model.onnx', sessionOptions);
    print('✅ ONNX model loaded');
    
    final vectorizer = await loadVectorizer('vocab.json');
    print('✅ Vectorizer loaded (vocab: ${vectorizer.vocabulary.length} words)');
    
    final classes = await loadClasses('scaler.json');
    print('✅ Classes loaded: ${classes.values.join(', ')}');
    print('');
    
    // Preprocess text
    final vector = preprocessText(testText, vectorizer);
    print('📊 TF-IDF shape: [1, ${vector.length}]');
    print('');
    
    // Run inference
    final predictions = await runInference(session, vector);
    
    // Display results
    print('📊 EMOTION ANALYSIS RESULTS:');
    final emotionResults = <MapEntry<String, double>>[];
    
    for (int i = 0; i < predictions.length; i++) {
      final className = classes[i.toString()] ?? 'Class $i';
      final probability = predictions[i];
      emotionResults.add(MapEntry(className, probability));
      print('   $className: ${probability.toStringAsFixed(3)}');
    }
    
    // Find dominant emotion
    final dominantEmotion = emotionResults.reduce((a, b) => a.value > b.value ? a : b);
    print('   🏆 Dominant Emotion: ${dominantEmotion.key} (${dominantEmotion.value.toStringAsFixed(3)})');
    
    print('   📝 Input Text: "$testText"');
    print('');
    
    // Performance metrics
    totalStopwatch.stop();
    final totalMs = totalStopwatch.elapsedMilliseconds;
    
    print('📈 PERFORMANCE SUMMARY:');
    print('   Total Processing Time: ${totalMs}ms');
    print('');
    
    // Throughput
    final throughput = 1000.0 / totalMs;
    print('🚀 THROUGHPUT:');
    print('   Texts per second: ${throughput.toStringAsFixed(1)}');
    print('');
    
    // Performance rating
    String rating;
    if (totalMs < 50) {
      rating = '🚀 EXCELLENT';
    } else if (totalMs < 100) {
      rating = '✅ GOOD';
    } else if (totalMs < 500) {
      rating = '⚠️ ACCEPTABLE';
    } else {
      rating = '🐌 SLOW';
    }
    
    print('🎯 PERFORMANCE RATING: $rating');
    print('   (${totalMs}ms total - Target: <100ms)');
    
    // Cleanup
    session.release();
    OrtEnv.instance.release();
    
  } catch (e) {
    print('❌ Error: $e');
    exit(1);
  }
} 