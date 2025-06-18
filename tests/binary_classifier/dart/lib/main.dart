import 'dart:convert';
import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

Future<Float32List> preprocessText(String text) async {
  final vocabJson = await rootBundle.loadString('vocab.json');
  final vocab = json.decode(vocabJson);
  final idf = (vocab['idf'] as List).map((e) => (e as num).toDouble()).toList();
  final word2idx = Map<String, int>.from(vocab['vocab']);

  final scalerJson = await rootBundle.loadString('scaler.json');
  final scaler = json.decode(scalerJson);
  final mean = (scaler['mean'] as List)
      .map((e) => (e as num).toDouble())
      .toList();
  final scale = (scaler['scale'] as List)
      .map((e) => (e as num).toDouble())
      .toList();

  final tf = List<double>.filled(word2idx.length, 0.0);
  final words = text.toLowerCase().split(' ');
  for (final word in words) {
    final idx = word2idx[word];
    if (idx != null) {
      tf[idx] += 1.0;
    }
  }
  final tfSum = tf.reduce((a, b) => a + b);
  if (tfSum > 0) {
    for (int i = 0; i < tf.length; i++) {
      tf[i] = tf[i] / tfSum;
    }
  }

  final tfidf = List<double>.generate(tf.length, (i) => tf[i] * idf[i]);

  final tfidfScaled = List<double>.generate(
    tfidf.length,
    (i) => (tfidf[i] - mean[i]) / scale[i],
  );

  return Float32List.fromList(tfidfScaled);
}

void printSystemInfo() {
  print('💻 SYSTEM INFORMATION:');
  print('   Platform: ${Platform.operatingSystem}');
  print('   Version: ${Platform.operatingSystemVersion}');
  print('   CPU Cores: ${Platform.numberOfProcessors} logical');
  print('   Runtime: Dart ${Platform.version}');
  print('');
}

void printPerformanceMetrics(String text, double probability, int totalTimeMs, int preprocessingMs, int inferenceMs) {
  final sentiment = probability > 0.5 ? "POSITIVE" : "NEGATIVE";
  final confidence = (probability * 100).toStringAsFixed(2);
  final throughput = (1000 / totalTimeMs).toStringAsFixed(1);
  
  print('📊 SENTIMENT ANALYSIS RESULTS:');
  print('   🏆 Predicted Sentiment: $sentiment');
  print('   📈 Confidence: $confidence% (${probability.toStringAsFixed(4)})');
  print('   📝 Input Text: "$text"');
  print('');
  
  print('📈 PERFORMANCE SUMMARY:');
  print('   Total Processing Time: ${totalTimeMs}ms');
  print('   ┣━ Preprocessing: ${preprocessingMs}ms (${(preprocessingMs / totalTimeMs * 100).toStringAsFixed(1)}%)');
  print('   ┣━ Model Inference: ${inferenceMs}ms (${(inferenceMs / totalTimeMs * 100).toStringAsFixed(1)}%)');
  print('   ┗━ Postprocessing: 0ms (0.0%)');
  print('');
  
  print('🚀 THROUGHPUT:');
  print('   Texts per second: $throughput');
  print('');
  
  print('💾 RESOURCE USAGE:');
  print('   Memory tracking not available in Dart Flutter');
  print('   CPU Usage: Not available in Dart Flutter');
  print('');
  
  final rating = totalTimeMs < 50 ? '🚀 EXCELLENT' : 
                 totalTimeMs < 100 ? '✅ GOOD' : 
                 totalTimeMs < 200 ? '⚠️ ACCEPTABLE' : '🐌 SLOW';
  print('🎯 PERFORMANCE RATING: $rating');
  print('   (${totalTimeMs}ms total - Target: <100ms)');
  print('');
}

Future<double> classifyTextBinary(String text) async {
  print('🤖 ONNX BINARY CLASSIFIER - DART IMPLEMENTATION');
  print('===========================================');
  print('🔄 Processing: $text');
  print('');
  
  printSystemInfo();
  
  final startTime = DateTime.now();
  
  // Preprocessing
  final preprocessStart = DateTime.now();
  final inputVector = await preprocessText(text);
  final preprocessEnd = DateTime.now();
  final preprocessingMs = preprocessEnd.difference(preprocessStart).inMilliseconds;

  // Model inference
  final inferenceStart = DateTime.now();
  
  OrtEnv.instance.init();
  final sessionOptions = OrtSessionOptions();
  final rawModel = await rootBundle.load('model.onnx');
  final session = OrtSession.fromBuffer(
    rawModel.buffer.asUint8List(),
    sessionOptions,
  );

  final inputNames = session.inputNames;
  if (inputNames.isEmpty) {
    throw Exception('No input names found in the model');
  }
  final inputName = inputNames[0];

  final inputTensor = OrtValueTensor.createTensorWithDataList(inputVector, [
    1,
    inputVector.length,
  ]);

  final result = await session.runAsync(OrtRunOptions(), {
    inputName: inputTensor,
  });
  
  final inferenceEnd = DateTime.now();
  final inferenceMs = inferenceEnd.difference(inferenceStart).inMilliseconds;
  
  final resultList = result?.toList();
  double probability = -1.0;
  if (resultList != null && resultList.isNotEmpty && resultList[0] != null) {
    final outputTensor = resultList[0] as OrtValueTensor;
    final List<dynamic> probabilities = outputTensor.value as List<dynamic>;
    final List<dynamic> flatProbs =
        (probabilities.isNotEmpty && probabilities.first is List)
        ? probabilities.first as List<dynamic>
        : probabilities;
    if (flatProbs.isNotEmpty) {
      probability = (flatProbs[0] as num).toDouble();
    }
  }
  
  inputTensor.release();
  session.release();
  OrtEnv.instance.release();
  
  final endTime = DateTime.now();
  final totalTimeMs = endTime.difference(startTime).inMilliseconds;
  
  printPerformanceMetrics(text, probability, totalTimeMs, preprocessingMs, inferenceMs);
  
  return probability;
}

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Binary Classification Demo',
      theme: ThemeData(primarySwatch: Colors.blue, useMaterial3: true),
      home: const ClassificationPage(),
    );
  }
}

class ClassificationPage extends StatefulWidget {
  const ClassificationPage({super.key});

  @override
  State<ClassificationPage> createState() => _ClassificationPageState();
}

class _ClassificationPageState extends State<ClassificationPage> {
  final TextEditingController _textController = TextEditingController();
  double _probability = -1.0;
  bool _isLoading = false;

  Future<void> _classifyText() async {
    if (_textController.text.isEmpty) return;

    setState(() {
      _isLoading = true;
    });

    try {
      final probability = await classifyTextBinary(_textController.text);
      setState(() {
        _probability = probability;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Binary Classification')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: _textController,
              decoration: const InputDecoration(
                labelText: 'Enter text to classify',
                border: OutlineInputBorder(),
              ),
              maxLines: 3,
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _isLoading ? null : _classifyText,
              child: _isLoading
                  ? const CircularProgressIndicator()
                  : const Text('Classify'),
            ),
            const SizedBox(height: 16),
            if (_probability >= 0)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      Text(
                        'Classification Result:',
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Probability: ${(_probability * 100).toStringAsFixed(2)}%',
                        style: Theme.of(context).textTheme.bodyLarge,
                      ),
                      Text(
                        'Class: ${_probability > 0.5 ? "Positive" : "Negative"}',
                        style: Theme.of(context).textTheme.bodyLarge,
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }
} 