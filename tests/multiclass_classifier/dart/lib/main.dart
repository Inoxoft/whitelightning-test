import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:convert';
import 'dart:io';
import 'package:onnxruntime/onnxruntime.dart';
import 'dart:typed_data';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Multiclass Classification Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Multiclass Text Classifier'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  String _prediction = '';
  String _inputText = '';
  bool _isLoading = false;

  Future<void> _classifyText() async {
    setState(() {
      _isLoading = true;
      _prediction = '';
    });
    print('Starting classification...');
    final prediction = await runTextClassifierWithResult(_inputText);
    print('Prediction result: ' + prediction);
    setState(() {
      _prediction = prediction;
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              TextField(
                decoration: const InputDecoration(
                  labelText: 'Enter text to classify',
                  border: OutlineInputBorder(),
                ),
                maxLines: 3,
                onChanged: (value) {
                  setState(() {
                    _inputText = value;
                  });
                },
              ),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: _isLoading || _inputText.isEmpty
                    ? null
                    : _classifyText,
                child: _isLoading
                    ? const CircularProgressIndicator()
                    : const Text('Classify'),
              ),
              const SizedBox(height: 24),
              if (_prediction.isNotEmpty)
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
                          _prediction,
                          style: Theme.of(context).textTheme.bodyLarge,
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                ),
              if (_prediction.isEmpty)
                Text(
                  'Prediction will appear here.',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
            ],
          ),
        ),
      ),
    );
  }
}

void printSystemInfo() {
  print('💻 SYSTEM INFORMATION:');
  print('   Platform: ${Platform.operatingSystem}');
  print('   Version: ${Platform.operatingSystemVersion}');
  print('   CPU Cores: ${Platform.numberOfProcessors} logical');
  print('   Total Memory: Not available in Dart Flutter');
  print('   Implementation: Dart with ONNX Runtime');
  print('');
}

void printMulticlassResults(String text, String prediction, List<dynamic> probabilities, Map<String, dynamic> scaler, int totalTimeMs, int preprocessingMs, int inferenceMs) {
  // Parse prediction to get category and score
  final parts = prediction.split(' (Score: ');
  final category = parts[0];
  final scoreStr = parts.length > 1 ? parts[1].replaceAll(')', '') : '0.0000';
  final score = double.tryParse(scoreStr) ?? 0.0;
  final confidence = (score * 100).toStringAsFixed(1);
  
  print('📊 TOPIC CLASSIFICATION RESULTS:');
  print('⏱️  Processing Time: ${totalTimeMs}ms');
  print('   🏆 Predicted Category: $category 📝');
  print('   📈 Confidence: $confidence%');
  print('   📝 Input Text: "$text"');
  print('');
  
  print('📊 DETAILED PROBABILITIES:');
  final categories = ['Business', 'Education', 'Entertainment', 'Environment', 'Health', 'Politics', 'Science', 'Sports', 'Technology', 'World'];
  
  for (int i = 0; i < probabilities.length && i < categories.length; i++) {
    final prob = (probabilities[i] as num).toDouble();
    final percentage = (prob * 100).toStringAsFixed(1);
    final isWinner = i == probabilities.indexOf(probabilities.reduce((a, b) => a > b ? a : b));
    final bar = '█' * ((prob * 20).round().clamp(0, 20));
    final winner = isWinner ? ' ⭐' : '';
    print('   📝 ${categories[i]}: $percentage% $bar$winner');
  }
  print('');
  
  final throughput = (1000 / totalTimeMs).toStringAsFixed(1);
  
  print('📈 PERFORMANCE SUMMARY:');
  print('   Total Processing Time: ${totalTimeMs}ms');
  print('   ┣━ Preprocessing: ${preprocessingMs}ms (${(preprocessingMs / totalTimeMs * 100).toStringAsFixed(1)}%)');
  print('   ┣━ Model Inference: ${inferenceMs}ms (${(inferenceMs / totalTimeMs * 100).toStringAsFixed(1)}%)');
  print('   ┗━ Post-processing: 0ms (0.0%)');
  print('   🧠 CPU Usage: Not available in Dart Flutter');
  print('   💾 Memory: Not available in Dart Flutter');
  print('   🚀 Throughput: $throughput texts/sec');
  
  final rating = totalTimeMs < 50 ? '🚀 EXCELLENT' : 
                 totalTimeMs < 100 ? '✅ GOOD' : 
                 totalTimeMs < 200 ? '⚠️ ACCEPTABLE' : '🐌 SLOW';
  print('   Performance Rating: $rating');
  print('');
}

Future<String> runTextClassifierWithResult(String text) async {
  print('🤖 ONNX MULTICLASS CLASSIFIER - DART IMPLEMENTATION');
  print('=============================================');
  print('🔄 Processing: $text');
  print('');
  
  printSystemInfo();
  
  final startTime = DateTime.now();
  
  print('Initializing ONNX...');
  OrtEnv.instance.init();
  final sessionOptions = OrtSessionOptions();

  print('Loading model...');
  final rawModel = await rootBundle.load('model.onnx');
  final session = OrtSession.fromBuffer(
    rawModel.buffer.asUint8List(),
    sessionOptions,
  );

  print('Loading vocab...');
  final vocabJson = await rootBundle.loadString('vocab.json');
  final vocab = json.decode(vocabJson) as Map<String, dynamic>;

  print('Loading scaler...');
  final scalerJson = await rootBundle.loadString('scaler.json');
  final scaler = json.decode(scalerJson) as Map<String, dynamic>;

  // Preprocessing
  final preprocessStart = DateTime.now();
  print('Tokenizing...');
  final words = text.toLowerCase().split(' ');
  final sequence = List<int>.filled(30, 0);
  for (int i = 0; i < words.length && i < 30; i++) {
    sequence[i] = vocab[words[i]] ?? vocab['<OOV>'] ?? 1;
  }
  final preprocessEnd = DateTime.now();
  final preprocessingMs = preprocessEnd.difference(preprocessStart).inMilliseconds;

  // Model inference
  final inferenceStart = DateTime.now();
  print('Running inference...');
  final inputTensor = OrtValueTensor.createTensorWithDataList(
    Int32List.fromList(sequence),
    [1, 30],
  );
  final result = await session.runAsync(OrtRunOptions(), {
    'input': inputTensor,
  });
  final inferenceEnd = DateTime.now();
  final inferenceMs = inferenceEnd.difference(inferenceStart).inMilliseconds;

  String resultString = 'No prediction';
  List<dynamic> probabilities = [];
  final resultList = result?.toList();
  if (resultList != null && resultList.isNotEmpty && resultList[0] != null) {
    final outputTensor = resultList[0] as OrtValueTensor;
    probabilities = outputTensor.value as List<dynamic>;
    print('Probabilities: ' + probabilities.toString());
    // Flatten if needed
    final List<dynamic> flatProbs =
        (probabilities.isNotEmpty && probabilities.first is List)
        ? probabilities.first as List<dynamic>
        : probabilities;
    if (flatProbs.isNotEmpty) {
      int maxIndex = 0;
      for (int i = 1; i < flatProbs.length; i++) {
        if (flatProbs[i] > flatProbs[maxIndex]) {
          maxIndex = i;
        }
      }
      final label = scaler[maxIndex.toString()];
      final score = flatProbs[maxIndex];
      resultString = '$label (Score: ${score.toStringAsFixed(4)})';
      probabilities = flatProbs;
    }
  }
  
  final endTime = DateTime.now();
  final totalTimeMs = endTime.difference(startTime).inMilliseconds;
  
  printMulticlassResults(text, resultString, probabilities, scaler, totalTimeMs, preprocessingMs, inferenceMs);
  
  inputTensor.release();
  session.release();
  OrtEnv.instance.release();
  print('Returning result: ' + resultString);
  return resultString;
} 