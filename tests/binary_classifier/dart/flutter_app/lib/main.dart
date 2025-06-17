import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart' show kIsWeb;

// Conditional imports for platform-specific ONNX Runtime support
import 'onnx_runner_stub.dart' if (dart.library.io) 'onnx_runner_native.dart' if (dart.library.html) 'onnx_runner_web.dart';

Future<Float32List> preprocessText(String text) async {
  final vocabJson = await rootBundle.loadString('assets/models/vocab.json');
  final vocab = json.decode(vocabJson);
  final idf = (vocab['idf'] as List).map((e) => (e as num).toDouble()).toList();
  final word2idx = Map<String, int>.from(vocab['vocab']);

  final scalerJson = await rootBundle.loadString('assets/models/scaler.json');
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

Future<double> classifyTextBinary(String text) async {
  try {
    print('🚀 Starting classification for: "$text"');
    print('🔧 Platform: ${OnnxRunner.platformInfo}');
    print('⚡ ONNX Available: ${OnnxRunner.isAvailable}');
    
    // Preprocess the text
    final processedVector = await preprocessText(text);
    print('📊 Preprocessed vector size: ${processedVector.length}');
    
    // Use platform-specific ONNX runner
    final probability = await OnnxRunner.classifyText(text, processedVector);
    
    print('🎯 Final classification result: ${(probability * 100).toStringAsFixed(2)}%');
    return probability;
    
  } catch (e) {
    print('❌ Classification error: $e');
    print('🔄 Using emergency fallback...');
    
    // Emergency fallback - simple keyword counting
    final words = text.toLowerCase().split(' ');
    final positiveWords = ['good', 'great', 'excellent', 'amazing', 'love', 'best'];
    final negativeWords = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible'];
    
    int positiveCount = 0;
    int negativeCount = 0;
    
    for (final word in words) {
      if (positiveWords.contains(word)) positiveCount++;
      if (negativeWords.contains(word)) negativeCount++;
    }
    
    double probability = 0.5;
    if (positiveCount > negativeCount) {
      probability = 0.7;
    } else if (negativeCount > positiveCount) {
      probability = 0.3;
    }
    
    print('🆘 Emergency fallback result: ${(probability * 100).toStringAsFixed(2)}%');
    return probability;
  }
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