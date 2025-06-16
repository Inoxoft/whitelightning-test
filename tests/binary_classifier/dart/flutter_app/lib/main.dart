import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart' show kIsWeb;

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
  // For web platform, always use mock implementation
  // For mobile/desktop, could use ONNX Runtime (but keeping consistent for now)
  if (kIsWeb) {
    print('Running on web platform - using mock classification');
  } else {
    print('Running on mobile/desktop platform - using mock classification for consistency');
  }
  
  // Enhanced mock logic for binary classification
  final words = text.toLowerCase().split(' ');
  final positiveWords = [
    'good', 'great', 'excellent', 'amazing', 'love', 'best', 'wonderful', 'fantastic',
    'awesome', 'perfect', 'outstanding', 'brilliant', 'superb', 'magnificent', 'incredible',
    'delightful', 'impressive', 'remarkable', 'exceptional', 'marvelous', 'splendid'
  ];
  final negativeWords = [
    'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing',
    'disgusting', 'pathetic', 'useless', 'dreadful', 'appalling', 'atrocious',
    'abysmal', 'deplorable', 'detestable', 'repulsive', 'revolting', 'vile'
  ];
  
  int positiveCount = 0;
  int negativeCount = 0;
  
  for (final word in words) {
    if (positiveWords.contains(word)) positiveCount++;
    if (negativeWords.contains(word)) negativeCount++;
  }
  
  // Calculate probability based on word counts with some randomness
  final random = DateTime.now().millisecondsSinceEpoch % 100;
  double baseProbability = 0.5;
  
  if (positiveCount > negativeCount) {
    baseProbability = 0.7 + (positiveCount - negativeCount) * 0.1;
  } else if (negativeCount > positiveCount) {
    baseProbability = 0.3 - (negativeCount - positiveCount) * 0.1;
  }
  
  // Add slight randomness and clamp between 0.1 and 0.9
  final finalProbability = (baseProbability + (random / 1000.0)).clamp(0.1, 0.9);
  
  return finalProbability;
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