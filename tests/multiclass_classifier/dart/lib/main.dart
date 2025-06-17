import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:convert';
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

Future<String> runTextClassifierWithResult(String text) async {
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

  print('Tokenizing...');
  final words = text.toLowerCase().split(' ');
  final sequence = List<int>.filled(30, 0);
  for (int i = 0; i < words.length && i < 30; i++) {
    sequence[i] = vocab[words[i]] ?? vocab['<OOV>'] ?? 1;
  }

  print('Running inference...');
  final inputTensor = OrtValueTensor.createTensorWithDataList(
    Int32List.fromList(sequence),
    [1, 30],
  );
  final result = await session.runAsync(OrtRunOptions(), {
    'input': inputTensor,
  });

  String resultString = 'No prediction';
  final resultList = result?.toList();
  if (resultList != null && resultList.isNotEmpty && resultList[0] != null) {
    final outputTensor = resultList[0] as OrtValueTensor;
    final List<dynamic> probabilities = outputTensor.value as List<dynamic>;
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
    }
  }
  inputTensor.release();
  session.release();
  OrtEnv.instance.release();
  print('Returning result: ' + resultString);
  return resultString;
} 