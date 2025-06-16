import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:convert';
import 'package:onnxruntime/onnxruntime.dart';
import 'dart:typed_data';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:system_info2/system_info2.dart';
import 'package:fl_chart/fl_chart.dart';
import 'dart:io';
import 'dart:math';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ONNX Binary Classifier',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'ONNX Binary Classifier'),
    );
  }
}

class SystemInfo {
  final String platform;
  final String deviceModel;
  final String cpuInfo;
  final int cpuCores;
  final double totalMemoryGb;
  final String dartVersion;

  SystemInfo({
    required this.platform,
    required this.deviceModel,
    required this.cpuInfo,
    required this.cpuCores,
    required this.totalMemoryGb,
    required this.dartVersion,
  });

  static Future<SystemInfo> create() async {
    final deviceInfo = DeviceInfoPlugin();
    String platform = Platform.operatingSystem;
    String deviceModel = 'Unknown';
    String cpuInfo = 'Unknown CPU';
    int cpuCores = Platform.numberOfProcessors;
    double totalMemoryGb = 0.0;

    try {
      if (Platform.isAndroid) {
        final androidInfo = await deviceInfo.androidInfo;
        deviceModel = '${androidInfo.brand} ${androidInfo.model}';
        platform = 'Android ${androidInfo.version.release}';
      } else if (Platform.isIOS) {
        final iosInfo = await deviceInfo.iosInfo;
        deviceModel = iosInfo.model;
        platform = 'iOS ${iosInfo.systemVersion}';
      } else if (Platform.isLinux || Platform.isMacOS || Platform.isWindows) {
        cpuInfo = SysInfo.processors.isNotEmpty ? SysInfo.processors.first.name : 'Unknown CPU';
        totalMemoryGb = SysInfo.totalPhysicalMemory / (1024 * 1024 * 1024);
        deviceModel = SysInfo.kernelArchitecture;
      }
    } catch (e) {
      // Fallback values if system info fails
    }

    return SystemInfo(
      platform: platform,
      deviceModel: deviceModel,
      cpuInfo: cpuInfo,
      cpuCores: cpuCores,
      totalMemoryGb: totalMemoryGb,
      dartVersion: Platform.version,
    );
  }
}

class PerformanceMetrics {
  final double totalTimeMs;
  final double preprocessingTimeMs;
  final double inferenceTimeMs;
  final double postprocessingTimeMs;
  final double memoryUsageMb;
  final DateTime timestamp;

  PerformanceMetrics({
    required this.totalTimeMs,
    required this.preprocessingTimeMs,
    required this.inferenceTimeMs,
    required this.postprocessingTimeMs,
    required this.memoryUsageMb,
    required this.timestamp,
  });
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> with TickerProviderStateMixin {
  String _prediction = '';
  String _inputText = '';
  bool _isLoading = false;
  SystemInfo? _systemInfo;
  PerformanceMetrics? _lastMetrics;
  final List<PerformanceMetrics> _performanceHistory = [];
  late TabController _tabController;
  
  // Performance tracking
  double _confidence = 0.0;
  String _classification = '';
  
  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    _loadSystemInfo();
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  Future<void> _loadSystemInfo() async {
    final systemInfo = await SystemInfo.create();
    setState(() {
      _systemInfo = systemInfo;
    });
  }

  Future<void> _classifyText() async {
    if (_inputText.isEmpty) return;
    
    setState(() {
      _isLoading = true;
      _prediction = '';
    });
    
    try {
      final result = await runTextClassifierWithResult(_inputText);
      setState(() {
        _prediction = result['prediction'] as String;
        _confidence = result['confidence'] as double;
        _classification = result['classification'] as String;
        _lastMetrics = result['metrics'] as PerformanceMetrics;
        _performanceHistory.add(_lastMetrics!);
        
        // Keep only last 20 measurements for chart
        if (_performanceHistory.length > 20) {
          _performanceHistory.removeAt(0);
        }
      });
    } catch (e) {
      setState(() {
        _prediction = 'Error: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _runBenchmark() async {
    setState(() {
      _isLoading = true;
    });
    
    final testTexts = [
      'This is a positive review of a great product',
      'Terrible service, would not recommend',
      'Amazing quality and fast delivery',
      'Poor customer support experience',
      'Excellent value for money',
    ];
    
    _performanceHistory.clear();
    
    for (int i = 0; i < 10; i++) {
      for (final text in testTexts) {
        final result = await runTextClassifierWithResult(text);
        final metrics = result['metrics'] as PerformanceMetrics;
        setState(() {
          _performanceHistory.add(metrics);
        });
        
        // Small delay to see progress
        await Future.delayed(Duration(milliseconds: 50));
      }
    }
    
    setState(() {
      _isLoading = false;
      _prediction = 'Benchmark completed: ${_performanceHistory.length} predictions';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(icon: Icon(Icons.text_fields), text: 'Classifier'),
            Tab(icon: Icon(Icons.analytics), text: 'Performance'),
            Tab(icon: Icon(Icons.info), text: 'System Info'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _buildClassifierTab(),
          _buildPerformanceTab(),
          _buildSystemInfoTab(),
        ],
      ),
    );
  }

  Widget _buildClassifierTab() {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Text Classification',
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  const SizedBox(height: 16),
                  TextField(
                    decoration: const InputDecoration(
                      labelText: 'Enter text to classify',
                      border: OutlineInputBorder(),
                      hintText: 'Type your text here...',
                    ),
                    maxLines: 3,
                    onChanged: (value) {
                      setState(() {
                        _inputText = value;
                      });
                    },
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _isLoading || _inputText.isEmpty ? null : _classifyText,
                          icon: _isLoading 
                              ? const SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(strokeWidth: 2),
                                )
                              : const Icon(Icons.psychology),
                          label: Text(_isLoading ? 'Classifying...' : 'Classify'),
                        ),
                      ),
                      const SizedBox(width: 8),
                      ElevatedButton.icon(
                        onPressed: _isLoading ? null : _runBenchmark,
                        icon: const Icon(Icons.speed),
                        label: const Text('Benchmark'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          if (_prediction.isNotEmpty) ...[
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Results',
                      style: Theme.of(context).textTheme.headlineSmall,
                    ),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        Icon(
                          _classification == 'Positive' ? Icons.thumb_up : Icons.thumb_down,
                          color: _classification == 'Positive' ? Colors.green : Colors.red,
                          size: 32,
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Classification: $_classification',
                                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              Text(
                                'Confidence: ${(_confidence * 100).toStringAsFixed(1)}%',
                                style: Theme.of(context).textTheme.bodyMedium,
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    LinearProgressIndicator(
                      value: _confidence,
                      backgroundColor: Colors.grey[300],
                      valueColor: AlwaysStoppedAnimation<Color>(
                        _classification == 'Positive' ? Colors.green : Colors.red,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
          if (_lastMetrics != null) ...[
            const SizedBox(height: 16),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Performance Metrics',
                      style: Theme.of(context).textTheme.headlineSmall,
                    ),
                    const SizedBox(height: 16),
                    _buildMetricRow('Total Time', '${_lastMetrics!.totalTimeMs.toStringAsFixed(2)} ms'),
                    _buildMetricRow('Preprocessing', '${_lastMetrics!.preprocessingTimeMs.toStringAsFixed(2)} ms'),
                    _buildMetricRow('Inference', '${_lastMetrics!.inferenceTimeMs.toStringAsFixed(2)} ms'),
                    _buildMetricRow('Postprocessing', '${_lastMetrics!.postprocessingTimeMs.toStringAsFixed(2)} ms'),
                    _buildMetricRow('Memory Usage', '${_lastMetrics!.memoryUsageMb.toStringAsFixed(2)} MB'),
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildMetricRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label),
          Text(
            value,
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget _buildPerformanceTab() {
    if (_performanceHistory.isEmpty) {
      return const Center(
        child: Text('No performance data available. Run some classifications first.'),
      );
    }

    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Performance Chart',
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  const SizedBox(height: 16),
                  SizedBox(
                    height: 200,
                    child: LineChart(
                      LineChartData(
                        gridData: FlGridData(show: true),
                        titlesData: FlTitlesData(
                          leftTitles: AxisTitles(
                            sideTitles: SideTitles(
                              showTitles: true,
                              reservedSize: 40,
                              getTitlesWidget: (value, meta) {
                                return Text('${value.toInt()}ms');
                              },
                            ),
                          ),
                          bottomTitles: AxisTitles(
                            sideTitles: SideTitles(
                              showTitles: true,
                              getTitlesWidget: (value, meta) {
                                return Text('${value.toInt()}');
                              },
                            ),
                          ),
                          rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                          topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                        ),
                        borderData: FlBorderData(show: true),
                        lineBarsData: [
                          LineChartBarData(
                            spots: _performanceHistory.asMap().entries.map((entry) {
                              return FlSpot(entry.key.toDouble(), entry.value.totalTimeMs);
                            }).toList(),
                            isCurved: true,
                            color: Colors.blue,
                            barWidth: 2,
                            dotData: FlDotData(show: false),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Statistics',
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  const SizedBox(height: 16),
                  _buildStatRow('Total Predictions', _performanceHistory.length.toString()),
                  _buildStatRow(
                    'Average Time',
                    '${(_performanceHistory.map((m) => m.totalTimeMs).reduce((a, b) => a + b) / _performanceHistory.length).toStringAsFixed(2)} ms',
                  ),
                  _buildStatRow(
                    'Min Time',
                    '${_performanceHistory.map((m) => m.totalTimeMs).reduce(min).toStringAsFixed(2)} ms',
                  ),
                  _buildStatRow(
                    'Max Time',
                    '${_performanceHistory.map((m) => m.totalTimeMs).reduce(max).toStringAsFixed(2)} ms',
                  ),
                  _buildStatRow(
                    'Throughput',
                    '${(1000.0 / (_performanceHistory.map((m) => m.totalTimeMs).reduce((a, b) => a + b) / _performanceHistory.length)).toStringAsFixed(2)} pred/sec',
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label),
          Text(
            value,
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget _buildSystemInfoTab() {
    if (_systemInfo == null) {
      return const Center(child: CircularProgressIndicator());
    }

    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'System Information',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 16),
              _buildInfoRow('Platform', _systemInfo!.platform),
              _buildInfoRow('Device', _systemInfo!.deviceModel),
              _buildInfoRow('CPU', _systemInfo!.cpuInfo),
              _buildInfoRow('CPU Cores', _systemInfo!.cpuCores.toString()),
              if (_systemInfo!.totalMemoryGb > 0)
                _buildInfoRow('Total Memory', '${_systemInfo!.totalMemoryGb.toStringAsFixed(2)} GB'),
              _buildInfoRow('Dart Version', _systemInfo!.dartVersion),
              _buildInfoRow('ONNX Runtime', '1.16.0'),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 120,
            child: Text(
              label,
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
          ),
          Expanded(
            child: Text(value),
          ),
        ],
      ),
    );
  }
}

Future<Map<String, dynamic>> runTextClassifierWithResult(String text) async {
  final totalStart = DateTime.now();
  
  // Initialize ONNX
  final preprocessStart = DateTime.now();
  OrtEnv.instance.init();
  final sessionOptions = OrtSessionOptions();

  // Load model
  final rawModel = await rootBundle.load('assets/models/model.onnx');
  final session = OrtSession.fromBuffer(
    rawModel.buffer.asUint8List(),
    sessionOptions,
  );

  // Load vocab
  final vocabJson = await rootBundle.loadString('assets/models/vocab.json');
  final vocabData = json.decode(vocabJson) as Map<String, dynamic>;
  final vocab = vocabData['vocab'] as Map<String, dynamic>;
  final idf = (vocabData['idf'] as List).map((e) => (e as num).toDouble()).toList();

  // Load scaler
  final scalerJson = await rootBundle.loadString('assets/models/scaler.json');
  final scalerData = json.decode(scalerJson) as Map<String, dynamic>;
  final mean = (scalerData['mean'] as List).map((e) => (e as num).toDouble()).toList();
  final scale = (scalerData['scale'] as List).map((e) => (e as num).toDouble()).toList();

  // Preprocess text
  final vector = List<double>.filled(5000, 0.0);
  final wordCounts = <String, int>{};
  final textLower = text.toLowerCase();
  
  for (final word in textLower.split(' ')) {
    wordCounts[word] = (wordCounts[word] ?? 0) + 1;
  }

  for (final entry in wordCounts.entries) {
    final word = entry.key;
    final count = entry.value;
    final idx = vocab[word];
    if (idx != null && idx is int && idx < 5000) {
      vector[idx] = count * idf[idx];
    }
  }

  for (int i = 0; i < 5000; i++) {
    vector[i] = (vector[i] - mean[i]) / scale[i];
  }
  
  final preprocessingTime = DateTime.now().difference(preprocessStart).inMicroseconds / 1000.0;

  // Run inference
  final inferenceStart = DateTime.now();
  final inputTensor = OrtValueTensor.createTensorWithDataList(
    Float32List.fromList(vector),
    [1, 5000],
  );
  
  final result = await session.runAsync(OrtRunOptions(), {
    'input': inputTensor,
  });
  final inferenceTime = DateTime.now().difference(inferenceStart).inMicroseconds / 1000.0;

  // Process results
  final postprocessStart = DateTime.now();
  double probability = 0.0;
  final resultList = result?.toList();
  if (resultList != null && resultList.isNotEmpty && resultList[0] != null) {
    final outputTensor = resultList[0] as OrtValueTensor;
    final probabilities = outputTensor.value as List<dynamic>;
    if (probabilities.isNotEmpty) {
      probability = (probabilities[0] as num).toDouble();
    }
  }

  final classification = probability > 0.5 ? 'Positive' : 'Negative';
  final confidence = probability > 0.5 ? probability : (1.0 - probability);
  
  final postprocessingTime = DateTime.now().difference(postprocessStart).inMicroseconds / 1000.0;
  final totalTime = DateTime.now().difference(totalStart).inMicroseconds / 1000.0;

  // Clean up
  inputTensor.release();
  session.release();
  OrtEnv.instance.release();

  // Create performance metrics
  final metrics = PerformanceMetrics(
    totalTimeMs: totalTime,
    preprocessingTimeMs: preprocessingTime,
    inferenceTimeMs: inferenceTime,
    postprocessingTimeMs: postprocessingTime,
    memoryUsageMb: 0.0, // Would need platform-specific implementation
    timestamp: DateTime.now(),
  );

  return {
    'prediction': '$classification (${(confidence * 100).toStringAsFixed(1)}%)',
    'classification': classification,
    'confidence': confidence,
    'probability': probability,
    'metrics': metrics,
  };
} 