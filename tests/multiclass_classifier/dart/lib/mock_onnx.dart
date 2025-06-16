// Mock ONNX Runtime implementation for CI testing
// This simulates the ONNX Runtime functionality when the actual package is not available

import 'dart:typed_data';
import 'dart:math';

class OrtSession {
  static Future<OrtSession> fromFile(String modelPath) async {
    print('🔧 Mock ONNX Session: Loading model from $modelPath');
    // Simulate loading delay
    await Future.delayed(Duration(milliseconds: 100));
    return OrtSession._();
  }
  
  static OrtSession fromBuffer(Uint8List buffer, OrtSessionOptions options) {
    print('🔧 Mock ONNX Session: Loading model from buffer');
    return OrtSession._();
  }
  
  OrtSession._();
  
  List<OrtValue> run(List<OrtValue> inputs) {
    // Simulate inference with realistic multiclass results
    final output = _generateRealisticPrediction(inputs);
    return [OrtValue.createTensorWithDataAsFloat32List([1, 10], output)];
  }
  
  Future<Map<String, OrtValue>> runAsync(OrtRunOptions options, Map<String, OrtValue> inputs) async {
    // Simulate async inference
    await Future.delayed(Duration(milliseconds: 10));
    
    final output = _generateRealisticPrediction(inputs.values.toList());
    
    return {
      'output': OrtValue.createTensorWithDataAsFloat32List([1, 10], output)
    };
  }
  
  Float32List _generateRealisticPrediction(List<OrtValue> inputs) {
    final random = Random();
    final output = Float32List(10); // 10 classes for multiclass classifier
    
    // Classes: Business, Education, Entertainment, Environment, Health, Politics, Science, Sports, Technology, World
    // Try to make somewhat realistic predictions based on simple heuristics
    
    // For demo purposes, create a biased but more realistic distribution
    // In a real scenario, this would be actual model inference
    
    // Create a somewhat realistic distribution (not completely random)
    final baseProbs = [0.1, 0.05, 0.15, 0.1, 0.05, 0.2, 0.1, 0.1, 0.1, 0.05]; // Slightly favor Politics and Entertainment
    
    for (int i = 0; i < 10; i++) {
      // Add some randomness to base probabilities
      output[i] = baseProbs[i] + (random.nextDouble() - 0.5) * 0.3;
      if (output[i] < 0) output[i] = 0.01; // Ensure positive
    }
    
    // Normalize to sum to 1.0
    double sum = 0.0;
    for (int i = 0; i < 10; i++) {
      sum += output[i];
    }
    for (int i = 0; i < 10; i++) {
      output[i] = output[i] / sum;
    }
    
    return output;
  }
  
  void release() {
    print('🔧 Mock ONNX Session: Released');
  }
}

class OrtValue {
  final List<int> _shape;
  final Float32List _data;
  
  OrtValue._(this._shape, this._data);
  
  static OrtValue createTensorWithDataAsFloat32List(List<int> shape, Float32List data) {
    return OrtValue._(shape, data);
  }
  
  Float32List get value => _data;
  List<int> get shape => _shape;
  
  void release() {
    // Mock release
  }
}

class OrtEnv {
  static OrtEnv? _instance;
  
  static OrtEnv instance() {
    _instance ??= OrtEnv._();
    return _instance!;
  }
  
  OrtEnv._();
  
  void init() {
    print('🔧 Mock ONNX Environment: Initialized');
  }
  
  void release() {
    print('🔧 Mock ONNX Environment: Released');
  }
}

class OrtSessionOptions {
  OrtSessionOptions();
}

class OrtRunOptions {
  OrtRunOptions();
}

class OrtValueTensor extends OrtValue {
  OrtValueTensor._(List<int> shape, Float32List data) : super._(shape, data);
  
  static OrtValueTensor createTensorWithDataList(List<int> shape, List<double> data) {
    final float32Data = Float32List.fromList(data);
    return OrtValueTensor._(shape, float32Data);
  }
} 