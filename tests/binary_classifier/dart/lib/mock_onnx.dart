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
    // Simulate inference with realistic binary classification results
    final output = _generateRealisticBinaryPrediction(inputs);
    return [OrtValue.createTensorWithDataAsFloat32List([1, 1], Float32List.fromList([output]))];
  }
  
  Future<Map<String, OrtValue>> runAsync(OrtRunOptions options, Map<String, OrtValue> inputs) async {
    // Simulate async inference
    await Future.delayed(Duration(milliseconds: 10));
    
    final output = _generateRealisticBinaryPrediction(inputs.values.toList());
    
    return {
      'output': OrtValue.createTensorWithDataAsFloat32List([1, 1], Float32List.fromList([output]))
    };
  }
  
  double _generateRealisticBinaryPrediction(List<OrtValue> inputs) {
    final random = Random();
    
    // For demo purposes, create a somewhat realistic binary classification
    // In a real scenario, this would be actual model inference
    
    // Create a bias towards certain probability ranges to make it more realistic
    // Most real predictions tend to be more confident (closer to 0 or 1)
    final baseProb = random.nextDouble();
    
    if (baseProb < 0.3) {
      // Negative prediction (0.0 - 0.4)
      return random.nextDouble() * 0.4;
    } else if (baseProb > 0.7) {
      // Positive prediction (0.6 - 1.0)
      return 0.6 + random.nextDouble() * 0.4;
    } else {
      // Uncertain prediction (0.4 - 0.6)
      return 0.4 + random.nextDouble() * 0.2;
    }
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