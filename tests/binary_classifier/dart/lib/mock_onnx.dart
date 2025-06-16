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
    // Simulate inference with random results
    final random = Random();
    final output = Float32List(2); // Binary classification output
    
    // Simulate realistic binary classification probabilities
    final prob1 = random.nextDouble();
    output[0] = prob1.toDouble();
    output[1] = (1.0 - prob1).toDouble();
    
    return [OrtValue.createTensorWithDataAsFloat32List([1, 2], output)];
  }
  
  Future<Map<String, OrtValue>> runAsync(OrtRunOptions options, Map<String, OrtValue> inputs) async {
    // Simulate async inference
    await Future.delayed(Duration(milliseconds: 10));
    
    final random = Random();
    final output = Float32List(2); // Binary classification output
    
    // Simulate realistic binary classification probabilities
    final prob1 = random.nextDouble();
    output[0] = prob1.toDouble();
    output[1] = (1.0 - prob1).toDouble();
    
    return {
      'output': OrtValue.createTensorWithDataAsFloat32List([1, 2], output)
    };
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