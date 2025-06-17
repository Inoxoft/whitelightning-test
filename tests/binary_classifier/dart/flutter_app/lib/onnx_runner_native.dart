import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart' as ort;

/// Native implementation using real ONNX Runtime for mobile/desktop platforms
class OnnxRunner {
  static Future<double> classifyText(String text, Float32List processedVector) async {
    print('🧠 Running ONNX inference on native platform (mobile/desktop)');
    
    try {
      // Initialize ONNX Runtime
      ort.OrtEnv.instance.init();
      
      // Load the ONNX model
      final modelBytes = await rootBundle.load('assets/models/model.onnx');
      final sessionOptions = ort.OrtSessionOptions();
      final session = ort.OrtSession.fromBuffer(modelBytes.buffer.asUint8List(), sessionOptions);
      
      // Get input name from model
      final inputNames = session.inputNames;
      if (inputNames.isEmpty) {
        throw Exception('No input names found in the model');
      }
      final inputName = inputNames[0];
      
      print('📊 Model input name: $inputName');
      print('📈 Input vector size: ${processedVector.length}');
      
      // Create input tensor
      final inputTensor = ort.OrtValueTensor.createTensorWithDataList(
        processedVector, // data
        [1, processedVector.length], // shape: [batch_size, features]
      );
      
      // Run inference
      final result = await session.runAsync(ort.OrtRunOptions(), {
        inputName: inputTensor,
      });
      
      // Extract prediction (probability)
      double probability = 0.5; // default fallback
      
      if (result != null && result.isNotEmpty) {
        final outputTensor = result.values.first as ort.OrtValueTensor;
        final List<dynamic> probabilities = outputTensor.value as List<dynamic>;
        
        // Handle different output formats
        final List<dynamic> flatProbs = (probabilities.isNotEmpty && probabilities.first is List)
            ? probabilities.first as List<dynamic>
            : probabilities;
            
        if (flatProbs.isNotEmpty) {
          probability = (flatProbs[0] as num).toDouble();
        }
      }
      
      // Cleanup
      inputTensor.release();
      session.release();
      ort.OrtEnv.instance.release();
      
      final clampedProbability = probability.clamp(0.0, 1.0);
      print('✅ ONNX inference successful: ${(clampedProbability * 100).toStringAsFixed(2)}%');
      print('🎯 Raw model output: $probability');
      
      return clampedProbability;
      
    } catch (e) {
      print('❌ ONNX Runtime error: $e');
      rethrow;
    }
  }
  
  static bool get isAvailable => true;
  
  static String get platformInfo => 'Native ONNX Runtime (mobile/desktop)';
} 