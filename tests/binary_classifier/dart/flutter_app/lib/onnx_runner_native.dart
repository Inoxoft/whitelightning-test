import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart' as ort;

/// Native implementation using real ONNX Runtime for mobile/desktop platforms
class OnnxRunner {
  static Future<double> classifyText(String text, Float32List processedVector) async {
    print('🧠 Running ONNX inference on native platform (mobile/desktop)');
    
    final totalStart = DateTime.now();
    
    try {
      // Initialize ONNX Runtime with timing
      final initStart = DateTime.now();
      ort.OrtEnv.instance.init();
      final initEnd = DateTime.now();
      print('⚡ ONNX Runtime initialization: ${initEnd.difference(initStart).inMicroseconds}μs');
      
      // Load the ONNX model with timing
      final loadStart = DateTime.now();
      final modelBytes = await rootBundle.load('assets/models/model.onnx');
      final sessionOptions = ort.OrtSessionOptions();
      final session = ort.OrtSession.fromBuffer(modelBytes.buffer.asUint8List(), sessionOptions);
      final loadEnd = DateTime.now();
      print('📦 Model loading time: ${loadEnd.difference(loadStart).inMilliseconds}ms');
      print('📊 Model size: ${(modelBytes.lengthInBytes / (1024 * 1024)).toStringAsFixed(2)} MB');
      
      // Get input name from model
      final inputNames = session.inputNames;
      if (inputNames.isEmpty) {
        throw Exception('No input names found in the model');
      }
      final inputName = inputNames[0];
      
      print('📊 Model input name: $inputName');
      print('📈 Input vector size: ${processedVector.length}');
      print('🔢 Input shape: [1, ${processedVector.length}] (batch_size, features)');
      
      // Analyze input vector statistics
      final minVal = processedVector.reduce((a, b) => a < b ? a : b);
      final maxVal = processedVector.reduce((a, b) => a > b ? a : b);
      final avgVal = processedVector.reduce((a, b) => a + b) / processedVector.length;
      print('📈 Input stats: min=${minVal.toStringAsFixed(4)}, max=${maxVal.toStringAsFixed(4)}, avg=${avgVal.toStringAsFixed(4)}');
      
      // Create input tensor with timing
      final tensorStart = DateTime.now();
      final inputTensor = ort.OrtValueTensor.createTensorWithDataList(
        processedVector, // data
        [1, processedVector.length], // shape: [batch_size, features]
      );
      final tensorEnd = DateTime.now();
      print('🔧 Tensor creation time: ${tensorEnd.difference(tensorStart).inMicroseconds}μs');
      
      // Run inference with detailed timing
      final inferenceStart = DateTime.now();
      final result = await session.runAsync(ort.OrtRunOptions(), {
        inputName: inputTensor,
      });
      final inferenceEnd = DateTime.now();
      final inferenceTime = inferenceEnd.difference(inferenceStart);
      
      print('🚀 Pure inference time: ${inferenceTime.inMicroseconds}μs (${inferenceTime.inMilliseconds}ms)');
      
      // Extract prediction with timing
      final extractStart = DateTime.now();
      double probability = 0.5; // default fallback
      
      if (result != null && result.isNotEmpty) {
        final outputTensor = result.values.first as ort.OrtValueTensor;
        final List<dynamic> probabilities = outputTensor.value as List<dynamic>;
        
        print('📊 Raw model output format: ${probabilities.runtimeType}');
        print('📈 Output shape: ${probabilities.length} elements');
        
        // Handle different output formats
        final List<dynamic> flatProbs = (probabilities.isNotEmpty && probabilities.first is List)
            ? probabilities.first as List<dynamic>
            : probabilities;
            
        if (flatProbs.isNotEmpty) {
          probability = (flatProbs[0] as num).toDouble();
          print('🎯 Raw probability output: $probability');
          
          // Show all outputs if multiple
          if (flatProbs.length > 1) {
            print('📊 All outputs: ${flatProbs.take(5).map((p) => (p as num).toStringAsFixed(4)).join(", ")}');
          }
        }
      }
      
      final extractEnd = DateTime.now();
      print('⚙️  Output extraction time: ${extractEnd.difference(extractStart).inMicroseconds}μs');
      
      // Cleanup with timing
      final cleanupStart = DateTime.now();
      inputTensor.release();
      session.release();
      ort.OrtEnv.instance.release();
      final cleanupEnd = DateTime.now();
      print('🧹 Cleanup time: ${cleanupEnd.difference(cleanupStart).inMicroseconds}μs');
      
      // Calculate total time and performance metrics
      final totalEnd = DateTime.now();
      final totalTime = totalEnd.difference(totalStart);
      final clampedProbability = probability.clamp(0.0, 1.0);
      
      print('⏱️  Total processing time: ${totalTime.inMicroseconds}μs (${totalTime.inMilliseconds}ms)');
      print('🎯 Final clamped probability: ${clampedProbability}');
      print('✅ ONNX inference completed successfully!');
      
      // Performance analysis
      if (totalTime.inMilliseconds < 10) {
        print('🏆 Performance Analysis: EXCELLENT - Mobile ready!');
      } else if (totalTime.inMilliseconds < 50) {
        print('🥇 Performance Analysis: VERY GOOD - Production ready!');
      } else if (totalTime.inMilliseconds < 100) {
        print('🥈 Performance Analysis: GOOD - Acceptable for most use cases');
      } else {
        print('🥉 Performance Analysis: SLOW - Consider optimization');
      }
      
      // Memory usage estimation
      final modelMemoryMB = modelBytes.lengthInBytes / (1024 * 1024);
      final inputMemoryKB = (processedVector.length * 4) / 1024; // 4 bytes per float32
      print('💾 Memory usage: Model=${modelMemoryMB.toStringAsFixed(1)}MB, Input=${inputMemoryKB.toStringAsFixed(1)}KB');
      
      return clampedProbability;
      
    } catch (e) {
      final totalEnd = DateTime.now();
      final totalTime = totalEnd.difference(totalStart);
      print('❌ ONNX Runtime error after ${totalTime.inMilliseconds}ms: $e');
      print('🔍 Error type: ${e.runtimeType}');
      print('📱 Platform: Native (mobile/desktop)');
      print('💡 Suggestion: Check model file compatibility and ONNX Runtime version');
      rethrow;
    }
  }
  
  static bool get isAvailable => true;
  
  static String get platformInfo => 'Native ONNX Runtime (mobile/desktop)';
} 