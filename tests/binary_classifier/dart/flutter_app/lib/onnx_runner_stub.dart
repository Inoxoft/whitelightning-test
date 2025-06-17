import 'dart:typed_data';

/// Stub implementation - should not be used directly
/// Platform-specific implementations will be used instead
abstract class OnnxRunner {
  static Future<double> classifyText(String text, Float32List processedVector) async {
    throw UnsupportedError('Platform-specific implementation not found');
  }
  
  static bool get isAvailable => false;
  
  static String get platformInfo => 'Stub implementation';
} 