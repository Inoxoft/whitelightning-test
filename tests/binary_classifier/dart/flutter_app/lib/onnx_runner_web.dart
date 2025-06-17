import 'dart:typed_data';

/// Web implementation using fallback keyword-based classification
/// ONNX Runtime is not available on web due to dart:ffi limitations
class OnnxRunner {
  static Future<double> classifyText(String text, Float32List processedVector) async {
    print('🌐 Running fallback classification on web platform');
    
    // Enhanced keyword-based classification for web
    final words = text.toLowerCase().split(' ');
    final positiveWords = [
      'good', 'great', 'excellent', 'amazing', 'love', 'best', 'wonderful', 'fantastic',
      'awesome', 'perfect', 'outstanding', 'brilliant', 'superb', 'magnificent', 'incredible',
      'delightful', 'impressive', 'remarkable', 'exceptional', 'marvelous', 'splendid',
      'nice', 'fine', 'cool', 'sweet', 'solid', 'lovely', 'beautiful', 'terrific'
    ];
    final negativeWords = [
      'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing',
      'disgusting', 'pathetic', 'useless', 'dreadful', 'appalling', 'atrocious',
      'abysmal', 'deplorable', 'detestable', 'repulsive', 'revolting', 'vile',
      'poor', 'weak', 'fail', 'broken', 'wrong', 'ugly', 'stupid', 'trash'
    ];
    
    int positiveCount = 0;
    int negativeCount = 0;
    
    for (final word in words) {
      if (positiveWords.contains(word)) positiveCount++;
      if (negativeWords.contains(word)) negativeCount++;
    }
    
    // Calculate probability based on word counts and text features
    double baseProbability = 0.5;
    
    // Factor in positive/negative word counts
    if (positiveCount > negativeCount) {
      baseProbability = 0.65 + (positiveCount - negativeCount) * 0.08;
    } else if (negativeCount > positiveCount) {
      baseProbability = 0.35 - (negativeCount - positiveCount) * 0.08;
    }
    
    // Factor in text length (longer text might be more thoughtful)
    final wordCount = words.length;
    if (wordCount > 10) {
      baseProbability += 0.05;
    } else if (wordCount < 3) {
      baseProbability -= 0.05;
    }
    
    // Factor in exclamation marks (usually positive)
    final exclamationCount = text.split('!').length - 1;
    baseProbability += exclamationCount * 0.02;
    
    // Factor in question marks (might indicate uncertainty)
    final questionCount = text.split('?').length - 1;
    baseProbability -= questionCount * 0.01;
    
    // Add slight variation for more realistic results
    final variation = (DateTime.now().millisecondsSinceEpoch % 100) / 1000.0 - 0.05;
    final finalProbability = (baseProbability + variation).clamp(0.1, 0.9);
    
    print('📊 Web fallback classification result: ${(finalProbability * 100).toStringAsFixed(2)}%');
    print('📈 Positive words: $positiveCount, Negative words: $negativeCount');
    print('📝 Word count: $wordCount, Exclamations: $exclamationCount');
    
    return finalProbability;
  }
  
  static bool get isAvailable => false; // ONNX not available on web
  
  static String get platformInfo => 'Web fallback (keyword-based classification)';
} 