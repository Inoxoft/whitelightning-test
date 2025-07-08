import json
import numpy as np
import onnxruntime as ort
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import time
import psutil
import os
import platform

def load_model_artifacts():
    """Load the ONNX model and preprocessing artifacts"""
    print("🔧 Loading components...")
    
    # Load ONNX model
    model_path = 'model.onnx'
    session = ort.InferenceSession(model_path)
    print("✅ ONNX model loaded")
    
    # Load vectorizer data
    vectorizer_path = 'vocab.json'
    with open(vectorizer_path, 'r') as f:
        vectorizer_data = json.load(f)
    
    # Reconstruct TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=vectorizer_data['max_features'])
    vectorizer.vocabulary_ = vectorizer_data['vocabulary']
    vectorizer.idf_ = np.array(vectorizer_data['idf'])
    print(f"✅ Vectorizer reconstructed (vocab: {len(vectorizer.vocabulary_)} words)")
    
    # Load classes
    classes_path = 'scaler.json'
    with open(classes_path, 'r') as f:
        classes = json.load(f)
    print(f"✅ Classes loaded: {classes}")
    
    return session, vectorizer, classes

def get_system_info():
    """Get system information for performance reporting"""
    return {
        'platform': platform.system(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
        'python_version': platform.python_version()
    }

def preprocess_text(text, vectorizer):
    """Preprocess text using TF-IDF vectorization"""
    start_time = time.time()
    
    # Transform using TF-IDF (exactly like sklearn does)
    X = vectorizer.transform([text]).toarray().astype(np.float32)
    
    preprocessing_time = (time.time() - start_time) * 1000
    
    return X, preprocessing_time

def run_inference(session, X):
    """Run ONNX model inference"""
    start_time = time.time()
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Predict using ONNX model (outputs probabilities with sigmoid)
    predictions = session.run([output_name], {input_name: X})[0]
    
    inference_time = (time.time() - start_time) * 1000
    
    return predictions, inference_time

def main():
    # Get command line argument for custom text
    if len(sys.argv) > 1:
        test_text = sys.argv[1]
    else:
        test_text = "I'm about to give birth, and I'm terrified. What if something goes wrong? What if I can't handle the pain? Received an unexpected compliment at work today. Small moments of happiness can make a big difference."
    
    print("🤖 ONNX MULTICLASS SIGMOID CLASSIFIER - PYTHON IMPLEMENTATION")
    print("=" * 63)
    print(f"🔄 Processing: {test_text}")
    print()
    
    # System information
    system_info = get_system_info()
    print("💻 SYSTEM INFORMATION:")
    print(f"   Platform: {system_info['platform']}")
    print(f"   Processor: {system_info['processor']}")
    print(f"   CPU Cores: {system_info['cpu_count']}")
    print(f"   Total Memory: {system_info['memory_gb']} GB")
    print(f"   Runtime: Python {system_info['python_version']}")
    print()
    
    # Track memory usage
    process = psutil.Process(os.getpid())
    memory_start = process.memory_info().rss / 1024 / 1024  # MB
    
    start_time = time.time()
    
    # Load model and artifacts
    session, vectorizer, classes = load_model_artifacts()
    
    # Preprocess text
    X, preprocessing_time = preprocess_text(test_text, vectorizer)
    
    print(f"📊 TF-IDF shape: {X.shape}")
    print(f"📊 Non-zero features: {np.count_nonzero(X)}")
    print(f"📊 Max TF-IDF value: {np.max(X):.4f}")
    print(f"📊 Min TF-IDF value: {np.min(X):.4f}")
    print()
    
    # Run inference
    predictions, inference_time = run_inference(session, X)
    
    # Post-processing
    postprocessing_start = time.time()
    
    print("📊 EMOTION ANALYSIS RESULTS:")
    emotion_results = []
    for i, (cls, prob) in enumerate(zip(classes, predictions[0])):
        emotion_results.append((cls, prob))
        print(f"   {cls}: {prob:.3f}")
    
    # Find dominant emotion
    dominant_emotion = max(emotion_results, key=lambda x: x[1])
    print(f"   🏆 Dominant Emotion: {dominant_emotion[0]} ({dominant_emotion[1]:.3f})")
    
    postprocessing_time = (time.time() - postprocessing_start) * 1000
    
    print(f"   📝 Input Text: \"{test_text}\"")
    print()
    
    # Performance metrics
    total_time = time.time() - start_time
    total_time_ms = total_time * 1000
    
    memory_end = process.memory_info().rss / 1024 / 1024  # MB
    memory_delta = memory_end - memory_start
    
    print("📈 PERFORMANCE SUMMARY:")
    print(f"   Total Processing Time: {total_time_ms:.2f}ms")
    print(f"   ┣━ Preprocessing: {preprocessing_time:.2f}ms ({preprocessing_time/total_time_ms*100:.1f}%)")
    print(f"   ┣━ Model Inference: {inference_time:.2f}ms ({inference_time/total_time_ms*100:.1f}%)")
    print(f"   ┗━ Postprocessing: {postprocessing_time:.2f}ms ({postprocessing_time/total_time_ms*100:.1f}%)")
    print()
    
    # Throughput
    throughput = 1000 / total_time_ms if total_time_ms > 0 else 0
    print("🚀 THROUGHPUT:")
    print(f"   Texts per second: {throughput:.1f}")
    print()
    
    # Memory usage
    print("💾 RESOURCE USAGE:")
    print(f"   Memory Start: {memory_start:.2f}MB")
    print(f"   Memory End: {memory_end:.2f}MB")
    print(f"   Memory Delta: {memory_delta:+.2f}MB")
    print()
    
    # Performance rating
    if total_time_ms < 50:
        rating = "🚀 EXCELLENT"
    elif total_time_ms < 100:
        rating = "✅ GOOD"
    elif total_time_ms < 500:
        rating = "⚠️ ACCEPTABLE"
    else:
        rating = "🐌 SLOW"
    
    print(f"🎯 PERFORMANCE RATING: {rating}")
    print(f"   ({total_time_ms:.2f}ms total - Target: <100ms)")

if __name__ == "__main__":
    main() 