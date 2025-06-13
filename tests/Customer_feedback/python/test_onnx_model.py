import os
import time
import json
import numpy as np
import onnxruntime as ort
import psutil
import pytest
from pathlib import Path

class ONNXModelTester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None
        self.vocab = None
        self.scaler = None
        self._load_vocab_and_scaler()
        
    def _load_vocab_and_scaler(self):
        """Load vocabulary and scaler files"""
        model_dir = self.model_path.parent
        with open(model_dir / 'vocab.json', 'r') as f:
            self.vocab = json.load(f)
        with open(model_dir / 'scaler.json', 'r') as f:
            self.scaler = json.load(f)
            
    def preprocess_text(self, text):
        """Preprocess text using TF-IDF and scaling"""
        # Load vocabulary and IDF weights
        idf = self.vocab['idf']
        word2idx = self.vocab['vocab']
        mean = np.array(self.scaler['mean'], dtype=np.float32)
        scale = np.array(self.scaler['scale'], dtype=np.float32)

        # Compute term frequency (TF)
        tf = np.zeros(len(word2idx), dtype=np.float32)
        words = text.lower().split()
        for word in words:
            idx = word2idx.get(word)
            if idx is not None:
                tf[idx] += 1
        if tf.sum() > 0:
            tf = tf / tf.sum()  # Normalize TF

        # TF-IDF
        tfidf = tf * np.array(idf, dtype=np.float32)

        # Standardize
        tfidf_scaled = (tfidf - mean) / scale
        return tfidf_scaled.astype(np.float32)
        
    def test_model_loading(self):
        """Test if the model can be loaded"""
        try:
            self.session = ort.InferenceSession(str(self.model_path))
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def test_inference(self, test_texts=None):
        """Test model inference on sample texts"""
        if test_texts is None:
            test_texts = [
                "This product is amazing!",
                "Terrible service, would not recommend.",
                "It's okay, nothing special.",
                "Best purchase ever!"
            ]
            
        results = []
        for text in test_texts:
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Preprocess text
            input_vector = self.preprocess_text(text)
            input_data = {self.session.get_inputs()[0].name: input_vector.reshape(1, -1)}
            
            # Run inference
            outputs = self.session.run(None, input_data)
            prediction = float(outputs[0][0][0])  # Probability of positive class
            
            print(f"Input: {text}\nPrediction: {prediction}\n")
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            results.append({
                'text': text,
                'prediction': prediction,
                'inference_time_ms': (end_time - start_time) * 1000,
                'memory_used_mb': end_memory - start_memory
            })
            
        return results
        
    def test_performance(self, num_runs=100):
        """Test model performance"""
        test_text = "This is a sample text for performance testing."
        input_vector = self.preprocess_text(test_text)
        input_data = {self.session.get_inputs()[0].name: input_vector.reshape(1, -1)}
        
        times = []
        memory_usage = []
        
        for _ in range(num_runs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            self.session.run(None, input_data)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            times.append((end_time - start_time) * 1000)
            memory_usage.append(end_memory - start_memory)
            
        return {
            'avg_inference_time_ms': sum(times) / len(times),
            'max_inference_time_ms': max(times),
            'min_inference_time_ms': min(times),
            'avg_memory_mb': sum(memory_usage) / len(memory_usage),
            'max_memory_mb': max(memory_usage)
        }
        
    def save_performance_results(self):
        """Save performance test results to a JSON file"""
        results = self.test_performance()
        with open('performance_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

def test_binary_classifier():
    """Main test function for binary classifier"""
    # Get the model path from the local directory
    model_path = Path(__file__).parent / "model.onnx"
    assert model_path.exists(), f"Model not found at {model_path}"
    
    # Initialize the tester
    tester = ONNXModelTester(model_path)
    
    # Run all tests
    tester.test_model_loading()
    inference_results = tester.test_inference()
    tester.test_performance()
    
    # Print all inference results
    print("\nAll inference results:")
    for result in inference_results:
        print(f"Input: {result['text']} | Prediction: {result['prediction']}")
    
    # Save performance results
    tester.save_performance_results()

def test_custom_text(text):
    """Test model with custom text input"""
    import time
    
    model_path = Path(__file__).parent / "model.onnx"
    assert model_path.exists(), f"Model not found at {model_path}"
    
    print("=" * 80)
    print("🤖 ONNX BINARY CLASSIFIER - DETAILED ANALYSIS")
    print("=" * 80)
    
    # Initialize tester and get model info
    start_time = time.time()
    tester = ONNXModelTester(model_path)
    tester.test_model_loading()
    
    # Get model information
    input_info = tester.session.get_inputs()[0]
    output_info = tester.session.get_outputs()[0]
    
    print(f"📝 INPUT TEXT:")
    print(f"   '{text}'")
    print(f"   Length: {len(text)} characters, {len(text.split())} words")
    print()
    
    print("🔧 MODEL INFORMATION:")
    print(f"   Model Path: {model_path}")
    print(f"   Input Shape: {input_info.shape}")
    print(f"   Input Type: {input_info.type}")
    print(f"   Output Shape: {output_info.shape}")
    print(f"   Output Type: {output_info.type}")
    print(f"   Vocabulary Size: {len(tester.vocab['vocab'])}")
    print()
    
    # Preprocessing analysis
    print("🔍 PREPROCESSING ANALYSIS:")
    words = text.lower().split()
    print(f"   Original words: {words}")
    
    # Check which words are in vocabulary
    vocab_words = []
    unknown_words = []
    for word in words:
        if word in tester.vocab['vocab']:
            vocab_words.append(word)
        else:
            unknown_words.append(word)
    
    print(f"   Words in vocabulary: {vocab_words}")
    print(f"   Unknown words: {unknown_words}")
    print(f"   Vocabulary coverage: {len(vocab_words)}/{len(words)} ({len(vocab_words)/len(words)*100:.1f}%)")
    
    # Preprocess and run inference
    preprocessing_start = time.time()
    input_vector = tester.preprocess_text(text)
    preprocessing_time = time.time() - preprocessing_start
    
    print(f"   TF-IDF vector shape: {input_vector.shape}")
    print(f"   Non-zero features: {np.count_nonzero(input_vector)}")
    print(f"   Vector min/max: {input_vector.min():.4f} / {input_vector.max():.4f}")
    print(f"   Preprocessing time: {preprocessing_time*1000:.2f}ms")
    print()
    
    # Model inference
    print("🚀 MODEL INFERENCE:")
    inference_start = time.time()
    input_data = {tester.session.get_inputs()[0].name: input_vector.reshape(1, -1)}
    outputs = tester.session.run(None, input_data)
    inference_time = time.time() - inference_start
    
    prediction = float(outputs[0][0][0])
    raw_output = outputs[0][0][0]
    
    print(f"   Raw model output: {raw_output}")
    print(f"   Sigmoid probability: {prediction:.6f}")
    print(f"   Inference time: {inference_time*1000:.2f}ms")
    print()
    
    # Results analysis
    print("📊 SENTIMENT ANALYSIS RESULTS:")
    print("   " + "─" * 50)
    
    if prediction > 0.8:
        sentiment_emoji = "😍"
        sentiment_desc = "Very Positive"
        color = "🟢"
    elif prediction > 0.6:
        sentiment_emoji = "😊"
        sentiment_desc = "Positive"
        color = "🟢"
    elif prediction > 0.4:
        sentiment_emoji = "😐"
        sentiment_desc = "Neutral"
        color = "🟡"
    elif prediction > 0.2:
        sentiment_emoji = "😞"
        sentiment_desc = "Negative"
        color = "🔴"
    else:
        sentiment_emoji = "😡"
        sentiment_desc = "Very Negative"
        color = "🔴"
    
    print(f"   {sentiment_emoji} SENTIMENT: {sentiment_desc}")
    print(f"   {color} CONFIDENCE: {prediction:.2%}")
    print(f"   📈 PROBABILITY SCORE: {prediction:.6f}")
    
    # Confidence bar
    bar_length = 40
    filled_length = int(bar_length * prediction)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    print(f"   📊 CONFIDENCE BAR: |{bar}| {prediction:.1%}")
    print()
    
    # Performance summary
    total_time = time.time() - start_time
    print("⚡ PERFORMANCE SUMMARY:")
    print(f"   Total processing time: {total_time*1000:.2f}ms")
    print(f"   Preprocessing: {preprocessing_time*1000:.2f}ms ({preprocessing_time/total_time*100:.1f}%)")
    print(f"   Model inference: {inference_time*1000:.2f}ms ({inference_time/total_time*100:.1f}%)")
    print(f"   Throughput: {1/total_time:.1f} texts/second")
    print()
    
    # Classification thresholds
    print("🎯 CLASSIFICATION THRESHOLDS:")
    print("   Negative: 0.0 ─────── 0.5 ─────── 1.0 :Positive")
    threshold_pos = int(prediction * 40)
    threshold_bar = " " * threshold_pos + "▲" + " " * (40 - threshold_pos)
    print(f"   Current: |{threshold_bar}|")
    print(f"   Your text is {abs(prediction - 0.5)*2:.1%} away from neutral")
    
    print("=" * 80)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # If text is provided as argument, test custom text
        custom_text = " ".join(sys.argv[1:])
        test_custom_text(custom_text)
    else:
        # Otherwise run the standard test suite
        pytest.main([__file__, "-v", "-s"]) 