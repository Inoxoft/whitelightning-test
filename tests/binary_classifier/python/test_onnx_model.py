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
    # Check for custom model files first, then fall back to default
    custom_model_path = Path(__file__).parent / "model_files" / "model.onnx"
    default_model_path = Path(__file__).parent / "model.onnx"
    
    model_path = custom_model_path if custom_model_path.exists() else default_model_path
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
    # Check for custom model files first, then fall back to default
    custom_model_path = Path(__file__).parent / "model_files" / "model.onnx"
    default_model_path = Path(__file__).parent / "model.onnx"
    
    model_path = custom_model_path if custom_model_path.exists() else default_model_path
    assert model_path.exists(), f"Model not found at {model_path}"
    
    tester = ONNXModelTester(model_path)
    tester.test_model_loading()
    
    # Preprocess and run inference
    input_vector = tester.preprocess_text(text)
    input_data = {tester.session.get_inputs()[0].name: input_vector.reshape(1, -1)}
    outputs = tester.session.run(None, input_data)
    prediction = float(outputs[0][0][0])
    
    print(f"\nCustom Text Analysis:")
    print(f"Input: {text}")
    print(f"Prediction: {prediction}")
    print(f"Sentiment: {'Positive' if prediction > 0.5 else 'Negative'} (confidence: {prediction:.2%})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # If text is provided as argument, test custom text
        custom_text = " ".join(sys.argv[1:])
        test_custom_text(custom_text)
    else:
        # Otherwise run the standard test suite
        pytest.main([__file__, "-v", "-s"]) 