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
            
            # Prepare input
            input_data = self._prepare_input(text)
            
            # Run inference
            outputs = self.session.run(None, input_data)
            prediction = float(outputs[0][0][0])  # Assuming binary classification
            
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
        input_data = self._prepare_input(test_text)
        
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
            
    def _prepare_input(self, text):
        """Prepare input for the model"""
        # This is a placeholder - you'll need to implement the actual input preparation
        # based on your model's requirements
        return {'input': np.array([text])}
        
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

def test_spam_detector():
    """Main test function for spam detector"""
    # Get the model path from the local directory
    model_path = Path(__file__).parent / "Customer feedback(B(P,N))" / "model.onnx"
    assert model_path.exists(), f"Model not found at {model_path}"
    
    # Initialize the tester
    tester = ONNXModelTester(model_path)
    
    # Run all tests
    tester.test_model_loading()
    tester.test_inference()
    tester.test_performance()
    
    # Save performance results
    tester.save_performance_results()

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 