import os
import time
import numpy as np
import onnxruntime as ort
import psutil
import pytest
from pathlib import Path

class ONNXModelTester:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.process = psutil.Process()
        self._load_model()

    def _load_model(self):
        """Load the ONNX model"""
        try:
            self.session = ort.InferenceSession(self.model_path)
            print(f"✓ Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise

    def get_model_info(self):
        """Get model metadata"""
        info = {
            'input_details': [],
            'output_details': [],
            'model_size_mb': os.path.getsize(self.model_path) / (1024 * 1024)
        }
        
        for input_meta in self.session.get_inputs():
            info['input_details'].append({
                'name': input_meta.name,
                'type': input_meta.type,
                'shape': input_meta.shape
            })
        
        for output_meta in self.session.get_outputs():
            info['output_details'].append({
                'name': output_meta.name,
                'type': output_meta.type,
                'shape': output_meta.shape
            })
        
        return info

    def test_model_loading(self):
        """Test if model loads correctly"""
        assert self.session is not None, "Model session should not be None"
        info = self.get_model_info()
        print("\nModel Information:")
        print(f"Model size: {info['model_size_mb']:.2f} MB")
        print("Input details:", info['input_details'])
        print("Output details:", info['output_details'])
        return True

    def test_inference(self, test_texts):
        """Test model inference on sample texts"""
        results = []
        for text in test_texts:
            start_time = time.time()
            start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            # Prepare input tensor
            input_tensor = self._preprocess_text(text)
            
            # Run inference
            output = self.session.run(None, {'input': input_tensor})
            
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            memory_used = end_memory - start_memory
            
            result = {
                'text': text,
                'prediction': output[0][0],
                'inference_time_ms': inference_time,
                'memory_used_mb': memory_used
            }
            results.append(result)
            
            print(f"\nText: {text}")
            print(f"Prediction: {output[0][0]:.4f}")
            print(f"Inference time: {inference_time:.2f} ms")
            print(f"Memory used: {memory_used:.2f} MB")
        
        return results

    def test_performance(self, num_iterations=100):
        """Test model performance"""
        total_time = 0
        max_memory = 0
        min_memory = float('inf')
        
        for i in range(num_iterations):
            text = f"Sample text for performance testing {i}"
            
            start_time = time.time()
            start_memory = self.process.memory_info().rss / 1024 / 1024
            
            # Run inference
            input_tensor = self._preprocess_text(text)
            self.session.run(None, {'input': input_tensor})
            
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            
            iteration_time = (end_time - start_time) * 1000
            memory_used = end_memory - start_memory
            
            total_time += iteration_time
            max_memory = max(max_memory, memory_used)
            min_memory = min(min_memory, memory_used)
        
        avg_time = total_time / num_iterations
        
        print("\nPerformance Test Results:")
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"Maximum memory usage: {max_memory:.2f} MB")
        print(f"Minimum memory usage: {min_memory:.2f} MB")
        
        return {
            'avg_inference_time_ms': avg_time,
            'max_memory_mb': max_memory,
            'min_memory_mb': min_memory
        }

    def _preprocess_text(self, text):
        """Preprocess text for model input"""
        # TODO: Implement actual text preprocessing based on your model's requirements
        # This is a placeholder that returns a dummy tensor
        return np.zeros((1, 512), dtype=np.float32)

def test_spam_detector():
    """Main test function for spam detector"""
    # Get the model path
    model_path = Path(__file__).parent.parent.parent / "models" / "spam_detector" / "model.onnx"
    assert model_path.exists(), f"Model not found at {model_path}"
    
    # Initialize tester
    tester = ONNXModelTester(str(model_path))
    
    # Test model loading
    assert tester.test_model_loading(), "Model loading test failed"
    
    # Test inference
    test_texts = [
        "Buy now! Limited time offer!",
        "Hello, how are you doing today?",
        "URGENT: Your account needs verification",
        "Meeting at 2 PM in the conference room"
    ]
    inference_results = tester.test_inference(test_texts)
    
    # Verify inference results
    for result in inference_results:
        assert 0 <= result['prediction'] <= 1, "Prediction should be between 0 and 1"
        assert result['inference_time_ms'] < 1000, "Inference time should be less than 1 second"
        assert result['memory_used_mb'] < 500, "Memory usage should be less than 500MB"
    
    # Test performance
    performance_results = tester.test_performance()
    assert performance_results['avg_inference_time_ms'] < 100, "Average inference time should be less than 100ms"
    assert performance_results['max_memory_mb'] < 500, "Maximum memory usage should be less than 500MB"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 