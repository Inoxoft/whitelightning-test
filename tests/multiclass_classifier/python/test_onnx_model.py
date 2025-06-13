import os
import time
import json
import numpy as np
import onnxruntime as ort
import psutil
import pytest
from pathlib import Path

class ONNXMulticlassModelTester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None
        self.vocab = None
        self.label_map = None
        self._load_vocab_and_labels()
        
    def _load_vocab_and_labels(self):
        """Load vocabulary and label mapping files"""
        model_dir = self.model_path.parent
        with open(model_dir / 'vocab.json', 'r') as f:
            self.vocab = json.load(f)
        with open(model_dir / 'scaler.json', 'r') as f:
            self.label_map = json.load(f)
            
    def preprocess_text(self, text):
        """Preprocess text for multiclass model using tokenization and padding"""
        oov_token = '<OOV>'
        words = text.lower().split()
        sequence = [self.vocab.get(word, self.vocab.get(oov_token, 1)) for word in words]
        sequence = sequence[:30]  # Truncate to max_len
        padded = np.zeros(30, dtype=np.int32)
        padded[:len(sequence)] = sequence  # Pad with zeros
        return padded
        
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
                "The government announced new policies to boost the economy",
                "Scientists discover new treatment for cancer",
                "Local team wins championship game",
                "Stock market shows significant gains today",
                "New movie breaks box office records"
            ]
            
        results = []
        for text in test_texts:
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Preprocess text
            input_vector = self.preprocess_text(text)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            input_data = input_vector.reshape(1, 30)
            outputs = self.session.run([output_name], {input_name: input_data})
            
            # Get prediction
            probabilities = outputs[0][0]
            predicted_idx = np.argmax(probabilities)
            label = self.label_map[str(predicted_idx)]
            score = probabilities[predicted_idx]
            
            print(f"Input: {text}")
            print(f"Prediction: {label} (Score: {score:.4f})")
            print(f"All probabilities: {dict(zip([self.label_map[str(i)] for i in range(len(probabilities))], probabilities))}")
            print()
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            results.append({
                'text': text,
                'predicted_label': label,
                'confidence_score': float(score),
                'all_probabilities': probabilities.tolist(),
                'inference_time_ms': (end_time - start_time) * 1000,
                'memory_used_mb': end_memory - start_memory
            })
            
        return results
        
    def test_performance(self, num_runs=100):
        """Test model performance"""
        test_text = "The government announced new policies to boost the economy"
        input_vector = self.preprocess_text(test_text)
        
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        input_data = input_vector.reshape(1, 30)
        
        times = []
        memory_usage = []
        
        for _ in range(num_runs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            self.session.run([output_name], {input_name: input_data})
            
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

def test_multiclass_classifier():
    """Main test function for multiclass classifier"""
    # Get the model path from the local directory
    model_path = Path(__file__).parent / "model.onnx"
    assert model_path.exists(), f"Model not found at {model_path}"
    
    # Initialize the tester
    tester = ONNXMulticlassModelTester(model_path)
    
    # Run all tests
    assert tester.test_model_loading(), "Model loading failed"
    inference_results = tester.test_inference()
    performance_results = tester.test_performance()
    
    # Print all inference results
    print("\nAll inference results:")
    for result in inference_results:
        print(f"Input: {result['text']} | Prediction: {result['predicted_label']} | Confidence: {result['confidence_score']:.4f}")
    
    # Save performance results
    tester.save_performance_results()
    
    # Performance assertions
    assert performance_results['avg_inference_time_ms'] < 100, f"Average inference time too slow: {performance_results['avg_inference_time_ms']:.2f}ms"
    assert performance_results['max_memory_mb'] < 500, f"Memory usage too high: {performance_results['max_memory_mb']:.2f}MB"

def test_custom_text(text):
    """Test model with custom text input"""
    import time
    
    model_path = Path(__file__).parent / "model.onnx"
    assert model_path.exists(), f"Model not found at {model_path}"
    
    print("=" * 80)
    print("🤖 ONNX MULTICLASS CLASSIFIER - DETAILED ANALYSIS")
    print("=" * 80)
    
    # Initialize tester and get model info
    start_time = time.time()
    tester = ONNXMulticlassModelTester(model_path)
    assert tester.test_model_loading(), "Model loading failed"
    
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
    print(f"   Vocabulary Size: {len(tester.vocab)}")
    print(f"   Number of Classes: {len(tester.label_map)}")
    print(f"   Available Labels: {list(tester.label_map.values())}")
    print()
    
    # Preprocessing analysis
    print("🔍 PREPROCESSING ANALYSIS:")
    words = text.lower().split()
    print(f"   Original words: {words}")
    
    # Check which words are in vocabulary
    vocab_words = []
    unknown_words = []
    for word in words:
        if word in tester.vocab:
            vocab_words.append(word)
        else:
            unknown_words.append(word)
    
    print(f"   Words in vocabulary: {vocab_words}")
    print(f"   Unknown words (OOV): {unknown_words}")
    
    # Show tokenization
    input_vector = tester.preprocess_text(text)
    print(f"   Tokenized sequence: {input_vector}")
    print(f"   Sequence length: {len(input_vector)}")
    print()
    
    # Run inference with detailed timing
    print("⚡ INFERENCE EXECUTION:")
    inference_start = time.time()
    
    input_name = tester.session.get_inputs()[0].name
    output_name = tester.session.get_outputs()[0].name
    input_data = input_vector.reshape(1, 30)
    outputs = tester.session.run([output_name], {input_name: input_data})
    
    inference_end = time.time()
    
    # Get prediction results
    probabilities = outputs[0][0]
    predicted_idx = np.argmax(probabilities)
    predicted_label = tester.label_map[str(predicted_idx)]
    confidence_score = probabilities[predicted_idx]
    
    print(f"   Inference time: {(inference_end - inference_start) * 1000:.2f}ms")
    print(f"   Memory usage: {tester._get_memory_usage():.2f}MB")
    print()
    
    # Results
    print("🎯 CLASSIFICATION RESULTS:")
    print(f"   Predicted Class: {predicted_label}")
    print(f"   Confidence Score: {confidence_score:.4f}")
    print()
    
    print("📊 ALL CLASS PROBABILITIES:")
    for i, prob in enumerate(probabilities):
        label = tester.label_map[str(i)]
        print(f"   {label}: {prob:.4f} {'⭐ PREDICTED' if i == predicted_idx else ''}")
    
    print()
    print("=" * 80)
    print(f"✅ Analysis completed in {(time.time() - start_time) * 1000:.2f}ms")
    print("=" * 80)

if __name__ == "__main__":
    test_multiclass_classifier() 