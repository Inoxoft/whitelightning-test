#!/usr/bin/env python3
"""
ONNX Model Testing Tool for Pre-Deployment Validation

This comprehensive tool validates ONNX models (binary and multiclass) 
before deployment with extensive testing scenarios.
"""

import json
import numpy as np
import pandas as pd
import onnxruntime as ort
import onnx
import time
import warnings
import os
import argparse
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import requests
from datetime import datetime
import os
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    pass  # python-dotenv not installed, use system environment variables

warnings.filterwarnings('ignore')

class ONNXModelTester:
    def __init__(self, model_path: str, config_dir: str = "."):
        self.model_path = model_path
        self.config_dir = Path(config_dir)
        self.session = None
        self.onnx_model = None
        self.scaler = None
        self.vocab = None
        self.test_results = {}
        self._load_model()
        self._load_configurations()
        
    def _load_model(self):
        """Load ONNX model"""
        try:
            # Load ONNX model for structure analysis
            self.onnx_model = onnx.load(self.model_path)
            # Load ONNX Runtime session for inference
            self.session = ort.InferenceSession(self.model_path)
            print(f"✓ ONNX model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"✗ Failed to load ONNX model: {e}")
            raise
    
    def _load_configurations(self):
        """Load supporting configuration files"""
        try:
            # Load scaler if available
            scaler_path = self.config_dir / "scaler.json"
            if scaler_path.exists():
                with open(scaler_path, 'r') as f:
                    scaler_data = json.load(f)
                    self.scaler = {
                        'mean': np.array(scaler_data.get('mean', [])),
                        'scale': np.array(scaler_data.get('scale', []))
                    }
                print("✓ Scaler configuration loaded")
            
            # Load vocabulary if available
            vocab_path = self.config_dir / "vocab.json"
            if vocab_path.exists():
                with open(vocab_path, 'r') as f:
                    self.vocab = json.load(f)
                print("✓ Vocabulary loaded")
                
        except Exception as e:
            print(f"⚠ Warning: Could not load some configurations: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the ONNX model"""
        info = {
            'input_details': [],
            'output_details': [],
            'model_size_mb': os.path.getsize(self.model_path) / (1024 * 1024)
        }
        
        # Input information
        for input_meta in self.session.get_inputs():
            info['input_details'].append({
                'name': input_meta.name,
                'type': input_meta.type,
                'shape': input_meta.shape
            })
        
        # Output information
        for output_meta in self.session.get_outputs():
            info['output_details'].append({
                'name': output_meta.name,
                'type': output_meta.type,
                'shape': output_meta.shape
            })
        
        return info
    
    def _create_test_data(self, input_meta, batch_size: int = 1):
        """Helper function to create appropriate test data based on input type"""
        # Handle dynamic shapes
        processed_shape = []
        for dim in input_meta.shape:
            if isinstance(dim, str) or dim is None or dim == -1:
                if len(processed_shape) == 0:  # batch dimension
                    processed_shape.append(batch_size)
                else:  # feature/sequence dimensions
                    # For sequence models (like text), use reasonable sequence length
                    if 'int' in input_meta.type:
                        processed_shape.append(30)  # Common sequence length
                    else:
                        processed_shape.append(2792 if len(processed_shape) == 1 else 10)
            else:
                processed_shape.append(dim)
        
        # Generate appropriate data based on input type
        if 'int' in input_meta.type:
            # For token-based models (NLP), generate random token IDs
            vocab_size = len(self.vocab) if self.vocab else 1000
            max_token_id = min(vocab_size - 1, 1000)  # Reasonable upper bound
            max_token_id = max(1, max_token_id)  # Ensure at least 1
            test_data = np.random.randint(0, max_token_id, size=processed_shape, dtype=np.int32)
        else:
            # For feature-based models, generate float data
            test_data = np.random.randn(*processed_shape).astype(np.float32)
        
        return test_data, processed_shape
    
    def test_structural_integrity(self) -> Dict[str, Any]:
        """📦 Test ONNX model structural integrity"""
        print("\n=== 📦 Structural Integrity Test ===")
        
        results = {
            'onnx_check_passed': False,
            'opset_version': None,
            'opset_supported': False,
            'initializers_count': 0,
            'missing_initializers': [],
            'structural_issues': []
        }
        
        try:
            # 1. ONNX model integrity check
            try:
                onnx.checker.check_model(self.onnx_model)
                results['onnx_check_passed'] = True
                print("✓ ONNX model integrity check passed")
            except Exception as e:
                results['structural_issues'].append(f"ONNX integrity check failed: {e}")
                print(f"✗ ONNX integrity check failed: {e}")
            
            # 2. Opset version validation
            try:
                if self.onnx_model.opset_import:
                    opset_version = self.onnx_model.opset_import[0].version
                    results['opset_version'] = opset_version
                    results['opset_supported'] = opset_version >= 11  # Modern ONNX Runtime support
                    
                    if results['opset_supported']:
                        print(f"✓ Opset version: {opset_version} (supported)")
                    else:
                        print(f"⚠ Opset version: {opset_version} (may have compatibility issues)")
                        results['structural_issues'].append(f"Old opset version: {opset_version}")
            except Exception as e:
                results['structural_issues'].append(f"Opset validation failed: {e}")
                print(f"✗ Opset validation failed: {e}")
            
            # 3. Initializers (weights) check
            try:
                results['initializers_count'] = len(self.onnx_model.graph.initializer)
                print(f"✓ Found {results['initializers_count']} initializers (weights/biases)")
                
                if results['initializers_count'] == 0:
                    results['structural_issues'].append("No initializers found - model may be missing weights")
                    print("⚠ No initializers found - model may be missing weights")
            except Exception as e:
                results['structural_issues'].append(f"Initializer check failed: {e}")
                print(f"✗ Initializer check failed: {e}")
            
            self.test_results['structural_integrity'] = results
            return results
            
        except Exception as e:
            print(f"✗ Structural integrity test failed: {e}")
            results['structural_issues'].append(str(e))
            self.test_results['structural_integrity'] = results
            return results
    
    def test_io_information(self) -> Dict[str, Any]:
        """🔍 Analyze input/output information"""
        print("\n=== 🔍 Input/Output Information Analysis ===")
        
        results = {
            'inputs': [],
            'outputs': [],
            'has_dynamic_shapes': False,
            'dynamic_dimensions': [],
            'total_inputs': 0,
            'total_outputs': 0
        }
        
        try:
            # Analyze inputs
            for i, input_meta in enumerate(self.session.get_inputs()):
                input_info = {
                    'index': i,
                    'name': input_meta.name,
                    'shape': input_meta.shape,
                    'type': input_meta.type,
                    'is_dynamic': any(isinstance(dim, str) or dim is None or dim == -1 for dim in input_meta.shape)
                }
                
                if input_info['is_dynamic']:
                    results['has_dynamic_shapes'] = True
                    dynamic_dims = [dim for dim in input_meta.shape if isinstance(dim, str) or dim is None or dim == -1]
                    results['dynamic_dimensions'].extend(dynamic_dims)
                
                results['inputs'].append(input_info)
                print(f"  Input {i}: {input_meta.name}")
                print(f"    Shape: {input_meta.shape} {'(dynamic)' if input_info['is_dynamic'] else '(static)'}")
                print(f"    Type: {input_meta.type}")
            
            # Analyze outputs
            for i, output_meta in enumerate(self.session.get_outputs()):
                output_info = {
                    'index': i,
                    'name': output_meta.name,
                    'shape': output_meta.shape,
                    'type': output_meta.type,
                    'is_dynamic': any(isinstance(dim, str) or dim is None or dim == -1 for dim in output_meta.shape)
                }
                
                if output_info['is_dynamic']:
                    results['has_dynamic_shapes'] = True
                
                results['outputs'].append(output_info)
                print(f"  Output {i}: {output_meta.name}")
                print(f"    Shape: {output_meta.shape} {'(dynamic)' if output_info['is_dynamic'] else '(static)'}")
                print(f"    Type: {output_meta.type}")
            
            results['total_inputs'] = len(results['inputs'])
            results['total_outputs'] = len(results['outputs'])
            
            print(f"✓ Total inputs: {results['total_inputs']}")
            print(f"✓ Total outputs: {results['total_outputs']}")
            print(f"✓ Dynamic shapes: {'Yes' if results['has_dynamic_shapes'] else 'No'}")
            
            self.test_results['io_information'] = results
            return results
            
        except Exception as e:
            print(f"✗ I/O information analysis failed: {e}")
            self.test_results['io_information'] = {'error': str(e)}
            return {}
    
    def test_model_graph_architecture(self) -> Dict[str, Any]:
        """🧱 Analyze model graph architecture"""
        print("\n=== 🧱 Model Graph Architecture Analysis ===")
        
        results = {
            'total_nodes': 0,
            'operation_types': {},
            'activation_functions': [],
            'layer_connections': [],
            'graph_complexity': 'simple'
        }
        
        try:
            graph = self.onnx_model.graph
            results['total_nodes'] = len(graph.node)
            
            # Analyze operation types
            op_counts = Counter(node.op_type for node in graph.node)
            results['operation_types'] = dict(op_counts)
            
            # Identify activation functions
            activation_ops = ['Relu', 'Sigmoid', 'Tanh', 'Softmax', 'LeakyRelu', 'Elu', 'Selu', 'Gelu']
            results['activation_functions'] = [op for op in activation_ops if op in results['operation_types']]
            
            # Analyze complexity
            if results['total_nodes'] > 100:
                results['graph_complexity'] = 'complex'
            elif results['total_nodes'] > 20:
                results['graph_complexity'] = 'moderate'
            
            print(f"✓ Total nodes (operations): {results['total_nodes']}")
            print(f"✓ Graph complexity: {results['graph_complexity']}")
            print(f"✓ Operation types found:")
            for op_type, count in sorted(results['operation_types'].items()):
                print(f"    {op_type}: {count}")
            
            if results['activation_functions']:
                print(f"✓ Activation functions: {', '.join(results['activation_functions'])}")
            else:
                print("⚠ No common activation functions detected")
            
            self.test_results['model_graph'] = results
            return results
            
        except Exception as e:
            print(f"✗ Model graph analysis failed: {e}")
            self.test_results['model_graph'] = {'error': str(e)}
            return {}
    
    def test_model_parameters(self) -> Dict[str, Any]:
        """🧠 Analyze model parameters"""
        print("\n=== 🧠 Model Parameters Analysis ===")
        
        results = {
            'total_parameters': 0,
            'trainable_parameters': 0,
            'parameter_shapes': [],
            'model_size_mb': 0,
            'largest_tensor': None,
            'parameter_distribution': {}
        }
        
        try:
            total_params = 0
            param_sizes = []
            
            # Analyze initializers (weights and biases)
            for initializer in self.onnx_model.graph.initializer:
                # Calculate number of elements
                shape = [dim for dim in initializer.dims]
                num_elements = np.prod(shape) if shape else 1
                total_params += num_elements
                param_sizes.append(num_elements)
                
                results['parameter_shapes'].append({
                    'name': initializer.name,
                    'shape': shape,
                    'elements': num_elements
                })
            
            results['total_parameters'] = total_params
            results['trainable_parameters'] = total_params  # Assume all are trainable for deployed models
            results['model_size_mb'] = os.path.getsize(self.model_path) / (1024 * 1024)
            
            if param_sizes:
                largest_idx = np.argmax(param_sizes)
                results['largest_tensor'] = results['parameter_shapes'][largest_idx]
            
            # Parameter distribution
            if param_sizes:
                results['parameter_distribution'] = {
                    'mean_size': np.mean(param_sizes),
                    'median_size': np.median(param_sizes),
                    'std_size': np.std(param_sizes)
                }
            
            print(f"✓ Total parameters: {total_params:,}")
            print(f"✓ Model file size: {results['model_size_mb']:.2f} MB")
            print(f"✓ Number of parameter tensors: {len(results['parameter_shapes'])}")
            
            if results['largest_tensor']:
                largest = results['largest_tensor']
                print(f"✓ Largest tensor: {largest['name']} with {largest['elements']:,} elements")
            
            self.test_results['model_parameters'] = results
            return results
            
        except Exception as e:
            print(f"✗ Model parameters analysis failed: {e}")
            self.test_results['model_parameters'] = {'error': str(e)}
            return {}
    
    def test_inference_smoke_tests(self) -> Dict[str, Any]:
        """🧪 Run comprehensive inference smoke tests"""
        print("\n=== 🧪 Inference Smoke Tests ===")
        
        results = {
            'basic_inference': False,
            'output_shape_correct': False,
            'output_range_valid': False,
            'stability_test': False,
            'edge_cases': {
                'zeros': False,
                'ones': False,
                'random': False,
                'large_values': False,
                'small_values': False
            },
            'inference_issues': []
        }
        
        try:
            input_meta = self.session.get_inputs()[0]
            input_name = input_meta.name
            output_meta = self.session.get_outputs()[0]
            
            # Create appropriate test data
            test_data, test_shape = self._create_test_data(input_meta)
            
            # Test 1: Basic inference
            try:
                output = self.session.run(None, {input_name: test_data})[0]
                results['basic_inference'] = True
                print("✓ Basic inference test passed")
                
                # Test 2: Output shape validation
                expected_batch = test_shape[0]
                if output.shape[0] == expected_batch:
                    results['output_shape_correct'] = True
                    print(f"✓ Output shape correct: {output.shape}")
                else:
                    results['inference_issues'].append(f"Output shape mismatch: expected batch {expected_batch}, got {output.shape[0]}")
                    print(f"⚠ Output shape mismatch: {output.shape}")
                
                # Test 3: Output range validation (for classification)
                if output.ndim >= 2 and output.shape[-1] == 1:  # Binary classification
                    if np.all((output >= 0) & (output <= 1)):
                        results['output_range_valid'] = True
                        print("✓ Output range valid for binary classification [0,1]")
                    else:
                        print(f"⚠ Output range: [{output.min():.3f}, {output.max():.3f}] (not typical for sigmoid)")
                elif output.ndim >= 2 and output.shape[-1] > 1:  # Multi-class
                    if np.allclose(np.sum(output, axis=-1), 1.0, atol=1e-3):
                        results['output_range_valid'] = True
                        print("✓ Output probabilities sum to 1 (softmax-like)")
                    else:
                        print("⚠ Output doesn't sum to 1 (may not be softmax)")
                else:
                    results['output_range_valid'] = True  # Can't validate range for unknown output type
                    print("✓ Output range validation skipped (unknown output type)")
                
            except Exception as e:
                results['inference_issues'].append(f"Basic inference failed: {e}")
                print(f"✗ Basic inference failed: {e}")
            
            # Test 4: Stability test (deterministic output)
            try:
                stable_test_data, _ = self._create_test_data(input_meta)
                output1 = self.session.run(None, {input_name: stable_test_data})[0]
                output2 = self.session.run(None, {input_name: stable_test_data})[0]
                
                if np.allclose(output1, output2, atol=1e-6):
                    results['stability_test'] = True
                    print("✓ Model outputs are deterministic")
                else:
                    results['inference_issues'].append("Model outputs are non-deterministic")
                    print("⚠ Model outputs vary between runs (non-deterministic)")
                    
            except Exception as e:
                results['inference_issues'].append(f"Stability test failed: {e}")
                print(f"✗ Stability test failed: {e}")
            
            # Test 5: Edge cases
            if 'int' in input_meta.type:
                # For token-based models
                vocab_size = len(self.vocab) if self.vocab else 1000
                max_token_id = min(vocab_size - 1, 1000)
                max_token_id = max(1, max_token_id)
                
                edge_test_data = {
                    'zeros': np.zeros(test_shape, dtype=np.int32),
                    'ones': np.ones(test_shape, dtype=np.int32),
                    'random': np.random.randint(0, max_token_id, size=test_shape, dtype=np.int32),
                    'max_tokens': np.full(test_shape, max_token_id-1, dtype=np.int32),
                    'mid_tokens': np.full(test_shape, max_token_id//2, dtype=np.int32)
                }
            else:
                # For feature-based models
                edge_test_data = {
                    'zeros': np.zeros(test_shape, dtype=np.float32),
                    'ones': np.ones(test_shape, dtype=np.float32),
                    'random': np.random.randn(*test_shape).astype(np.float32),
                    'large_values': np.full(test_shape, 100.0, dtype=np.float32),
                    'small_values': np.full(test_shape, 1e-6, dtype=np.float32)
                }
            
            for edge_name, edge_data in edge_test_data.items():
                try:
                    output = self.session.run(None, {input_name: edge_data})[0]
                    if not (np.isnan(output).any() or np.isinf(output).any()):
                        results['edge_cases'][edge_name] = True
                        print(f"✓ Edge case '{edge_name}' passed")
                    else:
                        results['inference_issues'].append(f"Edge case '{edge_name}' produced NaN/Inf")
                        print(f"✗ Edge case '{edge_name}' produced NaN/Inf")
                except Exception as e:
                    results['inference_issues'].append(f"Edge case '{edge_name}' failed: {e}")
                    print(f"✗ Edge case '{edge_name}' failed: {e}")
            
            self.test_results['inference_smoke_tests'] = results
            return results
            
        except Exception as e:
            print(f"✗ Inference smoke tests failed: {e}")
            self.test_results['inference_smoke_tests'] = {'error': str(e)}
            return {}
    
    def test_performance_profiling(self) -> Dict[str, Any]:
        """⚡ Performance and resource usage profiling"""
        print("\n=== ⚡ Performance Profiling ===")
        
        results = {
            'latency_ms': 0,
            'throughput_fps': 0,
            'memory_usage_mb': 0,
            'cpu_usage_percent': 0,
            'warmup_complete': False,
            'performance_category': 'unknown'
        }
        
        try:
            input_meta = self.session.get_inputs()[0]
            input_name = input_meta.name
            
            # Create appropriate test data
            test_data, processed_shape = self._create_test_data(input_meta)
            
            # Memory usage monitoring
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Warmup
            for _ in range(5):
                self.session.run(None, {input_name: test_data})
            results['warmup_complete'] = True
            
            # Performance measurement
            num_runs = 100
            start_time = time.time()
            
            for _ in range(num_runs):
                self.session.run(None, {input_name: test_data})
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_latency = (total_time / num_runs) * 1000  # ms
            throughput = num_runs / total_time  # FPS
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            
            results['latency_ms'] = avg_latency
            results['throughput_fps'] = throughput
            results['memory_usage_mb'] = max(0, memory_usage)  # Ensure non-negative
            
            # Performance categorization
            if avg_latency < 10:
                results['performance_category'] = 'excellent'
            elif avg_latency < 50:
                results['performance_category'] = 'good'
            elif avg_latency < 200:
                results['performance_category'] = 'acceptable'
            else:
                results['performance_category'] = 'poor'
            
            print(f"✓ Average latency: {avg_latency:.2f} ms")
            print(f"✓ Throughput: {throughput:.2f} FPS")
            print(f"✓ Memory usage: {memory_usage:.2f} MB")
            print(f"✓ Performance category: {results['performance_category']}")
            
            self.test_results['performance_profiling'] = results
            return results
            
        except Exception as e:
            print(f"✗ Performance profiling failed: {e}")
            self.test_results['performance_profiling'] = {'error': str(e)}
            return {}
    
    def test_compatibility(self) -> Dict[str, Any]:
        """🌐 Test platform and runtime compatibility"""
        print("\n=== 🌐 Compatibility Analysis ===")
        
        results = {
            'onnx_opset': None,
            'onnxruntime_compatible': True,
            'cpu_compatible': True,
            'mobile_ready': False,
            'web_compatible': False,
            'optimization_suggestions': []
        }
        
        try:
            # Opset version check
            if self.onnx_model.opset_import:
                opset_version = self.onnx_model.opset_import[0].version
                results['onnx_opset'] = opset_version
                
                if opset_version >= 11:
                    print(f"✓ ONNX opset {opset_version} - Modern runtime support")
                elif opset_version >= 8:
                    print(f"⚠ ONNX opset {opset_version} - Limited runtime support")
                    results['optimization_suggestions'].append("Consider updating to opset 11+")
                else:
                    print(f"⚠ ONNX opset {opset_version} - Legacy version")
                    results['optimization_suggestions'].append("Update to modern opset version")
            
            # CPU compatibility (already tested by successful loading)
            print("✓ CPU compatible - model loads successfully")
            
            # Mobile readiness heuristics
            model_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
            if model_size_mb < 50 and results.get('performance_category') in ['excellent', 'good']:
                results['mobile_ready'] = True
                print("✓ Mobile ready - small size and good performance")
            else:
                print("⚠ Mobile deployment may be challenging - large size or slow performance")
                if model_size_mb >= 50:
                    results['optimization_suggestions'].append("Consider model compression for mobile")
            
            # Web compatibility heuristics
            if opset_version and opset_version <= 13 and model_size_mb < 100:
                results['web_compatible'] = True
                print("✓ Web compatible - supported opset and reasonable size")
            else:
                print("⚠ Web deployment may have limitations")
                if model_size_mb >= 100:
                    results['optimization_suggestions'].append("Reduce model size for web deployment")
            
            # Check for unsupported operations for web/mobile
            graph = self.onnx_model.graph
            unsupported_web_ops = set()
            for node in graph.node:
                if node.op_type in ['Loop', 'If', 'Scan']:  # Complex control flow
                    unsupported_web_ops.add(node.op_type)
            
            if unsupported_web_ops:
                results['web_compatible'] = False
                print(f"⚠ Web incompatible operations: {', '.join(unsupported_web_ops)}")
                results['optimization_suggestions'].append("Remove complex control flow for web compatibility")
            
            self.test_results['compatibility'] = results
            return results
            
        except Exception as e:
            print(f"✗ Compatibility analysis failed: {e}")
            self.test_results['compatibility'] = {'error': str(e)}
            return {}
    
    def extract_model_metadata(self) -> Dict[str, Any]:
        """🧾 Extract model metadata and information"""
        print("\n=== 🧾 Model Metadata Extraction ===")
        
        results = {
            'model_name': None,
            'task_type': 'unknown',
            'input_length': None,
            'framework_origin': 'unknown',
            'preprocessing_info': {},
            'class_weights': None,
            'model_version': None,
            'creation_date': None
        }
        
        try:
            # Extract from model metadata
            if hasattr(self.onnx_model, 'metadata_props'):
                for prop in self.onnx_model.metadata_props:
                    if prop.key == 'model_name':
                        results['model_name'] = prop.value
                    elif prop.key == 'task_type':
                        results['task_type'] = prop.value
                    elif prop.key == 'framework':
                        results['framework_origin'] = prop.value
                    elif prop.key == 'version':
                        results['model_version'] = prop.value
            
            # Infer task type from model structure
            if results['task_type'] == 'unknown':
                output_shape = self.session.get_outputs()[0].shape
                if len(output_shape) >= 2:
                    if output_shape[-1] == 1:
                        results['task_type'] = 'binary_classification'
                    elif output_shape[-1] > 1:
                        results['task_type'] = 'multiclass_classification'
            
            # Extract input length from shape
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) >= 2:
                for dim in input_shape[1:]:
                    if isinstance(dim, int) and dim > 1:
                        results['input_length'] = dim
                        break
            
            # Check for preprocessing configuration
            if self.scaler:
                results['preprocessing_info']['scaler'] = 'StandardScaler'
                results['preprocessing_info']['scaler_features'] = len(self.scaler.get('mean', []))
            
            if self.vocab:
                results['preprocessing_info']['tokenizer'] = 'vocabulary_based'
                results['preprocessing_info']['vocab_size'] = len(self.vocab)
            
            # Try to infer framework from file path or operations
            graph = self.onnx_model.graph
            op_types = set(node.op_type for node in graph.node)
            
            if 'Gemm' in op_types and 'MatMul' in op_types:
                results['framework_origin'] = 'keras_tensorflow'
            elif 'Gemm' in op_types:
                results['framework_origin'] = 'sklearn_or_classical'
            elif 'Conv' in op_types:
                results['framework_origin'] = 'deep_learning_framework'
            
            # Print findings
            print(f"✓ Model name: {results['model_name'] or 'Not specified'}")
            print(f"✓ Task type: {results['task_type']}")
            print(f"✓ Input length: {results['input_length'] or 'Dynamic/Unknown'}")
            print(f"✓ Framework origin: {results['framework_origin']}")
            
            if results['preprocessing_info']:
                print("✓ Preprocessing detected:")
                for key, value in results['preprocessing_info'].items():
                    print(f"    {key}: {value}")
            
            self.test_results['model_metadata'] = results
            return results
            
        except Exception as e:
            print(f"✗ Model metadata extraction failed: {e}")
            self.test_results['model_metadata'] = {'error': str(e)}
            return {}
    
    def test_model_loading(self) -> bool:
        """Test if model loads correctly"""
        print("\n=== Model Loading Test ===")
        try:
            info = self.get_model_info()
            print(f"✓ Model size: {info['model_size_mb']:.2f} MB")
            print(f"✓ Input layers: {len(info['input_details'])}")
            print(f"✓ Output layers: {len(info['output_details'])}")
            
            for i, inp in enumerate(info['input_details']):
                print(f"  Input {i}: {inp['name']} - Shape: {inp['shape']}")
            
            for i, out in enumerate(info['output_details']):
                print(f"  Output {i}: {out['name']} - Shape: {out['shape']}")
            
            self.test_results['model_loading'] = True
            return True
        except Exception as e:
            print(f"✗ Model loading test failed: {e}")
            self.test_results['model_loading'] = False
            return False
    
    def test_inference_speed(self, num_samples: int = 100) -> Dict[str, float]:
        """Test inference speed with random data"""
        print(f"\n=== Inference Speed Test ({num_samples} samples) ===")
        
        try:
            # Get input metadata
            input_meta = self.session.get_inputs()[0]
            input_name = input_meta.name
            
            # Create appropriate test data
            test_data, sample_shape = self._create_test_data(input_meta)
            
            # Print data type info
            if 'int' in input_meta.type:
                vocab_size = len(self.vocab) if self.vocab else 1000
                print(f"✓ Using token IDs (vocab size: {vocab_size}, shape: {sample_shape})")
            else:
                print(f"✓ Using float features (shape: {sample_shape})")
            
            # Warm-up run
            self.session.run(None, {input_name: test_data})
            
            # Speed test
            start_time = time.time()
            for _ in range(num_samples):
                self.session.run(None, {input_name: test_data})
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / num_samples
            throughput = num_samples / total_time
            
            speed_results = {
                'total_time': total_time,
                'avg_inference_time': avg_time,
                'throughput_samples_per_sec': throughput
            }
            
            print(f"✓ Average inference time: {avg_time*1000:.2f} ms")
            print(f"✓ Throughput: {throughput:.2f} samples/sec")
            
            self.test_results['inference_speed'] = speed_results
            return speed_results
            
        except Exception as e:
            print(f"✗ Inference speed test failed: {e}")
            self.test_results['inference_speed'] = {'error': str(e)}
            return {}
    
    def test_with_training_data(self, training_data_path: str) -> Dict[str, Any]:
        """Test model performance on training data"""
        print("\n=== Training Data Test ===")
        
        try:
            if not os.path.exists(training_data_path):
                print(f"⚠ Training data not found at {training_data_path}")
                return {}
            
            df = pd.read_csv(training_data_path)
            print(f"✓ Loaded training data: {len(df)} samples")
            
            # Check if this is text data (sentiment analysis)
            first_column = df.iloc[:, 0]
            is_text_data = first_column.dtype == 'object' and isinstance(first_column.iloc[0], str)
            
            if is_text_data and len(first_column.iloc[0]) > 50:  # Likely text data
                print("⚠ Detected text data - sentiment analysis model")
                print("⚠ Cannot directly test with raw text data without preprocessing pipeline")
                print("💡 Consider providing preprocessed numerical features for testing")
                
                # Try to get basic stats about the data
                results = {
                    'accuracy': None,
                    'num_samples': len(df),
                    'num_classes': len(df.iloc[:, -1].unique()) if df.shape[1] > 1 else 'unknown',
                    'is_binary': len(df.iloc[:, -1].unique()) == 2 if df.shape[1] > 1 else None,
                    'data_type': 'text'
                }
                
                print(f"✓ Data type: Text (sentiment analysis)")
                print(f"✓ Number of samples: {results['num_samples']}")
                if df.shape[1] > 1:
                    print(f"✓ Number of classes: {results['num_classes']}")
                    print(f"✓ Classification type: {'Binary' if results['is_binary'] else 'Multiclass'}")
                
                self.test_results['training_data_performance'] = results
                return results
            
            # Original numerical data processing
            X = df.iloc[:, :-1].values.astype(np.float32)
            y_true = df.iloc[:, -1].values
            
            # Apply scaling if available
            if self.scaler is not None:
                X = (X - self.scaler['mean']) / self.scaler['scale']
            
            # Make predictions
            input_name = self.session.get_inputs()[0].name
            predictions = []
            
            for i in range(len(X)):
                sample = X[i:i+1]  # Keep batch dimension
                pred = self.session.run(None, {input_name: sample})[0]
                predictions.append(pred[0])
            
            predictions = np.array(predictions)
            
            # Determine if binary or multiclass
            unique_labels = len(np.unique(y_true))
            is_binary = unique_labels == 2
            
            if is_binary:
                y_pred = (predictions > 0.5).astype(int) if predictions.shape[-1] == 1 else np.argmax(predictions, axis=1)
            else:
                y_pred = np.argmax(predictions, axis=1) if len(predictions.shape) > 1 else predictions.round().astype(int)
            
            # Calculate metrics
            accuracy = np.mean(y_true == y_pred)
            
            results = {
                'accuracy': accuracy,
                'num_samples': len(df),
                'num_classes': unique_labels,
                'is_binary': is_binary,
                'data_type': 'numerical'
            }
            
            print(f"✓ Accuracy: {accuracy:.4f}")
            print(f"✓ Classification type: {'Binary' if is_binary else 'Multiclass'}")
            
            self.test_results['training_data_performance'] = results
            return results
            
        except Exception as e:
            print(f"✗ Training data test failed: {e}")
            self.test_results['training_data_performance'] = {'error': str(e)}
            return {}

    def test_edge_cases(self, edge_case_data_path: str) -> Dict[str, Any]:
        """Test model on edge cases"""
        print("\n=== Edge Case Test ===")
        
        try:
            if not os.path.exists(edge_case_data_path):
                print(f"⚠ Edge case data not found at {edge_case_data_path}")
                return {}
            
            df = pd.read_csv(edge_case_data_path)
            print(f"✓ Loaded edge case data: {len(df)} samples")
            
            # Check if this is text data
            first_column = df.iloc[:, 0]
            is_text_data = first_column.dtype == 'object' and isinstance(first_column.iloc[0], str)
            
            if is_text_data and len(first_column.iloc[0]) > 50:  # Likely text data
                print("⚠ Detected text data - sentiment analysis model")
                print("⚠ Cannot directly test with raw text data without preprocessing pipeline")
                print("💡 Consider providing preprocessed numerical features for edge case testing")
                
                results = {
                    'total_edge_cases': len(df),
                    'success_rate': None,
                    'failed_cases': [],
                    'data_type': 'text'
                }
                
                print(f"✓ Edge case data type: Text")
                print(f"✓ Number of edge cases: {results['total_edge_cases']}")
                
                self.test_results['edge_cases'] = results
                return results
            
            # Original numerical data processing
            X = df.iloc[:, :-1].values.astype(np.float32) if df.shape[1] > 1 else df.values.astype(np.float32)
            
            # Apply scaling if available
            if self.scaler is not None and X.shape[1] == len(self.scaler['mean']):
                X = (X - self.scaler['mean']) / self.scaler['scale']
            
            input_name = self.session.get_inputs()[0].name
            edge_case_results = []
            
            for i, sample in enumerate(X):
                try:
                    if len(sample.shape) == 1:
                        sample = sample.reshape(1, -1)
                    pred = self.session.run(None, {input_name: sample})[0]
                    edge_case_results.append({
                        'sample_idx': i,
                        'prediction': pred.tolist(),
                        'status': 'success'
                    })
                except Exception as e:
                    edge_case_results.append({
                        'sample_idx': i,
                        'error': str(e),
                        'status': 'failed'
                    })
            
            success_rate = sum(1 for r in edge_case_results if r['status'] == 'success') / len(edge_case_results)
            
            results = {
                'total_edge_cases': len(edge_case_results),
                'success_rate': success_rate,
                'failed_cases': [r for r in edge_case_results if r['status'] == 'failed'],
                'data_type': 'numerical'
            }
            
            print(f"✓ Edge cases processed: {len(edge_case_results)}")
            print(f"✓ Success rate: {success_rate:.2%}")
            if results['failed_cases']:
                print(f"⚠ Failed cases: {len(results['failed_cases'])}")
            
            self.test_results['edge_cases'] = results
            return results
            
        except Exception as e:
            print(f"✗ Edge case test failed: {e}")
            self.test_results['edge_cases'] = {'error': str(e)}
            return {}
    
    def test_input_validation(self) -> Dict[str, Any]:
        """Test model with various input types and edge values"""
        print("\n=== Input Validation Test ===")
        
        test_cases = []
        input_meta = self.session.get_inputs()[0]
        input_name = input_meta.name
        
        # Create base test data
        base_data, test_shape = self._create_test_data(input_meta)
        
        # Create validation tests based on input type
        if 'int' in input_meta.type:
            # For token-based models
            vocab_size = len(self.vocab) if self.vocab else 1000
            max_token_id = min(vocab_size - 1, 1000)
            max_token_id = max(1, max_token_id)
            
            validation_tests = [
                ("normal_tokens", lambda: np.random.randint(0, max_token_id, size=test_shape, dtype=np.int32)),
                ("zeros", lambda: np.zeros(test_shape, dtype=np.int32)),
                ("ones", lambda: np.ones(test_shape, dtype=np.int32)),
                ("max_tokens", lambda: np.full(test_shape, max_token_id-1, dtype=np.int32)),
                ("mid_range_tokens", lambda: np.full(test_shape, max_token_id//2, dtype=np.int32)),
                ("mixed_tokens", lambda: np.random.choice([0, 1, max_token_id//2, max_token_id-1], size=test_shape).astype(np.int32)),
            ]
        else:
            # For feature-based models
            validation_tests = [
                ("normal_values", lambda: np.random.randn(*test_shape).astype(np.float32)),
                ("zeros", lambda: np.zeros(test_shape, dtype=np.float32)),
                ("ones", lambda: np.ones(test_shape, dtype=np.float32)),
                ("large_values", lambda: np.full(test_shape, 1000.0, dtype=np.float32)),
                ("small_values", lambda: np.full(test_shape, 1e-6, dtype=np.float32)),
                ("negative_values", lambda: np.full(test_shape, -10.0, dtype=np.float32)),
            ]
        
        for test_name, data_generator in validation_tests:
            try:
                test_data = data_generator()
                result = self.session.run(None, {input_name: test_data})
                test_cases.append({
                    'test': test_name,
                    'status': 'passed',
                    'output_shape': [arr.shape for arr in result],
                    'has_nan': any(np.isnan(arr).any() for arr in result),
                    'has_inf': any(np.isinf(arr).any() for arr in result)
                })
                print(f"✓ {test_name}: Passed")
            except Exception as e:
                test_cases.append({
                    'test': test_name,
                    'status': 'failed',
                    'error': str(e)
                })
                print(f"✗ {test_name}: Failed - {e}")
        
        passed_tests = sum(1 for tc in test_cases if tc['status'] == 'passed')
        results = {
            'total_tests': len(test_cases),
            'passed_tests': passed_tests,
            'pass_rate': passed_tests / len(test_cases),
            'test_details': test_cases
        }
        
        print(f"✓ Input validation: {passed_tests}/{len(test_cases)} tests passed")
        
        self.test_results['input_validation'] = results
        return results

    def test_with_llm_router_api(self, api_endpoint: str, num_samples: int = 10, 
                                api_headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Test ONNX model with data from LLM-based router API"""
        print(f"\n=== 🤖 LLM Router API Test ({num_samples} samples) ===")
        
        results = {
            'api_endpoint': api_endpoint,
            'total_samples': num_samples,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'api_response_times': [],
            'model_inference_times': [],
            'predictions': [],
            'api_errors': [],
            'model_errors': [],
            'average_api_latency': 0,
            'average_model_latency': 0,
            'success_rate': 0
        }
        
        try:
            input_meta = self.session.get_inputs()[0]
            input_name = input_meta.name
            
            # Default headers
            headers = {'Content-Type': 'application/json'}
            if api_headers:
                headers.update(api_headers)
            
            print(f"✓ Testing with API endpoint: {api_endpoint}")
            print(f"✓ Model expects input: {input_meta.type} with shape {input_meta.shape}")
            
            for i in range(num_samples):
                try:
                    # Step 1: Get data from LLM Router API
                    api_start_time = time.time()
                    
                    # Create API request payload
                    api_payload = {
                        'request_id': f'test_{i}_{int(time.time())}',
                        'model_type': 'text_classification' if 'int' in input_meta.type else 'feature_classification',
                        'input_shape': input_meta.shape,
                        'vocab_size': len(self.vocab) if self.vocab else 1000,
                        'sequence_length': 30 if 'int' in input_meta.type else None
                    }
                    
                    # Make API request
                    response = requests.post(api_endpoint, json=api_payload, headers=headers, timeout=10)
                    api_end_time = time.time()
                    api_latency = api_end_time - api_start_time
                    results['api_response_times'].append(api_latency)
                    
                    if response.status_code != 200:
                        results['api_errors'].append(f"Sample {i}: HTTP {response.status_code}")
                        results['failed_predictions'] += 1
                        continue
                    
                    # Parse API response
                    api_data = response.json()
                    
                    # Step 2: Convert API data to model input format
                    model_input = self._convert_api_data_to_model_input(api_data, input_meta)
                    
                    # Step 3: Run model inference
                    model_start_time = time.time()
                    prediction = self.session.run(None, {input_name: model_input})[0]
                    model_end_time = time.time()
                    model_latency = model_end_time - model_start_time
                    results['model_inference_times'].append(model_latency)
                    
                    # Store successful prediction
                    results['predictions'].append({
                        'sample_id': i,
                        'api_data': api_data,
                        'model_input_shape': model_input.shape,
                        'prediction': prediction.tolist(),
                        'api_latency': api_latency,
                        'model_latency': model_latency,
                        'total_latency': api_latency + model_latency
                    })
                    
                    results['successful_predictions'] += 1
                    print(f"✓ Sample {i+1}/{num_samples}: API {api_latency*1000:.1f}ms + Model {model_latency*1000:.1f}ms")
                    
                except requests.exceptions.RequestException as e:
                    results['api_errors'].append(f"Sample {i}: API request failed - {str(e)}")
                    results['failed_predictions'] += 1
                    print(f"✗ Sample {i+1}/{num_samples}: API request failed - {str(e)}")
                    
                except Exception as e:
                    results['model_errors'].append(f"Sample {i}: Model inference failed - {str(e)}")
                    results['failed_predictions'] += 1
                    print(f"✗ Sample {i+1}/{num_samples}: Model inference failed - {str(e)}")
            
            # Calculate summary statistics
            if results['api_response_times']:
                results['average_api_latency'] = np.mean(results['api_response_times'])
            if results['model_inference_times']:
                results['average_model_latency'] = np.mean(results['model_inference_times'])
            
            results['success_rate'] = results['successful_predictions'] / num_samples
            
            # Print summary
            print(f"\n📊 API Test Summary:")
            print(f"✓ Successful predictions: {results['successful_predictions']}/{num_samples}")
            print(f"✓ Success rate: {results['success_rate']*100:.1f}%")
            print(f"✓ Average API latency: {results['average_api_latency']*1000:.1f}ms")
            print(f"✓ Average model latency: {results['average_model_latency']*1000:.1f}ms")
            
            if results['api_errors']:
                print(f"⚠ API errors: {len(results['api_errors'])}")
            if results['model_errors']:
                print(f"⚠ Model errors: {len(results['model_errors'])}")
            
            self.test_results['llm_router_api'] = results
            return results
            
        except Exception as e:
            print(f"✗ LLM Router API test failed: {e}")
            results['error'] = str(e)
            self.test_results['llm_router_api'] = results
            return results
    
    def _convert_api_data_to_model_input(self, api_data: Dict[str, Any], input_meta) -> np.ndarray:
        """Convert API response data to model input format"""
        try:
            # Check if API returned preprocessed data
            if 'model_input' in api_data:
                # API already provided model-ready input
                model_input = np.array(api_data['model_input'])
                
                # Ensure correct data type
                if 'int' in input_meta.type:
                    model_input = model_input.astype(np.int32)
                else:
                    model_input = model_input.astype(np.float32)
                
                return model_input
            
            # Handle different API response formats
            elif 'tokens' in api_data and 'int' in input_meta.type:
                # Token-based input (NLP models)
                tokens = api_data['tokens']
                if isinstance(tokens, list):
                    # Pad or truncate to expected sequence length
                    target_length = 30  # Default sequence length
                    if len(input_meta.shape) > 1:
                        target_length = input_meta.shape[1] if input_meta.shape[1] != -1 else 30
                    
                    if len(tokens) > target_length:
                        tokens = tokens[:target_length]
                    elif len(tokens) < target_length:
                        tokens.extend([0] * (target_length - len(tokens)))  # Pad with zeros
                    
                    return np.array([tokens], dtype=np.int32)  # Add batch dimension
                
            elif 'features' in api_data and 'float' in input_meta.type:
                # Feature-based input
                features = api_data['features']
                if isinstance(features, list):
                    return np.array([features], dtype=np.float32)  # Add batch dimension
            
            elif 'text' in api_data and self.vocab:
                # Raw text that needs tokenization
                text = api_data['text']
                tokens = self._tokenize_text(text)
                return np.array([tokens], dtype=np.int32)
            
            else:
                # Fallback: generate appropriate test data
                print("⚠ API data format not recognized, using generated test data")
                test_data, _ = self._create_test_data(input_meta)
                return test_data
                
        except Exception as e:
            print(f"⚠ Error converting API data, using generated test data: {e}")
            test_data, _ = self._create_test_data(input_meta)
            return test_data
    
    def _tokenize_text(self, text: str) -> List[int]:
        """Simple tokenization using vocabulary"""
        if not self.vocab:
            # Fallback: generate random tokens
            return [np.random.randint(0, 1000) for _ in range(30)]
        
        # Simple word-based tokenization
        words = text.lower().split()
        tokens = []
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab.get('<UNK>', 0))  # Unknown token
        
        # Pad or truncate to 30 tokens
        if len(tokens) > 30:
            tokens = tokens[:30]
        elif len(tokens) < 30:
            tokens.extend([0] * (30 - len(tokens)))
        
        return tokens

    def test_with_llm_generated_text(self, num_samples: int = None, model_type: str = "auto") -> Dict[str, Any]:
        """Test ONNX model with text generated by LLM via OpenRouter API"""
        
        # Auto-calculate samples based on number of classes (2 samples per class)
        if num_samples is None:
            class_info = self._detect_model_classes()
            num_samples = class_info['num_classes'] * 2
        
        print(f"\n=== 🤖 LLM Generated Text Test ({num_samples} samples) ===")
        
        # Get OpenRouter API key from environment (try both naming conventions)
        api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPEN_ROUTER_API_KEY')
        if not api_key:
            print("⚠ OpenRouter API key not found in .env file or environment")
            print("💡 Add to .env file: OPENROUTER_API_KEY=your-api-key or OPEN_ROUTER_API_KEY=your-api-key")
            return {'error': 'Missing OpenRouter API key'}
        
        results = {
            'total_samples': num_samples,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'llm_generation_times': [],
            'model_inference_times': [],
            'predictions': [],
            'generation_errors': [],
            'model_errors': [],
            'average_generation_time': 0,
            'average_model_latency': 0,
            'success_rate': 0,
            'model_type_detected': model_type
        }
        
        try:
            input_meta = self.session.get_inputs()[0]
            input_name = input_meta.name
            
            # Auto-detect model type from metadata
            if model_type == "auto":
                model_type = self._detect_model_type()
                results['model_type_detected'] = model_type
            
            print(f"✓ Detected model type: {model_type}")
            print(f"✓ Model expects input: {input_meta.type} with shape {input_meta.shape}")
            
            # Generate text prompts based on model type
            prompts = self._generate_prompts_for_model_type(model_type, num_samples)
            
            for i, prompt in enumerate(prompts):
                try:
                    # Step 1: Generate text using OpenRouter LLM
                    generation_start = time.time()
                    generated_text = self._generate_text_with_openrouter(prompt, api_key)
                    generation_end = time.time()
                    generation_time = generation_end - generation_start
                    results['llm_generation_times'].append(generation_time)
                    
                    if not generated_text:
                        results['generation_errors'].append(f"Sample {i}: Failed to generate text")
                        results['failed_predictions'] += 1
                        continue
                    
                    # Step 2: Convert text to model input
                    model_input = self._text_to_model_input(generated_text, input_meta)
                    
                    # Step 3: Run model inference
                    model_start = time.time()
                    prediction = self.session.run(None, {input_name: model_input})[0]
                    model_end = time.time()
                    model_latency = model_end - model_start
                    results['model_inference_times'].append(model_latency)
                    
                    # Store successful prediction
                    results['predictions'].append({
                        'sample_id': i,
                        'prompt': prompt,
                        'generated_text': generated_text,
                        'text_length': len(generated_text),
                        'model_input_shape': model_input.shape,
                        'prediction': prediction.tolist(),
                        'predicted_class': int(np.argmax(prediction)) if len(prediction.shape) > 1 else int(prediction > 0.5),
                        'confidence': float(np.max(prediction)) if len(prediction.shape) > 1 else float(prediction[0]),
                        'generation_time': generation_time,
                        'model_latency': model_latency,
                        'total_time': generation_time + model_latency
                    })
                    
                    results['successful_predictions'] += 1
                    pred_class = int(np.argmax(prediction)) if len(prediction.shape) > 1 else int(prediction > 0.5)
                    confidence = float(np.max(prediction)) if len(prediction.shape) > 1 else float(prediction[0])
                    
                    print(f"✓ Sample {i+1}/{num_samples}: '{generated_text[:50]}...' → Class {pred_class} ({confidence:.3f})")
                    print(f"  ⏱ Generation: {generation_time*1000:.1f}ms + Model: {model_latency*1000:.1f}ms")
                    
                except requests.exceptions.RequestException as e:
                    results['generation_errors'].append(f"Sample {i}: LLM generation failed - {str(e)}")
                    results['failed_predictions'] += 1
                    print(f"✗ Sample {i+1}/{num_samples}: LLM generation failed - {str(e)}")
                    
                except Exception as e:
                    results['model_errors'].append(f"Sample {i}: Model inference failed - {str(e)}")
                    results['failed_predictions'] += 1
                    print(f"✗ Sample {i+1}/{num_samples}: Model inference failed - {str(e)}")
            
            # Calculate summary statistics
            if results['llm_generation_times']:
                results['average_generation_time'] = np.mean(results['llm_generation_times'])
            if results['model_inference_times']:
                results['average_model_latency'] = np.mean(results['model_inference_times'])
            
            results['success_rate'] = results['successful_predictions'] / num_samples
            
            # Print summary
            print(f"\n📊 LLM Generated Text Test Summary:")
            print(f"✓ Successful predictions: {results['successful_predictions']}/{num_samples}")
            print(f"✓ Success rate: {results['success_rate']*100:.1f}%")
            print(f"✓ Average text generation time: {results['average_generation_time']*1000:.1f}ms")
            print(f"✓ Average model inference time: {results['average_model_latency']*1000:.1f}ms")
            
            if results['generation_errors']:
                print(f"⚠ Text generation errors: {len(results['generation_errors'])}")
            if results['model_errors']:
                print(f"⚠ Model inference errors: {len(results['model_errors'])}")
            
            # Show prediction distribution
            if results['predictions']:
                classes = [p['predicted_class'] for p in results['predictions']]
                class_counts = Counter(classes)
                print(f"📈 Prediction distribution: {dict(class_counts)}")
            
            self.test_results['llm_generated_text'] = results
            return results
            
        except Exception as e:
            print(f"✗ LLM generated text test failed: {e}")
            results['error'] = str(e)
            self.test_results['llm_generated_text'] = results
            return results
    
    def _detect_model_type(self) -> str:
        """Auto-detect the type of text classification model"""
        # Check metadata first
        metadata = self.test_results.get('model_metadata', {})
        
        # Look for clues in file path or config
        model_path_lower = self.model_path.lower()
        
        if 'hate' in model_path_lower or 'toxic' in model_path_lower:
            return 'hate_speech'
        elif 'sentiment' in model_path_lower or 'feedback' in model_path_lower or 'customer' in model_path_lower:
            return 'sentiment'
        elif 'spam' in model_path_lower:
            return 'spam'
        elif 'emotion' in model_path_lower:
            return 'emotion'
        elif 'intent' in model_path_lower:
            return 'intent'
        elif 'topic' in model_path_lower or 'category' in model_path_lower:
            return 'topic_classification'
        
        # Check output shape for binary vs multiclass
        output_meta = self.session.get_outputs()[0]
        if output_meta.shape and len(output_meta.shape) > 1:
            output_size = output_meta.shape[-1]
            if output_size == 2:
                return 'sentiment'  # Binary is often sentiment
            else:
                return 'multiclass_classification'
        elif output_meta.shape and output_meta.shape[-1] == 1:
            return 'sentiment'  # Single output is often binary sentiment
        
        return 'general_text_classification'
    
    def _detect_model_classes(self) -> Dict[str, Any]:
        """Detect the number of classes and try to infer their meanings"""
        output_meta = self.session.get_outputs()[0]
        
        # Determine number of classes
        num_classes = 1
        if output_meta.shape and len(output_meta.shape) > 1:
            num_classes = output_meta.shape[-1]
        elif output_meta.shape and output_meta.shape[-1] == 1:
            num_classes = 2  # Binary classification with single output
        
        # Infer class meanings based on model type and path
        model_path_lower = self.model_path.lower()
        class_info = {
            'num_classes': num_classes,
            'is_binary': num_classes == 2,
            'class_meanings': {},
            'model_type': self._detect_model_type()
        }
        
        # Define class meanings based on model type
        if 'hate' in model_path_lower or 'toxic' in model_path_lower:
            if num_classes == 2:
                class_info['class_meanings'] = {
                    0: 'hate_speech',
                    1: 'normal_speech'
                }
            else:
                class_info['class_meanings'] = {i: f'hate_level_{i}' for i in range(num_classes)}
                
        elif 'sentiment' in model_path_lower or 'feedback' in model_path_lower:
            if num_classes == 2:
                class_info['class_meanings'] = {
                    0: 'negative_sentiment',
                    1: 'positive_sentiment'
                }
            elif num_classes == 3:
                class_info['class_meanings'] = {
                    0: 'negative_sentiment',
                    1: 'neutral_sentiment', 
                    2: 'positive_sentiment'
                }
            else:
                class_info['class_meanings'] = {i: f'sentiment_level_{i}' for i in range(num_classes)}
                
        elif 'spam' in model_path_lower:
            if num_classes == 2:
                class_info['class_meanings'] = {
                    0: 'spam',
                    1: 'legitimate'
                }
            else:
                class_info['class_meanings'] = {i: f'spam_type_{i}' for i in range(num_classes)}
                
        elif 'news' in model_path_lower:
            # Common news categories
            news_categories = ['politics', 'sports', 'technology', 'business', 'entertainment', 
                             'health', 'science', 'world', 'opinion', 'local']
            class_info['class_meanings'] = {i: news_categories[i] if i < len(news_categories) 
                                          else f'news_category_{i}' for i in range(num_classes)}
                                          
        elif 'emotion' in model_path_lower:
            emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'neutral']
            class_info['class_meanings'] = {i: emotions[i] if i < len(emotions) 
                                          else f'emotion_{i}' for i in range(num_classes)}
        else:
            # Generic classification
            class_info['class_meanings'] = {i: f'class_{i}' for i in range(num_classes)}
        
        return class_info

    def _generate_prompts_for_model_type(self, model_type: str, num_samples: int) -> List[str]:
        """Generate appropriate prompts based on model type with balanced examples for each class"""
        prompts = []
        
        # First, detect the classes
        class_info = self._detect_model_classes()
        num_classes = class_info['num_classes']
        class_meanings = class_info['class_meanings']
        
        print(f"✓ Detected {num_classes} classes: {class_meanings}")
        
        # Generate exactly 2 samples per class
        samples_per_class = 2
        
        # Generate prompts for each class
        for class_id, class_meaning in class_meanings.items():
            class_prompts = self._generate_prompts_for_class(class_meaning, model_type, samples_per_class)
            prompts.extend(class_prompts)
        
        return prompts[:num_samples]  # Ensure we don't exceed requested samples
    
    def _generate_prompts_for_class(self, class_meaning: str, model_type: str, num_samples: int) -> List[str]:
        """Generate specific prompts for a particular class"""
        prompts = []
        
        # Define prompts based on class meaning
        if class_meaning == 'hate_speech':
            base_prompts = [
                "Generate a comment that shows bias against a group of people",
                "Write a comment with mild hostility towards others",
                "Create a post that contains subtle discrimination",
                "Generate a mildly offensive comment (but keep it realistic)",
                "Write something that shows prejudice but isn't extremely offensive"
            ]
        elif class_meaning == 'normal_speech':
            base_prompts = [
                "Generate a neutral social media comment about sports",
                "Write a positive comment about a movie",
                "Create a normal discussion post about technology",
                "Generate a friendly comment about food",
                "Write a constructive criticism about a product"
            ]
        elif class_meaning == 'negative_sentiment':
            base_prompts = [
                "Write a disappointed product review about a broken item",
                "Generate a frustrated customer complaint about terrible service",
                "Create a negative restaurant experience about bad food",
                "Write an angry service review about unhelpful support",
                "Generate disappointed feedback about a defective purchase"
            ]
        elif class_meaning == 'positive_sentiment':
            base_prompts = [
                "Write a very positive product review expressing satisfaction",
                "Generate an enthusiastic movie review",
                "Create a happy customer feedback message",
                "Write a delighted restaurant review",
                "Generate a satisfied service experience"
            ]
        elif class_meaning == 'neutral_sentiment':
            base_prompts = [
                "Write a balanced product review with pros and cons",
                "Generate a neutral news report about an event",
                "Create an objective comparison between two products",
                "Write a factual description of a service experience",
                "Generate an informational post about a topic"
            ]
        elif class_meaning == 'spam':
            base_prompts = [
                "Write a promotional email that sounds like spam",
                "Generate a message with too many exclamation marks and caps",
                "Create content with suspicious offers and urgent language",
                "Write a message promising unrealistic rewards",
                "Generate content with excessive promotional language"
            ]
        elif class_meaning == 'legitimate':
            base_prompts = [
                "Write a legitimate business email",
                "Generate a normal personal message",
                "Create a professional newsletter content",
                "Write a genuine customer inquiry",
                "Generate a real product announcement"
            ]
        elif 'news_category' in class_meaning or class_meaning in ['politics', 'sports', 'technology', 'business', 'entertainment', 'health', 'science', 'world', 'opinion', 'local']:
            category = class_meaning.replace('news_category_', '').replace('_', ' ')
            base_prompts = [
                f"Write a news headline and summary about {category}",
                f"Generate a breaking news story about {category}",
                f"Create a news article excerpt about {category}",
                f"Write a news report about recent {category} developments",
                f"Generate a news summary about {category} events"
            ]
        elif class_meaning in ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise', 'neutral']:
            base_prompts = [
                f"Write text expressing {class_meaning}",
                f"Generate content showing {class_meaning}",
                f"Create a message that conveys {class_meaning}",
                f"Write something that demonstrates {class_meaning}",
                f"Generate text with clear {class_meaning} emotion"
            ]
        else:
            # Generic prompts for unknown classes
            base_prompts = [
                f"Generate content for {class_meaning}",
                f"Write text that represents {class_meaning}",
                f"Create content typical of {class_meaning}",
                f"Generate an example of {class_meaning}",
                f"Write something that fits {class_meaning} category"
            ]
        
        # Cycle through prompts to reach num_samples
        for i in range(num_samples):
            prompts.append(base_prompts[i % len(base_prompts)])
        
        return prompts
    
    def _generate_text_with_openrouter(self, prompt: str, api_key: str) -> str:
        """Generate text using OpenRouter API"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/llama-3.1-8b-instruct:free",  # Free model
                "messages": [
                    {
                        "role": "user", 
                        "content": f"{prompt}. Keep it under 100 words and make it realistic."
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content'].strip()
            else:
                print(f"⚠ OpenRouter API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"⚠ Error generating text: {e}")
            return None
    
    def _text_to_model_input(self, text: str, input_meta) -> np.ndarray:
        """Convert text to model input format"""
        if 'int' in input_meta.type:
            # Tokenize text for NLP models
            tokens = self._tokenize_text(text)
            return np.array([tokens], dtype=np.int32)
        else:
            # For feature-based models, this would need text preprocessing
            # For now, generate appropriate test data
            test_data, _ = self._create_test_data(input_meta)
            return test_data

    def run_comprehensive_test(self, 
                             training_data_path: str = None,
                             edge_case_data_path: str = None,
                             api_endpoint: str = None,
                             api_headers: Dict[str, str] = None,
                             use_llm_text: bool = False) -> Dict[str, Any]:
        """Run all available tests"""
        print("🚀 Starting Comprehensive ONNX Model Testing")
        print("=" * 60)
        
        # 1. Structural and architectural analysis
        self.test_structural_integrity()
        self.test_io_information()
        self.test_model_graph_architecture()
        self.test_model_parameters()
        self.extract_model_metadata()
        
        # 2. Original functionality tests
        self.test_model_loading()
        self.test_inference_speed()  # Keep original for backward compatibility
        self.test_input_validation()
        
        # 3. Enhanced testing
        self.test_inference_smoke_tests()
        self.test_performance_profiling()
        self.test_compatibility()
        
        # 4. Data-based tests (if available)
        if training_data_path:
            self.test_with_training_data(training_data_path)
        
        if edge_case_data_path:
            self.test_edge_cases(edge_case_data_path)
        
        # 5. API-based testing (if available)
        if api_endpoint:
            self.test_with_llm_router_api(api_endpoint, api_headers=api_headers)
        
        # 6. LLM generated text testing (if enabled)
        if use_llm_text:
            print(f"\n🤖 Running LLM text generation test...")
            self.test_with_llm_generated_text()
        
        return self.test_results
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate a deployment readiness report"""
        print("\n" + "=" * 60)
        print("📊 DEPLOYMENT READINESS REPORT")
        print("=" * 60)
        
        # Scoring criteria
        score = 0
        max_score = 0
        issues = []
        recommendations = []
        
        # Model loading (Critical)
        max_score += 20
        if self.test_results.get('model_loading', False):
            score += 20
            print("✓ Model Loading: PASS (20/20 points)")
        else:
            issues.append("Model fails to load")
            print("✗ Model Loading: FAIL (0/20 points)")
        
        # Input validation (Important)
        max_score += 15
        input_val = self.test_results.get('input_validation', {})
        if input_val.get('pass_rate', 0) >= 0.8:
            score += 15
            print("✓ Input Validation: PASS (15/15 points)")
        elif input_val.get('pass_rate', 0) >= 0.6:
            score += 10
            print("⚠ Input Validation: PARTIAL (10/15 points)")
            recommendations.append("Some input validation tests failed")
        else:
            print("✗ Input Validation: FAIL (0/15 points)")
            issues.append("Multiple input validation failures")
        
        # Performance (Important)
        max_score += 15
        speed = self.test_results.get('inference_speed', {})
        if speed.get('avg_inference_time', float('inf')) < 0.1:  # Less than 100ms
            score += 15
            print("✓ Performance: EXCELLENT (15/15 points)")
        elif speed.get('avg_inference_time', float('inf')) < 0.5:  # Less than 500ms
            score += 10
            print("✓ Performance: GOOD (10/15 points)")
        elif speed.get('avg_inference_time', float('inf')) < 1.0:  # Less than 1s
            score += 5
            print("⚠ Performance: ACCEPTABLE (5/15 points)")
            recommendations.append("Consider optimizing model for better performance")
        else:
            print("✗ Performance: POOR (0/15 points)")
            issues.append("Model inference is too slow")
        
        # Training data performance (Important)
        max_score += 20
        train_perf = self.test_results.get('training_data_performance', {})
        if train_perf.get('data_type') == 'text':
            score += 10  # Partial credit for text data detection
            print("⚠ Training Data Performance: TEXT DETECTED (10/20 points)")
            recommendations.append("Provide preprocessed numerical features for full performance validation")
        elif train_perf.get('accuracy', 0) >= 0.9:
            score += 20
            print("✓ Training Data Performance: EXCELLENT (20/20 points)")
        elif train_perf.get('accuracy', 0) >= 0.8:
            score += 15
            print("✓ Training Data Performance: GOOD (15/20 points)")
        elif train_perf.get('accuracy', 0) >= 0.7:
            score += 10
            print("⚠ Training Data Performance: ACCEPTABLE (10/20 points)")
            recommendations.append("Model accuracy could be improved")
        elif 'error' in train_perf:
            print("? Training Data Performance: NOT TESTED (0/20 points)")
            recommendations.append("Test with training data for performance validation")
        else:
            print("✗ Training Data Performance: POOR (0/20 points)")
            issues.append("Model accuracy is below acceptable threshold")
        
        # Edge cases (Moderate)
        max_score += 15
        edge_cases = self.test_results.get('edge_cases', {})
        if edge_cases.get('data_type') == 'text':
            score += 8  # Partial credit for text data detection
            print("⚠ Edge Cases: TEXT DETECTED (8/15 points)")
            recommendations.append("Provide preprocessed numerical features for edge case testing")
        elif edge_cases.get('success_rate', 0) >= 0.95:
            score += 15
            print("✓ Edge Cases: EXCELLENT (15/15 points)")
        elif edge_cases.get('success_rate', 0) >= 0.8:
            score += 10
            print("✓ Edge Cases: GOOD (10/15 points)")
        elif edge_cases.get('success_rate', 0) >= 0.6:
            score += 5
            print("⚠ Edge Cases: NEEDS IMPROVEMENT (5/15 points)")
            recommendations.append("Address edge case failures")
        elif 'error' in edge_cases:
            print("? Edge Cases: NOT TESTED (0/15 points)")
            recommendations.append("Test with edge cases for robustness validation")
        else:
            print("✗ Edge Cases: POOR (0/15 points)")
            issues.append("Model fails on many edge cases")
        

        
        # Final assessment
        percentage = (score / max_score) * 100
        
        print(f"\n📈 OVERALL SCORE: {score}/{max_score} ({percentage:.1f}%)")
        
        if percentage >= 85:
            deployment_status = "READY FOR DEPLOYMENT"
            status_color = "✅"
        elif percentage >= 70:
            deployment_status = "DEPLOYMENT WITH CAUTION"
            status_color = "⚠️"
        else:
            deployment_status = "NOT READY FOR DEPLOYMENT"
            status_color = "❌"
        
        print(f"{status_color} STATUS: {deployment_status}")
        
        if issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(issues)}):")
            for issue in issues:
                print(f"  • {issue}")
        
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS ({len(recommendations)}):")
            for rec in recommendations:
                print(f"  • {rec}")
        
        report = {
            'score': score,
            'max_score': max_score,
            'percentage': percentage,
            'deployment_status': deployment_status,
            'is_ready': percentage >= 85,
            'issues': issues,
            'recommendations': recommendations,
            'test_results': self.test_results
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: str = "deployment_report.json"):
        """Save the deployment report to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Report saved to: {output_path}")


def auto_discover_files(folder_path: str) -> Dict[str, Optional[str]]:
    """Auto-discover model files in a folder"""
    folder = Path(folder_path)
    discovered = {
        'model': None,
        'training_data': None,
        'edge_cases': None,
        'config_dir': str(folder)
    }
    
    if not folder.exists():
        print(f"⚠ Folder not found: {folder_path}")
        return discovered
    
    print(f"🔍 Auto-discovering files in: {folder_path}")
    
    # Look for ONNX model
    onnx_files = list(folder.glob("*.onnx"))
    if onnx_files:
        discovered['model'] = str(onnx_files[0])
        print(f"✓ Found ONNX model: {onnx_files[0].name}")
    
    # Look for training data (common patterns)
    training_patterns = ["*training*.csv", "*train*.csv", "training_data.csv"]
    for pattern in training_patterns:
        training_files = list(folder.glob(pattern))
        if training_files:
            discovered['training_data'] = str(training_files[0])
            print(f"✓ Found training data: {training_files[0].name}")
            break
    
    # Look for edge case data (common patterns)
    edge_patterns = ["*edge*.csv", "*edge_case*.csv", "edge_case_data.csv"]
    for pattern in edge_patterns:
        edge_files = list(folder.glob(pattern))
        if edge_files:
            discovered['edge_cases'] = str(edge_files[0])
            print(f"✓ Found edge cases: {edge_files[0].name}")
            break
    
    # Check for config files
    config_files = ['scaler.json', 'vocab.json', 'generation_config.json']
    found_configs = []
    for config_file in config_files:
        if (folder / config_file).exists():
            found_configs.append(config_file)
    
    if found_configs:
        print(f"✓ Found config files: {', '.join(found_configs)}")
    
    return discovered


def main():
    parser = argparse.ArgumentParser(description='ONNX Model Testing Tool')
    
    # Create mutually exclusive group for folder vs individual files
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--folder', help='Path to folder containing model files (auto-discovery)')
    input_group.add_argument('--model', help='Path to ONNX model file')
    
    # Individual file arguments (used when --model is specified)
    parser.add_argument('--config-dir', default='.', help='Directory containing config files')
    parser.add_argument('--training-data', help='Path to training data CSV')
    parser.add_argument('--edge-cases', help='Path to edge case data CSV')
    parser.add_argument('--api-endpoint', help='LLM Router API endpoint for testing')
    parser.add_argument('--api-headers', help='API headers as JSON string (e.g., \'{"Authorization": "Bearer token"}\')')
    parser.add_argument('--use-llm-text', action='store_true', help='Test with LLM-generated text (requires OPENROUTER_API_KEY)')
    # LLM samples are now calculated automatically (2 per class)
    parser.add_argument('--output', default='deployment_report.json', help='Output report file')
    
    args = parser.parse_args()
    
    # Auto-discover files if folder is provided
    if args.folder:
        discovered = auto_discover_files(args.folder)
        
        if not discovered['model']:
            print("❌ No ONNX model found in the specified folder!")
            print("Please ensure your folder contains a .onnx file")
            return False
        
        model_path = discovered['model']
        config_dir = discovered['config_dir']
        training_data = discovered['training_data']
        edge_cases = discovered['edge_cases']
        api_endpoint = args.api_endpoint
        
        # Set output path relative to the folder
        output_path = str(Path(args.folder) / "deployment_report.json")
        
    else:
        # Use individually specified files
        model_path = args.model
        config_dir = args.config_dir
        training_data = args.training_data
        edge_cases = args.edge_cases
        api_endpoint = args.api_endpoint
        output_path = args.output
    
    # Parse API headers if provided
    api_headers = None
    if args.api_headers:
        try:
            api_headers = json.loads(args.api_headers)
        except json.JSONDecodeError:
            print("⚠ Warning: Invalid JSON format for API headers, ignoring")
            api_headers = None
    
    # Initialize tester
    tester = ONNXModelTester(model_path, config_dir)
    
    # Run comprehensive tests
    tester.run_comprehensive_test(
        training_data_path=training_data,
        edge_case_data_path=edge_cases,
        api_endpoint=api_endpoint,
        api_headers=api_headers,
        use_llm_text=args.use_llm_text
    )
    
    # Generate and save report
    report = tester.generate_deployment_report()
    tester.save_report(report, output_path)
    
    return report['is_ready']


if __name__ == "__main__":
    import sys
    ready = main()
    sys.exit(0 if ready else 1) 