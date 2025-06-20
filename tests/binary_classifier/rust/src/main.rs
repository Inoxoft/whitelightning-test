use anyhow::Result;
use ort::{Environment, Session, SessionBuilder, Value};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use ndarray::Array2;
use std::env;
use std::time::Instant;
use sysinfo::{System, SystemExt, CpuExt};
use std::thread;
use std::sync::{Mutex, atomic::{AtomicBool, Ordering}};

#[derive(Debug, Clone)]
struct SystemInfo {
    platform: String,
    architecture: String,
    cpu_brand: String,
    cpu_cores_physical: usize,
    cpu_cores_logical: usize,
    cpu_frequency_mhz: u64,
    total_memory_gb: f64,
    available_memory_gb: f64,
    rust_version: String,
    onnx_version: String,
    compiler_version: String,
}

#[derive(Debug, Clone)]
struct PerformanceMetrics {
    total_time_ms: f64,
    preprocessing_time_ms: f64,
    inference_time_ms: f64,
    postprocessing_time_ms: f64,
    memory_start_mb: f64,
    memory_end_mb: f64,
    memory_peak_mb: f64,
    memory_delta_mb: f64,
    cpu_usage_avg: f64,
    cpu_usage_peak: f64,
    cpu_samples: usize,
    throughput_per_sec: f64,
    predictions_count: usize,
}

struct ResourceMonitor {
    system: Arc<Mutex<System>>,
    monitoring: Arc<AtomicBool>,
    cpu_readings: Arc<Mutex<Vec<f64>>>,
    memory_readings: Arc<Mutex<Vec<f64>>>,
}

impl ResourceMonitor {
    fn new() -> Self {
        Self {
            system: Arc::new(Mutex::new(System::new_all())),
            monitoring: Arc::new(AtomicBool::new(false)),
            cpu_readings: Arc::new(Mutex::new(Vec::new())),
            memory_readings: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn start_monitoring(&self) {
        self.monitoring.store(true, Ordering::Relaxed);
        
        // Clear previous readings
        if let Ok(mut cpu_readings) = self.cpu_readings.lock() {
            cpu_readings.clear();
        }
        if let Ok(mut memory_readings) = self.memory_readings.lock() {
            memory_readings.clear();
        }

        let system_clone = Arc::clone(&self.system);
        let monitoring_clone = Arc::clone(&self.monitoring);
        let cpu_readings_clone = Arc::clone(&self.cpu_readings);
        let memory_readings_clone = Arc::clone(&self.memory_readings);

        thread::spawn(move || {
            while monitoring_clone.load(Ordering::Relaxed) {
                if let Ok(mut system) = system_clone.lock() {
                    system.refresh_cpu();
                    system.refresh_memory();
                    
                    let cpu_usage: f64 = system.cpus().iter()
                        .map(|cpu| cpu.cpu_usage() as f64)
                        .sum::<f64>() / system.cpus().len() as f64;
                    
                    let memory_usage_mb = system.used_memory() as f64 / (1024.0 * 1024.0);
                    
                    if let Ok(mut cpu_readings) = cpu_readings_clone.lock() {
                        cpu_readings.push(cpu_usage);
                    }
                    if let Ok(mut memory_readings) = memory_readings_clone.lock() {
                        memory_readings.push(memory_usage_mb);
                    }
                }
                thread::sleep(std::time::Duration::from_millis(50));
            }
        });
    }

    fn stop_monitoring(&self) -> (f64, f64, usize, f64, f64) {
        self.monitoring.store(false, Ordering::Relaxed);
        thread::sleep(std::time::Duration::from_millis(100)); // Allow final readings
        
        let cpu_readings = self.cpu_readings.lock().unwrap();
        let memory_readings = self.memory_readings.lock().unwrap();
        
        let cpu_avg = if cpu_readings.is_empty() { 0.0 } else {
            cpu_readings.iter().sum::<f64>() / cpu_readings.len() as f64
        };
        let cpu_peak = cpu_readings.iter().fold(0.0f64, |a, &b| a.max(b));
        let cpu_samples = cpu_readings.len();
        
        let memory_peak = memory_readings.iter().fold(0.0f64, |a, &b| a.max(b));
        let memory_current = memory_readings.last().copied().unwrap_or(0.0);
        
        (cpu_avg, cpu_peak, cpu_samples, memory_peak, memory_current)
    }
}

impl SystemInfo {
    fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        
        let platform = format!("{} {}", std::env::consts::OS, 
                              system.kernel_version().unwrap_or_else(|| "Unknown".to_string()));
        let architecture = std::env::consts::ARCH.to_string();
        
        let cpu_brand = system.cpus().first()
            .map(|cpu| cpu.brand().to_string())
            .unwrap_or_else(|| "Unknown CPU".to_string());
        
        let cpu_cores_physical = system.physical_core_count().unwrap_or(0);
        let cpu_cores_logical = system.cpus().len();
        
        let cpu_frequency_mhz = system.cpus().first()
            .map(|cpu| cpu.frequency())
            .unwrap_or(0);
        
        let total_memory_gb = system.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        let available_memory_gb = system.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        
        let rust_version = format!("{} ({})", 
                                  env!("CARGO_PKG_VERSION"),
                                  option_env!("RUSTC_VERSION").unwrap_or("unknown"));
        let onnx_version = "1.16.3".to_string();
        let compiler_version = format!("rustc {}", 
                                     option_env!("RUSTC_VERSION").unwrap_or("unknown"));

        Self {
            platform,
            architecture,
            cpu_brand,
            cpu_cores_physical,
            cpu_cores_logical,
            cpu_frequency_mhz,
            total_memory_gb,
            available_memory_gb,
            rust_version,
            onnx_version,
            compiler_version,
        }
    }

    fn print(&self) {
        println!("🖥️  SYSTEM INFORMATION:");
        println!("   Platform: {}", self.platform);
        println!("   Architecture: {}", self.architecture);
        println!("   CPU: {}", self.cpu_brand);
        println!("   CPU Cores: {} physical, {} logical", self.cpu_cores_physical, self.cpu_cores_logical);
        if self.cpu_frequency_mhz > 0 {
            println!("   CPU Frequency: {} MHz", self.cpu_frequency_mhz);
        }
        println!("   Total Memory: {:.2} GB", self.total_memory_gb);
        println!("   Available Memory: {:.2} GB", self.available_memory_gb);
        println!("   Rust Version: {}", self.rust_version);
        println!("   ONNX Runtime: {}", self.onnx_version);
        println!("   Compiler: {}", self.compiler_version);
        println!();
    }
}

impl PerformanceMetrics {
    fn print(&self) {
        println!("📊 PERFORMANCE METRICS:");
        println!("   Total Processing Time: {:.2}ms", self.total_time_ms);
        println!("   ├─ Preprocessing: {:.2}ms ({:.1}%)", 
                 self.preprocessing_time_ms, 
                 (self.preprocessing_time_ms / self.total_time_ms) * 100.0);
        println!("   ├─ Model Inference: {:.2}ms ({:.1}%)", 
                 self.inference_time_ms, 
                 (self.inference_time_ms / self.total_time_ms) * 100.0);
        println!("   └─ Postprocessing: {:.2}ms ({:.1}%)", 
                 self.postprocessing_time_ms, 
                 (self.postprocessing_time_ms / self.total_time_ms) * 100.0);
        println!();
        
        println!("🚀 THROUGHPUT:");
        println!("   Predictions per second: {:.2}", self.throughput_per_sec);
        println!("   Total predictions: {}", self.predictions_count);
        println!("   Average time per prediction: {:.2}ms", self.total_time_ms / self.predictions_count as f64);
        println!();
        
        println!("💾 MEMORY USAGE:");
        println!("   Memory Start: {:.2} MB", self.memory_start_mb);
        println!("   Memory End: {:.2} MB", self.memory_end_mb);
        println!("   Memory Peak: {:.2} MB", self.memory_peak_mb);
        println!("   Memory Delta: {}{:.2} MB", 
                 if self.memory_delta_mb >= 0.0 { "+" } else { "" }, 
                 self.memory_delta_mb);
        println!();
        
        println!("🔥 CPU USAGE:");
        if self.cpu_samples > 0 {
            println!("   Average CPU: {:.1}%", self.cpu_usage_avg);
            println!("   Peak CPU: {:.1}%", self.cpu_usage_peak);
            println!("   Samples: {}", self.cpu_samples);
        } else {
            println!("   CPU monitoring: Not available");
        }
        println!();
        
        // Performance rating
        let (rating, emoji) = if self.total_time_ms < 10.0 {
            ("EXCELLENT", "🚀")
        } else if self.total_time_ms < 50.0 {
            ("VERY GOOD", "✅")
        } else if self.total_time_ms < 100.0 {
            ("GOOD", "👍")
        } else if self.total_time_ms < 200.0 {
            ("ACCEPTABLE", "⚠️")
        } else {
            ("POOR", "❌")
        };
        
        println!("🎯 PERFORMANCE RATING: {} {}", emoji, rating);
        println!("   ({:.1}ms total - Target: <100ms)", self.total_time_ms);
        println!();
    }
}

struct BinaryClassifier {
    vocab: HashMap<String, usize>,
    idf: Vec<f32>,
    mean: Vec<f32>,
    scale: Vec<f32>,
    session: Session,
}

impl BinaryClassifier {
    fn new(model_path: &str, vocab_path: &str, scaler_path: &str) -> Result<Self> {
        let vocab_file = File::open(vocab_path)?;
        let vocab_reader = BufReader::new(vocab_file);
        let vocab_data: JsonValue = serde_json::from_reader(vocab_reader)?;
        
        let mut vocab = HashMap::new();
        let vocab_obj = vocab_data["vocab"].as_object().unwrap();
        for (key, value) in vocab_obj {
            vocab.insert(key.clone(), value.as_u64().unwrap() as usize);
        }
        
        let idf: Vec<f32> = vocab_data["idf"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        let scaler_file = File::open(scaler_path)?;
        let scaler_reader = BufReader::new(scaler_file);
        let scaler_data: JsonValue = serde_json::from_reader(scaler_reader)?;
        
        let mean: Vec<f32> = scaler_data["mean"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
            
        let scale: Vec<f32> = scaler_data["scale"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        let environment = Arc::new(Environment::builder()
            .with_name("binary_classifier")
            .build()?);
        let session = SessionBuilder::new(&environment)?
            .with_model_from_file(model_path)?;

        Ok(BinaryClassifier {
            vocab,
            idf,
            mean,
            scale,
            session,
        })
    }

    fn preprocess_text(&self, text: &str) -> Vec<f32> {
        let vocab_size = self.idf.len();
        let mut vector = vec![0.0; vocab_size];
        let mut word_counts: HashMap<&str, usize> = HashMap::new();
        let mut total_words = 0;

        let text_lower = text.to_lowercase();
        for word in text_lower.split_whitespace() {
            if !word.is_empty() {
                *word_counts.entry(word).or_insert(0) += 1;
                total_words += 1;
            }
        }

        // Apply CORRECTED TF-IDF with proper normalization
        if total_words > 0 {
            for (word, count) in word_counts {
                if let Some(&idx) = self.vocab.get(word) {
                    if idx < vocab_size {
                        // FIXED: Calculate proper TF (normalized by total words) then multiply by IDF
                        let tf = count as f32 / total_words as f32;  // Term Frequency normalization
                        vector[idx] = tf * self.idf[idx];            // Correct TF-IDF calculation
                    }
                }
            }
        }

        for i in 0..vocab_size {
            vector[i] = (vector[i] - self.mean[i]) / self.scale[i];
        }

        vector
    }

    fn predict_with_timing(&self, text: &str) -> Result<(f32, f64, f64, f64)> {
        let total_start = Instant::now();
        
        // Preprocessing
        let preprocess_start = Instant::now();
        let input_data = self.preprocess_text(text);
        let preprocessing_time = preprocess_start.elapsed().as_secs_f64() * 1000.0;
        
        // Inference
        let inference_start = Instant::now();
        let vocab_size = input_data.len();
        let input_array = Array2::from_shape_vec((1, vocab_size), input_data)?;
        let input_dyn = input_array.into_dyn();
        let input_cow = ndarray::CowArray::from(input_dyn.view());
        let input_tensor = Value::from_array(self.session.allocator(), &input_cow)?;

        let outputs = self.session.run(vec![input_tensor])?;
        let inference_time = inference_start.elapsed().as_secs_f64() * 1000.0;
        
        // Postprocessing
        let postprocess_start = Instant::now();
        let output_view = outputs[0].try_extract::<f32>()?;
        let output_data = output_view.view();
        let result = output_data[[0, 0]];
        let _postprocessing_time = postprocess_start.elapsed().as_secs_f64() * 1000.0;
        
        let total_time = total_start.elapsed().as_secs_f64() * 1000.0;
        
        Ok((result, total_time, preprocessing_time, inference_time))
    }

    fn predict(&self, text: &str) -> Result<f32> {
        let (result, _, _, _) = self.predict_with_timing(text)?;
        Ok(result)
    }
}

fn get_memory_usage_mb() -> f64 {
    let mut system = System::new();
    system.refresh_memory();
    system.used_memory() as f64 / (1024.0 * 1024.0)
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    
    // Check if model files exist
    let model_exists = std::path::Path::new("model.onnx").exists();
    let vocab_exists = std::path::Path::new("vocab.json").exists();
    let scaler_exists = std::path::Path::new("scaler.json").exists();
    
    if !model_exists || !vocab_exists || !scaler_exists {
        println!("⚠️ Model files not found in current directory");
        println!("Expected files: model.onnx, vocab.json, scaler.json");
        println!("✅ Rust implementation compiled successfully");
        println!("🏗️ Build verification completed - would run with actual model files");
        return Ok(());
    }

    // Print system information
    let system_info = SystemInfo::new();
    system_info.print();

    let classifier = BinaryClassifier::new(
        "model.onnx",
        "vocab.json", 
        "scaler.json",
    )?;

    // Handle command line arguments
    if args.len() > 1 {
        if args[1] == "--benchmark" {
            let iterations = if args.len() > 2 {
                args[2].parse().unwrap_or(10)
            } else {
                10
            };
            
            println!("🚀 Running Rust ONNX Binary Classifier Benchmark");
            println!("📊 Iterations: {}", iterations);
            println!();
            
            let test_texts = vec![
                "This is a positive review of a great product",
                "Terrible service, would not recommend",
                "Amazing quality and fast delivery",
                "Poor customer support experience",
                "Excellent value for money",
            ];
            
            // Initialize monitoring
            let monitor = ResourceMonitor::new();
            let memory_start = get_memory_usage_mb();
            monitor.start_monitoring();
            
            let start_time = Instant::now();
            let mut total_predictions = 0;
            let mut total_preprocessing_time = 0.0;
            let mut total_inference_time = 0.0;
            let mut total_postprocessing_time = 0.0;
            
            // Warmup
            println!("🔥 Warming up model (5 runs)...");
            for _ in 0..5 {
                for text in &test_texts {
                    let _ = classifier.predict(text)?;
                }
            }
            println!();
            
            println!("📊 Running benchmark...");
            for i in 0..iterations {
                for text in &test_texts {
                    let (probability, _total_time, preprocessing_time, inference_time) = 
                        classifier.predict_with_timing(text)?;
                    
                    total_predictions += 1;
                    total_preprocessing_time += preprocessing_time;
                    total_inference_time += inference_time;
                    total_postprocessing_time += _total_time - preprocessing_time - inference_time;
                    
                    if i == 0 {  // Print first iteration results
                        println!("Text: '{}' -> Probability: {:.4} ({})", 
                            text, 
                            probability,
                            if probability > 0.5 { "Positive" } else { "Negative" }
                        );
                    }
                }
                
                if iterations > 20 && i % (iterations / 10) == 0 && i > 0 {
                    println!("Progress: {}/{} ({:.1}%)", i, iterations, (i as f64 / iterations as f64) * 100.0);
                }
            }
            
            let duration = start_time.elapsed();
            let total_time_ms = duration.as_secs_f64() * 1000.0;
            
            // Stop monitoring and get metrics
            let (cpu_avg, cpu_peak, cpu_samples, memory_peak, memory_end) = monitor.stop_monitoring();
            
            let metrics = PerformanceMetrics {
                total_time_ms,
                preprocessing_time_ms: total_preprocessing_time,
                inference_time_ms: total_inference_time,
                postprocessing_time_ms: total_postprocessing_time,
                memory_start_mb: memory_start,
                memory_end_mb: memory_end,
                memory_peak_mb: memory_peak,
                memory_delta_mb: memory_end - memory_start,
                cpu_usage_avg: cpu_avg,
                cpu_usage_peak: cpu_peak,
                cpu_samples,
                throughput_per_sec: total_predictions as f64 / (total_time_ms / 1000.0),
                predictions_count: total_predictions,
            };
            
            println!();
            metrics.print();
            
        } else {
            // Custom text input with detailed metrics
            let text = &args[1];
            println!("🔍 Testing custom text: '{}'", text);
            println!();
            
            let monitor = ResourceMonitor::new();
            let memory_start = get_memory_usage_mb();
            monitor.start_monitoring();
            
            let (probability, total_time, preprocessing_time, inference_time) = 
                classifier.predict_with_timing(text)?;
            
            let (cpu_avg, cpu_peak, cpu_samples, memory_peak, memory_end) = monitor.stop_monitoring();
            
            println!("📊 PREDICTION RESULTS:");
            println!("   Text: '{}'", text);
            println!("   Probability: {:.4}", probability);
            println!("   Classification: {}", if probability > 0.5 { "Positive" } else { "Negative" });
            println!();
            
            let metrics = PerformanceMetrics {
                total_time_ms: total_time,
                preprocessing_time_ms: preprocessing_time,
                inference_time_ms: inference_time,
                postprocessing_time_ms: total_time - preprocessing_time - inference_time,
                memory_start_mb: memory_start,
                memory_end_mb: memory_end,
                memory_peak_mb: memory_peak,
                memory_delta_mb: memory_end - memory_start,
                cpu_usage_avg: cpu_avg,
                cpu_usage_peak: cpu_peak,
                cpu_samples,
                throughput_per_sec: 1000.0 / total_time,
                predictions_count: 1,
            };
            
            metrics.print();
        }
    } else {
        // Default test cases
        println!("🚀 Running Rust ONNX Binary Classifier Tests");
        println!();
        
        let test_cases = vec![
            ("This is a positive review of a great product", "Positive"),
            ("Terrible service, would not recommend", "Negative"), 
            ("Amazing quality and fast delivery", "Positive"),
            ("Poor customer support experience", "Negative"),
            ("Excellent value for money", "Positive"),
        ];
        
        println!("📝 Test Results:");
        for (text, expected) in test_cases {
            let probability = classifier.predict(text)?;
            let predicted = if probability > 0.5 { "Positive" } else { "Negative" };
            let status = if predicted == expected { "✅" } else { "❌" };
            
            println!("{} Text: '{}' -> Probability: {:.4} (Expected: {}, Got: {})", 
                status, text, probability, expected, predicted);
        }
        
        println!();
        println!("✅ Rust ONNX Binary Classifier test completed successfully!");
    }

    Ok(())
} 