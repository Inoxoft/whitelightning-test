use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;
use serde_json::Value;
use ort::{Environment, SessionBuilder, Value as OrtValue, tensor::InputTensor};
use regex::Regex;

#[derive(Debug)]
struct VectorizerData {
    vocabulary: HashMap<String, usize>,
    idf: Vec<f64>,
    max_features: usize,
}

impl VectorizerData {
    fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let json: Value = serde_json::from_reader(reader)?;
        
        let vocabulary = if let Some(vocab) = json.get("vocabulary") {
            serde_json::from_value(vocab.clone())?
        } else if let Some(vocab) = json.get("vocab") {
            serde_json::from_value(vocab.clone())?
        } else {
            return Err("No vocabulary found in JSON".into());
        };
        
        let idf: Vec<f64> = serde_json::from_value(json["idf"].clone())?;
        let max_features = json.get("max_features").and_then(|v| v.as_u64()).unwrap_or(5000) as usize;
        
        Ok(VectorizerData {
            vocabulary,
            idf,
            max_features,
        })
    }
}

fn load_classes(path: &str) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let classes: HashMap<String, String> = serde_json::from_reader(reader)?;
    Ok(classes)
}

fn preprocess_text(text: &str, vectorizer: &VectorizerData) -> Vec<f32> {
    let start = Instant::now();
    
    // Tokenize text (match sklearn's pattern)
    let token_regex = Regex::new(r"\b\w\w+\b").unwrap();
    let text_lower = text.to_lowercase();
    let tokens: Vec<&str> = token_regex.find_iter(&text_lower).map(|m| m.as_str()).collect();
    
    println!("📊 Tokens found: {}, First 10: {:?}", tokens.len(), &tokens[..tokens.len().min(10)]);
    
    // Count term frequencies
    let mut term_counts = HashMap::new();
    for token in &tokens {
        *term_counts.entry(token.to_string()).or_insert(0) += 1;
    }
    
    // Create TF-IDF vector
    let mut vector = vec![0.0f32; vectorizer.max_features];
    let mut found_in_vocab = 0;
    
    // Apply TF-IDF
    for (term, count) in &term_counts {
        if let Some(&index) = vectorizer.vocabulary.get(term) {
            if index < vectorizer.max_features {
                vector[index] = (*count as f32) * (vectorizer.idf[index] as f32);
                found_in_vocab += 1;
            }
        }
    }
    
    println!("📊 Found {} terms in vocabulary out of {} total tokens", found_in_vocab, tokens.len());
    
    // L2 normalization
    let norm: f32 = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    
    let duration = start.elapsed();
    println!("📊 TF-IDF: {} non-zero, norm: {:.4}", found_in_vocab, norm);
    println!("📊 Preprocessing completed in {:.2}ms", duration.as_millis());
    
    vector
}

fn run_inference(session: &ort::Session, vector: Vec<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let start = Instant::now();
    
    // Create input tensor
    let input_tensor = InputTensor::from_array(([1, vector.len()], vector.into_boxed_slice()));
    
    // Run inference
    let outputs = session.run([input_tensor])?;
    
    // Get output
    let output = outputs[0].try_extract::<f32>()?;
    let predictions = output.view().to_slice()?.to_vec();
    
    let duration = start.elapsed();
    println!("📊 Inference completed in {:.2}ms", duration.as_millis());
    
    Ok(predictions)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let test_text = if args.len() > 1 {
        &args[1]
    } else {
        "I'm about to give birth, and I'm terrified. What if something goes wrong? What if I can't handle the pain? Received an unexpected compliment at work today. Small moments of happiness can make a big difference."
    };
    
    println!("🤖 ONNX MULTICLASS SIGMOID CLASSIFIER - RUST IMPLEMENTATION");
    println!("{}", "=".repeat(62));
    println!("🔄 Processing: {}", test_text);
    println!();
    
    // System information
    println!("💻 SYSTEM INFORMATION:");
    println!("   Platform: Rust");
    println!("   CPU Cores: {}", num_cpus::get());
    println!("   Runtime: Rust {}", env!("RUSTC_VERSION"));
    println!();
    
    let total_start = Instant::now();
    
    // Load components
    println!("🔧 Loading components...");
    
    let environment = Environment::builder().with_name("MulticlassSigmoidTest").build()?;
    let session = SessionBuilder::new(&environment)?.with_model_from_file("model.onnx")?;
    println!("✅ ONNX model loaded");
    
    let vectorizer = VectorizerData::load("vocab.json")?;
    println!("✅ Vectorizer loaded (vocab: {} words)", vectorizer.vocabulary.len());
    
    let classes = load_classes("scaler.json")?;
    println!("✅ Classes loaded: {}", classes.values().cloned().collect::<Vec<_>>().join(", "));
    println!();
    
    // Preprocess text
    let vector = preprocess_text(test_text, &vectorizer);
    println!("📊 TF-IDF shape: [1, {}]", vector.len());
    println!();
    
    // Run inference
    let predictions = run_inference(&session, vector)?;
    
    // Display results
    println!("📊 EMOTION ANALYSIS RESULTS:");
    let mut emotion_results = Vec::new();
    
    for (i, &probability) in predictions.iter().enumerate() {
        let class_name = classes.get(&i.to_string()).cloned().unwrap_or_else(|| format!("Class {}", i));
        emotion_results.push((class_name.clone(), probability));
        println!("   {}: {:.3}", class_name, probability);
    }
    
    // Find dominant emotion
    let dominant_emotion = emotion_results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    println!("   🏆 Dominant Emotion: {} ({:.3})", dominant_emotion.0, dominant_emotion.1);
    
    println!("   📝 Input Text: \"{}\"", test_text);
    println!();
    
    // Performance metrics
    let total_time = total_start.elapsed();
    let total_ms = total_time.as_millis();
    
    println!("📈 PERFORMANCE SUMMARY:");
    println!("   Total Processing Time: {}ms", total_ms);
    println!();
    
    // Throughput
    let throughput = 1000.0 / total_ms as f64;
    println!("🚀 THROUGHPUT:");
    println!("   Texts per second: {:.1}", throughput);
    println!();
    
    // Performance rating
    let rating = if total_ms < 50 {
        "🚀 EXCELLENT"
    } else if total_ms < 100 {
        "✅ GOOD"
    } else if total_ms < 500 {
        "⚠️ ACCEPTABLE"
    } else {
        "🐌 SLOW"
    };
    
    println!("🎯 PERFORMANCE RATING: {}", rating);
    println!("   ({}ms total - Target: <100ms)", total_ms);
    
    Ok(())
} 