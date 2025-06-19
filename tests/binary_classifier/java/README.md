# ☕ Java Binary Classification ONNX Model

This directory contains a **Java implementation** for binary text classification using ONNX Runtime. The model performs **sentiment analysis** on text input using TF-IDF vectorization and a trained neural network with comprehensive performance monitoring and cross-platform support.

## 📋 System Requirements

### Minimum Requirements
- **CPU**: x86_64 or ARM64 architecture
- **RAM**: 4GB available memory (8GB recommended)
- **Storage**: 2GB free space
- **Java**: JDK 17+ (recommended: JDK 21 LTS)
- **Maven**: 3.6+ (for dependency management)
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+

### Supported Platforms
- ✅ **Windows**: 10, 11 (x64, ARM64)
- ✅ **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 9+, Fedora 30+
- ✅ **macOS**: 10.15+ (Intel & Apple Silicon)
- ✅ **Android**: API 21+ (with Android SDK)

## 📁 Directory Structure

```
java/
├── src/
│   ├── main/
│   │   └── java/
│   │       └── com/
│   │           └── whitelightning/
│   │               └── BinaryClassifierTest.java
│   └── test/
│       └── java/
│           └── com/
│               └── whitelightning/
│                   └── SpamDetectorTest.java
├── model.onnx                 # Binary classification ONNX model
├── vocab.json                 # TF-IDF vocabulary and IDF weights
├── scaler.json                # Feature scaling parameters
├── pom.xml                    # Maven configuration
└── README.md                  # This file
```

## 🛠️ Step-by-Step Installation

### 🪟 Windows Installation

#### Step 1: Install Java JDK
```powershell
# Option A: Download OpenJDK (Recommended)
# Visit: https://adoptium.net/temurin/releases/
# Download JDK 21 LTS for Windows x64

# Option B: Install via winget
winget install EclipseAdoptium.Temurin.21.JDK

# Option C: Install via Chocolatey
choco install temurin21jdk

# Option D: Install Oracle JDK
# Visit: https://www.oracle.com/java/technologies/downloads/
# Download JDK 21 for Windows

# Verify installation
java --version
javac --version
```

#### Step 2: Set JAVA_HOME Environment Variable
```powershell
# Find Java installation path
$javaPath = (Get-Command java).Source | Split-Path | Split-Path

# Set JAVA_HOME permanently
[Environment]::SetEnvironmentVariable("JAVA_HOME", $javaPath, "Machine")

# Add to PATH
$env:PATH += ";$javaPath\bin"

# Verify
echo $env:JAVA_HOME
```

#### Step 3: Install Maven
```powershell
# Option A: Download from Apache Maven
# Visit: https://maven.apache.org/download.cgi
# Download Binary zip archive and extract to C:\maven

# Option B: Install via Chocolatey
choco install maven

# Option C: Install via winget
winget install Apache.Maven

# Set Maven environment variables
[Environment]::SetEnvironmentVariable("MAVEN_HOME", "C:\maven", "Machine")
$env:PATH += ";C:\maven\bin"

# Verify installation
mvn --version
```

#### Step 4: Create Project Directory
```powershell
# Create project directory
mkdir C:\whitelightning-java
cd C:\whitelightning-java

# Create Maven project
mvn archetype:generate -DgroupId=com.whitelightning -DartifactId=binary-classifier -DarchetypeArtifactId=maven-archetype-quickstart -DinteractiveMode=false
cd binary-classifier
```

#### Step 5: Configure Dependencies
```powershell
# Edit pom.xml (see configuration section below)
# Download dependencies
mvn clean compile

# Verify dependencies
mvn dependency:tree
```

#### Step 6: Copy Source Files & Run
```powershell
# Copy your source files to the project
# src/main/java/com/whitelightning/BinaryClassifierTest.java
# model.onnx, vocab.json, scaler.json (to project root)

# Compile and run
mvn clean compile exec:java

# Run with custom text
mvn exec:java -Dexec.args="\"This product is amazing!\""

# Run benchmark
mvn exec:java -Dexec.args="--benchmark 100"
```

---

### 🐧 Linux Installation

#### Step 1: Install Java JDK
```bash
# Ubuntu/Debian - Install OpenJDK
sudo apt update
sudo apt install -y openjdk-21-jdk

# CentOS/RHEL 8+ - Install OpenJDK
sudo dnf install -y java-21-openjdk-devel

# CentOS/RHEL 7 - Install OpenJDK
sudo yum install -y java-21-openjdk-devel

# Fedora - Install OpenJDK
sudo dnf install -y java-21-openjdk-devel

# Alternative: Install via SDKMAN
curl -s "https://get.sdkman.io" | bash
source ~/.sdkman/bin/sdkman-init.sh
sdk install java 21.0.1-tem

# Verify installation
java --version
javac --version
```

#### Step 2: Set JAVA_HOME Environment Variable
```bash
# Find Java installation path
java_path=$(readlink -f $(which java) | sed "s:/bin/java::")

# Set JAVA_HOME permanently
echo "export JAVA_HOME=$java_path" >> ~/.bashrc
echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc

# Verify
echo $JAVA_HOME
```

#### Step 3: Install Maven
```bash
# Ubuntu/Debian
sudo apt install -y maven

# CentOS/RHEL/Fedora
sudo dnf install -y maven  # or yum for CentOS 7

# Alternative: Download and install manually
wget https://archive.apache.org/dist/maven/maven-3/3.9.5/binaries/apache-maven-3.9.5-bin.tar.gz
tar -xzf apache-maven-3.9.5-bin.tar.gz
sudo mv apache-maven-3.9.5 /opt/maven

# Add to PATH
echo "export MAVEN_HOME=/opt/maven" >> ~/.bashrc
echo "export PATH=\$MAVEN_HOME/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc

# Verify installation
mvn --version
```

#### Step 4: Create Project Directory
```bash
# Create project directory
mkdir -p ~/whitelightning-java
cd ~/whitelightning-java

# Create Maven project
mvn archetype:generate -DgroupId=com.whitelightning -DartifactId=binary-classifier -DarchetypeArtifactId=maven-archetype-quickstart -DinteractiveMode=false
cd binary-classifier
```

#### Step 5: Configure Dependencies
```bash
# Edit pom.xml (see configuration section below)
# Download dependencies
mvn clean compile

# Verify dependencies
mvn dependency:tree
```

#### Step 6: Copy Source Files & Run
```bash
# Copy your source files to the project
# src/main/java/com/whitelightning/BinaryClassifierTest.java
# model.onnx, vocab.json, scaler.json (to project root)

# Compile and run
mvn clean compile exec:java

# Run with custom text
mvn exec:java -Dexec.args="\"This product is amazing!\""

# Run benchmark
mvn exec:java -Dexec.args="--benchmark 100"
```

---

### 🍎 macOS Installation

#### Step 1: Install Java JDK
```bash
# Option A: Install via Homebrew (Recommended)
brew install openjdk@21

# Link to system Java
sudo ln -sfn /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-21.jdk

# Option B: Install via SDKMAN
curl -s "https://get.sdkman.io" | bash
source ~/.sdkman/bin/sdkman-init.sh
sdk install java 21.0.1-tem

# Option C: Download from Adoptium
# Visit: https://adoptium.net/temurin/releases/
# Download JDK 21 for macOS

# Verify installation
java --version
javac --version
```

#### Step 2: Set JAVA_HOME Environment Variable
```bash
# For Homebrew installation
echo 'export JAVA_HOME="/opt/homebrew/opt/openjdk@21"' >> ~/.zshrc
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# For SDKMAN installation
# JAVA_HOME is set automatically

# Verify
echo $JAVA_HOME
```

#### Step 3: Install Maven
```bash
# Install via Homebrew
brew install maven

# Alternative: Download and install manually
wget https://archive.apache.org/dist/maven/maven-3/3.9.5/binaries/apache-maven-3.9.5-bin.tar.gz
tar -xzf apache-maven-3.9.5-bin.tar.gz
sudo mv apache-maven-3.9.5 /opt/maven

# Add to PATH (if installed manually)
echo "export MAVEN_HOME=/opt/maven" >> ~/.zshrc
echo "export PATH=\$MAVEN_HOME/bin:\$PATH" >> ~/.zshrc
source ~/.zshrc

# Verify installation
mvn --version
```

#### Step 4: Create Project Directory
```bash
# Create project directory
mkdir -p ~/whitelightning-java
cd ~/whitelightning-java

# Create Maven project
mvn archetype:generate -DgroupId=com.whitelightning -DartifactId=binary-classifier -DarchetypeArtifactId=maven-archetype-quickstart -DinteractiveMode=false
cd binary-classifier
```

#### Step 5: Configure Dependencies
```bash
# Edit pom.xml (see configuration section below)
# Download dependencies
mvn clean compile

# Verify dependencies
mvn dependency:tree
```

#### Step 6: Copy Source Files & Run
```bash
# Copy your source files to the project
# src/main/java/com/whitelightning/BinaryClassifierTest.java
# model.onnx, vocab.json, scaler.json (to project root)

# Compile and run
mvn clean compile exec:java

# Run with custom text
mvn exec:java -Dexec.args="\"This product is amazing!\""

# Run benchmark
mvn exec:java -Dexec.args="--benchmark 100"
```

## 🔧 Advanced Configuration

### pom.xml Configuration
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.whitelightning</groupId>
    <artifactId>binary-classifier</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>

    <name>Binary Classifier</name>
    <description>ONNX Binary Classifier for Java</description>

    <properties>
        <maven.compiler.source>21</maven.compiler.source>
        <maven.compiler.target>21</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <onnxruntime.version>1.22.0</onnxruntime.version>
        <jackson.version>2.15.2</jackson.version>
        <slf4j.version>2.0.7</slf4j.version>
        <junit.version>5.10.0</junit.version>
    </properties>

    <dependencies>
        <!-- ONNX Runtime -->
        <dependency>
            <groupId>com.microsoft.onnxruntime</groupId>
            <artifactId>onnxruntime</artifactId>
            <version>${onnxruntime.version}</version>
        </dependency>

        <!-- JSON Processing -->
        <dependency>
            <groupId>com.fasterxml.jackson.core</groupId>
            <artifactId>jackson-core</artifactId>
            <version>${jackson.version}</version>
        </dependency>
        <dependency>
            <groupId>com.fasterxml.jackson.core</groupId>
            <artifactId>jackson-databind</artifactId>
            <version>${jackson.version}</version>
        </dependency>

        <!-- Logging -->
        <dependency>
            <groupId>org.slf4j</groupId>
            <artifactId>slf4j-api</artifactId>
            <version>${slf4j.version}</version>
        </dependency>
        <dependency>
            <groupId>ch.qos.logback</groupId>
            <artifactId>logback-classic</artifactId>
            <version>1.4.11</version>
        </dependency>

        <!-- Testing -->
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>${junit.version}</version>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <!-- Compiler Plugin -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.11.0</version>
                <configuration>
                    <source>21</source>
                    <target>21</target>
                    <encoding>UTF-8</encoding>
                </configuration>
            </plugin>

            <!-- Exec Plugin -->
            <plugin>
                <groupId>org.codehaus.mojo</groupId>
                <artifactId>exec-maven-plugin</artifactId>
                <version>3.1.0</version>
                <configuration>
                    <mainClass>com.whitelightning.BinaryClassifierTest</mainClass>
                    <options>
                        <option>-Xmx2g</option>
                        <option>-Dfile.encoding=UTF-8</option>
                    </options>
                </configuration>
            </plugin>

            <!-- Surefire Plugin (for tests) -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.1.2</version>
                <configuration>
                    <useSystemClassLoader>false</useSystemClassLoader>
                </configuration>
            </plugin>

            <!-- Shade Plugin (for fat JAR) -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-shade-plugin</artifactId>
                <version>3.5.0</version>
                <executions>
                    <execution>
                        <phase>package</phase>
                        <goals>
                            <goal>shade</goal>
                        </goals>
                        <configuration>
                            <transformers>
                                <transformer implementation="org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
                                    <mainClass>com.whitelightning.BinaryClassifierTest</mainClass>
                                </transformer>
                            </transformers>
                        </configuration>
                    </execution>
                </executions>
            </plugin>
        </plugins>
    </build>

    <profiles>
        <!-- Development Profile -->
        <profile>
            <id>dev</id>
            <properties>
                <maven.compiler.debug>true</maven.compiler.debug>
                <maven.compiler.optimize>false</maven.compiler.optimize>
            </properties>
        </profile>

        <!-- Production Profile -->
        <profile>
            <id>prod</id>
            <properties>
                <maven.compiler.debug>false</maven.compiler.debug>
                <maven.compiler.optimize>true</maven.compiler.optimize>
            </properties>
        </profile>
    </profiles>
</project>
```

### JVM Configuration
```bash
# JVM memory settings
export MAVEN_OPTS="-Xmx4g -Xms1g -XX:+UseG1GC"

# JVM performance tuning
export JAVA_OPTS="-server -XX:+UseG1GC -XX:MaxGCPauseMillis=100 -XX:+UseStringDeduplication"

# Enable JFR profiling
export JAVA_OPTS="$JAVA_OPTS -XX:+FlightRecorder -XX:StartFlightRecording=duration=60s,filename=profile.jfr"
```

### IDE Configuration

#### IntelliJ IDEA
```bash
# Import Maven project
# File → Open → Select pom.xml → Open as Project

# Set JDK version
# File → Project Structure → Project → Project SDK: 21

# Configure Maven
# File → Settings → Build → Build Tools → Maven → Maven home path
```

#### Eclipse
```bash
# Import Maven project
# File → Import → Existing Maven Projects → Select project folder

# Set JDK version
# Project → Properties → Java Build Path → Libraries → Modulepath/Classpath → JRE System Library
```

#### VS Code
```bash
# Install Java extensions
code --install-extension vscjava.vscode-java-pack

# Open project folder
code .
```

## 🎯 Usage Examples

### Basic Usage
```bash
# Default test
mvn clean compile exec:java

# Positive sentiment
mvn exec:java -Dexec.args="\"I love this product! It's amazing!\""

# Negative sentiment
mvn exec:java -Dexec.args="\"This is terrible and disappointing.\""

# Neutral sentiment
mvn exec:java -Dexec.args="\"The product is okay, nothing special.\""
```

### Performance Benchmarking
```bash
# Quick benchmark (10 iterations)
mvn exec:java -Dexec.args="--benchmark 10"

# Comprehensive benchmark (1000 iterations)
mvn exec:java -Dexec.args="--benchmark 1000"

# Save results to file
mvn exec:java -Dexec.args="--benchmark 100" > benchmark_results.txt
```

### Building and Running JAR
```bash
# Build fat JAR
mvn clean package

# Run JAR directly
java -jar target/binary-classifier-1.0.0.jar "This product is amazing!"

# Run with JVM options
java -Xmx2g -jar target/binary-classifier-1.0.0.jar "Custom text here"
```

### Testing
```bash
# Run all tests
mvn test

# Run specific test class
mvn test -Dtest=SpamDetectorTest

# Run tests with coverage
mvn jacoco:prepare-agent test jacoco:report
```

## 🐛 Troubleshooting

### Windows Issues

**1. "'java' is not recognized as an internal or external command"**
```powershell
# Check if Java is installed
java --version

# Add Java to PATH
$javaPath = (Get-Command java -ErrorAction SilentlyContinue).Source
if ($javaPath) {
    $env:PATH += ";$(Split-Path $javaPath)"
} else {
    # Install Java first
    winget install EclipseAdoptium.Temurin.21.JDK
}
```

**2. "JAVA_HOME environment variable is not set"**
```powershell
# Find Java installation
$javaHome = (Get-Command java).Source | Split-Path | Split-Path

# Set JAVA_HOME
[Environment]::SetEnvironmentVariable("JAVA_HOME", $javaHome, "Machine")
$env:JAVA_HOME = $javaHome
```

**3. "'mvn' is not recognized as an internal or external command"**
```powershell
# Install Maven
choco install maven

# Or add Maven to PATH
$env:PATH += ";C:\maven\bin"
```

**4. "Could not find or load main class"**
```powershell
# Clean and recompile
mvn clean compile

# Check classpath
mvn dependency:build-classpath

# Run with explicit classpath
mvn exec:java -Dexec.mainClass="com.whitelightning.BinaryClassifierTest"
```

### Linux Issues

**1. "java: command not found"**
```bash
# Install OpenJDK
sudo apt install openjdk-21-jdk  # Ubuntu/Debian
sudo dnf install java-21-openjdk-devel  # CentOS/RHEL/Fedora

# Verify installation
java --version
```

**2. "JAVA_HOME is not set"**
```bash
# Find Java installation
java_home=$(readlink -f $(which java) | sed "s:/bin/java::")

# Set JAVA_HOME
echo "export JAVA_HOME=$java_home" >> ~/.bashrc
source ~/.bashrc
```

**3. "mvn: command not found"**
```bash
# Install Maven
sudo apt install maven  # Ubuntu/Debian
sudo dnf install maven  # CentOS/RHEL/Fedora

# Verify installation
mvn --version
```

**4. "Permission denied" when running Maven**
```bash
# Fix Maven permissions
sudo chown -R $USER:$USER ~/.m2

# Or create .m2 directory
mkdir -p ~/.m2
```

### macOS Issues

**1. "java: command not found"**
```bash
# Install Java via Homebrew
brew install openjdk@21

# Link to system Java
sudo ln -sfn /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-21.jdk
```

**2. "JAVA_HOME is not set"**
```bash
# Set JAVA_HOME for Homebrew installation
echo 'export JAVA_HOME="/opt/homebrew/opt/openjdk@21"' >> ~/.zshrc
source ~/.zshrc
```

**3. "Maven not found"**
```bash
# Install Maven via Homebrew
brew install maven

# Verify installation
mvn --version
```

**4. "Apple Silicon compatibility issues"**
```bash
# Check Java architecture
java -XshowSettings:properties -version 2>&1 | grep "os.arch"

# Install native ARM64 Java
brew install openjdk@21
```

## 📊 Expected Output

```
🤖 ONNX BINARY CLASSIFIER - JAVA IMPLEMENTATION
===============================================
🔄 Processing: "This product is amazing!"

💻 SYSTEM INFORMATION:
   Platform: macOS 14.0 (Darwin)
   Processor: aarch64
   CPU Cores: 12 physical, 12 logical
   Total Memory: 32.0 GB
   Runtime: Java Implementation
   Java Version: 21.0.1
   ONNX Runtime Version: 1.22.0

📊 SENTIMENT ANALYSIS RESULTS:
   🏆 Predicted Sentiment: Positive ✅
   📈 Confidence: 87.34% (0.8734)
   📝 Input Text: "This product is amazing!"

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 45.23ms
   ┣━ Preprocessing: 12.45ms (27.5%)
   ┣━ Model Inference: 28.67ms (63.4%)
   ┗━ Postprocessing: 4.11ms (9.1%)

🚀 THROUGHPUT:
   Texts per second: 22.1

💾 RESOURCE USAGE:
   Memory Start: 156.78 MB
   Memory End: 162.34 MB
   Memory Delta: +5.56 MB
   CPU Usage: 15.2% avg, 45.7% peak (12 samples)

🎯 PERFORMANCE RATING: ✅ GOOD
   (45.2ms total - Target: <100ms)
```

## 🚀 Features

- **Enterprise Ready**: Robust, scalable, and maintainable code
- **Comprehensive Monitoring**: Detailed timing and resource tracking
- **Cross-Platform**: Runs on Windows, macOS, Linux, and Android
- **Maven Integration**: Modern dependency management and build system
- **JUnit Testing**: Comprehensive test suite with coverage reporting
- **Performance Benchmarking**: Statistical analysis with multiple runs
- **Memory Efficient**: Optimized for JVM garbage collection

## 🎯 Performance Characteristics

- **Total Time**: ~45ms (good for enterprise applications)
- **Memory Usage**: Moderate (~6MB additional)
- **JVM Optimizations**: G1GC, string deduplication, compressed OOPs
- **Thread Safe**: Safe for concurrent use in multi-threaded applications
- **Scalable**: Suitable for high-throughput server applications

## 🔧 Technical Details

### Model Architecture
- **Type**: Binary Classification
- **Input**: Text string
- **Features**: TF-IDF vectors (5000 dimensions)
- **Output**: Probability [0.0 - 1.0]
- **Threshold**: >0.5 = Positive, ≤0.5 = Negative

### Processing Pipeline
1. **Text Tokenization**: Split text into words and convert to lowercase
2. **TF-IDF Vectorization**: Convert to 5000-dimensional feature vector
3. **Feature Scaling**: Apply mean normalization and standard scaling
4. **Model Inference**: ONNX Runtime execution
5. **Post-processing**: Probability interpretation

### Java-Specific Optimizations
- **JIT Compilation**: HotSpot JVM optimizations
- **Garbage Collection**: G1GC for low-latency performance
- **String Interning**: Efficient string handling
- **NIO Buffers**: Efficient memory management for large arrays
- **Parallel Streams**: Multi-threaded processing when beneficial

## 🚀 Integration Example

```java
package com.whitelightning;

import ai.onnxruntime.*;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class BinaryClassifier {
    private final OrtEnvironment environment;
    private final OrtSession session;
    private final Map<String, Integer> vocab;
    private final float[] idfWeights;
    private final float[] meanValues;
    private final float[] scaleValues;
    
    public BinaryClassifier(String modelPath, String vocabPath, String scalerPath) 
            throws OrtException, IOException {
        
        // Initialize ONNX Runtime
        this.environment = OrtEnvironment.getEnvironment();
        this.session = environment.createSession(modelPath);
        
        // Load vocabulary and preprocessing data
        ObjectMapper mapper = new ObjectMapper();
        
        // Load vocabulary
        JsonNode vocabNode = mapper.readTree(new File(vocabPath));
        this.vocab = new HashMap<>();
        vocabNode.get("vocab").fields().forEachRemaining(entry -> 
            vocab.put(entry.getKey(), entry.getValue().asInt()));
        
        // Load IDF weights
        JsonNode idfArray = vocabNode.get("idf");
        this.idfWeights = new float[idfArray.size()];
        for (int i = 0; i < idfArray.size(); i++) {
            idfWeights[i] = (float) idfArray.get(i).asDouble();
        }
        
        // Load scaler data
        JsonNode scalerNode = mapper.readTree(new File(scalerPath));
        JsonNode meanArray = scalerNode.get("mean");
        JsonNode scaleArray = scalerNode.get("scale");
        
        this.meanValues = new float[meanArray.size()];
        this.scaleValues = new float[scaleArray.size()];
        
        for (int i = 0; i < meanArray.size(); i++) {
            meanValues[i] = (float) meanArray.get(i).asDouble();
            scaleValues[i] = (float) scaleArray.get(i).asDouble();
        }
    }
    
    public PredictionResult predict(String text) throws OrtException {
        long startTime = System.nanoTime();
        
        // Preprocess text to TF-IDF features
        float[] features = preprocessText(text);
        
        // Create input tensor
        float[][] input = {features};
        OnnxTensor inputTensor = OnnxTensor.createTensor(environment, input);
        
        // Run inference
        try (OrtSession.Result result = session.run(Map.of("input", inputTensor))) {
            float[][] output = (float[][]) result.get(0).getValue();
            float probability = output[0][0];
            
            long endTime = System.nanoTime();
            double processingTime = (endTime - startTime) / 1_000_000.0; // Convert to milliseconds
            
            String sentiment = probability > 0.5 ? "Positive" : "Negative";
            float confidence = probability > 0.5 ? probability : (1.0f - probability);
            
            return new PredictionResult(probability, sentiment, confidence, processingTime);
        }
    }
    
    private float[] preprocessText(String text) {
        float[] features = new float[5000];
        
        // Tokenize text
        String[] tokens = text.toLowerCase().split("\\s+");
        
        // Calculate TF-IDF
        for (String token : tokens) {
            Integer index = vocab.get(token);
            if (index != null && index < features.length) {
                features[index] += 1.0f * idfWeights[index];
            }
        }
        
        // Apply scaling
        for (int i = 0; i < features.length; i++) {
            features[i] = (features[i] - meanValues[i]) / scaleValues[i];
        }
        
        return features;
    }
    
    public void close() throws OrtException {
        session.close();
        environment.close();
    }
    
    public static class PredictionResult {
        public final float probability;
        public final String sentiment;
        public final float confidence;
        public final double processingTime;
        
        public PredictionResult(float probability, String sentiment, 
                              float confidence, double processingTime) {
            this.probability = probability;
            this.sentiment = sentiment;
            this.confidence = confidence;
            this.processingTime = processingTime;
        }
    }
    
    // Usage example
    public static void main(String[] args) {
        try (BinaryClassifier classifier = new BinaryClassifier(
                "model.onnx", "vocab.json", "scaler.json")) {
            
            String text = args.length > 0 ? args[0] : "This product is amazing!";
            PredictionResult result = classifier.predict(text);
            
            System.out.printf("🎯 PREDICTED SENTIMENT: %s%n", result.sentiment);
            System.out.printf("📈 Confidence: %.2f%%%n", result.confidence * 100);
            System.out.printf("⏱️  Processing time: %.1fms%n", result.processingTime);
            
        } catch (Exception e) {
            System.err.println("❌ Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
```

## 📈 Performance Tips

1. **JVM Tuning**: Use G1GC for better performance: `-XX:+UseG1GC`
2. **Memory Settings**: Allocate sufficient heap: `-Xmx4g -Xms1g`
3. **Warm-up**: Run several predictions to warm up the JIT compiler
4. **Connection Pooling**: Reuse ONNX sessions for better performance
5. **Profiling**: Use JFR for performance analysis: `-XX:+FlightRecorder`

## 🏗️ CI/CD Integration

### Maven Profiles
```xml
<profiles>
    <profile>
        <id>ci</id>
        <properties>
            <maven.test.skip>false</maven.test.skip>
            <maven.javadoc.skip>true</maven.javadoc.skip>
        </properties>
    </profile>
</profiles>
```

### GitHub Actions Integration
```yaml
- name: Test Java Binary Classifier
  run: |
    cd Test/binary_classifier/java
    mvn clean compile exec:java
```

### Docker Deployment
```dockerfile
FROM openjdk:21-jre-slim

WORKDIR /app
COPY target/binary-classifier-1.0.0.jar app.jar
COPY model.onnx vocab.json scaler.json ./

EXPOSE 8080
CMD ["java", "-jar", "app.jar"]
```

## 📝 Notes

- **Enterprise Grade**: Excellent for large-scale applications
- **Mature Ecosystem**: Rich libraries and tooling support
- **JVM Performance**: Good performance with JIT optimizations
- **Memory Management**: Automatic garbage collection

### When to Use Java Implementation
- ✅ **Enterprise Applications**: Large-scale, mission-critical systems
- ✅ **Spring Boot**: Microservices and web applications
- ✅ **Android Development**: Mobile applications
- ✅ **Big Data**: Integration with Hadoop, Spark, Kafka
- ✅ **Team Familiarity**: Existing Java expertise
- ❌ **Startup Time**: Slower startup compared to native languages
- ❌ **Memory Usage**: Higher memory footprint than native implementations

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 