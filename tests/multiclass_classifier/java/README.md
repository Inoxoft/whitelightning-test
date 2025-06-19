# ☕ Java Multiclass Classification ONNX Model

🤖 **ONNX Multiclass Classifier for News Category Classification**

This Java implementation provides comprehensive news text classification using ONNX Runtime with advanced performance monitoring, system information collection, and enterprise-grade cross-platform support.

## 📋 System Requirements

### Minimum Requirements
- **Java**: JDK 17+ (LTS recommended)
- **Maven**: 3.6+ for dependency management
- **RAM**: 4GB available memory
- **Storage**: 200MB free space
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+

### Recommended Versions
- **Java**: OpenJDK 17.0.7 or 21.0.1 LTS
- **Maven**: 3.9.4+
- **ONNX Runtime**: 1.22.0

### Supported Platforms
- ✅ **Windows**: 10, 11 (x64, ARM64)
- ✅ **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 9+, Fedora 30+
- ✅ **macOS**: 10.15+ (Intel & Apple Silicon)

## 📁 Directory Structure

```
java/
├── src/
│   ├── main/
│   │   └── java/
│   │       └── com/
│   │           └── whitelightning/
│   │               └── MulticlassClassifierTest.java
│   └── test/
│       └── java/
│           └── com/
│               └── whitelightning/
│                   └── SpamDetectorTest.java
├── target/                    # Compiled classes (generated)
├── pom.xml                    # Maven configuration
├── model.onnx                 # Multiclass classification ONNX model
├── vocab.json                 # Token vocabulary mapping
├── scaler.json                # Label mapping for categories
└── README.md                  # This file
```

## 🛠️ Step-by-Step Installation

### 🪟 Windows Installation

#### Step 1: Install Java JDK
```powershell
# Option A: Download from official website (Recommended)
# Visit: https://adoptium.net/temurin/releases/
# Download Eclipse Temurin JDK 17 or 21 (Windows x64)

# Option B: Using winget
winget install EclipseAdoptium.Temurin.17.JDK

# Option C: Using Chocolatey
choco install openjdk17

# Option D: Using Scoop
scoop install openjdk17

# Verify installation
java -version
javac -version
```

#### Step 2: Install Maven
```powershell
# Option A: Download from official website
# Visit: https://maven.apache.org/download.cgi
# Download Binary zip archive and extract to C:\apache-maven-3.9.4

# Option B: Using winget
winget install Apache.Maven

# Option C: Using Chocolatey
choco install maven

# Option D: Using Scoop
scoop install maven

# Set environment variables
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.7.7-hotspot"
$env:MAVEN_HOME = "C:\apache-maven-3.9.4"
$env:PATH += ";$env:JAVA_HOME\bin;$env:MAVEN_HOME\bin"

# Make permanent
[Environment]::SetEnvironmentVariable("JAVA_HOME", $env:JAVA_HOME, "User")
[Environment]::SetEnvironmentVariable("MAVEN_HOME", $env:MAVEN_HOME, "User")
[Environment]::SetEnvironmentVariable("PATH", $env:PATH, "User")

# Verify installation
mvn -version
```

#### Step 3: Create Project Directory
```powershell
# Create project directory
mkdir C:\whitelightning-java-multiclass
cd C:\whitelightning-java-multiclass

# Copy project files
# pom.xml, src/, model.onnx, vocab.json, scaler.json
```

#### Step 4: Build and Run
```powershell
# Clean and compile
mvn clean compile

# Run with default test texts
mvn exec:java

# Run with custom text
mvn exec:java -Dexec.args="`"France defeats Argentina in World Cup final`""

# Run performance benchmark
mvn exec:java -Dexec.args="--benchmark 100"

# Run tests
mvn test
```

---

### 🐧 Linux Installation

#### Step 1: Install Java JDK
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y openjdk-17-jdk

# CentOS/RHEL 8+
sudo dnf install -y java-17-openjdk java-17-openjdk-devel

# CentOS/RHEL 7
sudo yum install -y java-17-openjdk java-17-openjdk-devel

# Fedora
sudo dnf install -y java-17-openjdk java-17-openjdk-devel

# Alternative: Using SDKMAN
curl -s "https://get.sdkman.io" | bash
source ~/.bashrc
sdk install java 17.0.7-tem

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64  # Ubuntu/Debian
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk        # CentOS/RHEL/Fedora
echo "export JAVA_HOME=$JAVA_HOME" >> ~/.bashrc
source ~/.bashrc

# Verify installation
java -version
javac -version
```

#### Step 2: Install Maven
```bash
# Ubuntu/Debian
sudo apt install -y maven

# CentOS/RHEL 8+
sudo dnf install -y maven

# CentOS/RHEL 7
sudo yum install -y maven

# Fedora
sudo dnf install -y maven

# Alternative: Download manually
wget https://archive.apache.org/dist/maven/maven-3/3.9.4/binaries/apache-maven-3.9.4-bin.tar.gz
tar -xzf apache-maven-3.9.4-bin.tar.gz
sudo mv apache-maven-3.9.4 /opt/maven

# Set environment variables
export MAVEN_HOME=/opt/maven
export PATH=$MAVEN_HOME/bin:$PATH
echo "export MAVEN_HOME=/opt/maven" >> ~/.bashrc
echo "export PATH=\$MAVEN_HOME/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc

# Verify installation
mvn -version
```

#### Step 3: Create Project Directory
```bash
# Create project directory
mkdir -p ~/whitelightning-java-multiclass
cd ~/whitelightning-java-multiclass

# Copy project files
# pom.xml, src/, model.onnx, vocab.json, scaler.json
```

#### Step 4: Build and Run
```bash
# Clean and compile
mvn clean compile

# Run with default test texts
mvn exec:java

# Run with custom text
mvn exec:java -Dexec.args="\"France defeats Argentina in World Cup final\""

# Run performance benchmark
mvn exec:java -Dexec.args="--benchmark 100"

# Run tests
mvn test
```

---

### 🍎 macOS Installation

#### Step 1: Install Java JDK
```bash
# Option A: Using Homebrew (Recommended)
# Install Homebrew first if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH (Apple Silicon)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc

# Add to PATH (Intel)
echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc

# Install OpenJDK
brew install openjdk@17

# Link for system Java wrapper
sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk

# Option B: Download from official website
# Visit: https://adoptium.net/temurin/releases/
# Download Eclipse Temurin JDK 17 (macOS)

# Option C: Using SDKMAN
curl -s "https://get.sdkman.io" | bash
source ~/.zshrc
sdk install java 17.0.7-tem

# Set JAVA_HOME
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home  # Apple Silicon
export JAVA_HOME=/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home    # Intel
echo "export JAVA_HOME=$JAVA_HOME" >> ~/.zshrc
source ~/.zshrc

# Verify installation
java -version
javac -version
```

#### Step 2: Install Maven
```bash
# Using Homebrew
brew install maven

# Alternative: Download manually
wget https://archive.apache.org/dist/maven/maven-3/3.9.4/binaries/apache-maven-3.9.4-bin.tar.gz
tar -xzf apache-maven-3.9.4-bin.tar.gz
sudo mv apache-maven-3.9.4 /opt/maven

# Set environment variables (if installed manually)
export MAVEN_HOME=/opt/maven
export PATH=$MAVEN_HOME/bin:$PATH
echo "export MAVEN_HOME=/opt/maven" >> ~/.zshrc
echo "export PATH=\$MAVEN_HOME/bin:\$PATH" >> ~/.zshrc
source ~/.zshrc

# Verify installation
mvn -version
```

#### Step 3: Create Project Directory
```bash
# Create project directory
mkdir -p ~/whitelightning-java-multiclass
cd ~/whitelightning-java-multiclass

# Copy project files
# pom.xml, src/, model.onnx, vocab.json, scaler.json
```

#### Step 4: Build and Run
```bash
# Clean and compile
mvn clean compile

# Run with default test texts
mvn exec:java

# Run with custom text
mvn exec:java -Dexec.args="\"France defeats Argentina in World Cup final\""

# Run performance benchmark
mvn exec:java -Dexec.args="--benchmark 100"

# Run tests
mvn test
```

## 🔧 Advanced Configuration

### pom.xml Template
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.whitelightning</groupId>
    <artifactId>onnx-multiclass-classifier</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>

    <name>ONNX Multiclass Classifier</name>
    <description>Java implementation for ONNX multiclass text classification</description>

    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        
        <!-- Dependency versions -->
        <onnxruntime.version>1.22.0</onnxruntime.version>
        <jackson.version>2.15.2</jackson.version>
        <slf4j.version>2.0.7</slf4j.version>
        <logback.version>1.4.8</logback.version>
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
            <version>${logback.version}</version>
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
                    <source>17</source>
                    <target>17</target>
                    <encoding>UTF-8</encoding>
                </configuration>
            </plugin>

            <!-- Exec Plugin -->
            <plugin>
                <groupId>org.codehaus.mojo</groupId>
                <artifactId>exec-maven-plugin</artifactId>
                <version>3.1.0</version>
                <configuration>
                    <mainClass>com.whitelightning.MulticlassClassifierTest</mainClass>
                    <options>
                        <option>-Xmx2g</option>
                        <option>-Dfile.encoding=UTF-8</option>
                    </options>
                </configuration>
            </plugin>

            <!-- Surefire Plugin (Testing) -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.1.2</version>
                <configuration>
                    <includes>
                        <include>**/*Test.java</include>
                    </includes>
                </configuration>
            </plugin>

            <!-- JAR Plugin -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-jar-plugin</artifactId>
                <version>3.3.0</version>
                <configuration>
                    <archive>
                        <manifest>
                            <mainClass>com.whitelightning.MulticlassClassifierTest</mainClass>
                        </manifest>
                    </archive>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

### Environment Variables
```bash
# Linux/macOS
export JAVA_HOME=/path/to/jdk
export MAVEN_HOME=/path/to/maven
export PATH=$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH

# Windows (PowerShell)
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.7.7-hotspot"
$env:MAVEN_HOME = "C:\apache-maven-3.9.4"
$env:PATH += ";$env:JAVA_HOME\bin;$env:MAVEN_HOME\bin"
```

### JVM Performance Tuning
```bash
# Increase heap size
mvn exec:java -Dexec.args="-Xmx4g -Xms1g"

# Enable G1 garbage collector
mvn exec:java -Dexec.args="-XX:+UseG1GC -XX:MaxGCPauseMillis=200"

# Enable JIT compiler optimizations
mvn exec:java -Dexec.args="-XX:+TieredCompilation -XX:TieredStopAtLevel=4"
```

## 🎯 Usage Examples

### Basic Usage
```bash
# Default test suite
mvn exec:java

# Sports classification
mvn exec:java -Dexec.args="\"France Defeats Argentina in Thrilling World Cup Final\""

# Health classification
mvn exec:java -Dexec.args="\"New Healthcare Policy Announced by Government\""

# Politics classification
mvn exec:java -Dexec.args="\"President Signs New Legislation on Healthcare Reform\""

# Business classification
mvn exec:java -Dexec.args="\"Stock Market Reaches Record High\""

# Science classification
mvn exec:java -Dexec.args="\"Scientists Discover New Species in Amazon\""
```

### Performance Benchmarking
```bash
# Quick benchmark (10 iterations)
mvn exec:java -Dexec.args="--benchmark 10"

# Standard benchmark (100 iterations)
mvn exec:java -Dexec.args="--benchmark 100"

# Comprehensive benchmark (1000 iterations)
mvn exec:java -Dexec.args="--benchmark 1000"
```

### Testing and Development
```bash
# Run unit tests
mvn test

# Run specific test class
mvn test -Dtest=SpamDetectorTest

# Generate test reports
mvn surefire-report:report

# Package as JAR
mvn package

# Run packaged JAR
java -jar target/onnx-multiclass-classifier-1.0.0.jar
```

## 🐛 Troubleshooting

### Windows Issues

**1. "JAVA_HOME is not set"**
```powershell
# Set JAVA_HOME environment variable
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.7.7-hotspot"
[Environment]::SetEnvironmentVariable("JAVA_HOME", $env:JAVA_HOME, "User")
```

**2. "'mvn' is not recognized as an internal or external command"**
```powershell
# Add Maven to PATH
$env:PATH += ";C:\apache-maven-3.9.4\bin"
[Environment]::SetEnvironmentVariable("PATH", $env:PATH, "User")
```

**3. "The JAVA_HOME environment variable is not defined correctly"**
```powershell
# Ensure JAVA_HOME points to JDK, not JRE
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.7.7-hotspot"
```

**4. "Could not find or load main class"**
```powershell
# Ensure proper compilation
mvn clean compile

# Check classpath
mvn exec:java -Dexec.mainClass="com.whitelightning.MulticlassClassifierTest"
```

### Linux Issues

**1. "java: command not found"**
```bash
# Install OpenJDK
sudo apt install openjdk-17-jdk  # Ubuntu/Debian
sudo dnf install java-17-openjdk-devel  # CentOS/RHEL/Fedora

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
echo "export JAVA_HOME=$JAVA_HOME" >> ~/.bashrc
```

**2. "mvn: command not found"**
```bash
# Install Maven
sudo apt install maven  # Ubuntu/Debian
sudo dnf install maven  # CentOS/RHEL/Fedora
```

**3. "Permission denied" errors**
```bash
# Fix file permissions
chmod +x target/classes/com/whitelightning/MulticlassClassifierTest.class

# Or run with sudo if needed
sudo mvn exec:java
```

**4. "OutOfMemoryError: Java heap space"**
```bash
# Increase heap size
export MAVEN_OPTS="-Xmx4g -Xms1g"
mvn exec:java
```

### macOS Issues

**1. "Unable to find a $JAVA_HOME at /usr"**
```bash
# Set correct JAVA_HOME for Homebrew installation
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
echo "export JAVA_HOME=$JAVA_HOME" >> ~/.zshrc
```

**2. "Apple Silicon compatibility issues"**
```bash
# Use native ARM64 JDK
brew install openjdk@17

# Or use Rosetta for Intel JDK
arch -x86_64 java -version
```

**3. "Certificate verification failed"**
```bash
# Update certificates
/usr/libexec/java_home -V

# Or use specific JDK
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
```

**4. "dyld: Library not loaded"**
```bash
# Check library paths
otool -L $JAVA_HOME/lib/server/libjvm.dylib

# Reinstall JDK if needed
brew reinstall openjdk@17
```

## 📊 Expected Output

```
🤖 ONNX MULTICLASS CLASSIFIER - JAVA IMPLEMENTATION
==================================================

🔄 Processing: France Defeats Argentina in Thrilling World Cup Final

💻 SYSTEM INFORMATION:
   Platform: Linux
   Processor: amd64
   CPU Cores: 4
   Total Memory: 8.0 GB
   Runtime: Java Implementation
   Java Version: 17.0.7
   ONNX Runtime Version: 1.22.0

📊 MULTICLASS CLASSIFICATION RESULTS:
   🏆 Predicted Category: sports
   📈 Confidence: 94.67% (0.9467)
   📝 Input Text: "France Defeats Argentina in Thrilling World Cup Final"
   📋 All Class Probabilities:
      health: 0.0123 (1.2%)
      politics: 0.0234 (2.3%)
      sports: 0.9467 (94.7%)
      world: 0.0176 (1.8%)

📈 PERFORMANCE SUMMARY:
   Total Processing Time: 38.45ms
   ┣━ Preprocessing: 8.23ms (21.4%)
   ┣━ Model Inference: 25.67ms (66.8%)
   ┗━ Postprocessing: 4.55ms (11.8%)

🚀 THROUGHPUT:
   Texts per second: 26.0

💾 RESOURCE USAGE:
   Memory Start: 142.34 MB
   Memory End: 147.89 MB
   Memory Delta: +5.55 MB
   CPU Usage: 12.8% avg, 38.9% peak (10 samples)

🎯 PERFORMANCE RATING: 🚀 EXCELLENT
   (38.5ms total - Target: <100ms)
```

## 🚀 Features

- **Token-based Text Preprocessing** - Converts text to 30-token sequences with vocabulary mapping
- **ONNX Runtime Integration** - Uses Microsoft ONNX Runtime for Java
- **Dynamic Input Detection** - Automatically detects model input/output names
- **Comprehensive Performance Monitoring** - System info, timing breakdown, CPU/memory tracking
- **News Classification** - 4-class classification (health, politics, sports, world)
- **Benchmarking Mode** - Performance testing with detailed statistics

## 🎯 Performance Characteristics

- **Total Time**: ~38ms (excellent performance)
- **Memory Usage**: Moderate (~5.5MB additional)
- **CPU Efficiency**: Good CPU usage with high throughput
- **Platform**: Consistent performance across operating systems
- **Scalability**: Excellent for enterprise applications

## 🔧 Technical Details

### Model Architecture
- **Input Format**: [1, 30] (batch_size=1, sequence_length=30)
- **Type**: Int32
- **Preprocessing**: Tokenization with vocabulary mapping and zero-padding
- **Output Format**: [4] (4 class probabilities)
- **Type**: Float32
- **Classes**: health, politics, sports, world

### Text Preprocessing
1. **Tokenization** - Split text into words and convert to lowercase
2. **Vocabulary Mapping** - Convert words to token IDs using vocab.json
3. **OOV Handling** - Unknown words mapped to `<OOV>` token (ID: 1)
4. **Padding** - Sequences padded/truncated to exactly 30 tokens with zeros

## 🏗️ CI/CD Integration

The implementation is designed for GitHub Actions:

- **Safe CI Execution** - Graceful handling when model files are missing
- **Build Verification** - Confirms compilation and startup
- **Artifact Upload** - Saves compiled classes and logs
- **Custom Text Testing** - Supports parameterized text input

## 📝 Notes

- **Enterprise Ready**: Robust error handling and logging
- **Performance Optimized**: JVM tuning and efficient memory usage
- **Cross-Platform**: Consistent behavior across operating systems
- **Developer Friendly**: Maven integration and comprehensive testing

### When to Use Java Implementation
- ✅ **Enterprise Applications**: Large-scale production systems
- ✅ **Spring Boot**: Integration with Spring framework
- ✅ **Microservices**: Containerized classification services
- ✅ **Big Data**: Integration with Hadoop, Spark ecosystems
- ✅ **Legacy Systems**: Integration with existing Java infrastructure
- ✅ **Team Expertise**: Java development teams
- ❌ **Resource Constrained**: Higher memory usage than C/Rust
- ❌ **Startup Time**: JVM startup overhead

---

*For more information, see the main [README.md](../../../README.md) in the project root.* 