#!/bin/bash

echo "🧪 GitHub Actions C Multiclass Classifier Test Script"
echo "====================================================="

# Check system info
echo "📋 System Information:"
echo "   OS: $(uname -s)"
echo "   Architecture: $(uname -m)"
echo "   Kernel: $(uname -r)"

# Check compiler
echo ""
echo "🔧 Compiler Information:"
gcc --version | head -1
echo "   GCC Path: $(which gcc)"

# Check dependencies
echo ""
echo "📦 Dependencies Check:"
if command -v pkg-config >/dev/null 2>&1; then
    echo "   ✅ pkg-config available"
    if pkg-config --exists libcjson; then
        echo "   ✅ cJSON library found"
        echo "      Version: $(pkg-config --modversion libcjson)"
    else
        echo "   ⚠️ cJSON library not found via pkg-config"
    fi
else
    echo "   ⚠️ pkg-config not available"
fi

# Check for cJSON header
if [ -f /usr/include/cjson/cJSON.h ] || [ -f /usr/local/include/cjson/cJSON.h ]; then
    echo "   ✅ cJSON header found"
else
    echo "   ⚠️ cJSON header not found in standard locations"
fi

# Check ONNX Runtime setup
echo ""
echo "🤖 ONNX Runtime Setup:"
if [ -d "onnxruntime-osx-universal2-1.22.0" ]; then
    echo "   ✅ ONNX Runtime directory found"
    echo "   📁 Contents:"
    ls -la onnxruntime-osx-universal2-1.22.0/
    if [ -f "onnxruntime-osx-universal2-1.22.0/lib/libonnxruntime.so" ] || [ -f "onnxruntime-osx-universal2-1.22.0/lib/libonnxruntime.dylib" ]; then
        echo "   ✅ ONNX Runtime library found"
    else
        echo "   ⚠️ ONNX Runtime library not found"
    fi
else
    echo "   ❌ ONNX Runtime directory not found"
fi

# Check model files
echo ""
echo "📄 Model Files Check:"
for file in model.onnx vocab.json scaler.json; do
    if [ -f "$file" ]; then
        echo "   ✅ $file found ($(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null) bytes)"
    else
        echo "   ⚠️ $file not found"
    fi
done

# Check for Cyrillic support
echo ""
echo "🌐 Unicode/Cyrillic Support:"
echo "   Locale: $LANG"
echo "   LC_ALL: $LC_ALL"
if locale -a | grep -i utf >/dev/null 2>&1; then
    echo "   ✅ UTF-8 locales available"
else
    echo "   ⚠️ UTF-8 locales may not be available"
fi

# Try to build
echo ""
echo "🔨 Build Test:"
if make clean >/dev/null 2>&1; then
    echo "   ✅ Clean successful"
else
    echo "   ⚠️ Clean failed"
fi

if make >/dev/null 2>&1; then
    echo "   ✅ Build successful"
    if [ -f "test_onnx_model" ]; then
        echo "   ✅ Executable created"
        echo "   📊 Executable size: $(stat -f%z "test_onnx_model" 2>/dev/null || stat -c%s "test_onnx_model" 2>/dev/null) bytes"
    else
        echo "   ❌ Executable not found"
    fi
else
    echo "   ❌ Build failed"
    echo "   📝 Build output:"
    make 2>&1 | head -20
fi

echo ""
echo "🎯 Test Summary:"
echo "   This script verifies the GitHub Actions environment setup"
echo "   for the C multiclass classifier implementation with Cyrillic support."
echo ""
echo "✅ Setup verification completed!" 