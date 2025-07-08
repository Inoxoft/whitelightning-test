CXX = g++
CXXFLAGS = -std=c++14 -O2 -Wall -Wextra
TARGET = test_onnx_model
SOURCE = test_onnx_model.cpp

# Platform detection
UNAME_S := $(shell uname -s)

# ONNX Runtime paths
ifeq ($(UNAME_S),Darwin)
    # macOS
    ONNX_ROOT = ./onnxruntime-osx-universal2-1.22.0
    INCLUDES = -I$(ONNX_ROOT)/include
    LIBS = -L$(ONNX_ROOT)/lib -lonnxruntime -lpthread
else ifeq ($(UNAME_S),Linux)
    # Linux
    ONNX_ROOT = ./onnxruntime-linux-x64-1.22.0
    INCLUDES = -I$(ONNX_ROOT)/include
    LIBS = -L$(ONNX_ROOT)/lib -lonnxruntime -lpthread
    # Add rpath for runtime library loading
    LIBS += -Wl,-rpath,$(ONNX_ROOT)/lib
else
    # Fallback for other systems
    ONNX_ROOT = ./onnxruntime
    INCLUDES = -I$(ONNX_ROOT)/include
    LIBS = -L$(ONNX_ROOT)/lib -lonnxruntime -lpthread
endif

# Check if we have a symlinked onnxruntime directory
ifneq (,$(wildcard ./onnxruntime))
    ONNX_ROOT = ./onnxruntime
    INCLUDES = -I$(ONNX_ROOT)/include
    LIBS = -L$(ONNX_ROOT)/lib -lonnxruntime -lpthread
    ifeq ($(UNAME_S),Linux)
        LIBS += -Wl,-rpath,$(ONNX_ROOT)/lib
    endif
endif

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "🔨 Building multiclass sigmoid C++ implementation..."
	@echo "📍 Platform: $(UNAME_S)"
	@echo "🔗 ONNX Runtime: $(ONNX_ROOT)"
	@if [ ! -d "$(ONNX_ROOT)" ]; then \
		echo "❌ ONNX Runtime directory not found: $(ONNX_ROOT)"; \
		echo "📋 Available directories:"; \
		ls -la | grep onnxruntime || echo "No ONNX Runtime directories found"; \
	fi
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCE) $(LIBS) -o $(TARGET)
	@echo "✅ Build completed: $(TARGET)"

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -f $(TARGET)
	@echo "✅ Clean completed"

test: $(TARGET)
	@echo "🚀 Running multiclass sigmoid C++ tests..."
	@if [ -f model.onnx ] && [ -f vocab.json ] && [ -f scaler.json ]; then \
		echo "✅ Model files found, running full tests"; \
		./$(TARGET); \
	else \
		echo "⚠️ Model files not found, running build verification"; \
		./$(TARGET); \
	fi 