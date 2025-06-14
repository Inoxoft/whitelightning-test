CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
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

.PHONY: all clean test help

all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "🔨 Building multiclass classifier C++ implementation..."
	@echo "📍 Platform: $(UNAME_S)"
	@echo "🔗 ONNX Runtime: $(ONNX_ROOT)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCE) $(LIBS) -o $(TARGET)
	@echo "✅ Build completed: $(TARGET)"

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -f $(TARGET)
	@echo "✅ Clean completed"

test: $(TARGET)
	@echo "🚀 Running multiclass classifier tests..."
	./$(TARGET)

benchmark: $(TARGET)
	@echo "📊 Running performance benchmark..."
	./$(TARGET) --benchmark 100

help:
	@echo "🤖 Multiclass Classifier C++ Build System"
	@echo "=========================================="
	@echo "Available targets:"
	@echo "  all       - Build the executable (default)"
	@echo "  clean     - Remove build artifacts"
	@echo "  test      - Build and run tests"
	@echo "  benchmark - Build and run performance benchmark"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make                    # Build the project"
	@echo "  make test              # Build and test"
	@echo "  make benchmark         # Run performance tests"
	@echo "  ./$(TARGET) \"Custom text\"  # Test with custom text" 