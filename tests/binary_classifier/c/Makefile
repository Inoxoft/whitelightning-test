CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99

# Detect platform and set ONNX Runtime paths
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    ONNX_DIR = ./onnxruntime-osx-universal2-1.22.0
    PLATFORM_LIBS = -framework CoreFoundation -framework IOKit
else
    # Linux - check for ONNX Runtime in different locations
    ifneq (,$(wildcard ./onnxruntime-linux-x64-1.22.0))
        ONNX_DIR = ./onnxruntime-linux-x64-1.22.0
    else ifneq (,$(wildcard ../../../onnxruntime-linux-x64-1.22.0))
        ONNX_DIR = ../../../onnxruntime-linux-x64-1.22.0
    else
        ONNX_DIR = ./onnxruntime-linux-x64-1.22.0
    endif
    PLATFORM_LIBS = -ldl
    CFLAGS += -D_GNU_SOURCE
endif

INCLUDES = -I$(ONNX_DIR)/include
LIBS = -L$(ONNX_DIR)/lib -lonnxruntime -lcjson -lpthread -lm $(PLATFORM_LIBS)

TARGET = test_onnx_model
SOURCE = test_onnx_model.c

all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "🔨 Building $(TARGET)..."
	@echo "🖥️ Platform: $(UNAME_S)"
	@echo "📁 ONNX Runtime directory: $(ONNX_DIR)"
	@echo "🔗 Include path: $(INCLUDES)"
	@echo "📚 Libraries: $(LIBS)"
	@if [ ! -d "$(ONNX_DIR)" ]; then \
		echo "❌ ONNX Runtime directory not found: $(ONNX_DIR)"; \
		echo "📋 Available directories:"; \
		ls -la | grep onnxruntime || echo "No ONNX Runtime directories found"; \
	fi
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCE) $(LIBS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	@echo "🧪 Running C binary classifier tests..."
	@if [ -f model.onnx ] && [ -f vocab.json ] && [ -f scaler.json ]; then \
		echo "✅ Model files found, running full tests"; \
		./$(TARGET); \
	else \
		echo "⚠️ Model files not found, skipping tests"; \
		echo "📁 Required files: model.onnx, vocab.json, scaler.json"; \
		echo "🔧 Please ensure model files are present for testing"; \
	fi

benchmark: $(TARGET)
	@echo "📊 Running C binary classifier benchmark..."
	@if [ -f model.onnx ] && [ -f vocab.json ] && [ -f scaler.json ]; then \
		echo "✅ Model files found, running benchmark"; \
		./$(TARGET) --benchmark 100; \
	else \
		echo "⚠️ Model files not found, skipping benchmark"; \
		echo "📁 Required files: model.onnx, vocab.json, scaler.json"; \
	fi

test-ci: $(TARGET)
	@echo "🤖 Running CI tests (build verification only)..."
	@echo "✅ Binary classifier C implementation compiled successfully"
	@echo "🏗️ Build artifacts:"
	@ls -la $(TARGET)
	@echo "📋 System info:"
	@uname -a
	@gcc --version | head -1

install-deps:
	@echo "Installing dependencies..."
	@echo "Please ensure you have:"
	@echo "1. ONNX Runtime C API library"
	@echo "2. cJSON library (brew install cjson on macOS)"
	@echo "3. Required model files: model.onnx, vocab.json, scaler.json"

.PHONY: all clean test benchmark install-deps 