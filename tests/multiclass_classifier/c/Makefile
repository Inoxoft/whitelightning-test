CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
INCLUDES = -I./onnxruntime-osx-universal2-1.22.0/include
LIBS = -L./onnxruntime-osx-universal2-1.22.0/lib -lonnxruntime -lcjson -lpthread -lm

# Platform specific flags
ifeq ($(shell uname), Darwin)
    LIBS += -framework CoreFoundation -framework IOKit
else ifeq ($(shell uname), Linux)
    LIBS += -ldl
    CFLAGS += -D_GNU_SOURCE
endif

TARGET = test_onnx_model
SOURCE = test_onnx_model.c

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCE) $(LIBS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	@echo "🧪 Running C multiclass classifier tests..."
	@if [ -f model.onnx ] && [ -f vocab.json ] && [ -f scaler.json ]; then \
		echo "✅ Model files found, running full tests"; \
		./$(TARGET); \
	else \
		echo "⚠️ Model files not found, skipping tests"; \
		echo "📁 Required files: model.onnx, vocab.json, scaler.json"; \
		echo "🔧 Please ensure model files are present for testing"; \
	fi

benchmark: $(TARGET)
	@echo "📊 Running C multiclass classifier benchmark..."
	@if [ -f model.onnx ] && [ -f vocab.json ] && [ -f scaler.json ]; then \
		echo "✅ Model files found, running benchmark"; \
		./$(TARGET) --benchmark 100; \
	else \
		echo "⚠️ Model files not found, skipping benchmark"; \
		echo "📁 Required files: model.onnx, vocab.json, scaler.json"; \
	fi

test-ci: $(TARGET)
	@echo "🤖 Running CI tests (build verification only)..."
	@echo "✅ Multiclass classifier C implementation compiled successfully"
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