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
LIBS = -L$(ONNX_DIR)/lib -lonnxruntime -lpthread -lm $(PLATFORM_LIBS)

TARGET = test_onnx_model
SOURCE = test_onnx_model.c

all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "🔨 Building multiclass sigmoid C implementation..."
	@echo "🖥️ Platform: $(UNAME_S)"
	@echo "📁 ONNX Runtime directory: $(ONNX_DIR)"
	@if [ ! -d "$(ONNX_DIR)" ]; then \
		echo "❌ ONNX Runtime directory not found: $(ONNX_DIR)"; \
		echo "📋 Available directories:"; \
		ls -la | grep onnxruntime || echo "No ONNX Runtime directories found"; \
	fi
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCE) $(LIBS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	@echo "🧪 Running multiclass sigmoid C tests..."
	@if [ -f model.onnx ] && [ -f vocab.json ] && [ -f scaler.json ]; then \
		echo "✅ Model files found, running full tests"; \
		./$(TARGET); \
	else \
		echo "⚠️ Model files not found, running build verification"; \
		./$(TARGET); \
	fi

.PHONY: all clean test 